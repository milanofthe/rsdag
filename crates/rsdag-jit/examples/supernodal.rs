//! A harmonic-balance-shaped complex system (a ring of `nb` groups of `b`
//! unknowns, a dense group block on the diagonal of `dense_every`-th
//! group, diagonal couplings otherwise) as three programs: the scalar
//! elimination, the block elimination over the groups, and the supernodal
//! elimination over the scalar plan's panels. Size, build and native
//! factorization time of each.
//!
//!     cargo run --release -p rsdag-jit --example supernodal -- 42 13 1
//!     cargo run --release -p rsdag-jit --example supernodal -- pattern.txt 13
//!
//! A pattern file (`n`, then `row col` lines) replaces the ring.

use std::time::Instant;

use rsdag::symbolic::solve::{
    block_pattern, pattern_of, plan, solve_block_planned, solve_planned, solve_supernodal_planned,
    supernodes, Block, BlockRows, Cx, Num,
};
use rsdag::{ExprId, Graph, Node, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

struct Sys {
    g: Graph<F64>,
    syms: Vec<SymbolId>,
    vals: Vec<f64>,
    /// Scalar rows, and the same entries as group blocks.
    rows: Vec<Vec<(usize, Cx)>>,
    blocks: BlockRows<Cx>,
    rhs: Vec<Cx>,
    n: usize,
}

/// A system from a pattern file (`n` then `row col` lines, original
/// coordinates), diagonally dominant complex values; the groups are
/// unknowns `k / b`, blocks over them dense where the pattern has an
/// off-diagonal entry within the block, diagonal otherwise.
fn build_from_file(path: &str, b: usize) -> Sys {
    let text = std::fs::read_to_string(path).expect("pattern file");
    let mut lines = text.lines();
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let mut pos: Vec<(usize, usize)> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace().map(|t| t.parse::<usize>().unwrap());
            (it.next().unwrap(), it.next().unwrap())
        })
        .collect();
    pos.sort_unstable();
    pos.dedup();
    assert_eq!(n % b, 0, "the group size divides n");
    let nb = n / b;
    let mut g: Graph<F64> = Graph::new();
    let mut syms = Vec::new();
    let mut vals = Vec::new();
    let mut rows: Vec<Vec<(usize, Cx)>> = vec![Vec::new(); n];
    let mut by_block: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize, Cx)>> =
        std::collections::BTreeMap::new();
    for &(i, j) in &pos {
        let re = g.sym(&format!("a{i}_{j}r"));
        let im = g.sym(&format!("a{i}_{j}i"));
        for e in [re, im] {
            if let Node::Symbol(s) = g.node(e) {
                syms.push(*s);
            }
        }
        let big = if i == j { 4.0 * b as f64 } else { 0.0 };
        vals.push(((i * 7 + j * 3) % 11) as f64 * 0.1 - 0.5 + big);
        vals.push(((i * 5 + j) % 7) as f64 * 0.1 - 0.3 + 0.5 * big);
        let e = Cx::new(re, im);
        rows[i].push((j, e));
        by_block
            .entry((i / b, j / b))
            .or_default()
            .push((i % b, j % b, e));
    }
    let mut blocks: BlockRows<Cx> = vec![Vec::new(); nb];
    let zero = Cx::zero(&mut g);
    for ((bi, bj), list) in by_block {
        if list.iter().all(|&(r, c, _)| r == c) {
            let mut d = vec![zero; b];
            for &(r, _, e) in &list {
                d[r] = e;
            }
            blocks[bi].push((bj, Block::Diag(d)));
        } else {
            let mut v = vec![zero; b * b];
            for &(r, c, e) in &list {
                v[r * b + c] = e;
            }
            blocks[bi].push((bj, Block::Dense(v)));
        }
    }
    let mut rhs = Vec::new();
    for k in 0..n {
        let re = g.sym(&format!("b{k}r"));
        let im = g.sym(&format!("b{k}i"));
        for e in [re, im] {
            if let Node::Symbol(s) = g.node(e) {
                syms.push(*s);
            }
        }
        vals.push(1.0 + 0.01 * k as f64);
        vals.push(0.2);
        rhs.push(Cx::new(re, im));
    }
    Sys {
        g,
        syms,
        vals,
        rows,
        blocks,
        rhs,
        n,
    }
}

fn build(nb: usize, b: usize, dense_every: usize) -> Sys {
    let mut g: Graph<F64> = Graph::new();
    let mut syms = Vec::new();
    let mut vals = Vec::new();
    let n = nb * b;
    let mut rows: Vec<Vec<(usize, Cx)>> = vec![Vec::new(); n];
    let mut blocks: BlockRows<Cx> = vec![Vec::new(); nb];
    let entry = |g: &mut Graph<F64>,
                 syms: &mut Vec<SymbolId>,
                 vals: &mut Vec<f64>,
                 i: usize,
                 j: usize|
     -> Cx {
        let re = g.sym(&format!("a{i}_{j}r"));
        let im = g.sym(&format!("a{i}_{j}i"));
        for e in [re, im] {
            if let Node::Symbol(s) = g.node(e) {
                syms.push(*s);
            }
        }
        let big = if i == j { 4.0 * b as f64 } else { 0.0 };
        vals.push(((i * 7 + j * 3) % 11) as f64 * 0.1 - 0.5 + big);
        vals.push(((i * 5 + j) % 7) as f64 * 0.1 - 0.3 + 0.5 * big);
        Cx::new(re, im)
    };
    for gi in 0..nb {
        let dense = gi % dense_every == 0;
        // The group's own block.
        if dense {
            let mut blk = Vec::with_capacity(b * b);
            for r in 0..b {
                for c in 0..b {
                    let e = entry(&mut g, &mut syms, &mut vals, gi * b + r, gi * b + c);
                    rows[gi * b + r].push((gi * b + c, e));
                    blk.push(e);
                }
            }
            blocks[gi].push((gi, Block::Dense(blk)));
        } else {
            let mut d = Vec::with_capacity(b);
            for r in 0..b {
                let e = entry(&mut g, &mut syms, &mut vals, gi * b + r, gi * b + r);
                rows[gi * b + r].push((gi * b + r, e));
                d.push(e);
            }
            blocks[gi].push((gi, Block::Diag(d)));
        }
        // Diagonal couplings to the next group, both ways.
        let gj = (gi + 1) % nb;
        let (mut d1, mut d2) = (Vec::new(), Vec::new());
        for r in 0..b {
            let e = entry(&mut g, &mut syms, &mut vals, gi * b + r, gj * b + r);
            rows[gi * b + r].push((gj * b + r, e));
            d1.push(e);
            let e = entry(&mut g, &mut syms, &mut vals, gj * b + r, gi * b + r);
            rows[gj * b + r].push((gi * b + r, e));
            d2.push(e);
        }
        blocks[gi].push((gj, Block::Diag(d1)));
        blocks[gj].push((gi, Block::Diag(d2)));
    }
    let n_entries = vals.len();
    let mut rhs = Vec::new();
    for k in 0..n {
        let re = g.sym(&format!("b{k}r"));
        let im = g.sym(&format!("b{k}i"));
        for e in [re, im] {
            if let Node::Symbol(s) = g.node(e) {
                syms.push(*s);
            }
        }
        vals.push(1.0 + 0.01 * k as f64);
        vals.push(0.2);
        rhs.push(Cx::new(re, im));
    }
    let _ = n_entries;
    Sys {
        g,
        syms,
        vals,
        rows,
        blocks,
        rhs,
        n,
    }
}

fn measure(
    name: &str,
    g: &Graph<F64>,
    roots: &[ExprId],
    syms: &[SymbolId],
    vals: &[f64],
    n_entries: usize,
    fill: usize,
) -> Vec<f64> {
    let n_rhs = vals.len() - n_entries;
    let mut pure = vec![true; n_entries];
    pure.resize(n_entries + n_rhs, false);
    let t = Instant::now();
    let tape = Tape::compile_split(g, roots, syms, &pure);
    let t_tape = t.elapsed().as_secs_f64() * 1e3;
    let d = tape.dump();
    if let Ok(dir) = std::env::var("RSDAG_DUMP") {
        std::fs::write(format!("{dir}/{name}.txt"), &d).unwrap();
    }
    let kernels = d.matches("Gemm(").count()
        + d.matches("Gemv(").count()
        + d.matches("SolveMany(").count()
        + d.matches("Solve(").count();
    // Flops by kernel kind from the dump: a product 2mkn, a solve of k
    // right-hand sides 2/3 n^3 + 2 n^2 k; the rest scalar ops.
    let (mut f_gemm, mut f_solve, mut f_scalar) = (0f64, 0f64, 0f64);
    for line in d.lines() {
        let Some(text) = line.split(" <- ").nth(1) else {
            continue;
        };
        let dims = |s: &str| -> Vec<f64> {
            let mut v = Vec::new();
            let toks: Vec<&str> = s.split([' ', '(', ',']).collect();
            for (t, tok) in toks.iter().enumerate() {
                if let Some((a, b)) = tok.split_once('x') {
                    if let (Ok(a), Ok(b)) = (a.parse::<f64>(), b.parse::<f64>()) {
                        v.push(a);
                        v.push(b);
                    }
                } else if *tok == "rhs" && t > 0 {
                    v.push(toks[t - 1].parse().unwrap());
                }
            }
            v
        };
        let head = text.split(" -> ").next().unwrap();
        if let Some(rest) = head.strip_prefix("Gemm(") {
            let v = dims(rest);
            f_gemm += 2.0 * v[0] * v[1] * v[2];
        } else if let Some(rest) = head.strip_prefix("Gemv(") {
            let v = dims(rest);
            f_gemm += 2.0 * v[0] * v[1];
        } else if let Some(rest) = head.strip_prefix("SolveMany(") {
            let v = dims(rest);
            let (n, k) = (v[0], v[v.len() - 1]);
            f_solve += 2.0 / 3.0 * n * n * n + 2.0 * n * n * k;
        } else if let Some(rest) = head.strip_prefix("Solve(") {
            let v = dims(rest);
            f_solve += 2.0 / 3.0 * v[0] * v[0] * v[0] + 2.0 * v[0] * v[0];
        } else {
            f_scalar += 1.0;
        }
    }
    print!(
        "flops: products {:5.2}M solves {:5.2}M scalar {:4.0}k  ",
        f_gemm / 1e6,
        f_solve / 1e6,
        f_scalar / 1e3
    );
    let t = Instant::now();
    let native = NativeTape::compile(&tape).expect("compile");
    let t_native = t.elapsed().as_secs_f64() * 1e3;
    let time = |f: &mut dyn FnMut(), reps: usize| -> f64 {
        f();
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        t.elapsed().as_secs_f64() * 1e3 / reps as f64
    };
    let (mut w, mut o) = (Vec::new(), Vec::new());
    let t_p = time(&mut || native.eval_prolog(vals, &mut w), 20);
    let t_m = time(&mut || native.eval_main(vals, &mut w, &mut o), 50);
    println!(
        "{name:11} ops={:7} (prolog {:7}) kernels={kernels:4} fill={fill:6} build ms: tape {t_tape:6.1} native {t_native:5.1}   factor {t_p:.3} ms  solve {t_m:.3} ms",
        tape.n_ops(),
        tape.prolog_len()
    );
    o
}

fn main() {
    let mut args = std::env::args().skip(1);
    let first = args.next().unwrap_or_else(|| "42".to_string());
    let (file, nb) = match first.parse::<usize>() {
        Ok(nb) => (None, nb),
        Err(_) => (Some(first), 0),
    };
    let b: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(13);
    let dense_every: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(1);
    let build = |()| match &file {
        Some(path) => build_from_file(path, b),
        None => build(nb, b, dense_every),
    };
    let n_entries = |s: &Sys| s.vals.len() - 2 * s.n;
    // Scalar.
    let mut s = build(());
    let pattern = pattern_of(
        &s.rows
            .iter()
            .map(|r| r.iter().map(|&(j, e)| (j, e.re)).collect())
            .collect(),
    );
    let splan = plan(&pattern).expect("plan");
    println!(
        "b={b} n={} entries={} plan: fill {} flops {}",
        s.n,
        n_entries(&s) / 2,
        splan.cost.fill,
        splan.cost.flops
    );
    let t = Instant::now();
    let solved = solve_planned(&mut s.g, &s.rows, &splan, &s.rhs);
    let t_graph = t.elapsed().as_secs_f64() * 1e3;
    let roots: Vec<ExprId> = solved
        .x
        .iter()
        .flat_map(|c| [c.re, c.im])
        .chain([solved.pivots_ok])
        .collect();
    print!("graph {t_graph:6.1} ms  ");
    let x_scalar = measure(
        "scalar",
        &s.g,
        &roots,
        &s.syms,
        &s.vals,
        n_entries(&s),
        solved.fill,
    );
    // Block over the groups.
    let mut s = build(());
    let bplan = plan(&block_pattern(&s.blocks)).expect("plan");
    let t = Instant::now();
    let solved = solve_block_planned(&mut s.g, &s.blocks, b, &bplan, &s.rhs);
    let t_graph = t.elapsed().as_secs_f64() * 1e3;
    let roots: Vec<ExprId> = solved
        .x
        .iter()
        .flat_map(|c| [c.re, c.im])
        .chain([solved.pivots_ok])
        .collect();
    print!("graph {t_graph:6.1} ms  ");
    let x_block = measure(
        "block",
        &s.g,
        &roots,
        &s.syms,
        &s.vals,
        n_entries(&s),
        solved.fill,
    );
    // Supernodal over the scalar plan.
    let mut s = build(());
    let t = Instant::now();
    let sn = supernodes(&pattern, &splan);
    let solved = solve_supernodal_planned(&mut s.g, &s.rows, &splan, &sn, &s.rhs);
    let t_graph = t.elapsed().as_secs_f64() * 1e3;
    let roots: Vec<ExprId> = solved
        .x
        .iter()
        .flat_map(|c| [c.re, c.im])
        .chain([solved.pivots_ok])
        .collect();
    print!("graph {t_graph:6.1} ms  ");
    let x_super = measure(
        "supernodal",
        &s.g,
        &roots,
        &s.syms,
        &s.vals,
        n_entries(&s),
        solved.fill,
    );
    let mut widths = sn.widths();
    widths.sort_unstable();
    println!(
        "panels: {} of widths {}..{} (median {}), panel share {:.2}",
        sn.n_panels(),
        widths[0],
        widths[widths.len() - 1],
        widths[widths.len() / 2],
        sn.panel_share()
    );
    let n = 2 * s.n;
    let err = |a: &[f64], b: &[f64]| {
        a[..n]
            .iter()
            .zip(&b[..n])
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    };
    println!(
        "max diff: block vs scalar {:.2e}, supernodal vs scalar {:.2e}; guards {} {} {}",
        err(&x_block, &x_scalar),
        err(&x_super, &x_scalar),
        x_scalar[n],
        x_block[n],
        x_super[n]
    );
}
