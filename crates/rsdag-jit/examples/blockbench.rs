//! A harmonic-balance-shaped block system (a ring of `nb` block rows,
//! three blocks per row, `b` by `b` blocks) as a block-sparse program:
//! its size, build time and evaluation time, interpreted and native. The
//! couplings are dense or diagonal, the scalar real or complex (a complex
//! system of `b` by `b` blocks against a real one of `2b` by `2b`):
//!
//!     cargo run --release -p rsdag-jit --example blockbench -- 42 13 dense real
//!     cargo run --release -p rsdag-jit --example blockbench -- 42 13 diag complex

use std::time::Instant;

use rsdag::symbolic::solve::{block_pattern, plan, solve_block_planned, Block, BlockRows, Cx, Num};
use rsdag::{ExprId, Graph, Node, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

fn sym(g: &mut Graph<F64>, name: &str, syms: &mut Vec<SymbolId>) -> ExprId {
    let e = g.sym(name);
    if let Node::Symbol(s) = g.node(e) {
        syms.push(*s);
    }
    e
}

/// The system over the scalar `N`: `entry` makes one scalar of a block.
#[allow(clippy::too_many_arguments)]
fn build<N: Num>(
    g: &mut Graph<F64>,
    syms: &mut Vec<SymbolId>,
    vals: &mut Vec<f64>,
    nb: usize,
    b: usize,
    diag: bool,
    mut entry: impl FnMut(
        &mut Graph<F64>,
        &mut Vec<SymbolId>,
        &mut Vec<f64>,
        usize,
        usize,
        usize,
        usize,
    ) -> N,
    mut rhs_entry: impl FnMut(&mut Graph<F64>, &mut Vec<SymbolId>, &mut Vec<f64>, usize) -> N,
) -> (BlockRows<N>, Vec<N>) {
    let mut rows: BlockRows<N> = vec![Vec::new(); nb];
    let mut pattern = Vec::new();
    for i in 0..nb {
        pattern.push((i, i));
        pattern.push((i, (i + 1) % nb));
        pattern.push(((i + 1) % nb, i));
    }
    pattern.sort();
    pattern.dedup();
    for &(i, j) in &pattern {
        if diag && i != j {
            let d = (0..b).map(|r| entry(g, syms, vals, i, j, r, r)).collect();
            rows[i].push((j, Block::Diag(d)));
        } else {
            let mut block = Vec::with_capacity(b * b);
            for r in 0..b {
                for c in 0..b {
                    block.push(entry(g, syms, vals, i, j, r, c));
                }
            }
            rows[i].push((j, Block::Dense(block)));
        }
    }
    let rhs = (0..nb * b).map(|k| rhs_entry(g, syms, vals, k)).collect();
    (rows, rhs)
}

fn value(i: usize, j: usize, r: usize, c: usize, b: usize) -> f64 {
    ((i * 7 + j * 3 + r * 5 + c) % 11) as f64 * 0.1 - 0.5
        + if i == j && r == c {
            8.0 * b as f64
        } else {
            0.0
        }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let nb: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(42);
    let b: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(13);
    let diag = args.next().as_deref() == Some("diag");
    let complex = args.next().as_deref() == Some("complex");
    let reps: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(20);
    let mut g: Graph<F64> = Graph::new();
    let mut syms: Vec<SymbolId> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let t0 = Instant::now();
    let (roots, pattern_len, n, fill) = if complex {
        let (rows, rhs) = build(
            &mut g,
            &mut syms,
            &mut vals,
            nb,
            b,
            diag,
            |g, syms, vals, i, j, r, c| {
                let re = sym(g, &format!("a{i}_{j}_{r}_{c}r"), syms);
                let im = sym(g, &format!("a{i}_{j}_{r}_{c}i"), syms);
                vals.push(value(i, j, r, c, b));
                vals.push(0.5 * value(j, i, c, r, b));
                Cx::new(re, im)
            },
            |g, syms, vals, k| {
                let re = sym(g, &format!("b{k}r"), syms);
                let im = sym(g, &format!("b{k}i"), syms);
                vals.push(1.0 + k as f64 * 0.01);
                vals.push(0.3);
                Cx::new(re, im)
            },
        );
        let plan = plan(&block_pattern(&rows)).expect("plan");
        let solved = solve_block_planned(&mut g, &rows, b, &plan, &rhs);
        let roots: Vec<ExprId> = solved.x.iter().flat_map(|c| [c.re, c.im]).collect();
        (
            roots,
            rows.iter().map(|r| r.len()).sum::<usize>(),
            2 * nb * b,
            solved.fill,
        )
    } else {
        let (rows, rhs) = build(
            &mut g,
            &mut syms,
            &mut vals,
            nb,
            b,
            diag,
            |g, syms, vals, i, j, r, c| {
                vals.push(value(i, j, r, c, b));
                sym(g, &format!("a{i}_{j}_{r}_{c}"), syms)
            },
            |g, syms, vals, k| {
                vals.push(1.0 + k as f64 * 0.01);
                sym(g, &format!("b{k}"), syms)
            },
        );
        let plan = plan(&block_pattern(&rows)).expect("plan");
        let solved = solve_block_planned(&mut g, &rows, b, &plan, &rhs);
        (
            solved.x,
            rows.iter().map(|r| r.len()).sum::<usize>(),
            nb * b,
            solved.fill,
        )
    };
    let n_entries = vals.len() - n;
    let t_graph = t0.elapsed().as_secs_f64() * 1e3;
    let mut pure = vec![true; n_entries];
    pure.resize(n_entries + n, false);
    let t1 = Instant::now();
    let tape = Tape::compile_split(&g, &roots, &syms, &pure);
    let t_tape = t1.elapsed().as_secs_f64() * 1e3;
    let d = tape.dump();
    let kernels = d.matches("Gemm(").count()
        + d.matches("Gemv(").count()
        + d.matches("SolveMany(").count()
        + d.matches("Solve(").count();
    // Flops by kernel kind, from the dump: a product 2mkn, a solve of k
    // right-hand sides 2/3 n^3 + 2 n^2 k, a scalar op one.
    let (mut f_gemm, mut f_solve, mut f_scalar) = (0f64, 0f64, 0f64);
    for line in d.lines() {
        let Some(text) = line.split(" <- ").nth(1) else {
            continue;
        };
        // The "AxB" tokens of a kernel's text, and the count before "rhs".
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
        if let Some(rest) = text.strip_prefix("Gemm(") {
            let v = dims(rest.split(" -> ").next().unwrap());
            f_gemm += 2.0 * v[0] * v[1] * v[2]; // m, k, n
        } else if let Some(rest) = text.strip_prefix("Gemv(") {
            let v = dims(rest.split(" -> ").next().unwrap());
            f_gemm += 2.0 * v[0] * v[1];
        } else if let Some(rest) = text.strip_prefix("SolveMany(") {
            let v = dims(rest.split(" -> ").next().unwrap());
            let (n, k) = (v[0], v[v.len() - 1]);
            f_solve += 2.0 / 3.0 * n * n * n + 2.0 * n * n * k;
        } else if let Some(rest) = text.strip_prefix("Solve(") {
            let v = dims(rest.split(" -> ").next().unwrap());
            f_solve += 2.0 / 3.0 * v[0] * v[0] * v[0] + 2.0 * v[0] * v[0];
        } else {
            f_scalar += 1.0;
        }
    }
    println!(
        "flops: products {:.2}M solves {:.2}M scalar ops {:.0}k",
        f_gemm / 1e6,
        f_solve / 1e6,
        f_scalar / 1e3
    );
    let t2 = Instant::now();
    let native = NativeTape::compile(&tape).expect("compile");
    let t_native = t2.elapsed().as_secs_f64() * 1e3;
    println!(
        "nb={nb} b={b} {} {} blocks={pattern_len} n={n} entries={n_entries} nodes={} ops={} (prolog {}) kernels={kernels} fill={fill} build ms: graph {t_graph:.1} tape {t_tape:.1} native {t_native:.1}",
        if diag { "diag" } else { "dense" },
        if complex { "complex" } else { "real" },
        g.len(),
        tape.n_ops(),
        tape.prolog_len()
    );
    let time = |f: &mut dyn FnMut(), reps: usize| -> f64 {
        f();
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        t.elapsed().as_secs_f64() * 1e3 / reps as f64
    };
    let (mut w, mut o) = (Vec::new(), Vec::new());
    let t_i = time(&mut || tape.eval(&vals, &mut w, &mut o), 5);
    let (mut wn, mut on) = (Vec::new(), Vec::new());
    let t_n = time(&mut || native.eval(&vals, &mut wn, &mut on), reps);
    let t_np = time(&mut || native.eval_prolog(&vals, &mut wn), 20);
    let t_nm = time(&mut || native.eval_main(&vals, &mut wn, &mut on), 50);
    let same = o.iter().zip(&on).all(|(a, b)| a.to_bits() == b.to_bits());
    println!("eval ms: interp {t_i:.2} native {t_n:.3} (prolog {t_np:.3}, main {t_nm:.3}) bits equal {same}");
}
