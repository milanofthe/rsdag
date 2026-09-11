//! A harmonic-balance-shaped block system (a ring of `nb` block rows,
//! three blocks per row, `b` by `b` blocks) as a block-sparse program:
//! its size, build time and evaluation time, interpreted and native.
//!
//!     cargo run --release -p rsdag-jit --example blockbench --features rsdag/synth -- 42 26

use std::time::Instant;

use rsdag::symbolic::solve::{block_pattern, plan, solve_block_planned, BlockRows};
use rsdag::{ExprId, Graph, Node, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

fn main() {
    let mut args = std::env::args().skip(1);
    let nb: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(42);
    let b: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(26);
    let reps: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(20);
    let mut g: Graph<F64> = Graph::new();
    let mut syms: Vec<SymbolId> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut rows: BlockRows = vec![Vec::new(); nb];
    let mut pattern = Vec::new();
    for i in 0..nb {
        pattern.push((i, i));
        pattern.push((i, (i + 1) % nb));
        pattern.push(((i + 1) % nb, i));
    }
    pattern.sort();
    pattern.dedup();
    for &(i, j) in &pattern {
        let mut block = Vec::with_capacity(b * b);
        for r in 0..b {
            for c in 0..b {
                let e = g.sym(&format!("a{i}_{j}_{r}_{c}"));
                if let Node::Symbol(s) = g.node(e) {
                    syms.push(*s);
                }
                vals.push(
                    ((i * 7 + j * 3 + r * 5 + c) % 11) as f64 * 0.1 - 0.5
                        + if i == j && r == c {
                            8.0 * b as f64
                        } else {
                            0.0
                        },
                );
                block.push(e);
            }
        }
        rows[i].push((j, block));
    }
    let n = nb * b;
    let rhs: Vec<ExprId> = (0..n)
        .map(|k| {
            let e = g.sym(&format!("b{k}"));
            if let Node::Symbol(s) = g.node(e) {
                syms.push(*s);
            }
            vals.push(1.0 + k as f64 * 0.01);
            e
        })
        .collect();
    let n_entries = pattern.len() * b * b;
    let t0 = Instant::now();
    let plan = plan(&block_pattern(&rows)).expect("plan");
    let solved = solve_block_planned(&mut g, &rows, b, &plan, &rhs);
    let t_graph = t0.elapsed().as_secs_f64() * 1e3;
    let mut pure = vec![true; n_entries];
    pure.resize(n_entries + n, false);
    let t1 = Instant::now();
    let tape = Tape::compile_split(&g, &solved.x, &syms, &pure);
    let t_tape = t1.elapsed().as_secs_f64() * 1e3;
    let d = tape.dump();
    let kernels = d.matches("Gemm(").count()
        + d.matches("Gemv(").count()
        + d.matches("SolveMany(").count()
        + d.matches("Solve(").count();
    let t2 = Instant::now();
    let native = NativeTape::compile(&tape).expect("compile");
    let t_native = t2.elapsed().as_secs_f64() * 1e3;
    println!("nb={nb} b={b} n={n} entries={n_entries} nodes={} ops={} kernels={kernels} fill={} build ms: graph {t_graph:.1} tape {t_tape:.1} native {t_native:.1}", g.len(), tape.n_ops(), solved.fill);
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
