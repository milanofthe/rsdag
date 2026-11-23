//! Small dense matrices as models have them (controllers, filters, state
//! space, macromodels): `A x`, `A B` and the solve `A \ b` of `n` by `n`
//! inputs, compiled to a tape (the products fused into kernels) and to
//! native code; once for one instance and once for sixteen instances of
//! one shape, each with its own matrices (the tape batches them). Time per
//! instance, interpreted and native, as CSV on stdout
//! (`docs/bench/data/matrices.csv` for the README figures).
//!
//!     cargo run --release -p rsdag-jit --example matrices > docs/bench/data/matrices.csv

use rsdag::Program;
mod common;
use common::per_call;

use rsdag::{ExprId, Graph, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

/// The graph of `count` instances of `kind` at size `n`, its roots and its
/// inputs' values.
fn build(kind: &str, n: usize, count: usize) -> (Graph<F64>, Vec<ExprId>, Vec<SymbolId>, Vec<f64>) {
    let mut g: Graph<F64> = Graph::new();
    let mut syms = Vec::new();
    let mut vals = Vec::new();
    let mut roots = Vec::new();
    let mut sym = |g: &mut Graph<F64>, vals: &mut Vec<f64>, v: f64| {
        syms.push(SymbolId(syms.len() as u32));
        vals.push(v);
        g.sym(&format!("s{}", syms.len()))
    };
    for c in 0..count {
        // A diagonally dominant matrix, so the solve is well posed.
        let a: Vec<ExprId> = (0..n * n)
            .map(|k| {
                let v = if k / n == k % n {
                    4.0
                } else {
                    0.1 * (((k + c) % 7) as f64 - 3.0)
                };
                sym(&mut g, &mut vals, v)
            })
            .collect();
        match kind {
            "gemv" | "solve" => {
                let x: Vec<ExprId> = (0..n)
                    .map(|i| sym(&mut g, &mut vals, 1.0 - 0.05 * i as f64))
                    .collect();
                if kind == "gemv" {
                    for i in 0..n {
                        roots.push(g.dot(a[i * n..(i + 1) * n].to_vec(), x.clone()));
                    }
                } else {
                    roots.extend(g.solve_dense(a, x));
                }
            }
            _ => {
                let b: Vec<ExprId> = (0..n * n)
                    .map(|k| sym(&mut g, &mut vals, 1.0 - 0.01 * (k % 9) as f64))
                    .collect();
                for e in 0..n * n {
                    let (i, j) = (e / n, e % n);
                    let row = a[i * n..(i + 1) * n].to_vec();
                    roots.push(g.dot(row, b[j * n..(j + 1) * n].to_vec()));
                }
            }
        }
    }
    (g, roots, syms, vals)
}

fn main() {
    let _run = common::Run::new("matrices", vec![]);
    println!("kind,n,instances,interp_ns,native_ns");
    for count in [1usize, 16] {
        for kind in ["gemv", "gemm", "solve"] {
            for n in [2usize, 3, 4, 6, 8, 12, 16, 24, 32] {
                let (g, roots, syms, vals) = build(kind, n, count);
                let tape = Tape::compile(&g, &roots, &syms);
                let native = NativeTape::compile(&tape).expect("native code");
                // the slice forms, over buffers sized once, as a solver
                // calls them in its loop
                let (mut w, mut o) = (vec![0.0; tape.work_len()], vec![0.0; tape.out_len()]);
                tape.eval_into(&vals, &mut w, &mut o);
                let t_interp = per_call(|| tape.eval_into(&vals, &mut w, &mut o));
                let (mut wn, mut on) = (vec![0.0; native.work_len()], vec![0.0; native.out_len()]);
                native.eval_into(&vals, &mut wn, &mut on);
                assert!(
                    o.iter().zip(&on).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{kind} {n} x{count}: native differs from the interpreter"
                );
                let t_native = per_call(|| native.eval_into(&vals, &mut wn, &mut on));
                let per = 1e9 / count as f64;
                println!(
                    "{kind},{n},{count},{:.1},{:.1}",
                    t_interp * per,
                    t_native * per
                );
            }
        }
    }
}
