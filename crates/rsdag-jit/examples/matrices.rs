//! Small dense matrices as models have them (controllers, filters, state
//! space, macromodels): `A x`, `A B` and the solve `A \ b` of `n` by `n`
//! inputs, compiled to a tape (the products fused into kernels) and to
//! native code. Time per call, interpreted and native, as CSV on stdout
//! (`docs/bench/data/matrices.csv` for the README figures).
//!
//!     cargo run --release -p rsdag-jit --example matrices > docs/bench/data/matrices.csv

use std::time::Instant;

use rsdag::{ExprId, Graph, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

/// Seconds per call: the best of five batches of at least 20 ms.
fn per_call(mut f: impl FnMut()) -> f64 {
    let mut reps = 1usize;
    let mut dt;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        dt = t.elapsed().as_secs_f64();
        if dt > 0.02 {
            break;
        }
        reps *= 4;
    }
    let mut best = dt / reps as f64;
    for _ in 0..4 {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    best
}

/// The graph of `kind` at size `n`, its roots and its inputs' values.
pub fn build(kind: &str, n: usize) -> (Graph<F64>, Vec<ExprId>, Vec<SymbolId>, Vec<f64>) {
    let mut g: Graph<F64> = Graph::new();
    let mut syms = Vec::new();
    let mut sym = |g: &mut Graph<F64>, name: String| {
        syms.push(SymbolId(syms.len() as u32));
        g.sym(&name)
    };
    let a: Vec<ExprId> = (0..n * n).map(|k| sym(&mut g, format!("a{k}"))).collect();
    // A diagonally dominant matrix, so the solve is well posed.
    let mut vals: Vec<f64> = (0..n * n)
        .map(|k| {
            if k / n == k % n {
                4.0
            } else {
                0.1 * ((k % 7) as f64 - 3.0)
            }
        })
        .collect();
    let roots = match kind {
        "gemv" | "solve" => {
            let x: Vec<ExprId> = (0..n).map(|i| sym(&mut g, format!("x{i}"))).collect();
            vals.extend((0..n).map(|i| 1.0 - 0.05 * i as f64));
            if kind == "gemv" {
                (0..n)
                    .map(|i| g.dot(a[i * n..(i + 1) * n].to_vec(), x.clone()))
                    .collect()
            } else {
                g.solve_dense(a, x)
            }
        }
        _ => {
            let b: Vec<ExprId> = (0..n * n).map(|k| sym(&mut g, format!("b{k}"))).collect();
            vals.extend((0..n * n).map(|k| 1.0 - 0.01 * (k % 9) as f64));
            (0..n * n)
                .map(|e| {
                    let (i, j) = (e / n, e % n);
                    g.dot(
                        a[i * n..(i + 1) * n].to_vec(),
                        b[j * n..(j + 1) * n].to_vec(),
                    )
                })
                .collect()
        }
    };
    (g, roots, syms, vals)
}

fn main() {
    println!("kind,n,interp_ns,native_ns");
    for kind in ["gemv", "gemm", "solve"] {
        for n in [2usize, 3, 4, 6, 8, 12, 16, 24, 32] {
            let (g, roots, syms, vals) = build(kind, n);
            let tape = Tape::compile(&g, &roots, &syms);
            let native = NativeTape::compile(&tape).expect("native code");
            let (mut w, mut o) = (Vec::new(), Vec::new());
            tape.eval(&vals, &mut w, &mut o);
            let t_interp = per_call(|| tape.eval(&vals, &mut w, &mut o));
            let (mut wn, mut on) = (Vec::new(), Vec::new());
            native.eval(&vals, &mut wn, &mut on);
            assert!(
                o.iter().zip(&on).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{kind} {n}: native differs from the interpreter"
            );
            let t_native = per_call(|| native.eval(&vals, &mut wn, &mut on));
            println!("{kind},{n},{:.1},{:.1}", t_interp * 1e9, t_native * 1e9);
        }
    }
}
