//! The native backend's kernels: the matrix-vector and matrix-matrix
//! products and the dense solve, inputs read in place and gathered slots,
//! bit-identical to the interpreter.

use rsdag::{ExprId, Graph, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn the_gemv_kernel_matches_the_interpreter() {
    // `computed` 1: the matrix and the vector are computed and gathered;
    // 2: the matrix is an input run and only the vector is gathered.
    for (n, m, computed) in [(8usize, 2usize, 0u8), (40, 3, 0), (12, 1, 1), (16, 2, 2)] {
        let mut g: Graph<F64> = Graph::new();
        let a: Vec<ExprId> = (0..n * n).map(|k| g.sym(&format!("a{k}"))).collect();
        let x: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
        let b: Vec<ExprId> = (0..n * m).map(|k| g.sym(&format!("b{k}"))).collect();
        let u: Vec<ExprId> = (0..m).map(|i| g.sym(&format!("u{i}"))).collect();
        let syms: Vec<SymbolId> = (0..(n * n + n + n * m + m) as u32).map(SymbolId).collect();
        let (a, x) = match computed {
            1 => {
                let a2 = a.iter().map(|&e| g.exp(e)).collect();
                let x2 = x.iter().map(|&e| g.sin(e)).collect();
                (a2, x2)
            }
            2 => (a, x.iter().map(|&e| g.sin(e)).collect()),
            _ => (a, x),
        };
        let roots: Vec<ExprId> = (0..n)
            .map(|i| {
                let ax = g.dot(a[i * n..(i + 1) * n].to_vec(), x.clone());
                let bu = g.dot(b[i * m..(i + 1) * m].to_vec(), u.clone());
                g.add(ax, bu)
            })
            .collect();
        let tape = Tape::compile(&g, &roots, &syms);
        assert!(tape.dump().contains("Gemv"));
        let native = NativeTape::compile(&tape).expect("compile");
        let inputs: Vec<f64> = (0..syms.len())
            .map(|k| 0.03 * (k % 17) as f64 - 0.2)
            .collect();
        let (mut w1, mut o1, mut w2, mut o2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        tape.eval(&inputs, &mut w1, &mut o1);
        native.eval(&inputs, &mut w2, &mut o2);
        assert!(same(&o1, &o2), "n {n} computed {computed}");
    }
}

#[test]
fn the_gemm_kernel_matches_the_interpreter() {
    for (m, k, n, computed) in [
        (8usize, 5usize, 3usize, false),
        (33, 17, 6, false),
        (9, 4, 2, true),
    ] {
        let mut g: Graph<F64> = Graph::new();
        let a: Vec<ExprId> = (0..m * k).map(|i| g.sym(&format!("a{i}"))).collect();
        let b: Vec<ExprId> = (0..k * n).map(|i| g.sym(&format!("b{i}"))).collect();
        let syms: Vec<SymbolId> = (0..(m * k + k * n) as u32).map(SymbolId).collect();
        let (a, b) = if computed {
            let a2 = a.iter().map(|&e| g.exp(e)).collect();
            let b2 = b.iter().map(|&e| g.sin(e)).collect();
            (a2, b2)
        } else {
            (a, b)
        };
        let mut roots = Vec::new();
        for i in 0..m {
            for j in 0..n {
                let col: Vec<ExprId> = (0..k).map(|l| b[l * n + j]).collect();
                roots.push(g.dot(a[i * k..(i + 1) * k].to_vec(), col));
            }
        }
        let tape = Tape::compile(&g, &roots, &syms);
        assert!(tape.dump().contains("Gemm"), "{}", tape.dump());
        let native = NativeTape::compile(&tape).expect("compile");
        let inputs: Vec<f64> = (0..syms.len())
            .map(|i| 0.03 * (i % 17) as f64 - 0.2)
            .collect();
        let (mut w1, mut o1, mut w2, mut o2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        tape.eval(&inputs, &mut w1, &mut o1);
        native.eval(&inputs, &mut w2, &mut o2);
        assert!(same(&o1, &o2), "m {m} k {k} n {n} computed {computed}");
    }
}

#[test]
fn the_solve_kernel_matches_the_interpreter() {
    for (n, computed) in [(3usize, false), (16, false), (7, true)] {
        let mut g: Graph<F64> = Graph::new();
        let a: Vec<ExprId> = (0..n * n).map(|k| g.sym(&format!("a{k}"))).collect();
        let b: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("b{i}"))).collect();
        let syms: Vec<SymbolId> = (0..(n * n + n) as u32).map(SymbolId).collect();
        let (a, b) = if computed {
            (
                a.iter().map(|&e| g.tanh(e)).collect(),
                b.iter().map(|&e| g.exp(e)).collect(),
            )
        } else {
            (a, b)
        };
        let x = g.solve_dense(a, b);
        let tape = Tape::compile(&g, &x, &syms);
        assert!(tape.dump().contains("Solve"));
        let native = NativeTape::compile(&tape).expect("compile");
        let inputs: Vec<f64> = (0..syms.len())
            .map(|k| {
                if k < n * n && k / n == k % n {
                    5.0
                } else {
                    0.05 * (k % 11) as f64 - 0.2
                }
            })
            .collect();
        let (mut w1, mut o1, mut w2, mut o2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        tape.eval(&inputs, &mut w1, &mut o1);
        native.eval(&inputs, &mut w2, &mut o2);
        assert!(
            same(&o1, &o2),
            "n {n} computed {computed}: {o1:?} vs {o2:?}"
        );
    }
}

#[test]
fn the_multi_solve_kernel_matches_the_interpreter() {
    let (n, k) = (6usize, 3usize);
    let mut g: Graph<F64> = Graph::new();
    let a: Vec<ExprId> = (0..n * n).map(|i| g.sym(&format!("a{i}"))).collect();
    let bs: Vec<Vec<ExprId>> = (0..k)
        .map(|c| (0..n).map(|i| g.sym(&format!("b{c}_{i}"))).collect())
        .collect();
    let syms: Vec<SymbolId> = (0..(n * n + n * k) as u32).map(SymbolId).collect();
    let mut roots = Vec::new();
    for b in &bs {
        roots.extend(g.solve_dense(a.clone(), b.clone()));
    }
    let tape = Tape::compile(&g, &roots, &syms);
    assert!(tape.dump().contains("SolveMany"), "{}", tape.dump());
    let native = NativeTape::compile(&tape).expect("compile");
    let inputs: Vec<f64> = (0..syms.len())
        .map(|i| {
            if i < n * n && (i / n) == (i % n) {
                4.0 + i as f64 * 0.01
            } else {
                0.2 * ((i % 5) as f64 - 2.0)
            }
        })
        .collect();
    let (mut w1, mut o1, mut w2, mut o2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    tape.eval(&inputs, &mut w1, &mut o1);
    native.eval(&inputs, &mut w2, &mut o2);
    assert!(same(&o1, &o2));
}

/// Independent solves of one shape run as one batch (up to sixteen
/// unknowns): interpreted and natively, each solution bit-identical to its system solved alone. The
/// systems are plain random matrices (their eliminations swap rows), two
/// of them with a second right-hand side; a solve that reads another's
/// solution is no part of its batch.
#[test]
fn a_batch_of_solves_matches_each_solve_alone() {
    for (n, count, computed) in [
        (3usize, 5usize, false),
        (4, 8, true),
        (6, 3, false),
        (16, 4, false),
        (17, 4, false),
    ] {
        let mut g: Graph<F64> = Graph::new();
        let mut syms = Vec::new();
        let mut sym = |g: &mut Graph<F64>| {
            syms.push(SymbolId(syms.len() as u32));
            g.sym(&format!("s{}", syms.len()))
        };
        let mut systems = Vec::new();
        let mut roots = Vec::new();
        for c in 0..count {
            let a: Vec<ExprId> = (0..n * n).map(|_| sym(&mut g)).collect();
            let b: Vec<ExprId> = (0..n).map(|_| sym(&mut g)).collect();
            let (a, b) = if computed {
                (a.iter().map(|&e| g.tanh(e)).collect(), b)
            } else {
                (a, b)
            };
            roots.extend(g.solve_dense(a.clone(), b.clone()));
            if c < 2 {
                let b2: Vec<ExprId> = b.iter().map(|&e| g.sin(e)).collect();
                roots.extend(g.solve_dense(a.clone(), b2));
            }
            systems.push(a);
        }
        // A solve of the first solution against the last matrix: one level up.
        let first: Vec<ExprId> = roots[..n].to_vec();
        roots.extend(g.solve_dense(systems[count - 1].clone(), first));
        let tape = Tape::compile(&g, &roots, &syms);
        let d = tape.dump();
        // The systems with two right-hand sides are one batch, those with
        // one another (when there are two), the dependent solve alone;
        // past sixteen unknowns nothing batches.
        if n <= rsdag::semantics::SOLVE_BATCH_MAX_N {
            assert!(d.contains(&format!("SolveBatch(2 of {n}x{n}")), "{d}");
            if count >= 4 {
                let rest = format!("SolveBatch({} of {n}x{n}", count - 2);
                assert!(d.contains(&rest), "{d}");
            }
        } else {
            assert!(!d.contains("SolveBatch"), "{d}");
        }
        assert!(d.contains(&format!("Solve({n}x{n}")), "{d}");
        let native = NativeTape::compile(&tape).expect("compile");
        let inputs: Vec<f64> = (0..syms.len())
            .map(|k| ((k * 37 + 11) % 23) as f64 * 0.1 - 1.1)
            .collect();
        let (mut w1, mut o1, mut w2, mut o2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        tape.eval(&inputs, &mut w1, &mut o1);
        native.eval(&inputs, &mut w2, &mut o2);
        assert!(same(&o1, &o2), "n {n} count {count}: native differs");
        // Each solve compiled alone.
        let mut at = 0;
        let mut i = 0;
        while at < roots.len() {
            let alone = Tape::compile(&g, &roots[at..at + n], &syms);
            assert!(!alone.dump().contains("SolveBatch"));
            let (mut w, mut o) = (Vec::new(), Vec::new());
            alone.eval(&inputs, &mut w, &mut o);
            assert!(
                same(&o, &o1[at..at + n]),
                "n {n} count {count}: solve {i} differs"
            );
            at += n;
            i += 1;
        }
    }
}
