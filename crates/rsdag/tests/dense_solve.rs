//! The dense solve as a node: one pivoting kernel for all components,
//! exact against the numeric twin, differentiable in both modes, and the
//! route `Builder::solve` takes for a dense matrix.

use std::collections::HashMap;

use rsdag::synth::Spec;
use rsdag::{Builder, ExprId, Graph, Node, Numeric, SymbolId, Tape, F64};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

/// A random diagonally dominant system over symbols, with its values.
fn system(
    g: &mut Graph<F64>,
    n: usize,
    seed: u64,
) -> (Vec<ExprId>, Vec<ExprId>, Vec<SymbolId>, Vec<f64>) {
    let mut rng = Spec::new(seed).rng();
    let (mut syms, mut vals) = (Vec::new(), Vec::new());
    let a: Vec<ExprId> = (0..n * n)
        .map(|k| {
            let e = g.sym(&format!("a{k}"));
            syms.push(sym(g, e));
            vals.push(if k / n == k % n {
                4.0 + rng.val()
            } else {
                rng.val() - 0.5
            });
            e
        })
        .collect();
    let b: Vec<ExprId> = (0..n)
        .map(|i| {
            let e = g.sym(&format!("b{i}"));
            syms.push(sym(g, e));
            vals.push(rng.val());
            e
        })
        .collect();
    (a, b, syms, vals)
}

#[test]
fn the_kernel_is_the_numeric_twin_to_the_bit() {
    for (n, seed) in [(2usize, 1u64), (5, 2), (12, 3), (40, 4)] {
        let mut g: Graph<F64> = Graph::new();
        let (a, b, syms, vals) = system(&mut g, n, seed);
        let x = g.solve_dense(a.clone(), b.clone());
        let tape = Tape::compile(&g, &x, &syms);
        assert!(tape.dump().contains("Solve"), "{}", tape.dump());
        let (mut w, mut o) = (Vec::new(), Vec::new());
        tape.eval(&vals, &mut w, &mut o);
        let dense: Vec<Vec<f64>> = (0..n).map(|i| vals[i * n..(i + 1) * n].to_vec()).collect();
        let want = Numeric.solve(&dense, &vals[n * n..]);
        for (k, (p, q)) in want.iter().zip(&o).enumerate() {
            assert_eq!(p.to_bits(), q.to_bits(), "n {n} x[{k}]: {p} vs {q}");
        }
        // And the arena agrees.
        let env: HashMap<SymbolId, f64> = syms.iter().copied().zip(vals.iter().copied()).collect();
        let arena = rsdag::eval(&g, &x, &env);
        for (k, (p, q)) in want.iter().zip(&arena).enumerate() {
            assert_eq!(p.to_bits(), q.to_bits(), "arena x[{k}]");
        }
    }
}

#[test]
fn forward_and_reverse_derivatives_agree_with_finite_differences() {
    let n = 6;
    let mut g: Graph<F64> = Graph::new();
    let (a, b, syms, vals) = system(&mut g, n, 7);
    let x = g.solve_dense(a.clone(), b.clone());
    // A scalar of the solution, so the gradient has one root.
    let squares: Vec<ExprId> = x.iter().map(|&xi| g.mul(xi, xi)).collect();
    let f = g.reduce(rsdag::ReduceOp::Sum, squares);
    let grad = rsdag::gradient(&mut g, f, &syms);
    let fwd: Vec<ExprId> = syms
        .iter()
        .map(|&s| rsdag::differentiate(&mut g, f, s))
        .collect();
    let tape = Tape::compile(&g, &[f], &syms);
    let gtape = Tape::compile(&g, &grad, &syms);
    let ftape = Tape::compile(&g, &fwd, &syms);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    gtape.eval(&vals, &mut w, &mut o);
    let g_rev = o.clone();
    ftape.eval(&vals, &mut w, &mut o);
    let g_fwd = o.clone();
    for k in 0..syms.len() {
        let h = 1e-6;
        let mut vp = vals.clone();
        vp[k] += h;
        tape.eval(&vp, &mut w, &mut o);
        let fp = o[0];
        vp[k] -= 2.0 * h;
        tape.eval(&vp, &mut w, &mut o);
        let fm = o[0];
        let fd = (fp - fm) / (2.0 * h);
        assert!(
            (g_rev[k] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
            "reverse d/d{k}: {} vs fd {fd}",
            g_rev[k]
        );
        assert!(
            (g_fwd[k] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
            "forward d/d{k}: {} vs fd {fd}",
            g_fwd[k]
        );
    }
}

#[test]
fn a_dense_matrix_through_the_builder_is_one_kernel() {
    let n = 10;
    let mut g: Graph<F64> = Graph::new();
    let (a, b, syms, vals) = system(&mut g, n, 9);
    let rows: Vec<Vec<ExprId>> = (0..n).map(|i| a[i * n..(i + 1) * n].to_vec()).collect();
    let x = Builder::solve(&mut g, &rows, &b);
    let tape = Tape::compile(&g, &x, &syms);
    assert_eq!(tape.dump().matches("Solve").count(), 1, "{}", tape.dump());
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&vals, &mut w, &mut o);
    let dense: Vec<Vec<f64>> = (0..n).map(|i| vals[i * n..(i + 1) * n].to_vec()).collect();
    let want = Numeric.solve(&dense, &vals[n * n..]);
    for (p, q) in want.iter().zip(&o) {
        assert_eq!(p.to_bits(), q.to_bits());
    }
}

#[test]
fn solves_over_one_matrix_fuse_into_one_kernel_and_match_the_single_solves() {
    use rsdag::semantics::{solve, solve_many};
    let (n, k) = (5usize, 4usize);
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
    let d = tape.dump();
    assert_eq!(d.matches("SolveMany(").count(), 1, "{d}");
    assert_eq!(d.matches("Solve(").count(), 0, "{d}");
    let inputs: Vec<f64> = (0..syms.len())
        .map(|i| {
            let (r, c) = ((i / n) % n, i % n);
            if i < n * n && r == c {
                3.0 + i as f64 * 0.01
            } else {
                0.1 * ((i % 7) as f64 - 3.0)
            }
        })
        .collect();
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&inputs, &mut w, &mut o);
    // The reference: one solve per column, and the multi-solve routine.
    let av = &inputs[..n * n];
    let mut many = vec![0.0; n * k];
    solve_many(av, &inputs[n * n..], n, k, &mut many);
    for c in 0..k {
        let mut x = vec![0.0; n];
        solve(av, &inputs[n * n + c * n..n * n + (c + 1) * n], n, &mut x);
        for i in 0..n {
            assert_eq!(o[c * n + i].to_bits(), x[i].to_bits(), "column {c} row {i}");
            assert_eq!(
                many[c * n + i].to_bits(),
                x[i].to_bits(),
                "many column {c} row {i}"
            );
        }
    }
}
