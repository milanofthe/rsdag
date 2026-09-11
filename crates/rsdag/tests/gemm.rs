//! Row dots over one set of rows against several vectors fuse into a
//! matrix-matrix kernel: `A B` is one `Gemm`, bit-identical to its entries
//! as separate dots, whatever order the entries were built in, and
//! surviving specialization.

use std::collections::HashMap;

use rsdag::{ExprId, Graph, SymbolId, Tape, F64};

fn kernels_in(tape: &Tape, name: &str) -> usize {
    tape.dump().matches(name).count()
}

/// `A` (`m` by `k`) times `B` (`k` by `n`) over inputs, entry by entry,
/// row-major; `B` given by columns so a column is one vector.
fn matmul(
    g: &mut Graph<F64>,
    m: usize,
    k: usize,
    n: usize,
    column_major: bool,
) -> (Vec<ExprId>, Vec<SymbolId>) {
    let a: Vec<ExprId> = (0..m * k).map(|i| g.sym(&format!("a{i}"))).collect();
    let b: Vec<ExprId> = (0..k * n).map(|i| g.sym(&format!("b{i}"))).collect();
    let col = |j: usize| -> Vec<ExprId> { (0..k).map(|l| b[l * n + j]).collect() };
    let mut roots = vec![g.zero(); m * n];
    let entries: Vec<(usize, usize)> = if column_major {
        (0..n).flat_map(|j| (0..m).map(move |i| (i, j))).collect()
    } else {
        (0..m).flat_map(|i| (0..n).map(move |j| (i, j))).collect()
    };
    for (i, j) in entries {
        roots[i * n + j] = g.dot(a[i * k..(i + 1) * k].to_vec(), col(j));
    }
    let syms = (0..(m * k + k * n) as u32).map(SymbolId).collect();
    (roots, syms)
}

fn check(g: &Graph<F64>, roots: &[ExprId], syms: &[SymbolId], tape: &Tape, inputs: &[f64]) {
    let env: HashMap<SymbolId, f64> = syms.iter().copied().zip(inputs.iter().copied()).collect();
    let want = rsdag::eval(g, roots, &env);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(inputs, &mut w, &mut o);
    for (k, (a, b)) in want.iter().zip(&o).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "output {k}: {a} vs {b}");
    }
}

#[test]
fn a_matrix_product_is_one_kernel_whatever_the_entry_order() {
    for column_major in [false, true] {
        let (m, k, n) = (9, 7, 3);
        let mut g: Graph<F64> = Graph::new();
        let (roots, syms) = matmul(&mut g, m, k, n, column_major);
        let tape = Tape::compile(&g, &roots, &syms);
        assert_eq!(kernels_in(&tape, "Gemm(9x7"), 1, "{}", tape.dump());
        assert_eq!(kernels_in(&tape, "Gemv"), 0, "{}", tape.dump());
        let inputs: Vec<f64> = (0..syms.len())
            .map(|i| 0.01 * (i % 13) as f64 - 0.05)
            .collect();
        check(&g, &roots, &syms, &tape, &inputs);
    }
}

#[test]
fn computed_factors_are_gathered() {
    let (m, k, n) = (8, 5, 2);
    let mut g: Graph<F64> = Graph::new();
    let xs: Vec<ExprId> = (0..m * k + k * n)
        .map(|i| g.sym(&format!("x{i}")))
        .collect();
    let syms: Vec<SymbolId> = (0..xs.len() as u32).map(SymbolId).collect();
    let a: Vec<ExprId> = xs[..m * k].iter().map(|&e| g.exp(e)).collect();
    let b: Vec<ExprId> = xs[m * k..].iter().map(|&e| g.sin(e)).collect();
    let mut roots = Vec::new();
    for i in 0..m {
        for j in 0..n {
            let col: Vec<ExprId> = (0..k).map(|l| b[l * n + j]).collect();
            roots.push(g.dot(a[i * k..(i + 1) * k].to_vec(), col));
        }
    }
    let tape = Tape::compile(&g, &roots, &syms);
    assert_eq!(kernels_in(&tape, "Gemm(8x5"), 1, "{}", tape.dump());
    let inputs: Vec<f64> = (0..syms.len()).map(|i| 0.1 + 0.07 * i as f64).collect();
    check(&g, &roots, &syms, &tape, &inputs);
}

#[test]
fn a_single_vector_stays_a_gemv() {
    let mut g: Graph<F64> = Graph::new();
    let (roots, syms) = matmul(&mut g, 10, 4, 1, false);
    let tape = Tape::compile(&g, &roots, &syms);
    assert_eq!(kernels_in(&tape, "Gemv"), 1, "{}", tape.dump());
    assert_eq!(kernels_in(&tape, "Gemm"), 0);
}

#[test]
fn the_kernel_survives_specialization() {
    let (m, k, n) = (8, 6, 4);
    let mut g: Graph<F64> = Graph::new();
    let (entries, syms) = matmul(&mut g, m, k, n, false);
    let zero = g.zero();
    let roots: Vec<ExprId> = entries
        .iter()
        .map(|&r| {
            let c = g.cmp(rsdag::CmpOp::Gt, r, zero);
            let nr = g.neg(r);
            g.select(c, r, nr)
        })
        .collect();
    let tape = Tape::compile(&g, &roots, &syms);
    let inputs: Vec<f64> = (0..syms.len())
        .map(|i| 0.02 * (i % 11) as f64 - 0.09)
        .collect();
    let (mut w, mut o, mut choices) = (Vec::new(), Vec::new(), Vec::new());
    tape.eval_with(&inputs, &mut w, &mut o, &mut choices);
    let spec = tape.specialize(&choices, &vec![true; tape.n_selects()]);
    assert_eq!(
        kernels_in(spec.tape(), "Gemm(8x6"),
        1,
        "{}",
        spec.tape().dump()
    );
    let (mut w2, mut o2) = (Vec::new(), Vec::new());
    assert!(spec.eval_checked(&inputs, &mut w2, &mut o2));
    for (i, (a, b)) in o.iter().zip(&o2).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "output {i}");
    }
}
