//! Row dots against one vector fuse into a matrix-vector kernel, read in
//! place when the matrix and the vector are runs of inputs and gathered
//! otherwise, bit-identical to the rows as separate dots, and surviving
//! specialization.

mod common;
use common::assert_matches_eval;

use rsdag::Program;

use rsdag::{ExprId, Graph, SymbolId, Tape, F64};

fn gemvs_in(tape: &Tape) -> usize {
    tape.dump().matches("Gemv").count()
}

fn statespace(g: &mut Graph<F64>, n: usize, m: usize) -> (Vec<ExprId>, Vec<SymbolId>) {
    let a: Vec<ExprId> = (0..n * n).map(|k| g.sym(&format!("a{k}"))).collect();
    let x: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
    let b: Vec<ExprId> = (0..n * m).map(|k| g.sym(&format!("b{k}"))).collect();
    let u: Vec<ExprId> = (0..m).map(|i| g.sym(&format!("u{i}"))).collect();
    let roots = (0..n)
        .map(|i| {
            let ax = g.dot(a[i * n..(i + 1) * n].to_vec(), x.clone());
            let bu = g.dot(b[i * m..(i + 1) * m].to_vec(), u.clone());
            g.add(ax, bu)
        })
        .collect();
    let syms = (0..(n * n + n + n * m + m) as u32).map(SymbolId).collect();
    (roots, syms)
}

#[test]
fn a_state_space_block_is_two_kernels_over_inputs_in_place() {
    let (n, m) = (16, 3);
    let mut g: Graph<F64> = Graph::new();
    let (roots, syms) = statespace(&mut g, n, m);
    let tape = Tape::compile(&g, &roots, &syms);
    assert_eq!(gemvs_in(&tape), 2, "{}", tape.dump());
    assert!(
        tape.n_ops() <= 2 + 2 * n + n,
        "{} ops for {n} rows, more than the kernels, their picks and the adds:\n{}",
        tape.n_ops(),
        tape.dump()
    );
    let inputs: Vec<f64> = (0..syms.len())
        .map(|k| 0.01 * (k % 13) as f64 - 0.05)
        .collect();
    assert_matches_eval(&g, &roots, &syms, &tape, &inputs);
}

#[test]
fn computed_entries_are_gathered_and_short_inputs_read_as_nan() {
    let n = 10;
    let mut g: Graph<F64> = Graph::new();
    let xs: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
    let syms: Vec<SymbolId> = (0..n as u32).map(SymbolId).collect();
    let one = g.one();
    let a: Vec<ExprId> = (0..n * n)
        .map(|k| {
            let p = g.mul(xs[k / n], xs[k % n]);
            g.add(p, one)
        })
        .collect();
    let x_reversed_not_a_run: Vec<ExprId> = xs.iter().rev().copied().collect();
    let roots: Vec<ExprId> = (0..n)
        .map(|i| g.dot(a[i * n..(i + 1) * n].to_vec(), x_reversed_not_a_run.clone()))
        .collect();
    let tape = Tape::compile(&g, &roots, &syms);
    assert_eq!(gemvs_in(&tape), 1, "{}", tape.dump());
    let inputs: Vec<f64> = (0..n).map(|k| 0.1 + 0.07 * k as f64).collect();
    assert_matches_eval(&g, &roots, &syms, &tape, &inputs);
    let (roots2, syms2) = statespace(&mut g, 8, 1);
    let tape2 = Tape::compile(&g, &roots2, &syms2);
    let fewer_inputs_than_the_run_reads_in_place: Vec<f64> = vec![0.5; 20];
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape2.eval(&fewer_inputs_than_the_run_reads_in_place, &mut w, &mut o);
    assert!(o.iter().all(|v| v.is_nan()));
}

#[test]
fn a_kernel_survives_specialization() {
    let (n, m) = (12, 2);
    let mut g: Graph<F64> = Graph::new();
    let (rows, syms) = statespace(&mut g, n, m);
    let zero = g.zero();
    let roots: Vec<ExprId> = rows
        .iter()
        .map(|&r| {
            let c = g.cmp(rsdag::CmpOp::Gt, r, zero);
            let nr = g.neg(r);
            g.select(c, r, nr)
        })
        .collect();
    let tape = Tape::compile(&g, &roots, &syms);
    let inputs: Vec<f64> = (0..syms.len())
        .map(|k| 0.02 * (k % 11) as f64 - 0.09)
        .collect();
    let (mut w, mut o, mut choices) = (Vec::new(), Vec::new(), Vec::new());
    tape.eval_with(&inputs, &mut w, &mut o, &mut choices);
    let spec = tape.specialize(&choices, &vec![true; tape.n_selects()]);
    let (mut w2, mut o2) = (Vec::new(), Vec::new());
    assert!(spec.eval_checked(&inputs, &mut w2, &mut o2));
    for (k, (a, b)) in o.iter().zip(&o2).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "output {k}");
    }
}
