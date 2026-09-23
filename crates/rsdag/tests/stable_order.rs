//! A graph's numbers depend on its expressions, not on its history: equal
//! expressions have equal fingerprints in any graph, and a sum folds in the
//! same order however its terms were built, so it evaluates to the same
//! bits after any unrelated construction and in any build order.

use rsdag::synth::{build, inputs, Spec, Vocabulary};
use rsdag::{differentiate, sparse_jacobian, ExprId, Graph, Node, ReduceOp, SymbolId, Tape, F64};

fn syms(g: &mut Graph<F64>, n: usize) -> (Vec<ExprId>, Vec<SymbolId>) {
    let xs: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
    let ss = xs
        .iter()
        .map(|&e| match g.node(e) {
            Node::Symbol(s) => *s,
            _ => unreachable!(),
        })
        .collect();
    (xs, ss)
}

/// Terms whose sum rounds differently in different orders.
fn terms(g: &mut Graph<F64>, xs: &[ExprId], order: &[usize]) -> Vec<ExprId> {
    order
        .iter()
        .map(|&k| {
            let c = g.konst_f64(1.0 + (k as f64) * 1e-3);
            let e = g.exp(xs[k % xs.len()]);
            let p = g.mul(c, e);
            g.pow_i(p, 1 + (k % 3) as i64)
        })
        .collect()
}

fn value(g: &Graph<F64>, root: ExprId, ss: &[SymbolId], ins: &[f64]) -> u64 {
    let (mut w, mut o) = (Vec::new(), Vec::new());
    Tape::compile(g, &[root], ss).eval(ins, &mut w, &mut o);
    o[0].to_bits()
}

#[test]
fn a_sum_evaluates_the_same_whatever_was_built_before() {
    let n = 7;
    let ins: Vec<f64> = (0..n).map(|i| 0.3 + 0.37 * i as f64).collect();
    let forward: Vec<usize> = (0..40).collect();
    let backward: Vec<usize> = (0..40).rev().collect();

    let mut a: Graph<F64> = Graph::new();
    let (xa, sa) = syms(&mut a, n);
    let ta = terms(&mut a, &xa, &forward);
    let ra = a.reduce(ReduceOp::Sum, ta);

    // Another graph: unrelated nodes first, symbols in another order, the
    // terms built backwards.
    let mut b: Graph<F64> = Graph::new();
    for k in 0..500 {
        let q = b.sym(&format!("noise{k}"));
        let _ = b.sin(q);
    }
    let rev: Vec<ExprId> = (0..n).rev().map(|i| b.sym(&format!("x{i}"))).collect();
    let xb: Vec<ExprId> = rev.into_iter().rev().collect();
    let sb: Vec<SymbolId> = xb
        .iter()
        .map(|&e| match b.node(e) {
            Node::Symbol(s) => *s,
            _ => unreachable!(),
        })
        .collect();
    let tb = terms(&mut b, &xb, &backward);
    let rb = b.reduce(ReduceOp::Sum, tb);

    assert_eq!(a.fingerprint(ra), b.fingerprint(rb));
    assert_eq!(value(&a, ra, &sa, &ins), value(&b, rb, &sb, &ins));
}

#[test]
fn jacobian_bits_do_not_depend_on_how_it_was_taken() {
    // sparse_jacobian (column sweeps, reverse rows) against one
    // differentiate per entry in a fresh graph: the same bits.
    for seed in 0..12u64 {
        let spec = || {
            Spec::new(seed)
                .steps(800)
                .params(10)
                .outputs(5)
                .vocab(Vocabulary::Elementary)
                .width(12)
                .smooth()
        };
        let mut g: Graph<F64> = Graph::new();
        let (roots, ss) = build(&mut g, &mut spec());
        let jac = sparse_jacobian(&mut g, &roots, &ss);
        let fast: Vec<ExprId> = jac.iter().flat_map(|r| r.iter().map(|&(_, e)| e)).collect();

        let mut h: Graph<F64> = Graph::new();
        let (roots_h, ss_h) = build(&mut h, &mut spec());
        let mut slow = Vec::new();
        for (i, row) in jac.iter().enumerate() {
            for &(j, _) in row {
                slow.push(differentiate(&mut h, roots_h[i], ss_h[j]));
            }
        }
        let ins = inputs(&mut spec().rng(), ss.len());
        let (mut w, mut a, mut b) = (Vec::new(), Vec::new(), Vec::new());
        Tape::compile(&g, &fast, &ss).eval(&ins, &mut w, &mut a);
        Tape::compile(&h, &slow, &ss_h).eval(&ins, &mut w, &mut b);
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(
                x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
                "seed {seed} entry {k}: {x} vs {y}"
            );
        }
        for (&e, &f) in fast.iter().zip(&slow) {
            assert_eq!(g.fingerprint(e), h.fingerprint(f), "seed {seed}");
        }
    }
}
