//! Instance batching: the calls of one function evaluate as one batched
//! call whatever their arguments are, and instances that feed instances
//! batch by call depth.

use std::collections::HashMap;

use rsdag::{ExprId, Graph, Node, SymbolId, Tape, F64};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

/// `cell(a, b) = exp(a) - b*b + a`.
fn cell(g: &mut Graph<F64>) -> rsdag::FuncId {
    let (a, b) = (g.sym("a"), g.sym("b"));
    let e = g.exp(a);
    let bb = g.mul(b, b);
    let d = g.sub(e, bb);
    let out = g.add(d, a);
    let params = vec![sym(g, a), sym(g, b)];
    g.define_func("cell", params, vec![out])
}

fn batches_in(tape: &Tape) -> usize {
    tape.dump().matches("BundleBatch").count()
}

#[test]
fn instances_with_computed_arguments_batch_into_one_call() {
    let n = 50;
    let mut g: Graph<F64> = Graph::new();
    let f = cell(&mut g);
    let xs: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
    let syms: Vec<SymbolId> = xs.iter().map(|&e| sym(&g, e)).collect();
    // Arguments are expressions of the inputs, not the inputs themselves.
    let roots: Vec<ExprId> = (0..n)
        .map(|i| {
            let two = g.konst_f64(2.0);
            let a = g.mul(two, xs[i]);
            let b = g.add(xs[i], xs[(i + 1) % n]);
            g.call(f, 0, &[a, b])
        })
        .collect();
    let tape = Tape::compile(&g, &roots, &syms);
    assert_eq!(batches_in(&tape), 1, "one batched call:\n{}", tape.dump());
    let row: Vec<f64> = (0..n).map(|i| 0.1 + (i % 7) as f64 * 0.05).collect();
    let env: HashMap<SymbolId, f64> = syms.iter().copied().zip(row.iter().copied()).collect();
    let want = rsdag::eval(&g, &roots, &env);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&row, &mut w, &mut o);
    for (k, (a, b)) in want.iter().zip(&o).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "output {k}");
    }
}

#[test]
fn instances_that_feed_instances_batch_by_depth() {
    let n = 20;
    let mut g: Graph<F64> = Graph::new();
    let f = cell(&mut g);
    let xs: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
    let syms: Vec<SymbolId> = xs.iter().map(|&e| sym(&g, e)).collect();
    // A first layer of instances, then a second layer fed by the first.
    let first: Vec<ExprId> = (0..n)
        .map(|i| g.call(f, 0, &[xs[i], xs[(i + 1) % n]]))
        .collect();
    let second: Vec<ExprId> = (0..n)
        .map(|i| {
            let s = g.mul(first[i], first[(i + 3) % n]);
            let h = g.konst_f64(0.5);
            let t = g.mul(h, s);
            g.call(f, 0, &[t, xs[i]])
        })
        .collect();
    let roots: Vec<ExprId> = first.iter().chain(&second).copied().collect();
    let tape = Tape::compile(&g, &roots, &syms);
    assert_eq!(
        batches_in(&tape),
        2,
        "one batch per depth:\n{}",
        tape.dump()
    );
    let row: Vec<f64> = (0..n).map(|i| 0.05 + (i % 5) as f64 * 0.04).collect();
    let env: HashMap<SymbolId, f64> = syms.iter().copied().zip(row.iter().copied()).collect();
    let want = rsdag::eval(&g, &roots, &env);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&row, &mut w, &mut o);
    for (k, (a, b)) in want.iter().zip(&o).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "output {k}");
    }
}
