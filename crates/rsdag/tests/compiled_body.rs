//! A symbolic function runs as a body interpreted from its tape: built once
//! per set of outputs a program calls, shared by the programs over one
//! graph, and specialized to the constants a call passes.

mod common;
use common::constants::AGAINST_LIBM;
use rsdag::Program;
use rsdag::{ExprId, Graph, Node, OutputRole, ParamRole, Scope, Tape, F64};

#[test]
fn an_interpreted_body_computes_only_the_outputs_called() {
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let outs: Vec<ExprId> = (1..=5)
        .map(|k| {
            let c = g.konst_f64(k as f64);
            let e = g.exp(x);
            g.mul(c, e)
        })
        .collect();
    let xs = match *g.node(x) {
        Node::Symbol(s) => s,
        _ => unreachable!(),
    };
    let mut s = Scope::new(&mut g, "f");
    s.param(xs, ParamRole::Free);
    let f = s.close(outs.into_iter().map(|e| (e, OutputRole::Plain)));
    let y = g.sym("y");
    let (c1, c3) = (g.call(f, 1, &[y]), g.call(f, 3, &[y]));
    let r = g.add(c1, c3);
    let ys = match *g.node(y) {
        Node::Symbol(s) => s,
        _ => unreachable!(),
    };
    let tape = Tape::compile(&g, &[r], &[ys]);
    let body = tape.bundles()[0].clone();
    assert_eq!(body.n_outputs(), 2);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[0.5], &mut w, &mut o);
    let want = 6.0 * 0.5f64.exp();
    assert!((o[0] - want).abs() <= AGAINST_LIBM * want);
}

#[test]
fn calls_with_constant_arguments_run_a_specialized_copy() {
    let mut g: Graph<F64> = Graph::new();
    let sym = |g: &mut Graph<F64>, n: &str| {
        let e = g.sym(n);
        match *g.node(e) {
            Node::Symbol(s) => (e, s),
            _ => unreachable!(),
        }
    };
    let ((x, xs), (c, cs)) = (sym(&mut g, "x"), sym(&mut g, "c"));
    let ex = g.exp(x);
    let cex = g.mul(c, ex);
    let sc = g.sin(c);
    let o0 = g.add(cex, sc);
    let o1 = g.mul(c, x);
    let mut s = Scope::new(&mut g, "f");
    s.param(xs, ParamRole::Free);
    s.param(cs, ParamRole::Free);
    let f = s.close([(o0, OutputRole::Plain), (o1, OutputRole::Plain)]);
    let ((y, ys), (z, zs)) = (sym(&mut g, "y"), sym(&mut g, "z"));
    let (zero, two) = (g.zero(), g.konst_f64(2.0));
    let a = g.call(f, 0, &[y, zero]);
    let b = g.call(f, 0, &[z, zero]);
    let d = g.call(f, 1, &[y, two]);
    let s = g.add(a, b);
    let root = g.add(s, d);
    let n_funcs = g.n_funcs();
    let spec = g.specialize_calls(&[root])[0];
    assert_eq!(g.n_funcs(), n_funcs + 2, "one copy per constant pattern");
    for (callee, _) in g.free_calls_in(&[spec]).iter().map(|&o| g.output(o)) {
        assert_eq!(g.func(callee).params().len(), 1);
    }
    let vars = [ys, zs];
    let eval = |g: &Graph<F64>, e: ExprId| -> f64 {
        let tape = Tape::compile(g, &[e], &vars);
        let (mut w, mut o) = (Vec::new(), Vec::new());
        tape.eval(&[0.3, -0.7], &mut w, &mut o);
        o[0]
    };
    assert_eq!(eval(&g, root).to_bits(), eval(&g, spec).to_bits());
    let jr = rsdag::sparse_jacobian(&mut g, &[root], &vars);
    let js = rsdag::sparse_jacobian(&mut g, &[spec], &vars);
    for ((_, a), (_, b)) in jr[0].iter().zip(&js[0]) {
        assert_eq!(eval(&g, *a).to_bits(), eval(&g, *b).to_bits());
    }
}

#[test]
fn constant_parameters_stay_arguments() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = rsdag::Scope::new(&mut g, "f");
    let x = s.param("x", rsdag::ParamRole::State { id: 0 });
    let c = s.param("c", rsdag::ParamRole::Param);
    let ex = s.exp(x);
    let o = s.mul(c, ex);
    let f = s.close([(o, OutputRole::Plain)]);
    let y = g.sym("y");
    let (one, two, zero) = (g.one(), g.konst_f64(2.0), g.zero());
    let a = g.call(f, 0, &[y, one]);
    let b = g.call(f, 0, &[y, two]);
    let root = g.add(a, b);
    let n_funcs = g.n_funcs();
    let spec = g.specialize_calls(&[root])[0];
    assert_eq!(spec, root, "constant parameters are not specialized");
    assert_eq!(g.n_funcs(), n_funcs);
    let at_zero = g.call(f, 0, &[zero, two]);
    let spec = g.specialize_calls(&[at_zero])[0];
    assert_eq!(g.n_funcs(), n_funcs + 1, "a constant state still is");
    let callee = g
        .free_calls_in(&[spec])
        .iter()
        .map(|&o| g.output(o).0)
        .next()
        .unwrap();
    assert_eq!(
        g.func(callee).params().len(),
        1,
        "the copy keeps the parameter"
    );
}

#[test]
fn programs_over_one_graph_share_a_body() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = rsdag::Scope::new(&mut g, "f");
    let x = s.param("x", ParamRole::Free);
    let ex = s.exp(x);
    let f = s.close([(ex, OutputRole::Plain)]);
    let y = g.sym("y");
    let ys = match *g.node(y) {
        Node::Symbol(s) => s,
        _ => unreachable!(),
    };
    let c = g.call(f, 0, &[y]);
    let two = g.konst_f64(2.0);
    let c2 = g.mul(two, c);
    let a = Tape::compile(&g, &[c], &[ys]);
    let b = Tape::compile(&g, &[c2], &[ys]);
    assert_eq!(a.bundles().len(), 1);
    assert!(
        std::sync::Arc::ptr_eq(&a.bundles()[0], &b.bundles()[0]),
        "one body for both tapes"
    );
}
