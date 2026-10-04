use num_complex::Complex64;
use rsdag::graph::Graph;
use rsdag::BigRational;
use rsdag::*;
use std::collections::HashMap;

#[test]
fn complex_eval_is_linear_on_shared_dag() {
    // Balanced doubling DAG: x_{k+1} = x_k + x_k, one hash-consed node that
    // references x_k twice. 40 levels => 2^40 naive recursions but only 40
    // distinct nodes. Completing quickly proves `eval` memoizes (is linear in
    // reachable nodes, not exponential in paths).
    let mut ctx: Graph<BigRational> = Graph::new();
    let s = ctx.sym("s");
    let mut e = s;
    for _ in 0..40 {
        e = ctx.add(e, e);
    }
    let sid = match ctx.node(s) {
        Node::Symbol(id) => *id,
        _ => unreachable!(),
    };
    let mut env = HashMap::new();
    env.insert(sid, Complex64::new(1.0, 0.0));
    let v = eval(&ctx, &[e], &env)[0];
    assert!((v.re - 2f64.powi(40)).abs() < 1.0, "got {}", v.re);
}

/// A call evaluates in the caller's scalar: a complex argument reaches the
/// function's body as a complex number rather than being cut to its real part.
#[test]
fn complex_eval_runs_through_calls() {
    let mut g: Graph = Graph::new();
    let mut s = Scope::new(&mut g, "square_plus_one");
    let x = s.param("x");
    let x2 = s.mul(x, x);
    let one = s.one();
    let y = s.add(x2, one);
    let f = s.close(vec![y]);
    let z = g.sym("z");
    let call = g.call(f, 0, &[z]);
    let zid = match g.node(z) {
        Node::Symbol(id) => *id,
        _ => unreachable!(),
    };
    let w = Complex64::new(0.5, 2.0);
    let env: HashMap<SymbolId, Complex64> = [(zid, w)].into_iter().collect();
    let v = eval(&g, &[call], &env)[0];
    let want = w * w + 1.0;
    assert!((v - want).norm() < 1e-12, "got {v}, want {want}");
}
