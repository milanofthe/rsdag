use num_complex::Complex64;
use rsdag::graph::Graph;
use rsdag::*;
use std::collections::HashMap;

#[test]
fn complex_eval_is_linear_on_shared_dag() {
    // Balanced doubling DAG: x_{k+1} = x_k + x_k, one hash-consed node that
    // references x_k twice. 40 levels => 2^40 naive recursions but only 40
    // distinct nodes. Completing quickly proves `eval` memoizes (is linear in
    // reachable nodes, not exponential in paths).
    let mut ctx: Graph = Graph::new();
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
