use rsdag::eval::eval_real;
use rsdag::*;
use std::collections::HashMap;

#[test]
fn rebuild_drops_dead_nodes_and_keeps_values() {
    let mut g: Graph = Graph::new();
    let (x, y) = (g.sym("x"), g.sym("y"));
    let sx = g.sin(x);
    let _dead = g.exp(sx); // never reachable from the root
    let _dead2 = g.mul(x, y);
    let t = g.mul(sx, y);
    let root = g.add(t, x);
    let n_before = g.len();
    let (h, roots) = rebuild(&g, &[root]);
    assert!(h.len() < n_before, "{} < {n_before}", h.len());
    let mut env = HashMap::new();
    env.insert(rsdag::node::SymbolId(0), 0.7);
    env.insert(rsdag::node::SymbolId(1), -1.3);
    let a = eval_real(&g, &env, &[root]);
    let b = eval_real(&h, &env, &roots);
    assert_eq!(a[0].to_bits(), b[0].to_bits());
    assert_eq!(h.symbol_name(rsdag::node::SymbolId(1)), "y");
}

#[test]
fn rebuild_carries_functions_and_calls() {
    let mut g: Graph = Graph::new();
    let p = g.sym("p");
    let body = g.mul(p, p);
    let ps = match *g.node(p) {
        Node::Symbol(s) => s,
        _ => unreachable!(),
    };
    let f = g.define_func("sq", vec![ps], vec![body]);
    let x = g.sym("x");
    let c = g.call(f, 0, &[x]);
    let root = g.add(c, x);
    let (h, roots) = rebuild(&g, &[root]);
    let mut env = HashMap::new();
    env.insert(rsdag::node::SymbolId(1), 3.0);
    assert_eq!(eval_real(&h, &env, &roots)[0], 12.0);
    assert_eq!(h.n_funcs(), 1);
}
