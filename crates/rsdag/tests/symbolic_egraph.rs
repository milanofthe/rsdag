use num_rational::BigRational;
use rsdag::display::to_string;
use rsdag::*;

fn reachable(g: &Graph<BigRational>, root: ExprId) -> usize {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if seen.insert(id) {
            stack.extend_from_slice(&g.operands(id));
        }
    }
    seen.len()
}

#[test]
fn passes_through_unsupported_nodes() {
    let mut g: Graph = Graph::new();
    let (c, a, b) = (g.sym("c"), g.sym("a"), g.sym("b"));
    let sel = g.select(c, a, b);
    assert_eq!(simplify_egraph(&mut g, sel), sel);
}

#[test]
fn cancels_and_factors() {
    let mut g: Graph = Graph::new();
    let (x, y) = (g.sym("x"), g.sym("y"));
    // x*y + x*y*1 + x*(y - y) -> 2*x*y (or x*(2*y))
    let xy = g.mul(x, y);
    let yy = g.sub(y, y);
    let xyy = g.mul(x, yy);
    let s = g.add(xy, xy);
    let e = g.add(s, xyy);
    let out = simplify_egraph(&mut g, e);
    // The extracted form is one of the size-3 equivalents (which one is
    // an e-graph tie); it is smaller than the input and numerically equal.
    let text = to_string(&g, out);
    assert!(text.len() < to_string(&g, e).len(), "{text}");
    assert!(reachable(&g, out) < reachable(&g, e), "{text}");
    let mut env = std::collections::HashMap::new();
    env.insert(rsdag::node::SymbolId(0), 1.5);
    env.insert(rsdag::node::SymbolId(1), -2.0);
    let v = rsdag::eval::eval_real(&g, &env, &[e, out]);
    assert_eq!(v[0], v[1]);
    // x * (1/x) -> 1
    let inv = g.recip(x);
    let prod = g.mul(x, inv);
    let one = simplify_egraph(&mut g, prod);
    assert!(g.is_one(one));
}
