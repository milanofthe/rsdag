use rsdag::node::Node;
use rsdag::BigRational;
use rsdag::*;
use rustc_hash::FxHashMap as HashMap;

fn sid<K: Field>(ctx: &mut Graph<K>, name: &str) -> SymbolId {
    let e = ctx.sym(name);
    match ctx.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

#[test]
fn substitution_is_simultaneous() {
    // Swap x<->y in x - y: simultaneous, so the result is y - x (not 0).
    let mut ctx: Graph<BigRational> = Graph::new();
    let x = ctx.sym("x");
    let y = ctx.sym("y");
    let f = ctx.sub(x, y);
    let (xs, ys) = (sid(&mut ctx, "x"), sid(&mut ctx, "y"));
    let mut map: HashMap<SymbolId, ExprId> = HashMap::default();
    map.insert(xs, y);
    map.insert(ys, x);
    let g = substitute(&mut ctx, &[f], &map)[0];
    let want = ctx.sub(y, x);
    assert_eq!(g, want);
}
