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

/// Substituting and inlining walk the graph with an explicit stack: a chain
/// far deeper than a small thread stack could recurse through is fine.
#[test]
fn deep_chains_substitute_and_inline_on_a_small_stack() {
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let mut g: Graph = Graph::new();
            let x = g.sym("x");
            let mut acc = x;
            for _ in 0..100_000 {
                acc = g.sin(acc);
            }
            let f = g.close("f", vec![acc]);
            let y = g.sym("y");
            let xs = sid(&mut g, "x");
            let map: HashMap<SymbolId, ExprId> = [(xs, y)].into_iter().collect();
            let sub = substitute(&mut g, &[acc], &map)[0];
            let call = g.call(f, 0, &[y]);
            let inlined = g.inline_all(&[call])[0];
            assert_eq!(inlined, sub, "inlining is substituting the arguments");
        })
        .unwrap()
        .join()
        .unwrap();
}
