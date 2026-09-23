use rsdag::display::to_string;
use rsdag::symbolic::{prune_poly, rational_form};
use rsdag::BigRational;
use rsdag::*;
use std::collections::HashMap;

fn sym_id<K: Field>(g: &Graph<K>, e: ExprId) -> SymbolId {
    match *g.node(e) {
        Node::Symbol(s) => s,
        _ => unreachable!(),
    }
}

#[test]
fn collects_a_transfer_function() {
    // H = 1 / (1 + s R C): numerator [1], denominator [1, R C]
    let mut g: Graph<BigRational> = Graph::new();
    let (s, r, c) = (g.sym("s"), g.sym("R"), g.sym("C"));
    let rc = g.mul(r, c);
    let src = g.mul(s, rc);
    let one = g.one();
    let den = g.add(one, src);
    let h = g.recip(den);
    let sid = sym_id(&g, s);
    let (n, d) = rational_form(&mut g, h, sid).unwrap();
    assert_eq!(n.len(), 1);
    assert!(g.is_one(n[0]));
    assert_eq!(d.len(), 2);
    assert!(g.is_one(d[0]));
    assert_eq!(to_string(&g, d[1]), "R*C");
    assert!(collect(&mut g, h, sid).is_none());
    let p = collect(&mut g, den, sid).unwrap();
    assert_eq!(p.len(), 2);
}

#[test]
fn prunes_small_terms() {
    let mut g: Graph<BigRational> = Graph::new();
    let (a, b) = (g.sym("a"), g.sym("b"));
    let sum = g.add(a, b);
    let poly = vec![sum];
    let mut env = HashMap::new();
    env.insert(sym_id(&g, a), 1.0);
    env.insert(sym_id(&g, b), 1e-9);
    let (pruned, total, kept) = prune_poly(&mut g, &poly, &env, 1.0, 1e-6);
    assert_eq!((total, kept), (2, 1));
    assert_eq!(pruned[0], a);
}
