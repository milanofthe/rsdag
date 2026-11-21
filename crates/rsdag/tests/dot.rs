//! The graph view as data: one node per expression, a shared subexpression
//! once, the bodies of the calls in frames.

use rsdag::dot::{GraphView, Kind};
use rsdag::{Graph, OutputRole, ParamRole, F64};

#[test]
fn a_shared_subexpression_is_one_node() {
    let mut g: Graph<F64> = Graph::new();
    let (x, y) = (g.sym("x"), g.sym("y"));
    let xy = g.mul(x, y);
    let s = g.sin(xy);
    let f = g.add(s, xy);
    let d = GraphView::new(&g).root(f, "f").data();
    // x, y, x*y, sin and +, the result f, and the edges between them
    let exprs = d.nodes.iter().filter(|n| n.kind != Kind::Output).count();
    assert_eq!(exprs, 5, "{d:?}");
    assert_eq!(d.nodes.len(), 6, "one result node");
    let at = d
        .nodes
        .iter()
        .position(|n| n.id == format!("n{}", xy.0))
        .unwrap();
    assert_eq!(
        d.edges.iter().filter(|e| e.0 == at).count(),
        2,
        "x*y feeds sin and +"
    );
}

#[test]
fn numbers_are_short() {
    for (x, want) in [
        (0.0, "0"),
        (-3.0, "-3"),
        (0.5, "0.5"),
        (0.001, "0.001"),
        (5.5406e34, "5.541e34"),
        (1.0 / 3.0, "0.3333"),
    ] {
        let mut g: Graph<F64> = Graph::new();
        let c = g.konst_f64(x);
        let d = GraphView::new(&g).root(c, "").data();
        assert_eq!(d.nodes[0].label, want);
    }
}

#[test]
fn inline_draws_every_instance_as_its_body() {
    // leaf(a) = sin(a); mid(x, y) = leaf(x) * y; two calls of mid at the top
    let mut g: Graph<F64> = Graph::new();
    let mut s = rsdag::Scope::new(&mut g, "leaf");
    let a = s.param("a", ParamRole::Free);
    let sa = s.sin(a);
    let leaf = s.close([(sa, OutputRole::Plain)]);
    let mut s = rsdag::Scope::new(&mut g, "mid");
    let (x, y) = (s.param("x", ParamRole::Free), s.param("y", ParamRole::Free));
    let l = s.call(leaf, 0, &[x]);
    let m = s.mul(l, y);
    let mid = s.close([(m, OutputRole::Plain)]);
    let (p, q) = (g.sym("p"), g.sym("q"));
    let one = g.call(mid, 0, &[p, q]);
    let two = g.call(mid, 0, &[q, p]);
    let r = g.add(one, two);

    let data = GraphView::new(&g)
        .root(r, "r")
        .inline()
        .label(one, "X1")
        .label(two, "X2")
        .data();
    // no call is left: each instance is its body, the leaf nested in it
    assert!(data.nodes.iter().all(|n| n.kind != Kind::Call));
    let frame = |name: &str| data.clusters.iter().position(|c| c == name).unwrap();
    let (x1, x2) = (frame("X1"), frame("X2"));
    assert_eq!(data.clusters.len(), 4);
    assert_eq!((data.parents[x1], data.parents[x2]), (None, None));
    // a leaf frame in each instance, one sine in each
    let leaves: Vec<usize> = (0..4).filter(|&c| data.clusters[c] == "leaf").collect();
    let mut owners: Vec<_> = leaves.iter().map(|&c| data.parents[c]).collect();
    owners.sort();
    assert_eq!(owners, [Some(x1.min(x2)), Some(x1.max(x2))]);
    let mut sines: Vec<_> = data
        .nodes
        .iter()
        .filter(|n| n.label == "sin")
        .map(|n| n.cluster.unwrap())
        .collect();
    sines.sort();
    assert_eq!(sines, leaves);
    // the parameters are the arguments: p and q once each, at the top
    for name in ["p", "q"] {
        let n: Vec<_> = data.nodes.iter().filter(|n| n.label == name).collect();
        assert_eq!(n.len(), 1, "{name}");
        assert_eq!(n[0].cluster, None);
    }
    // p feeds X1's sine and X2's product
    let p_at = data.nodes.iter().position(|n| n.label == "p").unwrap();
    let readers: Vec<_> = data
        .edges
        .iter()
        .filter(|e| e.0 == p_at)
        .map(|e| data.nodes[e.1].cluster)
        .collect();
    let leaf_of = |x: usize| leaves.iter().copied().find(|&c| data.parents[c] == Some(x));
    assert!(
        readers.contains(&leaf_of(x1)) && readers.contains(&Some(x2)),
        "{readers:?}"
    );
}

#[test]
fn inline_binds_the_bound_parameters_of_a_call() {
    // leaf(a, k) = k sin(a), called over one argument list through two
    // bindings of k
    let mut g: Graph<F64> = Graph::new();
    let mut s = rsdag::Scope::new(&mut g, "leaf");
    let (a, k) = (s.param("a", ParamRole::Free), s.param("k", ParamRole::Free));
    let sa = s.sin(a);
    let m = s.mul(k, sa);
    let leaf = s.close([(m, OutputRole::Plain)]);
    let (p, c1, c2) = (g.sym("p"), g.sym("c1"), g.sym("c2"));
    let b1 = g.bind(leaf, &[(1, c1)]);
    let b2 = g.bind(leaf, &[(1, c2)]);
    let one = g.calls_bound(b1, &[0], &[p])[0];
    let two = g.calls_bound(b2, &[0], &[p])[0];
    let r = g.add(one, two);

    let data = GraphView::new(&g).root(r, "r").inline().data();
    // an instance per binding, its bound parameter the bound expression
    assert_eq!(data.clusters, ["leaf", "leaf"]);
    assert!(data.nodes.iter().all(|n| n.label != "k" && n.label != "a"));
    let reader = |name: &str| {
        let at = data.nodes.iter().position(|n| n.label == name).unwrap();
        let readers: Vec<_> = (data.edges.iter())
            .filter(|e| e.0 == at)
            .map(|e| data.nodes[e.1].cluster)
            .collect();
        assert_eq!(readers.len(), 1, "{name}");
        readers[0].expect("read in a frame")
    };
    assert_ne!(reader("c1"), reader("c2"));
}
