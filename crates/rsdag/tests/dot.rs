//! The graph view as data: one node per expression, a shared subexpression
//! once, faded nodes outside the focus, the bodies of the calls in frames.

use rsdag::dot::{number, reachable, GraphView, Kind};
use rsdag::{differentiate, ExprId, Graph, Node, SymbolId, F64};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

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
fn nodes_outside_the_focus_are_faded() {
    let mut g: Graph<F64> = Graph::new();
    let (x, y) = (g.sym("x"), g.sym("y"));
    let xy = g.mul(x, y);
    let s = g.sin(xy);
    let f = g.add(s, xy);
    let wrt = sym(&g, x);
    let df = differentiate(&mut g, f, wrt);
    let keep = reachable(&g, &[df]);
    let d = GraphView::new(&g)
        .root(f, "f")
        .root(df, "df/dx")
        .focus(keep.clone())
        .data();
    let faded = d.nodes.iter().filter(|n| n.faded).count();
    let all = reachable(&g, &[f, df]);
    // Every node only f reaches is faded, and f's result with them.
    assert_eq!(faded, all.len() - keep.len() + 1, "{d:?}");
}

#[test]
fn numbers_are_short() {
    assert_eq!(number(0.0), "0");
    assert_eq!(number(-3.0), "-3");
    assert_eq!(number(0.5), "0.5");
    assert_eq!(number(0.001), "0.001");
    assert_eq!(number(5.5406e34), "5.541e34");
    assert_eq!(number(1.0 / 3.0), "0.3333");
}

#[test]
fn bodies_draw_each_called_function_once() {
    // leaf(a) = sin(a); mid(x, y) = leaf(x) * y; two calls of mid at the top
    let mut g: Graph<F64> = Graph::new();
    let mut s = rsdag::Scope::new(&mut g, "leaf");
    let a = s.param("a");
    let sa = s.sin(a);
    let leaf = s.close(vec![sa]);
    let mut s = rsdag::Scope::new(&mut g, "mid");
    let (x, y) = (s.param("x"), s.param("y"));
    let l = s.call(leaf, 0, &[x]);
    let m = s.mul(l, y);
    let mid = s.close(vec![m]);
    let (p, q) = (g.sym("p"), g.sym("q"));
    let one = g.call(mid, 0, &[p, q]);
    let two = g.call(mid, 0, &[q, p]);
    let r = g.add(one, two);

    let flat = GraphView::new(&g).root(r, "r").data();
    assert!(flat.clusters.is_empty());

    // each body once in its frame, the sine of leaf's body once however many
    // calls reach it, a link per call (two of mid, one of leaf in mid's
    // body), and a call relabelled by its instance
    let data = GraphView::new(&g)
        .root(r, "r")
        .bodies()
        .label(one, "X1")
        .data();
    assert_eq!(data.clusters, ["mid", "leaf"]);
    assert_eq!(data.nodes.iter().filter(|n| n.label == "sin").count(), 1);
    assert_eq!(data.links.len(), 3);
    let sine = data.nodes.iter().find(|n| n.label == "sin").expect("sin");
    assert_eq!(sine.cluster, Some(1));
    assert!(data.nodes.iter().any(|n| n.label == "X1"));
    assert!(data
        .nodes
        .iter()
        .any(|n| n.label == "mid" && n.cluster.is_none()));
}

#[test]
fn inline_draws_every_instance_as_its_body() {
    // leaf(a) = sin(a); mid(x, y) = leaf(x) * y; two calls of mid at the top
    let mut g: Graph<F64> = Graph::new();
    let mut s = rsdag::Scope::new(&mut g, "leaf");
    let a = s.param("a");
    let sa = s.sin(a);
    let leaf = s.close(vec![sa]);
    let mut s = rsdag::Scope::new(&mut g, "mid");
    let (x, y) = (s.param("x"), s.param("y"));
    let l = s.call(leaf, 0, &[x]);
    let m = s.mul(l, y);
    let mid = s.close(vec![m]);
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
    // no call left to link to its body
    assert!(data.links.is_empty());
}
