//! The DOT views: one node per expression or instruction, a shared
//! subexpression once, faded nodes outside the focus, the prolog and main
//! phase as clusters with the state crossing between them as dashed edges.

use rsdag::dot::{number, reachable, Blocks, GraphView, Kind, Notation, Style, TapeView, Theme};
use rsdag::{differentiate, ExprId, Graph, Node, SymbolId, Tape, F64};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

fn count(s: &str, pat: &str) -> usize {
    s.matches(pat).count()
}

#[test]
fn a_shared_subexpression_is_one_node() {
    let mut g: Graph<F64> = Graph::new();
    let (x, y) = (g.sym("x"), g.sym("y"));
    let xy = g.mul(x, y);
    let s = g.sin(xy);
    let f = g.add(s, xy);
    let dot = GraphView::new(&g).root(f, "f").render();
    // x, y, x*y, sin and +, and five edges between them.
    let nodes = dot
        .lines()
        .filter(|l| l.starts_with("  n") && l.contains(" [label="))
        .count();
    assert_eq!(nodes, 5, "{dot}");
    assert_eq!(count(&dot, " -> n"), 5, "{dot}");
    assert_eq!(count(&dot, "out0 ["), 1);
    assert_eq!(
        count(&dot, &format!("n{} -> ", xy.0)),
        2,
        "x*y feeds sin and +"
    );
    assert!(dot.starts_with("digraph G {") && dot.ends_with("}\n"));
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
    let dot = GraphView::new(&g)
        .root(f, "f")
        .root(df, "df/dx")
        .focus(keep.clone())
        .render();
    let faded = dot
        .lines()
        .filter(|l| l.contains("fontcolor=\"#8b8b8b40\""))
        .count();
    let all = reachable(&g, &[f, df]);
    // Every node only f reaches is faded, and f's result with them.
    assert_eq!(faded, all.len() - keep.len() + 1, "{dot}");
}

#[test]
fn a_split_tape_has_two_phases_and_state_edges() {
    let mut g: Graph<F64> = Graph::new();
    let (v, p, q) = (g.sym("v"), g.sym("p"), g.sym("q"));
    let pq = g.mul(p, q);
    let r = g.pow_i(pq, -1);
    let u = g.mul(v, r);
    let e = g.exp(u);
    let syms = [sym(&g, v), sym(&g, p), sym(&g, q)];
    let pure = [false, true, true];
    let tape = Tape::compile_split(&g, &[e], &syms, &pure);
    assert!(tape.prolog_len() > 0);
    let dot = TapeView::new(&tape)
        .inputs(&["v", "p", "q"])
        .params(&pure)
        .outputs(&["e"])
        .render();
    assert_eq!(count(&dot, "subgraph cluster_"), 2, "{dot}");
    for i in 0..tape.n_ops() {
        assert!(dot.contains(&format!("o{i} [")), "op {i} drawn");
    }
    assert!(count(&dot, "style=dashed") >= 1, "the state crosses: {dot}");
    assert!(dot.contains("label=\"v\"") && dot.contains("label=<<B>e</B>>"));
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
fn a_theme_with_one_line_color_and_opaque_fills() {
    let mut g: Graph<F64> = Graph::new();
    let (x, y) = (g.sym("x"), g.sym("y"));
    let xy = g.mul(x, y);
    let s = g.sin(xy);
    let f = g.add(s, xy);
    let theme = Theme {
        style: Style::Filled,
        text: "#000000",
        line: Some("#000000"),
        fill_alpha: "",
        op: "#E0E0E0",
        notation: Notation::Math,
        ..Theme::default()
    };
    let dot = GraphView::new(&g).theme(theme).root(f, "F").render();
    assert!(dot.contains("label=\"\u{00d7}\", shape=circle"), "{dot}");
    assert!(dot.contains("label=\"sin(\u{00b7})\""), "{dot}");
    assert!(
        dot.contains("color=\"#000000\", fillcolor=\"#E0E0E0\""),
        "{dot}"
    );
    assert!(!dot.contains("#E0E0E026"));
}

#[test]
fn blocks_have_bold_titles_notes_and_patterns() {
    let dot = Blocks::new(Theme::default(), "LR")
        .block("a", "Graph", &["one line", "**bold line**"])
        .note("n", "guard", &["x < y"])
        .pattern("p", &["x.", ".x"], "pattern")
        .group("tape", &["a", "n"])
        .row(&["a", "p"])
        .edge("a", "p", "")
        .accent_edge("a", "n", "")
        .caption("under it all")
        .render();
    assert!(
        dot.contains("<B>Graph</B></FONT><BR/>one line<BR/><B>bold line</B>>"),
        "{dot}"
    );
    assert!(dot.contains("x &lt; y"), "{dot}");
    assert_eq!(count(&dot, "BGCOLOR="), 2, "{dot}");
    assert_eq!(count(&dot, "subgraph cluster_0"), 1);
    assert!(
        dot.contains("style=\"rounded,dashed\", color=\"#3b82f6\""),
        "{dot}"
    );
    assert!(dot.contains("labelloc=b"));
    // A row lines up across the group's border.
    assert!(dot.contains("{ rank=same; a; p; }") && dot.contains("newrank=true"));
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

    let flat = GraphView::new(&g).root(r, "r").render();
    assert_eq!(count(&flat, "subgraph cluster_"), 0);
    let dot = GraphView::new(&g).root(r, "r").bodies().render();
    assert_eq!(count(&dot, "subgraph cluster_"), 2, "{dot}");
    assert!(
        dot.contains("<B>mid</B>") && dot.contains("<B>leaf</B>"),
        "{dot}"
    );
    // the sine of leaf's body drawn once, however many calls reach it
    assert_eq!(count(&dot, "label=\"sin\""), 1, "{dot}");
    // a dashed link per call: two of mid, one of leaf in mid's body
    assert_eq!(count(&dot, "style=dashed"), 3, "{dot}");

    // the same as data, a call relabelled by its instance
    let data = GraphView::new(&g)
        .root(r, "r")
        .bodies()
        .label(one, "X1")
        .data();
    assert_eq!(data.clusters, ["mid", "leaf"]);
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
    // nested frames in the DOT
    let dot = GraphView::new(&g).root(r, "r").inline().render();
    assert_eq!(count(&dot, "subgraph cluster_"), 4, "{dot}");
    assert!(!dot.contains("style=dashed"), "{dot}");
}
