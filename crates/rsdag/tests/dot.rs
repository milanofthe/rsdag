//! The DOT views: one node per expression or instruction, a shared
//! subexpression once, faded nodes outside the focus, the prolog and main
//! phase as clusters with the state crossing between them as dashed edges.

use rsdag::dot::{number, reachable, GraphView, TapeView};
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
        .filter(|l| l.contains("fillcolor=\"#"))
        .filter(|l| l.contains("0d\""))
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
    assert!(dot.contains("label=\"v\"") && dot.contains("label=\"e\""));
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
