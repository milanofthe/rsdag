//! Function bodies under the native backend: a symbolic function's body is
//! compiled once and called per instance, batched when the calls share a
//! shape, nested when a body calls another, and the result is the
//! interpreter's to the bit.

use rsdag::Program;
use rsdag::{ExprId, Graph, Node, OutputRole, ParamRole, Scope, SymbolId, Tape, F64};
use rsdag_jit::{Batch, NativeTape, Options};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

/// `cell(l, c, r) = exp(c - l) - exp(r - c) + 2c - 1`, as a function.
fn cell(g: &mut Graph<F64>) -> rsdag::FuncId {
    let mut s = Scope::new(g, "cell");
    let [l, c, r] = ["l", "c", "r"].map(|n| s.param(n, ParamRole::Free));
    let d1 = s.sub(c, l);
    let d2 = s.sub(r, c);
    let e1 = s.exp(d1);
    let e2 = s.exp(d2);
    let e = s.sub(e1, e2);
    let two = s.konst_f64(2.0);
    let t = s.mul(two, c);
    let one = s.one();
    let u = s.sub(t, one);
    let out = s.add(e, u);
    s.close([(out, OutputRole::Plain)])
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

#[test]
fn batched_instances_of_a_body_match_the_interpreter() {
    for n in [3usize, 64, 5000] {
        let mut g: Graph<F64> = Graph::new();
        let f = cell(&mut g);
        let xs: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
        let syms: Vec<SymbolId> = xs.iter().map(|&e| sym(&g, e)).collect();
        // Leaf arguments: the calls batch into one call over n groups.
        let roots: Vec<ExprId> = (0..n)
            .map(|i| g.call(f, 0, &[xs[(i + n - 1) % n], xs[i], xs[(i + 1) % n]]))
            .collect();
        let tape = Tape::compile(&g, &roots, &syms);
        let native = NativeTape::compile(&tape).expect("compile");
        let row: Vec<f64> = (0..n).map(|i| 0.1 + (i % 9) as f64 * 0.07).collect();
        let (mut w1, mut o1, mut w2, mut o2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        tape.eval(&row, &mut w1, &mut o1);
        native.eval(&row, &mut w2, &mut o2);
        assert!(same(&o1, &o2), "n = {n}: {o1:?} vs {o2:?}");
    }
}

#[test]
fn a_parallel_batch_is_the_serial_loop() {
    for n in [2usize, 64, 5000] {
        let mut g: Graph<F64> = Graph::new();
        let f = cell(&mut g);
        let xs: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
        let syms: Vec<SymbolId> = xs.iter().map(|&e| sym(&g, e)).collect();
        let roots: Vec<ExprId> = (0..n)
            .map(|i| g.call(f, 0, &[xs[(i + n - 1) % n], xs[i], xs[(i + 1) % n]]))
            .collect();
        let tape = Tape::compile(&g, &roots, &syms);
        let opts = Options {
            batch: Batch::Always,
            ..Options::default()
        };
        let par = NativeTape::compile_opts(&tape, &opts, &[]).expect("compile");
        let ser = NativeTape::compile(&tape).expect("compile");
        let row: Vec<f64> = (0..n).map(|i| 0.1 + (i % 9) as f64 * 0.07).collect();
        let (mut w1, mut o1, mut w2, mut o2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        ser.eval(&row, &mut w1, &mut o1);
        par.eval(&row, &mut w2, &mut o2);
        assert!(same(&o1, &o2), "n = {n}");
    }
}

#[test]
fn a_body_called_on_expressions_and_a_body_calling_a_body() {
    let mut g: Graph<F64> = Graph::new();
    let f = cell(&mut g);
    // A function whose body calls `cell` twice on transformed arguments.
    let (a, b) = (g.sym("a"), g.sym("b"));
    let ab = g.mul(a, b);
    let c1 = g.call(f, 0, &[a, ab, b]);
    let c2 = g.call(f, 0, &[b, c1, a]);
    let (sa, sb) = (sym(&g, a), sym(&g, b));
    let mut s = Scope::new(&mut g, "outer");
    s.param(sa, ParamRole::Free);
    s.param(sb, ParamRole::Free);
    let outer = s.close([(c2, OutputRole::Plain)]);
    let (x, y, z) = (g.sym("x"), g.sym("y"), g.sym("z"));
    let syms = vec![sym(&g, x), sym(&g, y), sym(&g, z)];
    let xy = g.add(x, y);
    let o1 = g.call(outer, 0, &[xy, z]);
    let o2 = g.call(f, 0, &[o1, x, xy]);
    let s = g.sin(o2);
    let tape = Tape::compile(&g, &[o1, o2, s], &syms);
    let native = NativeTape::compile(&tape).expect("compile");
    let (mut w1, mut o1v, mut w2, mut o2v) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for row in [[0.3, 0.2, 0.5], [1.0, -0.4, 0.1], [0.0, 0.0, 0.0]] {
        tape.eval(&row, &mut w1, &mut o1v);
        native.eval(&row, &mut w2, &mut o2v);
        assert!(same(&o1v, &o2v), "{row:?}: {o1v:?} vs {o2v:?}");
    }
}
