//! Programs built apart, composed through the API: a function of one graph
//! imported into another with what it calls, called there, and compiled as
//! one program whose only calls are its leaves.

use rsdag::{ExprId, FuncId, Graph, Node, OutputRole, ParamRole, Scope, SymbolId, Tape, F64};
use rustc_hash::FxHashMap;

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    match *g.node(e) {
        Node::Symbol(s) => (e, s),
        _ => unreachable!(),
    }
}

/// A leaf `dev(a, b) = [a b, sin a]` and a composite
/// `blk(x, y, z) = [dev0(x, y) + dev1(y, z), dev0(y, z)]` over it, that is
/// `[x y + sin y, y z]`.
fn library() -> (Graph<F64>, FuncId, FuncId) {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "dev");
    let a = s.param_with_role("a", ParamRole::State { id: 0 });
    let b = s.param_with_role("b", ParamRole::Param);
    let ab = s.mul(a, b);
    let sa = s.sin(a);
    let dev = s.close(vec![ab, sa]);
    let mut s = Scope::new(&mut g, "blk");
    let (x, y, z) = (s.param("x"), s.param("y"), s.param("z"));
    let first = s.calls(dev, &[0, 1], &[x, y]);
    let second = s.calls(dev, &[0, 1], &[y, z]);
    let o0 = s.add(first[0], second[1]);
    let blk = s.close(vec![o0, second[0]]);
    // a derivative output comes along with the function
    g.derivative_output(blk, 0, 1);
    (g, dev, blk)
}

#[test]
fn an_imported_function_brings_what_it_calls_once() {
    let (lib, dev, blk) = library();
    let mut g: Graph<F64> = Graph::new();
    let mut imported = FxHashMap::default();
    let b = g.import(&lib, blk, &mut imported);
    let again = g.import(&lib, blk, &mut imported);
    assert_eq!(b, again);
    assert_eq!(g.n_funcs(), 2, "blk and the dev it calls");
    let d = imported[&dev];
    assert_eq!(g.func(d).param_roles(), lib.func(dev).param_roles());
    let k = g.func(b).derivative(0, 1).expect("the derivative output");
    assert!(matches!(
        g.func(b).output_roles()[k as usize],
        OutputRole::Derivative { of: 0, wrt: 1 }
    ));
}

#[test]
fn a_composed_program_runs_as_the_functions_do() {
    let (lib, dev, blk) = library();
    let mut g: Graph<F64> = Graph::new();
    let mut imported = FxHashMap::default();
    let b = g.import(&lib, blk, &mut imported);
    let ins: Vec<(ExprId, SymbolId)> = ["p", "q", "r", "s"]
        .iter()
        .map(|n| sym(&mut g, n))
        .collect();
    let e: Vec<ExprId> = ins.iter().map(|&(e, _)| e).collect();
    let syms: Vec<SymbolId> = ins.iter().map(|&(_, s)| s).collect();
    // two instances of the imported block, one sharing its leaf with the
    // other through the block's own calls
    let one = g.calls(b, &[0, 1], &[e[0], e[1], e[2]]);
    let two = g.calls(b, &[0, 1], &[e[1], e[2], e[3]]);
    let mut roots = one.clone();
    roots.extend(two);
    let composed = Tape::compile(&g, &roots, &syms);
    let hierarchical = Tape::compile(&g, &roots, &syms);
    let x: [f64; 4] = [0.3, -0.7, 1.1, 0.4];
    let (mut w, mut a, mut c) = (Vec::new(), Vec::new(), Vec::new());
    composed.eval(&x, &mut w, &mut a);
    hierarchical.eval(&x, &mut w, &mut c);
    // the values the library computes for the same arguments
    let (p, q, r, s) = (x[0], x[1], x[2], x[3]);
    let want = [p * q + q.sin(), q * r, q * r + r.sin(), r * s];
    for k in 0..4 {
        assert!(
            (a[k] - want[k]).abs() < 1e-15,
            "{k}: {} vs {}",
            a[k],
            want[k]
        );
        assert!(
            (c[k] - want[k]).abs() < 1e-15,
            "{k}: {} vs {}",
            c[k],
            want[k]
        );
    }
    // only the leaf is called in the composed program
    let flat = g.inline_composite(&roots);
    let called: Vec<FuncId> = g
        .free_calls_in(&flat)
        .iter()
        .map(|&o| g.output(o).0)
        .collect();
    assert!(!called.is_empty() && called.iter().all(|&f| f == imported[&dev]));
}
