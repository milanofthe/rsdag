//! Programs built apart, composed through the API: a function of one graph
//! imported into another with what it calls, called there, and compiled as
//! one program whose only calls are its leaves.

mod common;
use common::sym;

use rsdag::Program;
use rsdag::{
    differentiate, ExprId, FuncId, Graph, OutputRole, ParamRole, Scope, SymbolId, Tape, F64,
};
use rustc_hash::FxHashMap;

fn library() -> (Graph<F64>, FuncId, FuncId) {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "dev");
    let a = s.param("a", ParamRole::State { id: 0 });
    let b = s.param("b", ParamRole::Param);
    let ab = s.mul(a, b);
    let sa = s.sin(a);
    let dev = s.close([(ab, OutputRole::Plain), (sa, OutputRole::Plain)]);
    let mut s = Scope::new(&mut g, "blk");
    let (x, y, z) = (
        s.param("x", ParamRole::Free),
        s.param("y", ParamRole::Free),
        s.param("z", ParamRole::Free),
    );
    let first = s.calls(dev, &[0, 1], &[x, y]);
    let second = s.calls(dev, &[0, 1], &[y, z]);
    let o0 = s.add(first[0], second[1]);
    let blk = s.close([(o0, OutputRole::Plain), (second[0], OutputRole::Plain)]);
    let (p, q) = (g.sym("p"), g.sym("q"));
    let r = g.sym("r");
    let call = g.call(blk, 0, &[p, q, r]);
    let q = g.symbol_id("q");
    differentiate(&mut g, call, q);
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
    let roles = g.func(b).output_roles();
    assert!(
        roles.contains(&OutputRole::Derivative { of: 0, wrt: 1 }),
        "the derivative output comes along with the imported function"
    );
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
    let (p, q, r, s) = (x[0], x[1], x[2], x[3]);
    let want = [p * q + q.sin(), q * r, q * r + r.sin(), r * s];
    for k in 0..4 {
        assert_eq!(
            a[k].to_bits(),
            want[k].to_bits(),
            "{k}: {} vs {}",
            a[k],
            want[k]
        );
        assert_eq!(
            c[k].to_bits(),
            want[k].to_bits(),
            "{k}: {} vs {}",
            c[k],
            want[k]
        );
    }
    let flat = g.inline_composite(&roots);
    let called: Vec<FuncId> = g
        .free_calls_in(&flat)
        .iter()
        .map(|&o| g.output(o).0)
        .collect();
    assert!(
        !called.is_empty() && called.iter().all(|&f| f == imported[&dev]),
        "only the leaf is called in the composed program"
    );
}
