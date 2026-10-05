//! Functions that read globals: symbols of their body that are not their
//! parameters, the same for every call (a model card's parameters). A call
//! reads them besides its arguments, in every layer: evaluation, the tape
//! (split or not), support and dependence, the sparse Jacobian (in the
//! states and in the globals), substitution of a global, inlining, and a
//! caller binding a global by a parameter of its own name. Each against the
//! program with every call inlined, where the globals are plain symbols.

use std::collections::HashMap;

use rsdag::{
    eval, sparse_jacobian, substitute, ExprId, FuncId, Graph, Node, ParamRole, ReduceOp, Scope,
    SymbolId, Tape, F64,
};
use rustc_hash::FxHashMap;

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

/// `d(va, vb)` reading the card `t1`, `t2`: `t1 tanh(va - vb) + exp(t2 va)`
/// and its negation.
fn device(g: &mut Graph<F64>) -> FuncId {
    let mut s = Scope::new(g, "d");
    let t1 = s.global("t1");
    let t2 = s.global("t2");
    let va = s.param_with_role("va", ParamRole::State { id: 0 });
    let vb = s.param_with_role("vb", ParamRole::State { id: 1 });
    let dv = s.sub(va, vb);
    let th = s.tanh(dv);
    let i = s.mul(t1, th);
    let x = s.mul(t2, va);
    let e = s.exp(x);
    let i = s.add(i, e);
    let n = s.neg(i);
    s.close(vec![i, n])
}

/// `gate(a, b, c)`: a device from `a` to `b` and one from `b` to `c`; the
/// currents into `a`, `b`, `c`.
fn gate(g: &mut Graph<F64>, d: FuncId) -> FuncId {
    let mut s = Scope::new(g, "gate");
    let a = s.param_with_role("a", ParamRole::State { id: 0 });
    let b = s.param_with_role("b", ParamRole::State { id: 1 });
    let c = s.param_with_role("c", ParamRole::State { id: 2 });
    let ab = s.calls(d, &[0, 1], &[a, b]);
    let bc = s.calls(d, &[0, 1], &[b, c]);
    let ib = s.reduce(ReduceOp::Sum, vec![ab[1], bc[0]]);
    s.close(vec![ab[0], ib, bc[1]])
}

struct Circuit {
    g: Graph<F64>,
    roots: Vec<ExprId>,
    states: Vec<SymbolId>,
    card: Vec<SymbolId>,
}

/// A chain of `n` gates over nodes `v0..v2n`.
fn circuit(n: usize) -> Circuit {
    let mut g: Graph<F64> = Graph::new();
    let d = device(&mut g);
    let gt = gate(&mut g, d);
    let (_, t1) = sym(&mut g, "t1");
    let (_, t2) = sym(&mut g, "t2");
    let v: Vec<(ExprId, SymbolId)> = (0..=2 * n).map(|k| sym(&mut g, &format!("v{k}"))).collect();
    let mut terms: Vec<Vec<ExprId>> = vec![Vec::new(); 2 * n + 1];
    for k in 0..n {
        let i = g.calls(
            gt,
            &[0, 1, 2],
            &[v[2 * k].0, v[2 * k + 1].0, v[2 * k + 2].0],
        );
        for (j, ij) in i.into_iter().enumerate() {
            terms[2 * k + j].push(ij);
        }
    }
    let roots = terms
        .into_iter()
        .map(|t| g.reduce(ReduceOp::Sum, t))
        .collect();
    Circuit {
        g,
        roots,
        states: v.iter().map(|&(_, s)| s).collect(),
        card: vec![t1, t2],
    }
}

fn env(c: &Circuit) -> HashMap<SymbolId, f64> {
    let mut env: HashMap<SymbolId, f64> = c
        .states
        .iter()
        .enumerate()
        .map(|(k, &s)| (s, 0.1 + 0.07 * k as f64))
        .collect();
    env.insert(c.card[0], 0.8);
    env.insert(c.card[1], 0.3);
    env
}

fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= 1e-13 * (1.0 + x.abs()),
            "value {k}: {x:e} vs {y:e}"
        );
    }
}

#[test]
fn a_call_reads_the_globals_of_its_body() {
    let c = circuit(4);
    let d = FuncId(0);
    let gt = FuncId(1);
    let card: Vec<SymbolId> =
        c.g.globals(d)
            .iter()
            .map(|&e| match *c.g.node(e) {
                Node::Symbol(s) => s,
                _ => unreachable!(),
            })
            .collect();
    assert_eq!(card, c.card);
    assert_eq!(c.g.globals(gt).len(), 2, "the gate reads its devices' card");
    let roots = c.roots.clone();
    let support = c.g.support_in(&roots);
    assert!(support.contains(&c.card[0]) && support.contains(&c.card[1]));
    assert!(c.g.depends_on(&roots[..1], &[c.card[1]])[0]);
    // what the roots mention, not what the bodies they call read
    assert!(!c.g.free_symbols_in(&roots).contains(&c.card[0]));
}

#[test]
fn calls_with_globals_evaluate_as_inlined() {
    let mut c = circuit(4);
    let env = env(&c);
    let roots = c.roots.clone();
    let flat = c.g.inline_all(&roots);
    let want: Vec<f64> = eval(&c.g, &flat, &env);
    close(&eval(&c.g, &roots, &env), &want);
    let composed = c.g.inline_composite(&roots);
    close(&eval(&c.g, &composed, &env), &want);
    // the tape, plain and split over the card
    let syms: Vec<SymbolId> = c.states.iter().chain(&c.card).copied().collect();
    let x: Vec<f64> = syms.iter().map(|s| env[s]).collect();
    let tape = Tape::compile(&c.g, &roots, &syms);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&x, &mut w, &mut o);
    close(&o, &want);
    let pure: Vec<bool> = (0..syms.len()).map(|k| k >= c.states.len()).collect();
    let split = Tape::compile_split(&c.g, &roots, &syms, &pure);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    split.eval_prolog(&x, &mut w);
    split.eval_main(&x, &mut w, &mut o);
    close(&o, &want);
}

#[test]
fn the_jacobian_runs_through_the_globals() {
    let mut c = circuit(3);
    let env = env(&c);
    let roots = c.roots.clone();
    let wrt: Vec<SymbolId> = c.states.iter().chain(&c.card).copied().collect();
    let rows = sparse_jacobian(&mut c.g, &roots, &wrt);
    let flat = c.g.inline_all(&roots);
    let flat_rows = sparse_jacobian(&mut c.g, &flat, &wrt);
    for (i, (r, fr)) in rows.iter().zip(&flat_rows).enumerate() {
        let cols: Vec<usize> = r.iter().map(|&(j, _)| j).collect();
        let fcols: Vec<usize> = fr.iter().map(|&(j, _)| j).collect();
        assert_eq!(cols, fcols, "row {i}: the pattern");
        let es: Vec<ExprId> = r.iter().map(|&(_, e)| e).collect();
        let fes: Vec<ExprId> = fr.iter().map(|&(_, e)| e).collect();
        close(&eval(&c.g, &es, &env), &eval(&c.g, &fes, &env));
    }
    // every row reads the card
    let n = c.states.len();
    assert!(rows.iter().all(|r| r.iter().any(|&(j, _)| j >= n)));
}

#[test]
fn substituting_a_global_binds_the_bodies_reading_it() {
    let mut c = circuit(3);
    let mut env = env(&c);
    let roots = c.roots.clone();
    let (t1, v1) = (c.card[0], env[&c.card[0]]);
    let k = c.g.konst_f64(v1);
    let map: FxHashMap<SymbolId, ExprId> = [(t1, k)].into_iter().collect();
    let bound = substitute(&mut c.g, &roots, &map);
    let want = eval(&c.g, &roots, &env);
    env.remove(&t1);
    close(&eval(&c.g, &bound, &env), &want);
    assert!(!c.g.support_in(&bound).contains(&t1));
}

#[test]
fn a_parameter_binds_a_global_of_its_name_below() {
    // gate2(a, b, t1): its device reads `t1`, which gate2 takes as a
    // parameter: a call of gate2 binds it
    let mut g: Graph<F64> = Graph::new();
    let d = device(&mut g);
    let mut s = Scope::new(&mut g, "gate2");
    let a = s.param_with_role("a", ParamRole::State { id: 0 });
    let b = s.param_with_role("b", ParamRole::State { id: 1 });
    let t1 = s.param_with_role("t1", ParamRole::Param);
    let i = s.calls(d, &[0], &[a, b]);
    let twice = s.mul(i[0], t1);
    let gate2 = s.close(vec![twice]);
    assert_eq!(g.globals(gate2).len(), 1, "t2 only: t1 is a parameter");
    let (x, xs) = sym(&mut g, "x");
    let (y, ys) = sym(&mut g, "y");
    let (q, qs) = sym(&mut g, "q");
    let (_, t2s) = sym(&mut g, "t2");
    let root = g.call(gate2, 0, &[x, y, q]);
    let inlined = g.inline_all(&[root]);
    let env: HashMap<SymbolId, f64> = [(xs, 0.4), (ys, 0.1), (qs, 1.7), (t2s, 0.3)].into();
    // by hand: q * (q tanh(x - y) + exp(t2 x))
    let want = 1.7 * (1.7 * (0.4f64 - 0.1).tanh() + (0.3f64 * 0.4).exp());
    close(&eval(&g, &[root], &env), &[want]);
    close(&eval(&g, &inlined, &env), &[want]);
    let composed = g.inline_composite(&[root]);
    close(&eval(&g, &composed, &env), &[want]);
}
