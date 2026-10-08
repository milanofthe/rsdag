//! A bound function is its function with the binding: a call through it is
//! in every respect the call of the function with all its arguments. The
//! same circuit built both ways (the card passed by every call, or bound
//! once per card) evaluates, differentiates, specializes, inlines, compiles
//! and serializes to the same bits.

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

const CARD: usize = 6;

/// `d(va, vb, c0..c5, w)`: a current over a card of six and an instance
/// parameter.
fn device(g: &mut Graph<F64>) -> FuncId {
    let mut s = Scope::new(g, "d");
    let va = s.param_with_role("va", ParamRole::State { id: 0 });
    let vb = s.param_with_role("vb", ParamRole::State { id: 1 });
    let c: Vec<ExprId> = (0..CARD)
        .map(|k| s.param_with_role(&format!("c{k}"), ParamRole::Param))
        .collect();
    let w = s.param_with_role("w", ParamRole::Param);
    let dv = s.sub(va, vb);
    let k = s.reduce(ReduceOp::Sum, c.clone());
    let th = s.tanh(dv);
    let i = s.mul(k, th);
    let x = s.mul(c[1], va);
    let e = s.exp(x);
    let i = s.add(i, e);
    let i = s.mul(i, w);
    let q = s.mul(c[2], vb);
    s.close(vec![i, q])
}

struct Built {
    g: Graph<F64>,
    roots: Vec<ExprId>,
    states: Vec<SymbolId>,
    params: Vec<SymbolId>,
}

/// A ring of `n` devices over two cards (alternating), one with a ground
/// terminal; `bound` passes the card through a binding.
fn ring(n: usize, bound: bool) -> Built {
    let mut g: Graph<F64> = Graph::new();
    let d = device(&mut g);
    let v: Vec<(ExprId, SymbolId)> = (0..n).map(|k| sym(&mut g, &format!("v{k}"))).collect();
    let w: Vec<(ExprId, SymbolId)> = (0..n).map(|k| sym(&mut g, &format!("w{k}"))).collect();
    let cards: Vec<Vec<(ExprId, SymbolId)>> = (0..2)
        .map(|c| {
            (0..CARD)
                .map(|k| sym(&mut g, &format!("card{c}.{k}")))
                .collect()
        })
        .collect();
    let zero = g.konst_f64(0.0);
    let binds: Vec<_> = cards
        .iter()
        .map(|card| {
            let pairs: Vec<(u32, ExprId)> = card
                .iter()
                .enumerate()
                .map(|(k, &(e, _))| ((2 + k) as u32, e))
                .collect();
            g.bind(d, &pairs)
        })
        .collect();
    let mut terms: Vec<Vec<ExprId>> = vec![Vec::new(); n];
    for k in 0..n {
        let b = if k == 3 { zero } else { v[(k + 1) % n].0 };
        let card = k % 2;
        let outs = if bound {
            g.calls_bound(binds[card], &[0, 1], &[v[k].0, b, w[k].0])
        } else {
            let args: Vec<ExprId> = [v[k].0, b]
                .into_iter()
                .chain(cards[card].iter().map(|&(e, _)| e))
                .chain([w[k].0])
                .collect();
            g.calls(d, &[0, 1], &args)
        };
        terms[k].push(outs[0]);
        terms[k].push(outs[1]);
        let n1 = g.neg(outs[0]);
        terms[(k + 1) % n].push(n1);
    }
    let roots = terms
        .into_iter()
        .map(|t| g.reduce(ReduceOp::Sum, t))
        .collect();
    Built {
        g,
        roots,
        states: v.iter().map(|&(_, s)| s).collect(),
        params: w
            .iter()
            .chain(cards.iter().flatten())
            .map(|&(_, s)| s)
            .collect(),
    }
}

fn env(b: &Built) -> HashMap<SymbolId, f64> {
    b.states
        .iter()
        .chain(&b.params)
        .enumerate()
        .map(|(k, &s)| (s, 0.05 + 0.037 * k as f64))
        .collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The values of every entry of a sparse Jacobian, by row.
fn jacobian_values(b: &mut Built, wrt: &[SymbolId]) -> Vec<Vec<(usize, u64)>> {
    let roots = b.roots.clone();
    let rows = sparse_jacobian(&mut b.g, &roots, wrt);
    let env = env(b);
    rows.iter()
        .map(|r| {
            let es: Vec<ExprId> = r.iter().map(|&(_, e)| e).collect();
            let vals = eval(&b.g, &es, &env);
            r.iter().map(|&(j, _)| j).zip(bits(&vals)).collect()
        })
        .collect()
}

#[test]
fn a_bound_call_evaluates_and_differentiates_as_the_full_one() {
    let (mut full, mut bound) = (ring(9, false), ring(9, true));
    assert_eq!(
        bits(&eval(&full.g, &full.roots, &env(&full))),
        bits(&eval(&bound.g, &bound.roots, &env(&bound)))
    );
    let wrt_f: Vec<SymbolId> = full.states.iter().chain(&full.params).copied().collect();
    let wrt_b: Vec<SymbolId> = bound.states.iter().chain(&bound.params).copied().collect();
    assert_eq!(
        jacobian_values(&mut full, &wrt_f),
        jacobian_values(&mut bound, &wrt_b)
    );
    let rs = bound.roots.clone();
    let card0 = bound.params[9];
    assert!(bound.g.support_in(&rs).contains(&card0));
    assert!(
        bound.g.free_symbols_in(&rs).contains(&card0),
        "the binding is the caller's"
    );
}

#[test]
fn a_bound_call_compiles_and_inlines_as_the_full_one() {
    let (mut full, mut bound) = (ring(9, false), ring(9, true));
    let run = |b: &mut Built| -> Vec<Vec<u64>> {
        let syms: Vec<SymbolId> = b.states.iter().chain(&b.params).copied().collect();
        let env = env(b);
        let x: Vec<f64> = syms.iter().map(|s| env[s]).collect();
        let pure: Vec<bool> = (0..syms.len()).map(|k| k >= b.states.len()).collect();
        let roots = b.roots.clone();
        let mut out = Vec::new();
        for split in [false, true] {
            let tape = if split {
                Tape::compile_split(&b.g, &roots, &syms, &pure)
            } else {
                Tape::compile(&b.g, &roots, &syms)
            };
            let (mut w, mut o) = (Vec::new(), Vec::new());
            tape.eval_prolog(&x, &mut w);
            tape.eval_main(&x, &mut w, &mut o);
            out.push(bits(&o));
        }
        let spec = b.g.specialize_calls(&roots);
        out.push(bits(&eval(&b.g, &spec, &env)));
        let inl = b.g.inline_all(&roots);
        out.push(bits(&eval(&b.g, &inl, &env)));
        let comp = b.g.inline_composite(&roots);
        out.push(bits(&eval(&b.g, &comp, &env)));
        out
    };
    assert_eq!(run(&mut full), run(&mut bound));
}

#[test]
fn a_main_phase_gathers_what_it_reads() {
    // the device's main phase reads its states, `w` and the card entries
    // it multiplies a state by; the card's sum is the prolog's
    let b = ring(9, true);
    let syms: Vec<SymbolId> = b.states.iter().chain(&b.params).copied().collect();
    let pure: Vec<bool> = (0..syms.len()).map(|k| k >= b.states.len()).collect();
    let tape = Tape::compile_split(&b.g, &b.roots, &syms, &pure);
    let calls: Vec<(u32, u32)> = tape
        .ops()
        .iter()
        .filter_map(|op| match *op {
            rsdag::tape::Op::Call {
                n_args,
                n_in,
                state,
                ..
            } if state != rsdag::tape::NO_STATE => Some((n_args, n_in)),
            _ => None,
        })
        .collect();
    assert!(!calls.is_empty(), "the calls keep a state");
    assert!(calls.iter().all(|&(n_args, n_in)| n_in == 5 && n_args == 9));
}

#[test]
fn substitution_reaches_into_a_binding() {
    // a card parameter made a constant: the bound circuit as the full one
    let run = |mut b: Built| -> (Vec<u64>, bool) {
        let mut env = env(&b);
        let roots = b.roots.clone();
        let c = b.params[b.states.len() + 5];
        let k = b.g.konst_f64(env[&c]);
        let map: FxHashMap<SymbolId, ExprId> = [(c, k)].into_iter().collect();
        let sub = substitute(&mut b.g, &roots, &map);
        env.remove(&c);
        (
            bits(&eval(&b.g, &sub, &env)),
            b.g.free_symbols_in(&sub).contains(&c),
        )
    };
    let (full, bound) = (run(ring(5, false)), run(ring(5, true)));
    assert_eq!(full, bound);
    assert!(!bound.1, "the constant replaced the symbol");
}

#[test]
fn substitution_reaches_every_binding_of_one_argument_list() {
    // two devices on the same terminals with the same instance parameter,
    // each on its own card: one argument list, two bindings
    let mut g: Graph<F64> = Graph::new();
    let d = device(&mut g);
    let (v0, s0) = sym(&mut g, "v0");
    let (v1, _) = sym(&mut g, "v1");
    let (v2, _) = sym(&mut g, "v2");
    let (w, _) = sym(&mut g, "w");
    let cards: Vec<Vec<ExprId>> = (0..2)
        .map(|c| {
            (0..CARD)
                .map(|k| sym(&mut g, &format!("card{c}.{k}")).0)
                .collect()
        })
        .collect();
    let mut bound = Vec::new();
    let mut full = Vec::new();
    for card in &cards {
        let pairs: Vec<(u32, ExprId)> = (card.iter().enumerate())
            .map(|(k, &e)| ((2 + k) as u32, e))
            .collect();
        let b = g.bind(d, &pairs);
        bound.push(g.call_bound(b, 0, &[v0, v1, w]));
        let args: Vec<ExprId> = [v2, v1]
            .into_iter()
            .chain(card.iter().copied())
            .chain([w])
            .collect();
        full.push(g.call(d, 0, &args));
    }
    let map: FxHashMap<SymbolId, ExprId> = [(s0, v2)].into_iter().collect();
    let sub = substitute(&mut g, &bound, &map);
    let env: HashMap<SymbolId, f64> = (g.free_symbols_in(&full).into_iter().enumerate())
        .map(|(k, s)| (s, 0.05 + 0.037 * k as f64))
        .collect();
    assert_eq!(bits(&eval(&g, &sub, &env)), bits(&eval(&g, &full, &env)));
}

#[test]
fn inlining_reaches_every_binding_of_one_argument_list() {
    // a body of two devices on the same terminals, each on its own card;
    // the devices stay calls
    let mut g: Graph<F64> = Graph::new();
    let d = device(&mut g);
    let (a, sa) = sym(&mut g, "a");
    let (b, sb) = sym(&mut g, "b");
    let (w, sw) = sym(&mut g, "w");
    let terms: Vec<ExprId> = (0..2)
        .map(|c| {
            let pairs: Vec<(u32, ExprId)> = (0..CARD)
                .map(|k| ((2 + k) as u32, sym(&mut g, &format!("card{c}.{k}")).0))
                .collect();
            let bound = g.bind(d, &pairs);
            g.call_bound(bound, 0, &[a, b, w])
        })
        .collect();
    let body = g.reduce(ReduceOp::Sum, terms);
    let pair = g.define_func("pair", vec![sa, sb, sw], vec![body]);
    let args: Vec<ExprId> = ["v0", "v1", "w0"]
        .iter()
        .map(|n| sym(&mut g, n).0)
        .collect();
    let call = g.call(pair, 0, &args);
    let inlined = g.inline_composite(&[call]);
    let env: HashMap<SymbolId, f64> = (g.free_symbols_in(&[call]).into_iter().enumerate())
        .map(|(k, s)| (s, 0.05 + 0.037 * k as f64))
        .collect();
    assert_eq!(
        bits(&eval(&g, &inlined, &env)),
        bits(&eval(&g, &[call], &env))
    );
}

#[cfg(feature = "serde")]
#[test]
fn a_module_keeps_its_bindings() {
    let b = ring(5, true);
    let module = b.g.to_module();
    let (g2, map) = Graph::from_module(&module).expect("a valid module");
    let roots: Vec<ExprId> = b.roots.iter().map(|r| map.exprs[r.0 as usize]).collect();
    let env1 = env(&b);
    let env2: HashMap<SymbolId, f64> = env1
        .iter()
        .map(|(s, &v)| (map.symbols[s.0 as usize], v))
        .collect();
    assert_eq!(
        bits(&eval(&b.g, &b.roots, &env1)),
        bits(&eval(&g2, &roots, &env2))
    );
}
