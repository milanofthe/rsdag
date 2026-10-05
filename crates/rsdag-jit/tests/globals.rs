//! Native code over calls of a body that reads globals (a model card): the
//! interpreter's values bit for bit, plain and split over the card, on a
//! batch wide enough for lanes.

use rsdag::{ExprId, Graph, Node, ParamRole, ReduceOp, Scope, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

#[test]
fn native_calls_read_their_globals() {
    let mut g: Graph<F64> = Graph::new();
    let d = {
        let mut s = Scope::new(&mut g, "d");
        let t1 = s.global("t1");
        let t2 = s.global("t2");
        let va = s.param_with_role("va", ParamRole::State { id: 0 });
        let vb = s.param_with_role("vb", ParamRole::State { id: 1 });
        let w = s.param_with_role("w", ParamRole::Param);
        let k = s.mul(t1, w);
        let dv = s.sub(va, vb);
        let th = s.tanh(dv);
        let i = s.mul(k, th);
        let x = s.mul(t2, va);
        let e = s.exp(x);
        let i = s.add(i, e);
        s.close(vec![i])
    };
    let n = 37;
    let v: Vec<(ExprId, SymbolId)> = (0..=n).map(|k| sym(&mut g, &format!("v{k}"))).collect();
    let w: Vec<(ExprId, SymbolId)> = (0..n).map(|k| sym(&mut g, &format!("w{k}"))).collect();
    let (_, t1) = sym(&mut g, "t1");
    let (_, t2) = sym(&mut g, "t2");
    let mut terms: Vec<Vec<ExprId>> = vec![Vec::new(); n + 1];
    for k in 0..n {
        let i = g.call(d, 0, &[v[k].0, v[k + 1].0, w[k].0]);
        terms[k].push(i);
        let ni = g.neg(i);
        terms[k + 1].push(ni);
    }
    let roots: Vec<ExprId> = terms
        .into_iter()
        .map(|t| g.reduce(ReduceOp::Sum, t))
        .collect();
    let syms: Vec<SymbolId> = v
        .iter()
        .chain(&w)
        .map(|&(_, s)| s)
        .chain([t1, t2])
        .collect();
    let x: Vec<f64> = (0..syms.len())
        .map(|k| 0.05 + 0.031 * k as f64 - (k % 3) as f64 * 0.02)
        .collect();
    let pure: Vec<bool> = (0..syms.len()).map(|k| k > n).collect();
    for split in [false, true] {
        let tape = if split {
            Tape::compile_split(&g, &roots, &syms, &pure)
        } else {
            Tape::compile(&g, &roots, &syms)
        };
        let native = NativeTape::compile(&tape).expect("native code");
        let (mut w1, mut o1) = (Vec::new(), Vec::new());
        tape.eval_prolog(&x, &mut w1);
        tape.eval_main(&x, &mut w1, &mut o1);
        let (mut w2, mut o2) = (Vec::new(), Vec::new());
        native.eval_prolog(&x, &mut w2);
        native.eval_main(&x, &mut w2, &mut o2);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&o1), bits(&o2), "split {split}");
    }
}

#[test]
fn native_bound_calls_are_the_full_ones() {
    // the same ring, its card passed by every call or bound once per card
    let build = |bound: bool| {
        let mut g: Graph<F64> = Graph::new();
        let d = {
            let mut s = Scope::new(&mut g, "d");
            let va = s.param_with_role("va", ParamRole::State { id: 0 });
            let vb = s.param_with_role("vb", ParamRole::State { id: 1 });
            let c0 = s.param_with_role("c0", ParamRole::Param);
            let c1 = s.param_with_role("c1", ParamRole::Param);
            let dv = s.sub(va, vb);
            let th = s.tanh(dv);
            let i = s.mul(c0, th);
            let x = s.mul(c1, va);
            let e = s.exp(x);
            let i = s.add(i, e);
            s.close(vec![i])
        };
        let n = 29;
        let v: Vec<(ExprId, SymbolId)> = (0..n).map(|k| sym(&mut g, &format!("v{k}"))).collect();
        let card: Vec<(ExprId, SymbolId)> =
            (0..4).map(|k| sym(&mut g, &format!("card{k}"))).collect();
        let binds = [
            g.bind(d, &[(2, card[0].0), (3, card[1].0)]),
            g.bind(d, &[(2, card[2].0), (3, card[3].0)]),
        ];
        let mut terms: Vec<Vec<ExprId>> = vec![Vec::new(); n];
        for k in 0..n {
            let (a, b) = (v[k].0, v[(k + 1) % n].0);
            let c = 2 * (k % 2);
            let i = if bound {
                g.call_bound(binds[k % 2], 0, &[a, b])
            } else {
                g.call(d, 0, &[a, b, card[c].0, card[c + 1].0])
            };
            terms[k].push(i);
            let ni = g.neg(i);
            terms[(k + 1) % n].push(ni);
        }
        let roots: Vec<ExprId> = terms
            .into_iter()
            .map(|t| g.reduce(ReduceOp::Sum, t))
            .collect();
        let syms: Vec<SymbolId> = v.iter().chain(&card).map(|&(_, s)| s).collect();
        (g, roots, syms, n)
    };
    let mut outs = Vec::new();
    for bound in [false, true] {
        let (g, roots, syms, n) = build(bound);
        let x: Vec<f64> = (0..syms.len()).map(|k| 0.1 + 0.029 * k as f64).collect();
        let pure: Vec<bool> = (0..syms.len()).map(|k| k >= n).collect();
        let tape = Tape::compile_split(&g, &roots, &syms, &pure);
        let native = NativeTape::compile(&tape).expect("native code");
        let (mut w, mut o) = (Vec::new(), Vec::new());
        native.eval_prolog(&x, &mut w);
        native.eval_main(&x, &mut w, &mut o);
        outs.push(o.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }
    assert_eq!(outs[0], outs[1]);
}
