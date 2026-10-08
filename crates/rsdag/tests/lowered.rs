//! The tapes of one system over one signature are views of one lowered
//! program: a view computes what its own compilation computes, value for
//! value, and its calls run bodies of the outputs it reads only.

use rsdag::{
    sparse_jacobian, tape::Op, ExprId, FuncId, Graph, Lowered, Node, ParamRole, ReduceOp, Scope,
    SymbolId, Tape, F64,
};

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

/// `d(va, vb, k, w)`: a current and a charge over a card parameter and an
/// instance parameter.
fn device(g: &mut Graph<F64>) -> FuncId {
    let mut s = Scope::new(g, "d");
    let va = s.param_with_role("va", ParamRole::State { id: 0 });
    let vb = s.param_with_role("vb", ParamRole::State { id: 1 });
    let k = s.param_with_role("k", ParamRole::Param);
    let w = s.param_with_role("w", ParamRole::Param);
    let dv = s.sub(va, vb);
    let kw = s.mul(k, w);
    let th = s.tanh(dv);
    let i = s.mul(kw, th);
    let e = s.exp(va);
    let i = s.add(i, e);
    let q = s.mul(kw, vb);
    let q = s.mul(q, va);
    s.close(vec![i, q])
}

struct System {
    g: Graph<F64>,
    currents: Vec<ExprId>,
    charges: Vec<ExprId>,
    jac_i: Vec<ExprId>,
    jac_q: Vec<ExprId>,
    inputs: Vec<SymbolId>,
    pure: Vec<bool>,
}

/// A ring of `n` cells, each a composite function calling the device twice
/// (so its calls are expanded from a template and merged), with the
/// Jacobians of its currents and charges.
fn ring(n: usize) -> System {
    let mut g: Graph<F64> = Graph::new();
    let d = device(&mut g);
    let (a, sa) = sym(&mut g, "cell.a");
    let (b, sb) = sym(&mut g, "cell.b");
    let (w, sw) = sym(&mut g, "cell.w");
    let (k, _) = sym(&mut g, "k");
    let half = g.konst_f64(0.5);
    let w2 = g.mul(w, half);
    let first = g.calls(d, &[0, 1], &[a, b, k, w]);
    let second = g.calls(d, &[0, 1], &[b, a, k, w2]);
    let i = g.add(first[0], second[0]);
    let q = g.add(first[1], second[1]);
    let cell = g.define_func("cell", vec![sa, sb, sw], vec![i, q]);
    let v: Vec<(ExprId, SymbolId)> = (0..n).map(|j| sym(&mut g, &format!("v{j}"))).collect();
    let ws: Vec<(ExprId, SymbolId)> = (0..n).map(|j| sym(&mut g, &format!("w{j}"))).collect();
    let mut terms: Vec<Vec<ExprId>> = vec![Vec::new(); n];
    let mut stores: Vec<Vec<ExprId>> = vec![Vec::new(); n];
    for j in 0..n {
        let next = v[(j + 1) % n].0;
        let outs = g.calls(cell, &[0, 1], &[v[j].0, next, ws[j].0]);
        terms[j].push(outs[0]);
        stores[j].push(outs[1]);
        let back = g.neg(outs[0]);
        terms[(j + 1) % n].push(back);
    }
    let currents: Vec<ExprId> = terms
        .into_iter()
        .map(|t| g.reduce(ReduceOp::Sum, t))
        .collect();
    let charges: Vec<ExprId> = stores
        .into_iter()
        .map(|t| g.reduce(ReduceOp::Sum, t))
        .collect();
    let states: Vec<SymbolId> = v.iter().map(|&(_, s)| s).collect();
    let flat = |rows: Vec<Vec<(usize, ExprId)>>| -> Vec<ExprId> {
        rows.into_iter().flatten().map(|(_, e)| e).collect()
    };
    let jac_i = flat(sparse_jacobian(&mut g, &currents, &states));
    let jac_q = flat(sparse_jacobian(&mut g, &charges, &states));
    let Node::Symbol(sk) = *g.node(k) else {
        unreachable!()
    };
    let params: Vec<SymbolId> = ws.iter().map(|&(_, s)| s).chain([sk]).collect();
    let pure: Vec<bool> = states
        .iter()
        .map(|_| false)
        .chain(params.iter().map(|_| true))
        .collect();
    let inputs: Vec<SymbolId> = states.into_iter().chain(params).collect();
    System {
        g,
        currents,
        charges,
        jac_i,
        jac_q,
        inputs,
        pure,
    }
}

fn bits(tape: &Tape, inputs: &[f64]) -> Vec<u64> {
    let (mut work, mut out) = (Vec::new(), Vec::new());
    tape.eval(inputs, &mut work, &mut out);
    out.iter().map(|v| v.to_bits()).collect()
}

/// Per call of the tape, the outputs its body computes per instance.
fn call_widths(tape: &Tape) -> Vec<u32> {
    let mut w: Vec<u32> = (tape.ops().iter())
        .filter_map(|op| match *op {
            Op::Call { n_out, .. } => Some(n_out),
            _ => None,
        })
        .collect();
    w.sort_unstable();
    w
}

#[test]
fn a_view_computes_what_its_own_compilation_does() {
    let s = ring(7);
    let values: Vec<f64> = (0..s.inputs.len()).map(|j| 0.1 + 0.07 * j as f64).collect();
    let all: Vec<ExprId> = (s.currents.iter())
        .chain(&s.charges)
        .chain(&s.jac_i)
        .chain(&s.jac_q)
        .copied()
        .collect();
    let views: Vec<Vec<ExprId>> = vec![
        s.currents.clone(),
        s.currents.iter().chain(&s.jac_i).copied().collect(),
        s.currents.iter().chain(&s.charges).copied().collect(),
        all.clone(),
        s.jac_q.clone(),
    ];
    for split in [false, true] {
        let pure = split.then_some(s.pure.as_slice());
        let lowered = Lowered::new(&s.g, &all, &s.inputs, pure);
        for roots in &views {
            let view = lowered.tape(&s.g, roots);
            let own = match pure {
                Some(p) => Tape::compile_split(&s.g, roots, &s.inputs, p),
                None => Tape::compile(&s.g, roots, &s.inputs),
            };
            assert_eq!(bits(&view, &values), bits(&own, &values), "split {split}");
            assert_eq!(call_widths(&view), call_widths(&own), "split {split}");
        }
    }
}

#[test]
fn a_view_of_every_root_is_the_compilation() {
    let s = ring(5);
    let all: Vec<ExprId> = s.currents.iter().chain(&s.jac_i).copied().collect();
    let lowered = Lowered::new(&s.g, &all, &s.inputs, Some(&s.pure));
    let view = lowered.tape(&s.g, &all);
    let own = Tape::compile_split(&s.g, &all, &s.inputs, &s.pure);
    assert_eq!(view.dump(), own.dump());
}
