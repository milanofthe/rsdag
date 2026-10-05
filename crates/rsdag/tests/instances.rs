//! The calls of one instance share one argument list, and everything that
//! walks calls (rewriting, specializing, differentiating, compiling) works
//! on the list once per instance. The results are the inlined graph's.

use rsdag::{sparse_jacobian, ExprId, FuncId, Graph, Node, ParamRole, Scope, SymbolId, Tape, F64};

const N: usize = 40;

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    match *g.node(e) {
        Node::Symbol(s) => (e, s),
        _ => unreachable!(),
    }
}

/// body(v, p): output k is p_k v_k^2 + sin(v_{k+1}) through a callee
/// + 3 v_{k+2}, every output reading four of the 2 N parameters.
fn body(g: &mut Graph<F64>) -> FuncId {
    let mut s = Scope::new(g, "dev");
    let (a, b) = (s.param("a"), s.param("b"));
    let sb = s.sin(b);
    let dev = s.close(vec![sb, a]);
    let mut s = Scope::new(g, "body");
    let v: Vec<ExprId> = (0..N)
        .map(|k| s.param_with_role(&format!("v{k}"), ParamRole::State { id: k as u32 }))
        .collect();
    let p: Vec<ExprId> = (0..N)
        .map(|k| s.param_with_role(&format!("p{k}"), ParamRole::Param))
        .collect();
    let outs = (0..N)
        .map(|k| {
            let sq = s.pow_i(v[k], 2);
            let q = s.mul(p[k], sq);
            let c = s.call(dev, 0, &[v[k], v[(k + 1) % N]]);
            let three = s.konst_f64(3.0);
            let lin = s.mul(three, v[(k + 2) % N]);
            let qc = s.add(q, c);
            s.add(qc, lin)
        })
        .collect();
    s.close(outs)
}

#[test]
fn calls_of_one_instance_are_the_calls_one_by_one() {
    let mut g: Graph<F64> = Graph::new();
    let f = body(&mut g);
    let args: Vec<ExprId> = (0..2 * N).map(|k| g.sym(&format!("x{k}"))).collect();
    let outs: Vec<u32> = (0..N as u32).collect();
    let n0 = g.len();
    let together = g.calls(f, &outs, &args);
    let one_by_one: Vec<ExprId> = outs.iter().map(|&k| g.call(f, k, &args)).collect();
    assert_eq!(together, one_by_one, "the same nodes");
    assert_eq!(g.len(), n0 + N, "one node per output, the list shared");

    // the same calls in another graph, built in another order: the same
    // fingerprints
    let mut h: Graph<F64> = Graph::new();
    let fh = body(&mut h);
    let hargs: Vec<ExprId> = (0..2 * N).map(|k| h.sym(&format!("x{k}"))).collect();
    let back: Vec<ExprId> = outs.iter().rev().map(|&k| h.call(fh, k, &hargs)).collect();
    for (a, b) in together.iter().zip(back.iter().rev()) {
        assert_eq!(g.fingerprint(*a), h.fingerprint(*b));
    }
}

#[test]
fn two_instances_rewrite_differentiate_and_compile_as_inlined() {
    let mut g: Graph<F64> = Graph::new();
    let f = body(&mut g);
    // instance 1 over free symbols; instance 2 with its first state at ground
    // and its parameters shared, so it is specialized
    let mut ins1 = Vec::new();
    let mut ins2 = Vec::new();
    let mut wrt = Vec::new();
    for k in 0..N {
        let (e, s) = sym(&mut g, &format!("a{k}"));
        ins1.push(e);
        wrt.push(s);
    }
    for k in 0..N {
        if k == 0 {
            ins2.push(g.zero());
        } else {
            let (e, s) = sym(&mut g, &format!("b{k}"));
            ins2.push(e);
            wrt.push(s);
        }
    }
    let mut params = Vec::new();
    for k in 0..N {
        let (e, s) = sym(&mut g, &format!("p{k}"));
        ins1.push(e);
        ins2.push(e);
        params.push(s);
    }
    let outs: Vec<u32> = (0..N as u32).collect();
    let mut rows = g.calls(f, &outs, &ins1);
    rows.extend(g.calls(f, &outs, &ins2));
    let jac = sparse_jacobian(&mut g, &rows, &wrt);
    for (i, row) in jac.iter().enumerate() {
        assert!(
            row.len() <= 3,
            "row {i} reads three states, not {}",
            row.len()
        );
    }
    let mut roots = rows.clone();
    roots.extend(jac.iter().flatten().map(|&(_, e)| e));
    // the parameters at a value, then the calls specialized
    let half = g.konst_f64(0.5);
    let at: rustc_hash::FxHashMap<SymbolId, ExprId> = params.iter().map(|&s| (s, half)).collect();
    let fixed = rsdag::substitute(&mut g, &roots, &at);
    let spec = g.specialize_calls(&fixed);
    let flat = g.inline_all(&fixed);
    // what depends on the states, through the calls and inlined alike
    let through = g.depends_on(&fixed, &wrt);
    let inlined: Vec<bool> = flat
        .iter()
        .map(|&e| g.free_symbols(e).iter().any(|s| wrt.contains(s)))
        .collect();
    assert_eq!(through, inlined);
    assert!(
        through.iter().any(|&d| !d),
        "some Jacobian entries are constant"
    );
    let ins: Vec<f64> = (0..wrt.len()).map(|k| 0.2 + 0.03 * k as f64).collect();
    let (mut w, mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    Tape::compile(&g, &fixed, &wrt).eval(&ins, &mut w, &mut a);
    Tape::compile(&g, &spec, &wrt).eval(&ins, &mut w, &mut b);
    Tape::compile(&g, &flat, &wrt).eval(&ins, &mut w, &mut c);
    for k in 0..a.len() {
        let tol = 1e-13 * (1.0 + c[k].abs());
        assert!(
            (a[k] - c[k]).abs() <= tol,
            "entry {k}: {} vs {}",
            a[k],
            c[k]
        );
        assert!(
            (b[k] - c[k]).abs() <= tol,
            "entry {k}: {} vs {}",
            b[k],
            c[k]
        );
    }
}

/// A hierarchy compiled as it should run: the composite functions (a body of
/// calls, a wrapper around two bodies) inlined, the leaf's calls from every
/// instance left as calls, the values those of the hierarchy.
#[test]
fn composite_functions_inline_and_leaves_stay_calls() {
    let mut g: Graph<F64> = Graph::new();
    let f = body(&mut g);
    // top(u, p) = body(u, p) + body(u shifted, p), one instance each
    let mut s = Scope::new(&mut g, "top");
    let u: Vec<ExprId> = (0..N).map(|k| s.param(&format!("u{k}"))).collect();
    let p: Vec<ExprId> = (0..N).map(|k| s.param(&format!("q{k}"))).collect();
    let a: Vec<ExprId> = u.iter().chain(&p).copied().collect();
    let b: Vec<ExprId> = (0..N)
        .map(|k| u[(k + 1) % N])
        .chain(p.iter().copied())
        .collect();
    let outs: Vec<u32> = (0..N as u32).collect();
    let fa = s.calls(f, &outs, &a);
    let fb = s.calls(f, &outs, &b);
    let sums: Vec<ExprId> = fa.iter().zip(&fb).map(|(&x, &y)| s.add(x, y)).collect();
    let top = s.close(sums);
    let mut args = Vec::new();
    let mut ins = Vec::new();
    for k in 0..2 * N {
        let (e, s) = sym(&mut g, &format!("z{k}"));
        args.push(e);
        ins.push(s);
    }
    let rows = g.calls(top, &outs, &args);
    let flat = g.inline_composite(&rows);
    let leaves: std::collections::BTreeSet<FuncId> = g
        .free_calls_in(&flat)
        .iter()
        .map(|&o| g.output(o).0)
        .collect();
    assert_eq!(leaves.len(), 1, "only the leaf is called");
    assert!(!leaves.contains(&f) && !leaves.contains(&top));
    let all = g.inline_all(&rows);
    let x: Vec<f64> = (0..2 * N).map(|k| 0.1 + 0.02 * k as f64).collect();
    let (mut w, mut r0, mut r1, mut r2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    Tape::compile(&g, &rows, &ins).eval(&x, &mut w, &mut r0);
    Tape::compile(&g, &flat, &ins).eval(&x, &mut w, &mut r1);
    Tape::compile(&g, &all, &ins).eval(&x, &mut w, &mut r2);
    for k in 0..N {
        assert!(
            (r0[k] - r1[k]).abs() <= 1e-13 * (1.0 + r0[k].abs()),
            "row {k}"
        );
        assert!(
            (r0[k] - r2[k]).abs() <= 1e-13 * (1.0 + r0[k].abs()),
            "row {k}"
        );
    }
}
