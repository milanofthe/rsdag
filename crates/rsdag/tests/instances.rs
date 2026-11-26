//! The calls of one instance share one argument list, and everything that
//! walks calls (rewriting, specializing, differentiating, compiling) works
//! on the list once per instance. The results are the inlined graph's.

mod common;
use common::constants::REORDERED;
use common::sym;

use rsdag::Program;
use rsdag::{
    sparse_jacobian, ExprId, FuncId, Graph, OutputRole, ParamRole, Scope, SymbolId, Tape, F64,
};

const N: usize = 40;

fn body(g: &mut Graph<F64>) -> FuncId {
    let mut s = Scope::new(g, "dev");
    let (a, b) = (s.param("a", ParamRole::Free), s.param("b", ParamRole::Free));
    let sb = s.sin(b);
    let dev = s.close([(sb, OutputRole::Plain), (a, OutputRole::Plain)]);
    let mut s = Scope::new(g, "body");
    let v: Vec<ExprId> = (0..N)
        .map(|k| s.param(&format!("v{k}"), ParamRole::State { id: k as u32 }))
        .collect();
    let p: Vec<ExprId> = (0..N)
        .map(|k| s.param(&format!("p{k}"), ParamRole::Param))
        .collect();
    let outs: Vec<ExprId> = (0..N)
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
    s.close(outs.into_iter().map(|e| (e, OutputRole::Plain)))
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

    let mut h: Graph<F64> = Graph::new();
    let fh = body(&mut h);
    let hargs: Vec<ExprId> = (0..2 * N).map(|k| h.sym(&format!("x{k}"))).collect();
    let back: Vec<ExprId> = outs.iter().rev().map(|&k| h.call(fh, k, &hargs)).collect();
    for (a, b) in together.iter().zip(back.iter().rev()) {
        assert_eq!(
            rsdag::to_string(&g, *a),
            rsdag::to_string(&h, *b),
            "the same calls built in another order in another graph"
        );
    }
}

#[test]
fn two_instances_rewrite_differentiate_and_compile_as_inlined() {
    let mut g: Graph<F64> = Graph::new();
    let f = body(&mut g);
    let mut ins_over_free_symbols = Vec::new();
    let mut ins_grounded_and_sharing_params = Vec::new();
    let mut wrt = Vec::new();
    for k in 0..N {
        let (e, s) = sym(&mut g, &format!("a{k}"));
        ins_over_free_symbols.push(e);
        wrt.push(s);
    }
    for k in 0..N {
        if k == 0 {
            ins_grounded_and_sharing_params.push(g.zero());
        } else {
            let (e, s) = sym(&mut g, &format!("b{k}"));
            ins_grounded_and_sharing_params.push(e);
            wrt.push(s);
        }
    }
    let mut params = Vec::new();
    for k in 0..N {
        let (e, s) = sym(&mut g, &format!("p{k}"));
        ins_over_free_symbols.push(e);
        ins_grounded_and_sharing_params.push(e);
        params.push(s);
    }
    let outs: Vec<u32> = (0..N as u32).collect();
    let mut rows = g.calls(f, &outs, &ins_over_free_symbols);
    rows.extend(g.calls(f, &outs, &ins_grounded_and_sharing_params));
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
    let half = g.konst_f64(0.5);
    let at: rustc_hash::FxHashMap<SymbolId, ExprId> = params.iter().map(|&s| (s, half)).collect();
    let fixed = rsdag::substitute(&mut g, &roots, &at);
    let spec = g.specialize_calls(&fixed);
    let flat = g.inline_all(&fixed);
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
        let tol = REORDERED * (1.0 + c[k].abs());
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

#[test]
fn composite_functions_inline_and_leaves_stay_calls() {
    let mut g: Graph<F64> = Graph::new();
    let f = body(&mut g);
    let mut s = Scope::new(&mut g, "top");
    let u: Vec<ExprId> = (0..N)
        .map(|k| s.param(&format!("u{k}"), ParamRole::Free))
        .collect();
    let p: Vec<ExprId> = (0..N)
        .map(|k| s.param(&format!("q{k}"), ParamRole::Free))
        .collect();
    let a: Vec<ExprId> = u.iter().chain(&p).copied().collect();
    let b: Vec<ExprId> = (0..N)
        .map(|k| u[(k + 1) % N])
        .chain(p.iter().copied())
        .collect();
    let outs: Vec<u32> = (0..N as u32).collect();
    let fa = s.calls(f, &outs, &a);
    let fb = s.calls(f, &outs, &b);
    let sums: Vec<ExprId> = fa.iter().zip(&fb).map(|(&x, &y)| s.add(x, y)).collect();
    let top = s.close(sums.into_iter().map(|e| (e, OutputRole::Plain)));
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
            (r0[k] - r1[k]).abs() <= REORDERED * (1.0 + r0[k].abs()),
            "row {k}"
        );
        assert!(
            (r0[k] - r2[k]).abs() <= REORDERED * (1.0 + r0[k].abs()),
            "row {k}"
        );
    }
}

#[test]
fn leaf_calls_with_the_same_operands_merge_into_one_instance() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "leaf");
    let v = s.param("v", ParamRole::State { id: 0 });
    let p = s.param("p", ParamRole::Param);
    let ep = s.exp(p);
    let sv = s.sin(v);
    let o = s.mul(sv, ep);
    let leaf = s.close([(o, OutputRole::Plain)]);
    let mut s = Scope::new(&mut g, "top");
    let v = s.param("v", ParamRole::State { id: 0 });
    let p = s.param("p", ParamRole::Param);
    let w = s.param("w", ParamRole::State { id: 1 });
    let c = s.call(leaf, 0, &[v, p]);
    let o = s.add(c, w);
    let top = s.close([(o, OutputRole::Plain)]);
    let (x0, sx0) = sym(&mut g, "x0");
    let (x1, sx1) = sym(&mut g, "x1");
    let (p0, sp0) = sym(&mut g, "p0");
    let (p1, sp1) = sym(&mut g, "p1");
    let ws: Vec<(ExprId, SymbolId)> = (0..4).map(|k| sym(&mut g, &format!("w{k}"))).collect();
    let pairs = [(x0, p0), (x0, p0), (x1, p0), (x0, p0), (x0, p1)];
    let roots: Vec<ExprId> = (pairs.iter().enumerate())
        .map(|(k, &(x, q))| g.call(top, 0, &[x, q, ws[k % 4].0]))
        .collect();
    let mut syms = vec![sx0, sx1, sp0, sp1];
    syms.extend(ws.iter().map(|w| w.1));
    let pure: Vec<bool> = syms
        .iter()
        .map(|&s| g.symbol_name(s).starts_with('p'))
        .collect();
    let tape = Tape::compile_split(&g, &roots, &syms, &pure);
    let instances: u32 = (tape.ops().iter())
        .map(|op| match *op {
            rsdag::tape::Op::Call { n_groups, .. } => n_groups,
            _ => 0,
        })
        .sum();
    assert_eq!(instances, 3, "one instance per distinct operand list");
    let all = g.inline_all(&roots);
    let flat = Tape::compile(&g, &all, &syms);
    let x = [0.3, -0.7, 0.2, -0.4, 1.0, 2.0, 3.0, 4.0];
    let (mut w, mut a, mut b) = (Vec::new(), Vec::new(), Vec::new());
    tape.eval_prolog(&x, &mut w);
    tape.eval_main(&x, &mut w, &mut a);
    flat.eval(&x, &mut w, &mut b);
    for (k, (a, b)) in a.iter().zip(&b).enumerate() {
        assert!((a - b).abs() <= REORDERED * (1.0 + b.abs()), "root {k}");
    }
}
