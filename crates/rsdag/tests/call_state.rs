//! Calls keep their instance state in the caller: a function whose
//! parameters carry the `Param` role has a prolog over them, and a caller
//! split over those parameters runs that prolog in its own prolog, once
//! per binding, and only the rest per evaluation.

use rsdag::{ExprId, FuncId, Graph, Node, ParamRole, SymbolId, Tape, F64};

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

/// `f(v, p) = [tanh(v) * k(p), v^2 + k(p)]` with `k(p)` an expensive chain
/// over `p` alone: the prolog's work.
fn device(g: &mut Graph<F64>) -> FuncId {
    let (v, vs) = sym(g, "dev.v");
    let (p, ps) = sym(g, "dev.p");
    let mut k = p;
    for i in 0..40 {
        let c = g.konst_f64(1.0 + 0.01 * i as f64);
        let e = g.exp(k);
        let t = g.mul(c, e);
        k = g.ln(t);
    }
    let tv = g.tanh(v);
    let o0 = g.mul(tv, k);
    let v2 = g.mul(v, v);
    let o1 = g.add(v2, k);
    let f = g.define_func("dev", vec![vs, ps], vec![o0, o1]);
    g.set_param_role(f, 0, ParamRole::State { id: 0 });
    g.set_param_role(f, 1, ParamRole::Param);
    f
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn a_split_caller_runs_the_callees_prolog_in_its_own() {
    for n in [1usize, 12] {
        let mut g: Graph<F64> = Graph::new();
        let f = device(&mut g);
        let mut syms = Vec::new();
        let mut roots = Vec::new();
        for i in 0..n {
            let (x, xs) = sym(&mut g, &format!("x{i}"));
            let (q, qs) = sym(&mut g, &format!("q{i}"));
            syms.push(xs);
            syms.push(qs);
            let a = g.call(f, 0, &[x, q]);
            let b = g.call(f, 1, &[x, q]);
            roots.push(g.add(a, b));
        }
        let pure: Vec<bool> = (0..syms.len()).map(|k| k % 2 == 1).collect();
        let split = Tape::compile_split(&g, &roots, &syms, &pure);
        let whole = Tape::compile(&g, &roots, &syms);
        assert!(
            split.dump().contains("CallProlog"),
            "{n} instances:\n{}",
            split.dump()
        );
        assert!(!whole.dump().contains("CallProlog"));

        let ins = |t: f64| -> Vec<f64> {
            (0..syms.len())
                .map(|k| {
                    if k % 2 == 1 {
                        0.2 + 0.01 * k as f64
                    } else {
                        t + 0.1 * k as f64
                    }
                })
                .collect()
        };
        let mut w = vec![0.0; split.work_len()];
        let mut got = vec![0.0; split.out_len()];
        let (mut ww, mut want) = (Vec::new(), Vec::new());
        split.eval_prolog_into(&ins(0.0), &mut w);
        for t in [0.0, 0.5, -1.25] {
            // Only the state changes between these: the prolog stands.
            split.eval_main_into(&ins(t), &mut w, &mut got);
            whole.eval(&ins(t), &mut ww, &mut want);
            assert!(
                same(&got, &want),
                "{n} instances at t = {t}: {got:?} vs {want:?}"
            );
        }
    }
}
