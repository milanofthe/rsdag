//! Calls keep their instance state in the caller: a function whose
//! parameters carry the `Param` role has a prolog over them, and a caller
//! split over those parameters runs that prolog in its own prolog, once
//! per binding, and only the rest per evaluation.

mod common;
use common::{identical, sym};

use rsdag::Program;
use rsdag::{FuncId, Graph, OutputRole, ParamRole, Scope, Tape, F64};

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
    let mut s = Scope::new(g, "dev");
    s.param(vs, ParamRole::State { id: 0 });
    s.param(ps, ParamRole::Param);
    s.close([(o0, OutputRole::Plain), (o1, OutputRole::Plain)])
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
                identical(&got, &want),
                "{n} instances at t = {t}: {got:?} vs {want:?}"
            );
        }
    }
}

#[test]
fn instances_of_the_same_parameters_share_a_state_with_the_same_results() {
    // Twelve instances over five parameter values, some repeated; 0.0 and
    // -0.0 are apart (not the same bits). Serially and on a pool, the split
    // tape's outputs are the whole one's, bit for bit.
    let qs = [0.2, 0.3, 0.2, 0.2, 0.0, -0.0, 0.3, 0.7, 0.2, 0.0, 0.7, 0.3];
    let mut g: Graph<F64> = Graph::new();
    let f = device(&mut g);
    let (mut syms, mut roots) = (Vec::new(), Vec::new());
    for i in 0..qs.len() {
        let (x, xs) = sym(&mut g, &format!("x{i}"));
        let (q, qsym) = sym(&mut g, &format!("q{i}"));
        syms.push(xs);
        syms.push(qsym);
        let a = g.call(f, 0, &[x, q]);
        let b = g.call(f, 1, &[x, q]);
        roots.push(g.add(a, b));
    }
    let pure: Vec<bool> = (0..syms.len()).map(|k| k % 2 == 1).collect();
    let split = Tape::compile_split(&g, &roots, &syms, &pure);
    let whole = Tape::compile(&g, &roots, &syms);
    let ins = |t: f64| -> Vec<f64> {
        (0..syms.len())
            .map(|k| match k % 2 {
                1 => qs[k / 2],
                _ => t + 0.1 * k as f64,
            })
            .collect()
    };
    let run = || {
        let mut w = vec![0.0; split.work_len()];
        let mut got = vec![0.0; split.out_len()];
        let (mut ww, mut want) = (Vec::new(), Vec::new());
        split.eval_prolog_into(&ins(0.0), &mut w);
        for t in [0.0, 0.5, -1.25] {
            split.eval_main_into(&ins(t), &mut w, &mut got);
            whole.eval(&ins(t), &mut ww, &mut want);
            assert!(identical(&got, &want), "at t = {t}: {got:?} vs {want:?}");
        }
    };
    run();
    let pool = rsdag::parallel::Lent {
        pool: std::sync::Arc::new(rsdag::parallel::Workers::new(3)),
        dispatch: rsdag::parallel::Dispatch::free(3),
    };
    rsdag::parallel::install_lent(pool, run);
}

/// `f(v, c, q)`: a card parameter `c` and an instance parameter `q`, the
/// prolog's chain over both.
fn carded(g: &mut Graph<F64>) -> FuncId {
    let (v, vs) = sym(g, "dev.v");
    let (c, cs) = sym(g, "dev.c");
    let (q, qs) = sym(g, "dev.q");
    let mut k = g.mul(c, q);
    for i in 0..10 {
        let a = g.konst_f64(1.0 + 0.01 * i as f64);
        let e = g.exp(k);
        let t = g.mul(a, e);
        k = g.ln(t);
    }
    let tv = g.tanh(v);
    let o = g.mul(tv, k);
    let mut s = Scope::new(g, "dev");
    s.param(vs, ParamRole::State { id: 0 });
    s.param(cs, ParamRole::Param);
    s.param(qs, ParamRole::Param);
    s.close([(o, OutputRole::Plain)])
}

#[test]
fn instances_share_rows_by_the_arguments_they_own() {
    // One card for all twelve instances, an instance parameter each (some
    // repeated, 0.0 and -0.0 apart), or one for all. The rows found are
    // those of keying every pure argument, and the outputs the whole
    // tape's, bit for bit.
    let qs = [0.2, 0.3, 0.2, 0.2, 0.0, -0.0, 0.3, 0.7, 0.2, 0.0, 0.7, 0.3];
    for shared_q in [false, true] {
        let mut g: Graph<F64> = Graph::new();
        let f = carded(&mut g);
        let (card, cs) = sym(&mut g, "card");
        let (q_all, q_all_s) = sym(&mut g, "q");
        let (mut syms, mut roots) = (vec![cs, q_all_s], Vec::new());
        let mut vals = vec![1.5, 0.4];
        let mut pure = vec![true, true];
        for (i, &qv) in qs.iter().enumerate() {
            let (x, xs) = sym(&mut g, &format!("x{i}"));
            syms.push(xs);
            vals.push(0.1 * i as f64);
            pure.push(false);
            let q = match shared_q {
                true => q_all,
                false => {
                    let (q, qsym) = sym(&mut g, &format!("q{i}"));
                    syms.push(qsym);
                    vals.push(qv);
                    pure.push(true);
                    q
                }
            };
            roots.push(g.call(f, 0, &[x, card, q]));
        }
        let split = Tape::compile_split(&g, &roots, &syms, &pure);
        let whole = Tape::compile(&g, &roots, &syms);
        let prolog = (0..split.ops().len())
            .find(|&i| matches!(split.ops()[i], rsdag::tape::Op::CallProlog { .. }))
            .expect("a prolog call");
        let c = split.call(prolog);
        assert_eq!(c.n_groups, qs.len());
        let own: &[u32] = if shared_q { &[] } else { &[1] };
        assert_eq!(c.own, Some(own), "{}", split.dump());

        let mut w = vec![0.0; split.work_len()];
        split.eval_prolog_into(&vals, &mut w);
        let rows: Vec<usize> = (0..c.n_groups)
            .map(|k| w[c.out + c.n_groups * c.n_out + k] as usize)
            .collect();
        // every pure argument keyed: the first instance with the same bits
        let key = |k: usize| -> [u64; 2] {
            let q = if shared_q { 0.4 } else { qs[k] };
            [1.5f64.to_bits(), f64::to_bits(q)]
        };
        let want: Vec<usize> = (0..qs.len())
            .map(|k| (0..=k).find(|&j| key(j) == key(k)).unwrap())
            .collect();
        assert_eq!(rows, want);

        let (mut got, mut ww, mut out) = (vec![0.0; split.out_len()], Vec::new(), Vec::new());
        split.eval_main_into(&vals, &mut w, &mut got);
        whole.eval(&vals, &mut ww, &mut out);
        assert!(identical(&got, &out), "{got:?} vs {out:?}");
    }
}
