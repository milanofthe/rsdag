//! Calls keep their instance state in the caller: a function whose
//! parameters carry the `Param` role has a prolog over them, and a caller
//! split over those parameters runs that prolog in its own prolog, once
//! per binding, and only the rest per evaluation.

mod common;
use common::{circuit, exp_ln_device, identical, sym, Circuit};

use rsdag::Program;
use rsdag::{FuncId, Graph, OutputRole, ParamRole, Scope, Tape, F64};

const PARAMS: [f64; 12] = [0.2, 0.3, 0.2, 0.2, 0.0, -0.0, 0.3, 0.7, 0.2, 0.0, 0.7, 0.3];

fn params() -> [f64; 12] {
    let has = |v: f64| PARAMS.iter().any(|q| q.to_bits() == v.to_bits());
    assert!(
        has(0.0) && has(-0.0),
        "the parameters hold both signed zeros"
    );
    assert!(
        (1..PARAMS.len()).any(|i| PARAMS[..i].contains(&PARAMS[i])),
        "the parameters repeat"
    );
    PARAMS
}

#[test]
fn a_split_caller_runs_the_callees_prolog_in_its_own() {
    for n in [1usize, 12] {
        let Circuit {
            g,
            roots,
            syms,
            pure,
        } = circuit(n, |g| exp_ln_device(g, 40), &["q"]);
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
        let states_after_one_prolog = [0.0, 0.5, -1.25];
        for t in states_after_one_prolog {
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
    let qs = params();
    let Circuit {
        g,
        roots,
        syms,
        pure,
    } = circuit(qs.len(), |g| exp_ln_device(g, 40), &["q"]);
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
    let qs = params();
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
        let key_of_every_pure_argument = |k: usize| -> [u64; 2] {
            let q = if shared_q { 0.4 } else { qs[k] };
            [1.5f64.to_bits(), f64::to_bits(q)]
        };
        let want: Vec<usize> = (0..qs.len())
            .map(|k| {
                (0..=k)
                    .find(|&j| key_of_every_pure_argument(j) == key_of_every_pure_argument(k))
                    .unwrap()
            })
            .collect();
        assert_eq!(rows, want);

        let (mut got, mut ww, mut out) = (vec![0.0; split.out_len()], Vec::new(), Vec::new());
        split.eval_main_into(&vals, &mut w, &mut got);
        whole.eval(&vals, &mut ww, &mut out);
        assert!(identical(&got, &out), "{got:?} vs {out:?}");
    }
}
