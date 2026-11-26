//! A body whose main phase branches on its state specializes over the
//! choices its instances take: a batch runs on the main phase shortened
//! against a trace, an instance whose guards fail runs again on the full
//! one, and the outputs are bit for bit those of the body unspecialized,
//! while instances cross their regions, while they thrash, and through the
//! derivatives.

mod common;
use common::{same, sym};

use rsdag::Program;

use rsdag::variant::VariantPolicy;
use rsdag::{
    BodyBackend, ChoicePolicy, CmpOp, ExprId, FuncId, Graph, OutputRole, ParamRole, Scope,
    SymbolId, Tape, F64,
};

const REGIONS: usize = 24;
const REGIONS_ABOVE_THE_STARTING_STATES: usize = 20;

fn device(g: &mut Graph<F64>) -> FuncId {
    let (v, vs) = sym(g, "dev.v");
    let (p, ps) = sym(g, "dev.p");
    let mut y = g.zero();
    for k in 0..REGIONS {
        let th = match k {
            0..REGIONS_ABOVE_THE_STARTING_STATES => 1.0 + 0.1 * k as f64,
            _ => 0.78 + 0.02 * (k - REGIONS_ABOVE_THE_STARTING_STATES) as f64,
        };
        let th = g.konst_f64(th);
        let c = g.cmp(CmpOp::Gt, v, th);
        let mut heavy = g.mul(v, p);
        for i in 0..4 {
            let t = g.tanh(heavy);
            let w = g.konst_f64(0.9 + 0.01 * (k + i) as f64);
            heavy = g.mul(w, t);
        }
        let kk = g.konst_f64(0.5 + 0.1 * k as f64);
        let light = g.mul(kk, v);
        let s = g.select(c, heavy, light);
        y = g.add(y, s);
    }
    let vv = g.mul(v, v);
    let o1 = g.mul(vv, p);
    let mut s = Scope::new(g, "dev");
    s.param(vs, ParamRole::State { id: 0 });
    s.param(ps, ParamRole::Param);
    s.close([(y, OutputRole::Plain), (o1, OutputRole::Plain)])
}

struct Circuit {
    g: Graph<F64>,
    roots: Vec<ExprId>,
    syms: Vec<SymbolId>,
    pure: Vec<bool>,
}

fn circuit(n: usize) -> Circuit {
    let mut g: Graph<F64> = Graph::new();
    let f = device(&mut g);
    let (mut syms, mut roots, mut pure) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let (x, xs) = sym(&mut g, &format!("x{i}"));
        let (p, ps) = sym(&mut g, &format!("p{i}"));
        syms.extend([xs, ps]);
        pure.extend([false, true]);
        let a = g.call(f, 0, &[x, p]);
        let b = g.call(f, 1, &[x, p]);
        roots.push(g.add(a, b));
    }
    Circuit {
        g,
        roots,
        syms,
        pure,
    }
}

fn tape(c: &Circuit, specialize_over_choices: bool) -> Tape {
    let mut t = Tape::compile_split(&c.g, &c.roots, &c.syms, &c.pure);
    t.with_backend(&BodyBackend {
        variants: Some(VariantPolicy {
            choices: ChoicePolicy {
                enabled: specialize_over_choices,
                ..ChoicePolicy::default()
            },
            ..VariantPolicy::default()
        }),
        ..BodyBackend::default()
    });
    t
}

fn specializations_built_and_off_where_rsprobe_records(f: impl FnOnce()) -> Option<[u64; 2]> {
    let ctx = rsprobe::Context::new();
    {
        let _on = ctx.enter();
        let _s = rsprobe::scope!("counted");
        f();
    }
    let r = ctx.record();
    r.find(&["counted"])?;
    let sum = |m: &str| r.nodes.iter().map(|n| n.get(m)).sum();
    Some([sum("specializations built"), sum("specializations off")])
}

fn rng(s: &mut u64) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn run_and_match_the_plain_tape_at_every_step(
    n: usize,
    steps: usize,
    step: impl Fn(usize, usize) -> f64,
) {
    let c = circuit(n);
    let (spec, plain) = (tape(&c, true), tape(&c, false));
    let mut ins: Vec<f64> = (0..n)
        .flat_map(|i| [0.7 + 0.2 * i as f64 / n as f64, 0.6 + 0.001 * i as f64])
        .collect();
    let (mut ws, mut wp) = (vec![0.0; spec.work_len()], vec![0.0; plain.work_len()]);
    let (mut got, mut want) = (vec![0.0; spec.out_len()], vec![0.0; plain.out_len()]);
    spec.eval_prolog_into(&ins, &mut ws);
    plain.eval_prolog_into(&ins, &mut wp);
    for k in 0..steps {
        for i in 0..n {
            ins[2 * i] += step(i, k);
        }
        spec.eval_main_into(&ins, &mut ws, &mut got);
        plain.eval_main_into(&ins, &mut wp, &mut want);
        assert!(same(&got, &want), "{n} instances, step {k}");
    }
}

#[test]
fn bodies_specialize_over_their_choices_and_stay_the_full_bodies() {
    let drift = specializations_built_and_off_where_rsprobe_records(|| {
        run_and_match_the_plain_tape_at_every_step(64, 60, |i, _| {
            0.004 * if i % 2 == 0 { 1.0 } else { -1.0 }
        })
    });
    let change = specializations_built_and_off_where_rsprobe_records(|| {
        run_and_match_the_plain_tape_at_every_step(64, 30, |_, k| if k == 4 { 1.2 } else { 0.0 })
    });
    let seed = std::cell::Cell::new(11u64);
    let jumps = specializations_built_and_off_where_rsprobe_records(|| {
        run_and_match_the_plain_tape_at_every_step(64, 40, |_, k| {
            if k < 3 {
                return 0.0;
            }
            let mut s = seed.get();
            let d = 2.4 * rng(&mut s) - 1.2;
            seed.set(s);
            d
        })
    });
    if let (Some(drift), Some(change), Some(jumps)) = (drift, change, jumps) {
        let [built, off] = drift;
        assert!(
            built >= 2,
            "built, and built again without what flipped: {built}"
        );
        assert_eq!(off, 0, "drift keeps it on");
        assert!(change[0] > 0, "rebuilt after the change");
        assert_eq!(change[1], 0, "one change of every instance keeps it on");
        assert!(jumps[1] > 0, "jumps turn it off");
    }
}

#[test]
fn derivatives_through_the_specialized_bodies_are_the_full_bodies() {
    let n = 32;
    let mut c = circuit(n);
    let jac = rsdag::sparse_jacobian(&mut c.g, &c.roots, &c.syms);
    let entries: Vec<ExprId> = jac.iter().flat_map(|r| r.iter().map(|&(_, e)| e)).collect();
    let d = Circuit {
        roots: entries,
        ..c
    };
    let (spec, plain) = (tape(&d, true), tape(&d, false));
    let mut ins: Vec<f64> = (0..n)
        .flat_map(|i| [0.7 + 0.2 * i as f64 / n as f64, 0.6])
        .collect();
    let (mut ws, mut wp) = (vec![0.0; spec.work_len()], vec![0.0; plain.work_len()]);
    let (mut got, mut want) = (vec![0.0; spec.out_len()], vec![0.0; plain.out_len()]);
    spec.eval_prolog_into(&ins, &mut ws);
    plain.eval_prolog_into(&ins, &mut wp);
    for k in 0..30 {
        for i in 0..n {
            ins[2 * i] += 0.005 * if (i + k) % 3 == 0 { 1.0 } else { -0.5 };
        }
        spec.eval_main_into(&ins, &mut ws, &mut got);
        plain.eval_main_into(&ins, &mut wp, &mut want);
        assert!(same(&got, &want), "step {k}");
    }
}
