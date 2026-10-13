//! A body whose main phase branches on its state specializes over the
//! choices its instances take: a batch runs on the main phase shortened
//! against a trace, an instance whose guards fail runs again on the full
//! one, and the outputs are bit for bit those of the body unspecialized,
//! while instances cross their regions, while they thrash, and through the
//! derivatives.

use std::sync::atomic::{AtomicUsize, Ordering};

use rsdag::hooks::{self, Level, Log};
use rsdag::variant::VariantPolicy;
use rsdag::{
    BodyBackend, ChoicePolicy, CmpOp, ExprId, FuncId, Graph, Node, ParamRole, SymbolId, Tape, F64,
};

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

/// `f(v, p)`, `p` a parameter: a sum of regions of `v`, each a select
/// between a heavy arm (above its threshold) and a light one. The
/// instances start between 0.7 and 0.9: most thresholds lie above them,
/// every instance on the light arm, four among them, the instances on
/// either side.
fn device(g: &mut Graph<F64>) -> FuncId {
    let (v, vs) = sym(g, "dev.v");
    let (p, ps) = sym(g, "dev.p");
    let mut y = g.zero();
    for k in 0..24 {
        let th = match k {
            0..20 => 1.0 + 0.1 * k as f64,
            _ => 0.78 + 0.02 * (k - 20) as f64,
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
    let f = g.define_func("dev", vec![vs, ps], vec![y, o1]);
    g.set_param_role(f, 0, ParamRole::State { id: 0 });
    g.set_param_role(f, 1, ParamRole::Param);
    f
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

/// The circuit's tape, its bodies specializing over their choices or not.
fn tape(c: &Circuit, choices: bool) -> Tape {
    let mut t = Tape::compile_split(&c.g, &c.roots, &c.syms, &c.pure);
    t.with_backend(&BodyBackend {
        variants: Some(VariantPolicy {
            choices: ChoicePolicy {
                enabled: choices,
                ..ChoicePolicy::default()
            },
            ..VariantPolicy::default()
        }),
        ..BodyBackend::default()
    });
    t
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

/// The reports of the bodies' specialization, counted.
struct Count {
    built: AtomicUsize,
    /// Specialization turned off: the choices thrash, or so many selects
    /// flipped that pinning the rest does not pay.
    off: AtomicUsize,
}
impl Log for Count {
    fn log(&self, _: Level, message: &str) {
        if message.contains("specialized over its run-time choices") {
            self.built.fetch_add(1, Ordering::Relaxed);
        }
        if message.contains("run-time choices thrash")
            || message.contains("run-time choices shorten it too little")
        {
            self.off.fetch_add(1, Ordering::Relaxed);
        }
    }
}
static SINK: Count = Count {
    built: AtomicUsize::new(0),
    off: AtomicUsize::new(0),
};

fn rng(s: &mut u64) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

/// `steps` main phases at one binding, each instance's state moved by
/// `step(i, k)`, on the specializing tape and on the plain one: the same
/// outputs at every step.
fn run(n: usize, steps: usize, step: impl Fn(usize, usize) -> f64) {
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
    hooks::set_log(&SINK);
    // Instances drifting slowly, half of them up across the lowest
    // thresholds above them: most hold their regions from one step to the
    // next, a few cross and run again full, and the selects they flip are
    // unpinned in a rebuild.
    run(64, 60, |i, _| 0.004 * if i % 2 == 0 { 1.0 } else { -1.0 });
    let built = SINK.built.load(Ordering::Relaxed);
    assert!(
        built >= 2,
        "built, and built again without what flipped: {built}"
    );
    assert_eq!(SINK.off.load(Ordering::Relaxed), 0, "drift keeps it on");
    // Instances holding their regions, then jumping across every region at
    // every step: the selects flip until pinning the rest does not pay,
    // and the specialization goes off.
    let seed = std::cell::Cell::new(11u64);
    run(64, 40, |_, k| {
        if k < 3 {
            return 0.0;
        }
        let mut s = seed.get();
        let d = 2.4 * rng(&mut s) - 1.2;
        seed.set(s);
        d
    });
    assert!(SINK.off.load(Ordering::Relaxed) > 0, "jumps turn it off");
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
