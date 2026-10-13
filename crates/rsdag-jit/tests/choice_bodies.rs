//! Native bodies specialized over their run-time choices: the shortened
//! main phase compiled and run in lanes, an instance whose guards fail run
//! again on the full one, the outputs bit for bit the interpreter's with
//! no specialization, whether the shortened bodies are compiled where they
//! are first needed or in the background while the evaluations go on.

use rsdag::variant::VariantPolicy;
use rsdag::{
    BodyBackend, ChoicePolicy, CmpOp, ExprId, FuncId, Graph, Node, ParamRole, SymbolId, Tape, F64,
};
use rsdag_jit::{Lanes, NativeTape, Options};

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

/// `f(v, p)`: a sum of regions of `v`, a heavy arm above each threshold
/// and a light one below; the instances start between 0.7 and 0.9, most
/// thresholds above them, four among them.
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
    let e = g.exp(v);
    let o1 = g.mul(e, p);
    let f = g.define_func("dev", vec![vs, ps], vec![y, o1]);
    g.set_param_role(f, 0, ParamRole::State { id: 0 });
    g.set_param_role(f, 1, ParamRole::Param);
    f
}

/// `n` instances, each `f(x_i, p_i)`, both outputs summed; specializing
/// over the choices or not.
fn tape(n: usize, choices: bool) -> (Tape, Vec<f64>) {
    let mut g: Graph<F64> = Graph::new();
    let f = device(&mut g);
    let (mut syms, mut roots, mut pure, mut ins) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let (x, xs) = sym(&mut g, &format!("x{i}"));
        let (p, ps) = sym(&mut g, &format!("p{i}"));
        syms.extend([xs, ps]);
        pure.extend([false, true]);
        ins.extend([0.7 + 0.2 * i as f64 / n as f64, 0.6 + 0.001 * i as f64]);
        let a = g.call(f, 0, &[x, p]);
        let b = g.call(f, 1, &[x, p]);
        roots.push(g.add(a, b));
    }
    let mut t = Tape::compile_split(&g, &roots, &syms, &pure);
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
    (t, ins)
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

fn run(n: usize, background: bool, lanes: Lanes) {
    let (spec, mut ins) = tape(n, true);
    let (plain, _) = tape(n, false);
    let opts = Options {
        background,
        lanes,
        ..Options::default()
    };
    let native = NativeTape::compile_opts(&spec, &opts, &[]).expect("native code");
    let (mut wn, mut wp) = (Vec::new(), Vec::new());
    let (mut got, mut want) = (Vec::new(), Vec::new());
    native.eval_prolog(&ins, &mut wn);
    plain.eval_prolog(&ins, &mut wp);
    for k in 0..80 {
        for i in 0..n {
            // drift, half of them up across the thresholds above
            ins[2 * i] += 0.004 * if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        native.eval_main(&ins, &mut wn, &mut got);
        plain.eval_main(&ins, &mut wp, &mut want);
        assert!(
            same(&got, &want),
            "{n} instances, step {k}, {lanes:?}, background {background}"
        );
        if background && k % 20 == 0 {
            // let the background compiles land now and then
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
}

#[test]
fn native_bodies_specialized_over_their_choices_are_the_full_bodies() {
    for lanes in [Lanes::Never, Lanes::Always] {
        for n in [5usize, 64] {
            run(n, false, lanes);
            run(n, true, lanes);
        }
    }
}
