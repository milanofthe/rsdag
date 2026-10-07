//! Per-binding variants against the full body: a device whose arms branch
//! on its parameters (a polarity and a model level, each picking one of two
//! chains of transcendental steps), `--n` instances, the main phase natively
//! with the variants (`rsdag::variant`) and with both arms of every branch
//! computed. One CSV row per length of the chains.
//!
//!     cargo run --release -p rsdag-jit --example variants -- [--n <instances>]

use std::time::Instant;

use rsdag::{CmpOp, ExprId, Graph, Node, ParamRole, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

fn chain(g: &mut Graph<F64>, x: ExprId, n: usize, c: f64) -> ExprId {
    let mut y = x;
    for i in 0..n {
        let k = g.konst_f64(c + 0.001 * i as f64);
        let t = g.tanh(y);
        let u = g.mul(k, t);
        y = g.add(u, x);
    }
    y
}

/// `n` instances of a device of two parameter branches with arms of
/// `arm` steps, and its inputs: states, then per instance `p` and `q`.
fn circuit(n: usize, arm: usize) -> (Graph<F64>, Vec<ExprId>, Vec<SymbolId>, Vec<bool>) {
    let mut g: Graph<F64> = Graph::new();
    let (v, vs) = sym(&mut g, "dev.v");
    let (p, ps) = sym(&mut g, "dev.p");
    let (q, qs) = sym(&mut g, "dev.q");
    let (zero, one) = (g.zero(), g.one());
    let pos = g.cmp(CmpOp::Gt, p, zero);
    let lvl = g.cmp(CmpOp::Gt, q, one);
    let arms: Vec<ExprId> = (0..4)
        .map(|k| chain(&mut g, v, arm, 0.5 + 0.1 * k as f64))
        .collect();
    let a = g.select(pos, arms[0], arms[1]);
    let b = g.select(lvl, arms[2], arms[3]);
    let pa = g.mul(p, a);
    let o = g.add(pa, b);
    let f = g.define_func("dev", vec![vs, ps, qs], vec![o]);
    g.set_param_role(f, 0, ParamRole::State { id: 0 });
    g.set_param_role(f, 1, ParamRole::Param);
    g.set_param_role(f, 2, ParamRole::Param);
    let (mut roots, mut syms, mut pure) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let (x, xs) = sym(&mut g, &format!("x{i}"));
        let (pp, pps) = sym(&mut g, &format!("p{i}"));
        let (qq, qqs) = sym(&mut g, &format!("q{i}"));
        syms.extend([xs, pps, qqs]);
        pure.extend([false, true, true]);
        roots.push(g.call(f, 0, &[x, pp, qq]));
    }
    (g, roots, syms, pure)
}

/// Seconds per call: the best of five batches of at least 50 ms.
fn per_call(mut f: impl FnMut()) -> f64 {
    let mut reps = 1usize;
    let mut dt;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        dt = t.elapsed().as_secs_f64();
        if dt > 0.05 {
            break;
        }
        reps *= 4;
    }
    let mut best = dt / reps as f64;
    for _ in 0..4 {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    best
}

fn main_us(n: usize, arm: usize, variants: bool) -> f64 {
    rsdag::variant::set_enabled(variants);
    let (g, roots, syms, pure) = circuit(n, arm);
    let tape = Tape::compile_split(&g, &roots, &syms, &pure);
    let native = NativeTape::compile(&tape).expect("native");
    // Half the instances one polarity, a third on the higher level.
    let ins: Vec<f64> = (0..n)
        .flat_map(|i| {
            let p = if i % 2 == 0 { 0.7 } else { -0.4 };
            let q = if i % 3 == 0 { 2.0 } else { 0.5 };
            [0.1 * i as f64 - 0.5, p, q]
        })
        .collect();
    let (mut w, mut out) = (Vec::new(), Vec::new());
    native.eval_prolog(&ins, &mut w);
    1e6 * per_call(|| native.eval_main(&ins, &mut w, &mut out))
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = match args.iter().position(|a| a == "--n") {
        Some(i) => {
            let v = args.remove(i + 1);
            args.remove(i);
            v.parse().expect("--n <instances>")
        }
        None => 64,
    };
    println!("instances,arm_ops,main_full_us,main_variants_us,speedup");
    for arm in [4usize, 16, 64, 256] {
        let full = main_us(n, arm, false);
        let var = main_us(n, arm, true);
        println!("{n},{arm},{full:.3},{var:.3},{:.2}", full / var);
    }
}
