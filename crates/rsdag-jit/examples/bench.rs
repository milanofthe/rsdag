//! Interpreter against native code on a few programs: nanoseconds per op,
//! and the compile time per op that the native backend costs.
//!
//!     cargo run --release -p rsdag-jit --example bench --features rsdag/synth

use rsdag::synth::{build, inputs, Spec, Vocabulary};
use rsdag::{ExprId, Graph, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;
use std::time::Instant;

fn per_call<F: FnMut()>(mut f: F, reps: usize) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e9 / reps as f64
}

fn lorenz(g: &mut Graph<F64>) -> (Vec<ExprId>, Vec<SymbolId>) {
    let (x, y, z) = (g.sym("x"), g.sym("y"), g.sym("z"));
    let (s, r, b) = (g.konst_f64(10.0), g.konst_f64(28.0), g.konst_f64(8.0 / 3.0));
    let dx = {
        let d = g.sub(y, x);
        g.mul(s, d)
    };
    let dy = {
        let d = g.sub(r, z);
        let p = g.mul(x, d);
        g.sub(p, y)
    };
    let dz = {
        let p = g.mul(x, y);
        let q = g.mul(b, z);
        g.sub(p, q)
    };
    let syms = [x, y, z]
        .iter()
        .map(|&e| match *g.node(e) {
            rsdag::Node::Symbol(s) => s,
            _ => unreachable!(),
        })
        .collect();
    (vec![dx, dy, dz], syms)
}

fn synthetic(seed: u64, steps: usize, vocab: Vocabulary) -> (Tape, Vec<f64>) {
    let mut g: Graph<F64> = Graph::new();
    let mut spec = Spec::new(seed)
        .steps(steps)
        .params(8)
        .outputs(4)
        .vocab(vocab)
        .width(64)
        .smooth();
    let (roots, syms) = build(&mut g, &mut spec);
    let tape = Tape::compile(&g, &roots, &syms);
    let row = inputs(&mut spec.rng(), syms.len());
    (tape, row)
}

fn row(name: &str, tape: &Tape, ins: &[f64]) {
    let n = tape.n_ops().max(1);
    let reps = (2_000_000 / n).clamp(20, 20_000);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    let t_i = per_call(|| tape.eval(ins, &mut w, &mut o), reps) / n as f64;
    let t0 = Instant::now();
    let native = NativeTape::compile(tape).expect("native compile");
    let t_c = t0.elapsed().as_secs_f64() * 1e9 / n as f64;
    let (mut nw, mut no) = (Vec::new(), Vec::new());
    let t_n = per_call(|| native.eval(ins, &mut nw, &mut no), reps) / n as f64;
    tape.eval(ins, &mut w, &mut o);
    // A NaN's payload is no part of any guarantee.
    let same = o
        .iter()
        .zip(&no)
        .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
    println!(
        "{name:<22} {n:>8} ops  interp {t_i:>6.2} ns/op  native {t_n:>6.2} ns/op  compile {t_c:>7.1} ns/op  {}",
        if same { "bits equal" } else { "MISMATCH" }
    );
}

fn main() {
    let mut g: Graph<F64> = Graph::new();
    let (roots, syms) = lorenz(&mut g);
    let tape = Tape::compile(&g, &roots, &syms);
    row("lorenz", &tape, &[1.0, 1.0, 1.0]);
    for (name, vocab) in [
        ("ring", Vocabulary::Ring),
        ("elementary", Vocabulary::Elementary),
        ("full", Vocabulary::Full),
    ] {
        for steps in [4096, 65536] {
            let (tape, ins) = synthetic(7, steps, vocab);
            row(&format!("{name} {steps}"), &tape, &ins);
        }
    }
}
