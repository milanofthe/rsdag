//! Size sweep of the interpreter and the native code: ns per op evaluated
//! and ns per op compiled, per program size and vocabulary, as CSV on
//! stdout (`docs/bench/data/ops.csv` for the README figures).
//!
//!     cargo run --release -p rsdag-jit --example sweep --features rsdag/synth > docs/bench/data/ops.csv

use rsdag::synth::{build, inputs, Spec, Vocabulary};
use rsdag::{Graph, Tape, F64};
use rsdag_jit::NativeTape;
use std::time::Instant;
fn per_call(mut f: impl FnMut(), reps: usize) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e9 / reps as f64
}
fn main() {
    println!("vocab,steps,ops,interp_ns_per_op,native_ns_per_op,compile_ns_per_op");
    for (name, vocab) in [
        ("ring", Vocabulary::Ring),
        ("elementary", Vocabulary::Elementary),
        ("full", Vocabulary::Full),
    ] {
        for steps in [1024usize, 4096, 16384, 65536, 262144, 1048576] {
            let mut g: Graph<F64> = Graph::new();
            let mut spec = Spec::new(7)
                .steps(steps)
                .params(8)
                .outputs(4)
                .vocab(vocab)
                .width(64)
                .smooth();
            let (roots, syms) = build(&mut g, &mut spec);
            let tape = Tape::compile(&g, &roots, &syms);
            let ins = inputs(&mut spec.rng(), syms.len());
            let n = tape.n_ops().max(1);
            let reps = (4_000_000 / n).clamp(3, 20_000);
            let (mut w, mut o) = (Vec::new(), Vec::new());
            let t_i = per_call(|| tape.eval(&ins, &mut w, &mut o), reps) / n as f64;
            let t0 = Instant::now();
            let native = NativeTape::compile(&tape).expect("compile");
            let t_c = t0.elapsed().as_secs_f64() * 1e9 / n as f64;
            let (mut nw, mut no) = (Vec::new(), Vec::new());
            let t_n = per_call(|| native.eval(&ins, &mut nw, &mut no), reps) / n as f64;
            println!("{name},{steps},{n},{t_i:.3},{t_n:.3},{t_c:.2}");
        }
    }
}
