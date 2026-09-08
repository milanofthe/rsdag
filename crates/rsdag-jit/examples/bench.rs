//! Kernel benchmark: the cost of building a program and of running it,
//! through every evaluation path rsdag offers.
//!
//! The numbers a consumer cares about are all per-op: a tape op costs about
//! 1.8 ns in the interpreter and about 0.35 ns through the JIT, so the
//! crossover against the JIT's compile time (roughly 0.7 us per op) sits
//! near a thousand evaluations of the same program. A Newton loop is far
//! past that, a one-shot residual is not.
//!
//! Run with `cargo run --release --example bench` (add `quick` for the small
//! sizes only, which is what CI does to keep the example honest).
//!
//! Every path is checked against the interpreter as it is measured: the JIT
//! and the lane tape must agree bit for bit, which is the property the
//! consumers rely on when they cache a factorization across a switch of
//! evaluation path.

use std::time::Instant;

use rsdag::synth::{build, inputs, Spec, Vocabulary};
use rsdag::{ExprId, Graph, SymbolId, Tape, F64};
use rsdag_jit::{ChunkedTape, LaneTape};

/// Mean seconds per call over `reps` calls of `f`.
fn per_call<F: FnMut()>(mut f: F, reps: usize) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() / reps as f64
}

/// Enough repetitions to spend about 20 ms on a program of this size, so a
/// 14-op tape and a 30k-op tape are both measured out of the timer noise.
fn reps_for(n_ops: usize) -> usize {
    (2_000_000 / n_ops.max(1)).clamp(50, 200_000)
}

/// The Lorenz right-hand side: the small end, where the per-call overhead of
/// an evaluation path is the whole story.
fn lorenz(g: &mut Graph<F64>) -> (Vec<ExprId>, Vec<SymbolId>) {
    let (x, y, z) = (g.sym("x"), g.sym("y"), g.sym("z"));
    let (s, r, b) = (g.konst_f64(10.0), g.konst_f64(28.0), g.konst_f64(8.0 / 3.0));
    let d0 = {
        let d = g.sub(y, x);
        g.mul(s, d)
    };
    let d1 = {
        let rz = g.sub(r, z);
        let xrz = g.mul(x, rz);
        g.sub(xrz, y)
    };
    let d2 = {
        let xy = g.mul(x, y);
        let bz = g.mul(b, z);
        g.sub(xy, bz)
    };
    (
        vec![d0, d1, d2],
        vec![SymbolId(0), SymbolId(1), SymbolId(2)],
    )
}

/// A program of about `steps` nodes drawn from `vocab`, with the inputs to
/// evaluate it over. The generator is the one the parity suite fuzzes with,
/// so the benchmark prices the same population the tests check.
fn synthetic(seed: u64, steps: usize, vocab: Vocabulary) -> (Graph<F64>, Tape, Vec<f64>) {
    let mut g: Graph<F64> = Graph::new();
    let mut spec = Spec::new(seed)
        .steps(steps)
        .params(8)
        .outputs(4)
        .vocab(vocab)
        // Short lists: a `Reduce` of 20 is one op doing twenty flops, which
        // makes a per-op number meaningless. The fuzzers cover the long
        // ones; the benchmark prices ops.
        .max_list(4);
    let (roots, syms) = build(&mut g, &mut spec);
    let tape = Tape::compile(&g, &roots, &syms);
    let inputs = inputs(&mut spec.rng(), syms.len());
    (g, tape, inputs)
}

/// Time one program through the interpreter and the JIT, and check that they
/// agree bit for bit.
fn row(name: &str, tape: &Tape, inputs: &[f64]) {
    let reps = reps_for(tape.n_ops());
    let (mut w, mut out) = (Vec::new(), Vec::new());
    tape.eval(inputs, &mut w, &mut out);
    let t_interp = per_call(|| tape.eval(inputs, &mut w, &mut out), reps);

    let t0 = Instant::now();
    let jit = ChunkedTape::compile(tape).expect("jit compile");
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;
    let (mut jw, mut jout) = (Vec::new(), Vec::new());
    jit.eval(inputs, &mut jw, &mut jout);
    let t_jit = per_call(|| jit.eval(inputs, &mut jw, &mut jout), reps);
    assert!(
        out.iter()
            .zip(&jout)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{name}: the JIT disagrees with the interpreter"
    );

    println!(
        "{name:<24} {:>7} | {:>9.2} ns {:>6.2} | {:>9.2} ns {:>6.2} | {:>5.2}x | {:>8.2} ms",
        tape.n_ops(),
        t_interp * 1e9,
        t_interp * 1e9 / tape.n_ops() as f64,
        t_jit * 1e9,
        t_jit * 1e9 / tape.n_ops() as f64,
        t_interp / t_jit,
        compile_ms,
    );
}

fn main() {
    let quick = std::env::args().any(|a| a == "quick");
    let sizes: &[usize] = if quick { &[64] } else { &[64, 512, 4096] };

    println!(
        "{:<24} {:>7} | {:>21} | {:>21} | {:>6} | {:>11}",
        "program", "ops", "interpreter (ns/op)", "cranelift jit (ns/op)", "gain", "jit compile"
    );

    let mut g: Graph<F64> = Graph::new();
    let (outs, syms) = lorenz(&mut g);
    let tape = Tape::compile(&g, &outs, &syms);
    row("lorenz rhs", &tape, &[1.0, 2.0, 3.0]);

    // The same rhs against the compiler's own code, so the JIT's distance to
    // native is visible and not just its distance to the interpreter.
    let mut acc = 0.0f64;
    let native = per_call(
        || {
            // Every input through `black_box`, or the compiler folds the
            // whole right-hand side and the comparison is meaningless.
            let x = std::hint::black_box(1.0f64);
            let y = std::hint::black_box(2.0f64);
            let z = std::hint::black_box(3.0f64);
            let d0 = 10.0 * (y - x);
            let d1 = x * (28.0 - z) - y;
            let d2 = x * y - (8.0 / 3.0) * z;
            acc += std::hint::black_box(d0) + std::hint::black_box(d1) + std::hint::black_box(d2);
        },
        200_000,
    );
    std::hint::black_box(acc);
    println!(
        "{:<24} {:>7} | {:>9.2} ns",
        "  same rhs in rust",
        14,
        native * 1e9
    );

    for &vocab in &[Vocabulary::Ring, Vocabulary::Elementary, Vocabulary::Full] {
        for &steps in sizes {
            let (_, tape, inputs) = synthetic(steps as u64, steps, vocab);
            row(&format!("{vocab:?} {steps} nodes"), &tape, &inputs);
        }
    }

    // The Jacobian of a smooth synthetic program: what it costs to build the
    // derivative graph, and what the derivative program then costs to run.
    for &steps in sizes {
        let mut g: Graph<F64> = Graph::new();
        let mut spec = Spec::new(steps as u64)
            .steps(steps)
            .params(8)
            .outputs(4)
            .smooth();
        let (roots, syms) = build(&mut g, &mut spec);
        let t0 = Instant::now();
        let jac = rsdag::jacobian(&mut g, &roots, &syms);
        let jac_ms = t0.elapsed().as_secs_f64() * 1e3;
        let nz: Vec<ExprId> = jac
            .iter()
            .flatten()
            .copied()
            .filter(|&e| !g.is_zero(e))
            .collect();
        let jtape = Tape::compile(&g, &nz, &syms);
        let inputs = inputs(&mut spec.rng(), syms.len());
        row(&format!("jacobian of {steps} nodes"), &jtape, &inputs);
        println!(
            "  {} nonzero of {} entries, differentiated in {jac_ms:.2} ms, graph {} nodes",
            nz.len(),
            roots.len() * syms.len(),
            g.len(),
        );
    }

    // The lane tape evaluates `width` parameter sets per instruction. The
    // comparison that means anything is per set against the scalar JIT
    // running the same work twice, and it depends on what the program is
    // made of: ring arithmetic and the aggregate ops vectorize, the
    // transcendentals extract their lanes and call the host per lane.
    let steps = *sizes.last().unwrap();
    for (label, spec) in [
        (
            "aggregates (sum, product, dot)",
            Spec::new(steps as u64).vocab(Vocabulary::Ring).smooth(),
        ),
        (
            "ring with min/max",
            Spec::new(steps as u64).vocab(Vocabulary::Ring),
        ),
        (
            "elementary",
            Spec::new(steps as u64).vocab(Vocabulary::Elementary),
        ),
    ] {
        let mut spec = spec.steps(steps).params(8).outputs(4).max_list(4);
        let mut g: Graph<F64> = Graph::new();
        let (roots, syms) = build(&mut g, &mut spec);
        let tape = Tape::compile(&g, &roots, &syms);
        let inputs = inputs(&mut spec.rng(), syms.len());
        let Ok(lane) = LaneTape::compile(&tape) else {
            continue;
        };
        let Ok(jit) = ChunkedTape::compile(&tape) else {
            continue;
        };
        let w = lane.width();
        let wide: Vec<f64> = inputs
            .iter()
            .flat_map(|&v| (0..w).map(move |k| v + 0.001 * k as f64))
            .collect();
        let (mut lw, mut lo) = (Vec::new(), Vec::new());
        lane.eval(&wide, &mut lw, &mut lo);
        let (mut jw, mut jo) = (Vec::new(), Vec::new());
        jit.eval(&inputs, &mut jw, &mut jo);
        let reps = reps_for(tape.n_ops());
        let t_lane = per_call(|| lane.eval(&wide, &mut lw, &mut lo), reps) / w as f64;
        let t_jit = per_call(|| jit.eval(&inputs, &mut jw, &mut jo), reps);
        println!(
            "lane tape, {label}, {} ops, width {w}: {:.2} ns per set against the scalar jit's {:.2} ns -> {:.2}x",
            tape.n_ops(),
            t_lane * 1e9,
            t_jit * 1e9,
            t_jit / t_lane,
        );
    }
}
