//! Kernel benchmark: the cost of building a program and of running it,
//! through every evaluation path rsgb offers.
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

use rsgb::{ExprId, Graph, SymbolId, Tape, F64};
use rsgb_jit::{ChunkedTape, LaneTape};

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

/// A coupled nonlinear residual of `n` rows over `n` unknowns, each row
/// touching three of them through an exponential, a square root and a
/// rational term: the shape of a circuit or FEM residual, and the shape that
/// makes a Jacobian sparse.
fn residual(g: &mut Graph<F64>, n: usize) -> (Vec<ExprId>, Vec<SymbolId>) {
    let xs: Vec<_> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
    let vt = g.konst_f64(0.025);
    let is = g.konst_f64(1e-14);
    let mut outs = Vec::with_capacity(n);
    for i in 0..n {
        let (a, b, c) = (xs[i], xs[(i + 1) % n], xs[(i + 7) % n]);
        let diode = {
            let q = g.div(a, vt);
            let e = g.exp(q);
            let one = g.one();
            let m = g.sub(e, one);
            g.mul(is, m)
        };
        let root = {
            let s = g.add(a, b);
            let p = g.mul(s, s);
            g.sqrt(p)
        };
        let rational = {
            let s = g.sub(b, c);
            let l = g.mul(s, s);
            let one = g.one();
            let den = g.add(one, l);
            g.div(s, den)
        };
        let row = {
            let u = g.add(diode, root);
            g.add(u, rational)
        };
        outs.push(row);
    }
    (outs, (0..n as u32).map(SymbolId).collect())
}

/// Inputs inside the domain of every row (the exponential would overflow far
/// outside it, which says nothing about the kernels).
fn sample_inputs(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.05 + 0.4 * (i % 7) as f64 / 7.0).collect()
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
    let sizes: &[usize] = if quick { &[32] } else { &[32, 256, 1024] };

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

    for &n in sizes {
        let mut g: Graph<F64> = Graph::new();
        let (outs, syms) = residual(&mut g, n);
        let inputs = sample_inputs(n);

        let t0 = Instant::now();
        let tape = Tape::compile(&g, &outs, &syms);
        let tape_ms = t0.elapsed().as_secs_f64() * 1e3;
        row(&format!("residual n={n}"), &tape, &inputs);

        let t0 = Instant::now();
        let jac = rsgb::jacobian(&mut g, &outs, &syms);
        let jac_ms = t0.elapsed().as_secs_f64() * 1e3;
        let nz: Vec<ExprId> = jac
            .iter()
            .flatten()
            .copied()
            .filter(|&e| !g.is_zero(e))
            .collect();
        let jtape = Tape::compile(&g, &nz, &syms);
        row(&format!("residual n={n} jacobian"), &jtape, &inputs);
        println!(
            "  graph {} nodes, tape compile {tape_ms:.2} ms | jacobian {} of {} entries nonzero, built in {jac_ms:.2} ms",
            g.len(),
            nz.len(),
            n * n,
        );
    }

    // The lane tape evaluates `width` parameter sets per instruction. Report
    // it per set, which is the only comparison that means anything.
    let n = *sizes.last().unwrap();
    let mut g: Graph<F64> = Graph::new();
    let (outs, syms) = residual(&mut g, n);
    let tape = Tape::compile(&g, &outs, &syms);
    let inputs = sample_inputs(n);
    if let Ok(lane) = LaneTape::compile(&tape) {
        let w = lane.width();
        // Lane k gets the k-th parameter set; set 0 repeats the scalar run,
        // so its outputs must come back bit-identical.
        let wide: Vec<f64> = inputs
            .iter()
            .flat_map(|&v| (0..w).map(move |k| v + 0.001 * k as f64))
            .collect();
        let (mut lw, mut lout) = (Vec::new(), Vec::new());
        lane.eval(&wide, &mut lw, &mut lout);
        let (mut sw, mut sout) = (Vec::new(), Vec::new());
        tape.eval(&inputs, &mut sw, &mut sout);
        assert!(
            sout.iter()
                .zip(lout.chunks(w))
                .all(|(a, b)| a.to_bits() == b[0].to_bits()),
            "the lane tape disagrees with the interpreter on lane 0"
        );
        let reps = reps_for(tape.n_ops());
        let t_lane = per_call(|| lane.eval(&wide, &mut lw, &mut lout), reps);
        let t_scalar = per_call(|| tape.eval(&inputs, &mut sw, &mut sout), reps);
        println!(
            "lane tape (residual n={n}, width {w}): {:.2} ns for {w} sets, {:.2} ns per set against the interpreter's {:.2} ns -> {:.2}x",
            t_lane * 1e9,
            t_lane * 1e9 / w as f64,
            t_scalar * 1e9,
            t_scalar / (t_lane / w as f64),
        );
    }
}
