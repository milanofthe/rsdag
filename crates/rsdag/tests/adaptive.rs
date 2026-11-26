//! The adaptive program without native code (a wasm build, say): the
//! interpreter and its choice specialization, bit for bit the interpreter.

mod common;
use common::same;

use rsdag::synth::{build, inputs, Rng, Spec, Vocabulary};
use rsdag::Program;
use rsdag::{Adaptive, Graph, Policy, Tape, F64};

#[test]
fn specialization_alone_is_the_interpreter() {
    let mut flips = 0;
    for seed in 0..6u64 {
        let mut g: Graph<F64> = Graph::new();
        let mut spec = Spec::new(seed)
            .steps(900)
            .params(10)
            .outputs(5)
            .vocab(Vocabulary::Full)
            .width(14)
            .selects(2);
        let (roots, syms) = build(&mut g, &mut spec);
        let pure: Vec<bool> = (0..syms.len()).map(|k| k % 3 == 0).collect();
        let mut ins = inputs(&mut spec.rng(), syms.len());
        let policy = Policy {
            choices: rsdag::ChoicePolicy {
                min_selects: 1,
                min_shrink_pct: 0,
                ..Default::default()
            },
            ..Policy::default()
        };
        let a = Adaptive::new(Tape::compile_split(&g, &roots, &syms, &pure), policy, None);
        let mut rng = Rng::new(seed + 7);
        let (mut w, mut o, mut rw, mut ro) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for episode in 0..40 {
            for (x, &p) in ins.iter_mut().zip(&pure) {
                let step = ((rng.next_u64() % 2001) as f64 / 1000.0 - 1.0) * 0.3;
                *x += if p { step } else { 0.0 };
            }
            let mut ep = a.eval_prolog(&ins, &mut w);
            for _ in 0..6 {
                for (x, &p) in ins.iter_mut().zip(&pure) {
                    if !p {
                        *x += ((rng.next_u64() % 2001) as f64 / 1000.0 - 1.0) * 0.3;
                    }
                }
                a.eval_main(&mut ep, &ins, &mut w, &mut o);
                a.tape().eval(&ins, &mut rw, &mut ro);
                assert!(same(&o, &ro), "seed {seed} episode {episode}");
            }
        }
        let st = a.stats();
        assert!(!st.native && st.specializations > 0, "seed {seed}: {st:?}");
        flips += st.flips;
    }
    assert!(flips > 0, "some region flipped");
}

/// A session binds the pure inputs once and evaluates as the program does
/// at every point: alone, and point by point into an output-major block.
#[test]
fn a_session_evaluates_as_the_program() {
    use rsdag::Session;
    let mut g: Graph<F64> = Graph::new();
    let mut spec = Spec::new(7).steps(400).params(6).outputs(3).width(10);
    let (roots, syms) = build(&mut g, &mut spec);
    let pure: Vec<bool> = (0..syms.len()).map(|k| k % 2 == 0).collect();
    let ins = inputs(&mut spec.rng(), syms.len());
    let tape = Tape::compile_split(&g, &roots, &syms, &pure);
    let program = Adaptive::new(tape, Policy::default(), None);
    let (mut work, mut want) = (Vec::new(), Vec::new());
    let at = |k: usize| -> Vec<f64> {
        let mut x = ins.clone();
        for (j, v) in x.iter_mut().enumerate().filter(|(j, _)| !pure[*j]) {
            *v += 0.01 * (k * (j + 1)) as f64;
        }
        x
    };
    let mut session = Session::new(&program, ins.clone());
    let (mut row, mut block) = (Vec::new(), Vec::new());
    session.eval_points(
        &program,
        4,
        |k, x| x.copy_from_slice(&at(k)),
        &mut row,
        &mut block,
    );
    for k in 0..4 {
        program.eval(&at(k), &mut work, &mut want);
        let got: Vec<f64> = (0..want.len()).map(|o| block[o * 4 + k]).collect();
        assert!(same(&got, &want), "point {k}");
        session.inputs_mut().copy_from_slice(&at(k));
        session.eval(&program, &mut row);
        assert!(same(&row, &want), "point {k} alone");
    }
}

/// Native code for the test: the interpreter on a copy of the tape (the
/// tape emitted again, nothing pinned), compiled on a thread of its own.
struct Copying;

impl rsdag::Compiler for Copying {
    fn compile(&self, tape: &Tape, _live: &[u32]) -> Option<Box<dyn rsdag::Program>> {
        let n = tape.n_selects();
        Some(Box::new(
            tape.specialize(&vec![0; n], &vec![false; n]).into_tape(),
        ))
    }
    fn submit(&self, job: Box<dyn FnOnce() + Send>) {
        std::thread::spawn(job);
    }
}

/// Sixteen regions of `x` above and below their thresholds, a heavy arm
/// and a light one each, and as much again that no region decides: the
/// specialization at a point removes about half of the tape.
fn regions() -> (Graph<F64>, Vec<rsdag::ExprId>, Vec<rsdag::SymbolId>) {
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let rsdag::Node::Symbol(xs) = *g.node(x) else {
        unreachable!()
    };
    let chain = |g: &mut Graph<F64>, k: usize| {
        let mut y = x;
        for i in 0..8 {
            let w = g.konst_f64(0.9 + 1e-3 * (8 * k + i) as f64);
            let t = g.tanh(y);
            y = g.mul(w, t);
        }
        y
    };
    let mut roots = Vec::new();
    for k in 0..16 {
        let th = g.konst_f64(0.1 * k as f64 - 0.72);
        let c = g.cmp(rsdag::CmpOp::Gt, x, th);
        let heavy = chain(&mut g, k);
        let w = g.konst_f64(0.5 + 0.01 * k as f64);
        let light = g.mul(w, x);
        roots.push(g.select(c, heavy, light));
        roots.push(chain(&mut g, 100 + k));
    }
    (g, roots, vec![xs])
}

/// Once the full tape's native code serves, an interpreted specialization
/// that is not several times shorter does not run, so it holds no guards
/// to wait for: it is compiled, and its native code serves.
#[test]
fn a_specialization_is_compiled_where_the_interpreted_one_does_not_serve() {
    let (g, roots, syms) = regions();
    let tape = Tape::compile(&g, &roots, &syms);
    let full = tape.n_ops();
    let a = Adaptive::new(tape, Policy::default(), Some(std::sync::Arc::new(Copying)));
    let (mut w, mut o, mut rw, mut ro) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for k in 0..400 {
        let x = 0.01 * (k % 7) as f64;
        a.eval(&[x], &mut w, &mut o);
        a.tape().eval(&[x], &mut rw, &mut ro);
        assert!(same(&o, &ro), "call {k}");
        // The full tape's native code lands well before the probation of
        // an interpreted specialization would end.
        while k == 3 && a.native().is_none() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
    let st = a.stats();
    let mut trace = Vec::new();
    a.tape().eval_with(&[0.0], &mut rw, &mut ro, &mut trace);
    let spec = a.tape().specialize(&trace, &[true; 16]).n_ops();
    assert!(
        4 * spec > full && 100 * spec < 85 * full,
        "{spec} of {full}"
    );
    assert!(st.native && st.specializations > 0, "{st:?}");
    assert!(
        st.spec_native,
        "the specialization's native code serves: {st:?}"
    );
}
