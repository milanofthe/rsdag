//! The adaptive program without native code (a wasm build, say): the
//! interpreter and its choice specialization, bit for bit the interpreter.

use rsdag::synth::{build, inputs, Rng, Spec, Vocabulary};
use rsdag::{Adaptive, Graph, Policy, Tape, F64};

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

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
