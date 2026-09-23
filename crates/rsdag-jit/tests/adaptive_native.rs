//! The adaptive program is the interpreter's result, bit for bit, whichever
//! rung serves a call: across region flips, parameter changes between
//! episodes, and the native code landing in the middle of an episode.

use rsdag::synth::{build, inputs, Rng, Spec, Vocabulary};
use rsdag::{Adaptive, Policy};
use rsdag::{Graph, Tape, F64};

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

fn program(seed: u64) -> (Tape, Vec<f64>, Vec<bool>) {
    let mut g: Graph<F64> = Graph::new();
    let mut spec = Spec::new(seed)
        .steps(900)
        .params(10)
        .outputs(5)
        .vocab(Vocabulary::Full)
        .width(14);

    let (roots, syms) = build(&mut g, &mut spec);
    let pure: Vec<bool> = (0..syms.len()).map(|k| k % 3 == 0).collect();
    let ins = inputs(&mut spec.rng(), syms.len());
    (Tape::compile_split(&g, &roots, &syms, &pure), ins, pure)
}

/// Drift the impure inputs a little (regions flip now and then), and every
/// `episode_len` calls the pure ones.
fn drift(rng: &mut Rng, ins: &mut [f64], pure: &[bool], params_too: bool) {
    for (x, &p) in ins.iter_mut().zip(pure) {
        if !p || params_too {
            *x += ((rng.next_u64() % 2001) as f64 / 1000.0 - 1.0) * 0.3;
        }
    }
}

#[test]
fn every_rung_is_the_interpreter() {
    let mut seen_flip = false;
    for seed in 0..6u64 {
        let (tape, mut ins, pure) = program(seed);
        let policy = Policy {
            spec_min_selects: 1,
            spec_min_shrink_pct: 0,
            kick_after: 2,
            ..Policy::default()
        };
        let a = Adaptive::new(tape, policy, Some(rsdag_jit::compiler()));
        let reference = a.tape();
        let mut rng = Rng::new(seed + 100);
        let (mut w, mut o, mut rw, mut ro) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for episode in 0..60 {
            drift(&mut rng, &mut ins, &pure, true);
            if episode % 3 == 0 {
                // Whole calls.
                a.eval(&ins, &mut w, &mut o);
                reference.eval(&ins, &mut rw, &mut ro);
                assert!(same(&o, &ro), "seed {seed} episode {episode}: eval");
                continue;
            }
            let mut ep = a.eval_prolog(&ins, &mut w);
            for it in 0..8 {
                drift(&mut rng, &mut ins, &pure, false);
                a.eval_main(&mut ep, &ins, &mut w, &mut o);
                reference.eval(&ins, &mut rw, &mut ro);
                assert!(
                    same(&o, &ro),
                    "seed {seed} episode {episode} iteration {it}"
                );
            }
            // Give the background compiles a moment now and then.
            if episode % 10 == 5 {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
        let st = a.stats();
        assert!(st.native, "seed {seed}: the native code landed");
        assert!(
            st.specializations > 0 || st.spec_off,
            "seed {seed}: specialization ran: {st:?}"
        );
        seen_flip |= st.flips > 0;
    }
    assert!(seen_flip, "some region flipped in some run");
}

#[test]
fn without_the_jit_it_stays_interpreted_and_correct() {
    let (tape, ins, _) = program(7);
    let a = Adaptive::new(
        tape,
        Policy {
            jit: false,
            ..Policy::default()
        },
        Some(rsdag_jit::compiler()),
    );
    let reference = a.tape();
    let (mut w, mut o, mut rw, mut ro) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..20 {
        a.eval(&ins, &mut w, &mut o);
    }
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(a.native().is_none());
    reference.eval(&ins, &mut rw, &mut ro);
    assert!(same(&o, &ro));
}
