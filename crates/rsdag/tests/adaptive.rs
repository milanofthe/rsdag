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
            .width(14);
        let (roots, syms) = build(&mut g, &mut spec);
        let pure: Vec<bool> = (0..syms.len()).map(|k| k % 3 == 0).collect();
        let mut ins = inputs(&mut spec.rng(), syms.len());
        let policy = Policy {
            spec_min_selects: 1,
            spec_min_shrink_pct: 0,
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
