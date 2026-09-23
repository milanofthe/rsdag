//! A split tape's state: after a prolog, `work[..state_len]` is everything
//! the main phase needs of it, so an instance's prolog result can be saved
//! as that prefix and the main phase run on any other buffer holding it.

use rsdag::synth::{build, inputs, Spec, Vocabulary};
use rsdag::{Graph, Tape, F64};

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

#[test]
fn the_state_prefix_carries_the_prolog() {
    for (seed, vocab) in (0..40u64).zip(
        [Vocabulary::Ring, Vocabulary::Elementary, Vocabulary::Full]
            .iter()
            .cycle(),
    ) {
        let mut g: Graph<F64> = Graph::new();
        let mut spec = Spec::new(seed)
            .steps(600)
            .params(12)
            .outputs(5)
            .vocab(*vocab)
            .width(16)
            .smooth();
        let (roots, syms) = build(&mut g, &mut spec);
        let pure: Vec<bool> = (0..syms.len()).map(|k| k % 2 == 0).collect();
        let tape = Tape::compile_split(&g, &roots, &syms, &pure);
        let ins = inputs(&mut spec.rng(), syms.len());
        let (mut w, mut want) = (Vec::new(), Vec::new());
        tape.eval(&ins, &mut w, &mut want);

        let n = tape.work_len();
        let s = tape.state_len();
        assert!(s <= n, "seed {seed}");
        let mut a = vec![0.0; n];
        tape.eval_prolog_into(&ins, &mut a);
        // Another buffer: garbage everywhere but the state.
        let mut b = vec![f64::NAN; n];
        b[..s].copy_from_slice(&a[..s]);
        let mut got = vec![0.0; tape.out_len()];
        tape.eval_main_into(&ins, &mut b, &mut got);
        assert!(same(&got, &want), "seed {seed}: state {s} of {n}");

        // The same for its specialization, whose layout is its own.
        let mut trace: Vec<u8> = Vec::new();
        let (mut w2, mut o2) = (Vec::new(), Vec::new());
        tape.eval_with(&ins, &mut w2, &mut o2, &mut trace);
        let spec_tape = tape.specialize(&trace, &vec![true; trace.len()]);
        let t = spec_tape.tape();
        let (n, s) = (t.work_len(), t.state_len());
        let mut a = vec![0.0; n];
        t.eval_prolog_into(&ins, &mut a);
        let mut b = vec![f64::NAN; n];
        b[..s].copy_from_slice(&a[..s]);
        let mut got = vec![0.0; t.out_len()];
        t.eval_main_into(&ins, &mut b, &mut got);
        assert!(
            same(&got[..want.len()], &want),
            "seed {seed}: specialized, state {s} of {n}"
        );
    }
}
