//! The interpreter and the native code behind one interface: the same
//! outputs through `Program`, and a prolog by either serves the other's
//! main phase.

use rsdag::synth::{build, inputs, Spec, Vocabulary};
use rsdag::{Graph, Program, Tape, F64};
use rsdag_jit::NativeTape;

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

#[test]
fn both_backends_through_one_interface() {
    for (seed, vocab) in (0..24u64).zip(
        [Vocabulary::Ring, Vocabulary::Elementary, Vocabulary::Full]
            .iter()
            .cycle(),
    ) {
        let mut g: Graph<F64> = Graph::new();
        let mut spec = Spec::new(seed)
            .steps(500)
            .params(10)
            .outputs(4)
            .vocab(*vocab)
            .width(12)
            .smooth();
        let (roots, syms) = build(&mut g, &mut spec);
        let pure: Vec<bool> = (0..syms.len()).map(|k| k % 3 == 0).collect();
        let tape = Tape::compile_split(&g, &roots, &syms, &pure);
        let native = NativeTape::compile(&tape).expect("native");
        let backends: [&dyn Program; 2] = [&tape, &native];
        let ins = inputs(&mut spec.rng(), syms.len());
        let mut outs = Vec::new();
        for p in backends {
            assert!(ins.len() >= p.n_inputs());
            let mut w = vec![0.0; p.work_len()];
            let mut o = vec![0.0; p.out_len()];
            p.eval_into(&ins, &mut w, &mut o);
            outs.push(o);
        }
        assert!(same(&outs[0], &outs[1]), "seed {seed}: eval_into");
        for (a, b) in [(0, 1), (1, 0)] {
            let (pa, pb) = (backends[a], backends[b]);
            assert_eq!(pa.state_len(), pb.state_len());
            let mut wa = vec![0.0; pa.work_len()];
            pa.eval_prolog_into(&ins, &mut wa);
            let mut wb = vec![f64::NAN; pb.work_len()];
            let s = pb.state_len();
            wb[..s].copy_from_slice(&wa[..s]);
            let mut o = vec![0.0; pb.out_len()];
            pb.eval_main_into(&ins, &mut wb, &mut o);
            assert!(
                same(&o, &outs[0]),
                "seed {seed}: prolog by {a}, main by {b}"
            );
        }
    }
}
