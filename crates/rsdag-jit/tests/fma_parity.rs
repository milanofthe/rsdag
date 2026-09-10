//! With contraction on, the interpreter and the native backend fuse the
//! same multiply-adds with a real fused instruction, so they still agree
//! with each other bit for bit.

use rsdag::synth::{build, inputs, Spec, Vocabulary};
use rsdag::{CompileOptions, Graph, Tape, F64};
use rsdag_jit::{NativeTape, CHUNK_OPS};

/// NaN is equal to NaN here: a fused multiply-add that produces one carries
/// whatever payload the platform's instruction or `fma` routine gives it,
/// and the payload is no part of any guarantee.
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[test]
fn backends_agree_on_a_contracted_program() {
    for seed in 0..40u64 {
        let mut g: Graph<F64> = Graph::new();
        let mut spec = Spec::new(seed)
            .steps(60 + (seed as usize % 60))
            .params(4)
            .outputs(2)
            .vocab(Vocabulary::Ring)
            .smooth();
        let (roots, syms) = build(&mut g, &mut spec);
        let tape = Tape::compile_with(&g, &roots, &syms, None, CompileOptions { contract: true });
        let row = inputs(&mut spec.rng(), syms.len());
        let (mut w, mut want) = (Vec::new(), Vec::new());
        tape.eval(&row, &mut w, &mut want);

        let jit = NativeTape::compile_with(&tape, [3, CHUNK_OPS][seed as usize % 2]).unwrap();
        let (mut jw, mut jo) = (Vec::new(), Vec::new());
        jit.eval(&row, &mut jw, &mut jo);
        let bits = |v: &[f64]| -> Vec<String> {
            v.iter().map(|x| format!("{:016x}", x.to_bits())).collect()
        };
        assert!(
            want.iter().zip(&jo).all(|(&a, &b)| same(a, b)),
            "seed {seed}: jit {jo:?} {:?} vs interpreter {want:?} {:?}",
            bits(&jo),
            bits(&want)
        );
    }
}
