use rsdag::synth::{build, cases, inputs, Spec, Vocabulary};
use rsdag::*;
use rsdag::{Tape, F64};

#[test]
fn a_seed_reproduces_the_same_program_and_values() {
    let run = || {
        let mut g: Graph<F64> = Graph::new();
        let mut spec = Spec::new(42).steps(300).params(5).outputs(3);
        let (roots, syms) = build(&mut g, &mut spec);
        let tape = Tape::compile(&g, &roots, &syms);
        let (mut w, mut out) = (Vec::new(), Vec::new());
        tape.eval(&inputs(&mut spec.rng(), syms.len()), &mut w, &mut out);
        (g.len(), tape.n_ops(), out)
    };
    let (nodes, ops, out) = run();
    let (nodes2, ops2, out2) = run();
    assert_eq!((nodes, ops), (nodes2, ops2));
    // Bit patterns, not values: a deep chain of `exp` overflows to
    // infinity and `0 * inf` gives NaN. Those are legitimate results
    // that every evaluation path must reproduce, so they are coverage
    // rather than a defect, and the corpus compares bit patterns
    // everywhere for that reason.
    assert!(out
        .iter()
        .zip(&out2)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn every_in_crate_path_matches_the_arena() {
    // The paths rsdag itself offers, over a corpus that varies size and
    // vocabulary with the seed. The backends add their own paths to the
    // same corpus in their own tests.
    for case in cases(0..24, |seed| {
        let spec = Spec::new(seed)
            .steps(20 + 12 * (seed as usize % 8))
            .params(1 + seed as usize % 4);
        match seed % 3 {
            0 => spec.vocab(Vocabulary::Ring).max_list(20),
            1 => spec.vocab(Vocabulary::Elementary),
            _ => spec.vocab(Vocabulary::Full),
        }
    }) {
        let (mut w, mut o) = (Vec::new(), Vec::new());
        case.expect_bits("tape", |row| {
            case.tape.eval(row, &mut w, &mut o);
            o.clone()
        });
        // Mark every other input parameter-pure, so the prolog split
        // actually has a prefix to hoist.
        let pure: Vec<bool> = (0..case.syms.len()).map(|k| k % 2 == 0).collect();
        let split = Tape::compile_split(&case.graph, &case.roots, &case.syms, &pure);
        let (mut sw, mut so) = (Vec::new(), Vec::new());
        case.expect_bits("split tape", |row| {
            split.eval(row, &mut sw, &mut so);
            so.clone()
        });
        let (mut tw, mut to) = (Vec::new(), Vec::new());
        case.expect_bits("typed tape (f64)", |row| {
            case.tape.eval_typed::<f64>(row, &mut tw, &mut to);
            to.clone()
        });
    }
}

#[test]
fn smooth_programs_differentiate_and_evaluate_like_the_arena() {
    for seed in 0..8u64 {
        let mut g: Graph<F64> = Graph::new();
        let mut spec = Spec::new(seed).steps(120).smooth();
        let (roots, syms) = build(&mut g, &mut spec);
        let d = rsdag::differentiate(&mut g, roots[0], syms[0]);
        let tape = Tape::compile(&g, &[roots[0], d], &syms);
        let row = inputs(&mut spec.rng(), syms.len());
        let (mut w, mut out) = (Vec::new(), Vec::new());
        tape.eval(&row, &mut w, &mut out);
        let env: std::collections::HashMap<_, _> =
            syms.iter().copied().zip(row.iter().copied()).collect();
        let want = rsdag::eval_real(&g, &env, &[roots[0], d]);
        assert!(
            out.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())),
            "seed {seed}: tape {out:?} vs arena {want:?}"
        );
    }
}
