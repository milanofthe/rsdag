//! The C backend against the arena sweep, over synthetic programs.
//!
//! The ring vocabulary must come back bit for bit: the emitted C is
//! generated to compute the same IEEE operation sequence, without FMA
//! contraction or fast-math. The full vocabulary calls the platform `libm`,
//! whose transcendentals need not agree with the ones the interpreter uses
//! to the last bit, so it is checked within a tolerance.

use rsdag::synth::{cases, Spec, Vocabulary};
use rsdag_c::verify::{find_compiler, run_c};

/// Evaluate a corpus through the C backend. Returns the number of programs
/// that agreed bit for bit and the number that needed the tolerance.
fn run(vocab: Vocabulary, seeds: std::ops::Range<u64>, tol: f64) -> (usize, usize) {
    let Some(cc) = find_compiler() else {
        eprintln!("no C compiler on the path, skipping");
        return (0, 0);
    };
    let (mut exact, mut close) = (0, 0);
    for case in cases(seeds, |seed| {
        Spec::new(seed)
            .steps(4 + (seed as usize % 16))
            .params(1 + seed as usize % 3)
            .vocab(vocab)
            // Lists on both sides of the backends' SIMD reduction threshold.
            .max_list(20)
    }) {
        let got = run_c(&cc, &case.tape, &case.rows).expect("C build");
        let want = case.reference();
        let bits_equal = got
            .iter()
            .zip(&want)
            .all(|(g, w)| g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()));
        if bits_equal {
            exact += 1;
        } else {
            close += 1;
        }
        let mut rows = got.into_iter();
        case.expect_close("c source", tol, |_| {
            rows.next().expect("one result per row")
        });
    }
    (exact, close)
}

#[test]
fn ring_vocabulary_is_bit_exact() {
    let (exact, close) = run(Vocabulary::Ring, 1..40, 0.0);
    assert_eq!(close, 0, "the ring must be bit-exact ({exact} programs)");
}

#[test]
fn full_vocabulary_is_within_libm_tolerance() {
    let (exact, close) = run(Vocabulary::Full, 100..140, 1e-12);
    assert!(
        exact + close > 0 || find_compiler().is_none(),
        "no programs ran"
    );
    eprintln!("exact {exact}, within tolerance {close}");
}
