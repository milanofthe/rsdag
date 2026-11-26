//! The special functions rsdag defines itself: total on every argument
//! (no recurrence that never ends), and the reflection for `x <= 0` agrees
//! with the recurrence it replaces.

mod common;
use common::constants::{DIGAMMA_ERROR, POLYGAMMA_MPMATH, TRIGAMMA_ERROR};
use rsdag::semantics::{digamma, trigamma};

#[test]
fn digamma_and_trigamma_end_everywhere() {
    for x in [f64::NEG_INFINITY, -1e300, -1e17, -3.0, 0.0, f64::NAN] {
        let _ = (digamma(x), trigamma(x));
    }
    assert!(digamma(f64::NEG_INFINITY).is_nan());
    assert!(digamma(-3.0).is_nan(), "a pole");
    assert_eq!(trigamma(-3.0), f64::INFINITY);
    assert_eq!(digamma(f64::INFINITY), f64::INFINITY);
    assert_eq!(trigamma(f64::INFINITY), 0.0);
}

fn scale(f: fn(f64) -> f64, x: f64, value: f64) -> f64 {
    let reflected = if x <= 0.0 { f(1.0 - x).abs() } else { 0.0 };
    1.0 + value.abs() + reflected
}

#[test]
fn values_match_the_exact_ones() {
    for (x, psi, psi1) in POLYGAMMA_MPMATH {
        let (a, b) = (digamma(x), trigamma(x));
        assert!(
            (a - psi).abs() <= DIGAMMA_ERROR * scale(digamma, x, psi),
            "psi({x}): {a} vs {psi}"
        );
        assert!(
            (b - psi1).abs() <= TRIGAMMA_ERROR * scale(trigamma, x, psi1),
            "psi1({x}): {b} vs {psi1}"
        );
    }
}
