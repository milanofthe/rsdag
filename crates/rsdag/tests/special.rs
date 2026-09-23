//! The special functions rsdag defines itself: total on every argument
//! (no recurrence that never ends), and the reflection for `x <= 0` agrees
//! with the recurrence it replaces.

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

/// Against exact values (mpmath), negative arguments through the reflection;
/// the asymptotic series is good to about 1e-10 where it starts (x = 6).
#[test]
fn values_match_the_exact_ones() {
    let exact: [(f64, f64, f64); 9] = [
        (-18.7556, -0.2962093295988112, 20.407003394699764),
        (-7.3, 4.33730730551005, 14.951383181433922),
        (-2.5, 1.103156640645243, 9.539246644989124),
        (-0.75, -2.894120200042932, 18.975106932284888),
        (-0.1, 9.245073050052948, 101.92253995947719),
        (-0.001, 999.4211381978913, 1000001.6473414317),
        (0.3, -3.502524222200133, 12.245364546107732),
        (2.5, 0.7031566406452432, 0.49035775610023485),
        (11.0, 2.351752589066721, 0.09516633568168574),
    ];
    for (x, psi, psi1) in exact {
        let (a, b) = (digamma(x), trigamma(x));
        assert!(
            (a - psi).abs() <= 1e-9 * (1.0 + psi.abs()),
            "psi({x}): {a} vs {psi}"
        );
        assert!(
            (b - psi1).abs() <= 1e-9 * (1.0 + psi1.abs()),
            "psi1({x}): {b} vs {psi1}"
        );
    }
}
