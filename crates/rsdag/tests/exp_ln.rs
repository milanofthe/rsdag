//! rsdag's own exponential and logarithm (`semantics::exp_core`,
//! `semantics::ln_core`) against the platform's, within an ulp, and their
//! special values.

use rsdag::semantics::{exp_core, exp_guarded, ln_core, ln_guarded, EXP_LIMIT, LN_FLOOR};

/// Ulps between two finite doubles of one sign.
fn ulps(a: f64, b: f64) -> u64 {
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
}

/// A xorshift stream of uniforms in `[0, 1)`.
fn uniforms(mut s: u64) -> impl FnMut() -> f64 {
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn exp_is_within_an_ulp_of_the_platforms() {
    let mut u = uniforms(0x9E37_79B9_7F4A_7C15);
    let mut worst = 0;
    for _ in 0..2_000_000 {
        // Uniform over the range, and log-uniform near zero.
        for x in [
            -708.0 + u() * (EXP_LIMIT + 708.0),
            (u() - 0.5) * 10f64.powf(-12.0 * u()),
        ] {
            worst = worst.max(ulps(exp_core(x), x.exp()));
        }
    }
    assert!(worst <= 1, "exp off by {worst} ulps");
}

#[test]
fn ln_is_within_an_ulp_of_the_platforms() {
    let mut u = uniforms(0xD1B5_4A32_D192_ED03);
    let mut worst = 0;
    for _ in 0..2_000_000 {
        // Log-uniform over the normals above the floor, and near one.
        for x in [
            10f64.powf(-30.0 + 330.0 * u()),
            1.0 + (u() - 0.5) * 10f64.powf(-10.0 * u()),
        ] {
            worst = worst.max(ulps(ln_core(x), x.ln()));
        }
    }
    assert!(worst <= 1, "ln off by {worst} ulps");
}

#[test]
fn special_values() {
    assert_eq!(exp_core(0.0), 1.0);
    assert_eq!(exp_core(-0.0), 1.0);
    assert_eq!(ln_core(1.0), 0.0);
    assert_eq!(ln_core(f64::INFINITY), f64::INFINITY);
    assert!(exp_core(f64::NAN).is_nan());
    assert!(ln_core(f64::NAN).is_nan());
    assert_eq!(exp_core(f64::NEG_INFINITY), 0.0);
    assert_eq!(exp_core(-800.0), 0.0);
    // Subnormal results round once.
    for x in [-710.0, -720.5, -740.0, -744.9] {
        assert!(ulps(exp_core(x), x.exp()) <= 1, "exp({x})");
    }
    // The guards.
    assert_eq!(exp_guarded(EXP_LIMIT + 1.0), exp_core(EXP_LIMIT) * 2.0);
    assert_eq!(ln_guarded(0.0), ln_core(LN_FLOOR));
    assert_eq!(ln_guarded(-1.0), ln_core(LN_FLOOR));
    assert!(ln_guarded(f64::NAN).is_nan() && exp_guarded(f64::NAN).is_nan());
}
