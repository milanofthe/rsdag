//! rsdag's own elementary functions: within their error bounds of the
//! exact values, and the same bits on every platform, which is what the
//! pinned hash below asserts on each CI runner.

use rsdag::math;
use rsdag::semantics::unary_f64;
use rsdag::synth::Rng;

/// Uniform in `[0, 1)`.
fn uniform(rng: &mut Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// Against exact values (mpmath, `scripts/math_ref.py`), the errors the
/// README states: the kernels' own, on the ordinary range and toward the
/// edges.
#[test]
fn kernels_are_within_their_error_bounds() {
    let data = include_str!("data/math_ref.txt");
    let mut worst: std::collections::BTreeMap<&str, f64> = Default::default();
    for line in data.lines() {
        let mut it = line.split(' ');
        let name = it.next().unwrap();
        let mut p = || f64::from_bits(u64::from_str_radix(it.next().unwrap(), 16).unwrap());
        let (x, hi, lo) = (p(), p(), p());
        let got = match name {
            "exp" => math::exp(x),
            "ln" => math::ln(x),
            "sinh" => math::sinh(x),
            "cosh" => math::cosh(x),
            "tanh" => math::tanh(x),
            _ => unreachable!(),
        };
        let ulp = hi.abs().next_up() - hi.abs();
        let err = if got == hi && lo == 0.0 {
            0.0
        } else {
            ((got - hi) - lo).abs() / ulp
        };
        let key = if name == "exp" && x.abs() > 700.0 {
            "exp near the edges"
        } else {
            name
        };
        let (bound, w) = (
            match key {
                "exp" => 0.52,
                "exp near the edges" => 1.0,
                "ln" => 0.78,
                "sinh" => 1.75,
                "cosh" => 1.01,
                _ => 2.09,
            },
            worst.entry(key).or_default(),
        );
        *w = w.max(err);
        assert!(err <= bound, "{key}({x:e}): {err} ulp");
    }
    assert_eq!(worst.len(), 6, "every function and region sampled");
}

#[test]
fn the_edges_are_the_ieee_ones() {
    assert_eq!(math::exp(0.0), 1.0);
    assert_eq!(math::exp(f64::NEG_INFINITY), 0.0);
    assert_eq!(math::exp(f64::INFINITY), f64::INFINITY);
    assert_eq!(math::exp(710.0), f64::INFINITY);
    assert_eq!(math::exp(-746.0), 0.0);
    assert!(math::exp(-740.0) > 0.0, "subnormal, not flushed");
    assert!(math::exp(f64::NAN).is_nan());
    assert_eq!(math::ln(1.0), 0.0);
    assert_eq!(math::ln(0.0), f64::NEG_INFINITY);
    assert!(math::ln(-1.0).is_nan());
    assert_eq!(math::ln(f64::INFINITY), f64::INFINITY);
    assert!((math::ln(f64::from_bits(1)) + 744.4400719213812).abs() < 1e-12);
    let c = f64::from_bits(COSH_710);
    assert!(
        (math::cosh(710.0) - c).abs() <= c.next_up() - c,
        "cosh(710) within an ulp"
    );
    assert!(math::cosh(711.0).is_infinite());
    assert_eq!(math::sinh(-0.0).to_bits(), (-0.0f64).to_bits());
    assert_eq!(math::tanh(30.0), 1.0);
    assert_eq!(math::tanh(-30.0), -1.0);
}

#[test]
fn powi_is_square_and_multiply() {
    let mut rng = Rng::new(3);
    for _ in 0..100_000 {
        let x = (uniform(&mut rng) - 0.5) * 8.0;
        let n = (uniform(&mut rng) * 80.0) as i32 - 40;
        assert_eq!(math::powi(x, n).to_bits(), x.powi(n).to_bits(), "{x}^{n}");
    }
}

/// The bits of every unary op on a fixed sample set, hashed. The kernels
/// are IEEE arithmetic in a fixed order, so the hash is the same on every
/// platform; CI runs this on Linux (x86-64, AArch64), macOS and Windows.
/// A kernel changed on purpose changes this constant.
#[test]
fn every_platform_computes_the_same_bits() {
    let mut rng = Rng::new(5);
    let mut xs: Vec<f64> = (0..20_000)
        .map(|_| (uniform(&mut rng) - 0.5) * 60.0)
        .collect();
    xs.extend((0..2_000).map(|_| (uniform(&mut rng) - 0.5) * 1500.0));
    xs.extend([
        0.0,
        -0.0,
        1e-300,
        5e-324,
        1.0,
        -1.0,
        0.5,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ]);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for spec in rsdag::node::UNARY_OPS {
        for &x in &xs {
            let v = unary_f64(spec.op, x);
            let bits = if v.is_nan() {
                0x7ff8_0000_0000_0000
            } else {
                v.to_bits()
            };
            h = (h ^ bits).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    assert_eq!(
        h, PINNED,
        "bits of the elementary functions moved: {h:#018x}"
    );
}

const PINNED: u64 = 0x0ebc_5daf_2175_861b;

/// `cosh(710)` correctly rounded (mpmath).
const COSH_710: u64 = 0x7fe3_e21a_4645_07f9;
