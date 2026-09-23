//! rsdag's own elementary functions: within a small error of the platform
//! library, and the same bits on every platform, which is what the pinned
//! hash below asserts on each CI runner.

use rsdag::math;
use rsdag::semantics::unary_f64;
use rsdag::synth::Rng;

fn ulps(a: f64, b: f64) -> f64 {
    if a == b || (a.is_nan() && b.is_nan()) {
        return 0.0;
    }
    let ulp = b.abs().next_up() - b.abs();
    (a - b).abs() / ulp
}

/// Uniform in `[0, 1)`.
fn uniform(rng: &mut Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

fn samples(rng: &mut Rng, lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n).map(|_| lo + (hi - lo) * uniform(rng)).collect()
}

#[test]
fn kernels_are_close_to_the_platform_library() {
    let mut rng = Rng::new(11);
    let wide = samples(&mut rng, -740.0, 709.0, 200_000);
    let near = samples(&mut rng, -3.0, 3.0, 200_000);
    let pos: Vec<f64> = samples(&mut rng, -300.0, 300.0, 200_000)
        .iter()
        .map(|e| 10f64.powf(*e))
        .collect();
    // The platform is within about half an ulp; the kernels within the
    // bound, so these are the kernels' own errors plus that half.
    let cases: [(&str, &[f64], fn(f64) -> f64, fn(f64) -> f64, f64); 6] = [
        ("exp", &wide, math::exp, f64::exp, 1.6),
        ("exp", &near, math::exp, f64::exp, 1.6),
        ("ln", &pos, math::ln, f64::ln, 1.4),
        ("sinh", &near, math::sinh, f64::sinh, 2.4),
        ("cosh", &near, math::cosh, f64::cosh, 1.6),
        ("tanh", &near, math::tanh, f64::tanh, 2.7),
    ];
    for (name, xs, ours, platform, bound) in cases {
        let worst = xs
            .iter()
            .map(|&x| (ulps(ours(x), platform(x)), x))
            .fold((0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a });
        assert!(worst.0 <= bound, "{name}: {} ulp at {:e}", worst.0, worst.1);
    }
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
    assert_eq!(math::cosh(710.0), 710f64.cosh());
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
