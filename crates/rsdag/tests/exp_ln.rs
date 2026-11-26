//! rsdag's own exponential and logarithm (`semantics::exp_core`,
//! `semantics::ln_core`) against a reference in double-double arithmetic
//! (about 100 bits), within the error the algorithm states
//! (`rsdag::constants::exp_ln`), with the multiply-adds fused and not; and
//! their special values.

mod common;
use common::constants::LN2_LO;
use rsdag::semantics::{
    exp_core, exp_guarded, exp_with, ln_core, ln_guarded, ln_with, MulAdd, Split, EXP_AT_LIMIT,
    EXP_LIMIT, LN_AT_FLOOR, LN_FLOOR,
};

const EXP_ULPS: f64 = 0.51;
const EXP_SUBNORMAL_ROUNDED_TWICE_ULPS: f64 = 0.76;
const LN_ULPS: f64 = 0.63;

#[derive(Clone, Copy)]
struct DoubleDouble(f64, f64);

fn two_sum(a: f64, b: f64) -> DoubleDouble {
    let s = a + b;
    let bb = s - a;
    DoubleDouble(s, (a - (s - bb)) + (b - bb))
}
fn two_prod(a: f64, b: f64) -> DoubleDouble {
    let p = a * b;
    DoubleDouble(p, a.mul_add(b, -p))
}
fn add(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
    let s = two_sum(a.0, b.0);
    let t = two_sum(a.1, b.1);
    let s = two_sum(s.0, s.1 + t.0);
    two_sum(s.0, s.1 + t.1)
}
fn mul(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
    let p = two_prod(a.0, b.0);
    two_sum(p.0, p.1 + (a.0 * b.1 + a.1 * b.0))
}
fn neg(a: DoubleDouble) -> DoubleDouble {
    DoubleDouble(-a.0, -a.1)
}

const LN2: DoubleDouble = DoubleDouble(std::f64::consts::LN_2, LN2_LO);

fn exp_by_series_for_r_up_to_0_36(r: DoubleDouble) -> DoubleDouble {
    let (mut term, mut sum) = (DoubleDouble(1.0, 0.0), DoubleDouble(1.0, 0.0));
    for n in 1..=30 {
        term = div_by_integer(mul(term, r), n as f64);
        sum = add(sum, term);
    }
    sum
}

fn div_by_integer(t: DoubleDouble, n: f64) -> DoubleDouble {
    let q = t.0 / n;
    let p = two_prod(q, n);
    two_sum(q, (((t.0 - p.0) - p.1) + t.1) / n)
}

fn exp_ref_as_k_and_exp_r(x: f64) -> (i32, DoubleDouble) {
    let k = (x / std::f64::consts::LN_2).round();
    let r = add(DoubleDouble(x, 0.0), neg(mul(DoubleDouble(k, 0.0), LN2)));
    (k as i32, exp_by_series_for_r_up_to_0_36(r))
}

fn exp_ulps(y: f64, k: i32, v: DoubleDouble) -> f64 {
    let h = k / 2;
    let y_over_2_to_k_in_two_exact_steps = y * 2f64.powi(-h) * 2f64.powi(-(k - h));
    let ev = v.0.log2().floor() as i32;
    let d = (y_over_2_to_k_in_two_exact_steps - v.0) - v.1;
    let e = (k + ev).max(-1022) - k;
    d.abs() * 2f64.powi(52 - e)
}

fn ln_ref(x: f64) -> DoubleDouble {
    let e = x.log2().floor() as i32;
    let mut m = x * 2f64.powi(-e);
    let mut e = e;
    if m > std::f64::consts::SQRT_2 {
        m /= 2.0;
        e += 1;
    }
    let t = m - 1.0;
    let lm = if t.abs() < 1.0 / 16.0 {
        let (mut p, mut sum) = (DoubleDouble(1.0, 0.0), DoubleDouble(0.0, 0.0));
        for n in 1..=40 {
            p = mul(p, DoubleDouble(t, 0.0));
            let term = div_by_integer(p, n as f64);
            sum = add(sum, if n % 2 == 1 { term } else { neg(term) });
        }
        sum
    } else {
        let y = m.ln();
        let ey = exp_by_series_for_r_up_to_0_36(DoubleDouble(-y, 0.0));
        add(
            DoubleDouble(y, 0.0),
            add(mul(DoubleDouble(m, 0.0), ey), DoubleDouble(-1.0, 0.0)),
        )
    };
    add(mul(DoubleDouble(e as f64, 0.0), LN2), lm)
}

fn ln_ulps(y: f64, l: DoubleDouble) -> f64 {
    let e = l.0.abs().log2().floor() as i32;
    ((y - l.0) - l.1).abs() * 2f64.powi(52 - e)
}

fn xorshift_uniforms(mut s: u64) -> impl FnMut() -> f64 {
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn exp_worst<M: MulAdd>() -> (f64, f64) {
    let mut u = xorshift_uniforms(0x9E37_79B9_7F4A_7C15);
    let (mut normal, mut subnormal) = (0.0f64, 0.0f64);
    for _ in 0..100_000 {
        for x in [
            -708.3 + u() * (EXP_LIMIT + 708.3),
            -40.0 + 80.0 * u(),
            (u() - 0.5) * 10f64.powf(-12.0 * u()),
        ] {
            let (k, v) = exp_ref_as_k_and_exp_r(x);
            normal = normal.max(exp_ulps(exp_with::<M>(x), k, v));
        }
        let x = -745.13 + u() * (745.13 - 708.4);
        let (k, v) = exp_ref_as_k_and_exp_r(x);
        subnormal = subnormal.max(exp_ulps(exp_with::<M>(x), k, v));
    }
    (normal, subnormal)
}

fn ln_worst<M: MulAdd>() -> f64 {
    let mut u = xorshift_uniforms(0xD1B5_4A32_D192_ED03);
    let mut worst = 0.0f64;
    for _ in 0..100_000 {
        for x in [
            10f64.powf(-30.0 + 338.0 * u()),
            1.0 + (u() - 0.5) / 4.0,
            1.0 + (u() - 0.5) * 10f64.powf(-15.0 * u()),
        ] {
            if x != 1.0 {
                worst = worst.max(ln_ulps(ln_with::<M>(x), ln_ref(x)));
            }
        }
    }
    worst
}

fn check_exp<M: MulAdd>(how: &str) {
    let (normal, subnormal) = exp_worst::<M>();
    assert!(normal < EXP_ULPS, "exp {how}: {normal} ulps");
    assert!(
        subnormal < EXP_SUBNORMAL_ROUNDED_TWICE_ULPS,
        "exp {how}, subnormal: {subnormal} ulps"
    );
}

#[test]
fn exp_is_within_its_stated_error() {
    check_exp::<Split>("unfused");
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    check_exp::<rsdag::semantics::Fused>("fused");
}

#[test]
fn ln_is_within_its_stated_error() {
    let w = ln_worst::<Split>();
    assert!(w < LN_ULPS, "ln unfused: {w} ulps");
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let w = ln_worst::<rsdag::semantics::Fused>();
        assert!(w < LN_ULPS, "ln fused: {w} ulps");
    }
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
    assert_eq!(exp_core(f64::MIN_POSITIVE.ln() - 40.0), 0.0);
    assert_eq!(exp_guarded(EXP_LIMIT + 1.0), EXP_AT_LIMIT * 2.0);
    assert_eq!(exp_guarded(f64::INFINITY), f64::INFINITY);
    for x in [
        0.0,
        -0.0,
        -1.0,
        5e-324,
        f64::MIN_POSITIVE,
        LN_FLOOR,
        f64::NEG_INFINITY,
    ] {
        assert_eq!(ln_guarded(x), LN_AT_FLOOR, "ln({x:e})");
    }
    assert!(ln_guarded(f64::NAN).is_nan() && exp_guarded(f64::NAN).is_nan());
    let (k, v) = exp_ref_as_k_and_exp_r(EXP_LIMIT);
    assert!(
        exp_ulps(EXP_AT_LIMIT, k, v) <= 0.5,
        "EXP_AT_LIMIT is not the correctly rounded value"
    );
    assert!(
        ln_ulps(LN_AT_FLOOR, ln_ref(LN_FLOOR)) <= 0.5,
        "LN_AT_FLOOR is not the correctly rounded value"
    );
}
