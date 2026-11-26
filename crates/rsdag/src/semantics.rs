//! The reference semantics: what every op computes, in one place.
//!
//! The arena sweep, the tape interpreter and the native backend's host
//! routines all take their arithmetic from here, so a value cannot depend on which of them
//! computed it. That covers the domain guards (the `exp` cap at
//! [`EXP_LIMIT`], the `ln` floor at [`LN_FLOOR`], the `sqrt` clamp), the
//! special functions, which are defined here rather than taken from a
//! platform library, and the fold orders of reductions and dot products. A
//! backend that reproduces these bit for bit reproduces every program.

use crate::constants::{
    DIGAMMA_COEFFS, EXP_INV_LN2_N, EXP_LN2_HI_N, EXP_LN2_LO_N, EXP_POLY, EXP_SCALE_BIAS, EXP_TABLE,
    EXP_TABLE_BITS, EXP_UNDERFLOW, F64_EXP_BIAS, F64_FRAC_BITS, LN_LN2_HI, LN_LN2_LO, LN_OFF,
    LN_POLY, LN_TABLE, LN_TABLE_BITS, LU_PANEL_LARGE, LU_PANEL_SMALL, LU_PANEL_SWITCH,
    LU_TRAILING_ROWS, LU_UNBLOCKED_MAX, MAGIC, POLYGAMMA_SHIFT, TRIGAMMA_COEFFS,
};
use crate::node::{BinOp, CmpOp, Elementary, ReduceOp, UnaryOp};
use crate::tape::Fold;

pub use crate::constants::FOLD_LANES;

/// The reduction of a slice: four accumulators ([`FOLD_LANES`]), merged
/// as `(a0 + a1) + (a2 + a3)`, then the tail in order. The one fold order
/// every evaluator uses, so results agree to the bit; for fewer than four
/// operands the accumulators start at the identity and the result is the
/// sequential fold's exactly.
pub fn reduce_slice(op: ReduceOp, xs: &[f64]) -> f64 {
    match op {
        ReduceOp::Sum => fold_lanes(xs, 0.0, |a, x| a + x),
        ReduceOp::Product => fold_lanes(xs, 1.0, |a, x| a * x),
        ReduceOp::Min => fold_first(xs, f64::INFINITY, f64::min),
        ReduceOp::Max => fold_first(xs, f64::NEG_INFINITY, f64::max),
    }
}

#[inline(always)]
fn fold_lanes(xs: &[f64], id: f64, f: impl Fn(f64, f64) -> f64) -> f64 {
    let mut a = [id; FOLD_LANES];
    let ch = xs.len() / FOLD_LANES;
    for c in 0..ch {
        for l in 0..FOLD_LANES {
            a[l] = f(a[l], xs[FOLD_LANES * c + l]);
        }
    }
    let acc = f(f(a[0], a[1]), f(a[2], a[3]));
    (xs[ch * FOLD_LANES..].iter()).fold(acc, |acc, &x| f(acc, x))
}

#[inline(always)]
fn fold_first(xs: &[f64], empty: f64, f: impl Fn(f64, f64) -> f64) -> f64 {
    match xs.split_first() {
        None => empty,
        Some((&x, rest)) => rest.iter().fold(x, |acc, &x| f(acc, x)),
    }
}

/// Canonical evaluation of an [`Elementary`] at `args` (as many as its
/// arity).
pub fn elementary_f64(f: Elementary, args: &[f64]) -> f64 {
    debug_assert_eq!(
        args.len(),
        f.arity(),
        "{} takes {} arguments",
        f.name(),
        f.arity()
    );
    match f {
        Elementary::Unary(op) => unary_f64(op, args[0]),
        Elementary::Binary(op) => binary_f64(op, args[0], args[1]),
        Elementary::Min => reduce_slice(ReduceOp::Min, &args[..2]),
        Elementary::Max => reduce_slice(ReduceOp::Max, &args[..2]),
    }
}

/// Canonical evaluation of a [`BinOp`], the one reference for every backend.
pub fn binary_f64(op: BinOp, x: f64, y: f64) -> f64 {
    match op {
        BinOp::Powf => libm::pow(x, y),
        BinOp::Mod => libm::fmod(x, y),
        BinOp::Atan2 => libm::atan2(x, y),
        BinOp::Hypot => libm::hypot(x, y),
        BinOp::RandStream => rand_stream(x, y),
    }
}

/// Digamma `psi(x)`: the recurrence `psi(x) = psi(x + 1) - 1 / x` up to
/// [`POLYGAMMA_SHIFT`], then the asymptotic expansion with the coefficients
/// [`DIGAMMA_COEFFS`]; for `x <= 0` the reflection
/// `psi(x) = psi(1 - x) - pi / tan(pi x)`, where `tan(pi x)` is evaluated as
/// `tan(pi (x - round(x)))`. The difference `x - round(x)` is exact, so the
/// argument of `tan` carries one rounding relative to itself and the term
/// keeps its relative accuracy near the poles and for large `|x|`.
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        return digamma(1.0 - x) - std::f64::consts::PI / libm::tan(pi_x_reduced(x));
    }
    let (mut x, mut result) = (x, 0.0);
    while x < POLYGAMMA_SHIFT {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + libm::log(x) - 0.5 * inv - inv2 * horner(&DIGAMMA_COEFFS, inv2)
}

/// Trigamma `psi'(x)`: the recurrence `psi'(x) = psi'(x + 1) + 1 / x^2` up
/// to [`POLYGAMMA_SHIFT`], then the asymptotic expansion with the
/// coefficients [`TRIGAMMA_COEFFS`]; for `x <= 0` the reflection
/// `psi'(x) = pi^2 / sin^2(pi x) - psi'(1 - x)`, with `sin(pi x)` evaluated
/// at the exactly reduced argument as in [`digamma`].
pub fn trigamma(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return if x.is_nan() { x } else { 0.0 };
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::INFINITY;
        }
        let s = libm::sin(pi_x_reduced(x));
        let pi2 = std::f64::consts::PI * std::f64::consts::PI;
        return pi2 / (s * s) - trigamma(1.0 - x);
    }
    let (mut x, mut result) = (x, 0.0);
    while x < POLYGAMMA_SHIFT {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + inv + 0.5 * inv2 + inv * inv2 * horner(&TRIGAMMA_COEFFS, inv2)
}

fn pi_x_reduced(x: f64) -> f64 {
    std::f64::consts::PI * (x - x.round())
}

fn horner(coeffs: &[f64], t: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |p, &c| p * t + c)
}

// https://prng.di.unimi.it/splitmix64.c
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(z: u64) -> f64 {
    ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// Counter-based uniform noise in `[0, 1)` from the bits of `key`
/// (splitmix64 finalizer; the top 53 bits become the mantissa).
pub fn rand_uniform(key: f64) -> f64 {
    unit(splitmix64(key.to_bits()))
}

/// Counter-based uniform noise in `[0, 1)` of stream `key` at `counter`:
/// the stream's own key (the mixed bits of `key`) combined with the
/// counter's bits, mixed again. Distinct pairs draw independently.
pub fn rand_stream(key: f64, counter: f64) -> f64 {
    let k = splitmix64(key.to_bits());
    unit(splitmix64(
        k ^ counter.to_bits().wrapping_mul(0xD1B5_4A32_D192_ED03),
    ))
}

pub use crate::constants::{EXP_AT_LIMIT, EXP_LIMIT, LN_AT_FLOOR, LN_FLOOR};
#[cfg(target_arch = "x86_64")]
pub use crate::simd::sse2_baseline;

/// Whether a multiply-add rounds once on this machine: where the CPU has a
/// fused multiply-add (x86-64 with AVX and FMA, unless `RSDAG_SSE2` is set;
/// AArch64 always). Every evaluator decides by this, so on one machine they
/// agree to the bit; across machines a result may differ in its last bits.
pub fn fused() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| {
            !sse2_baseline()
                && std::is_x86_feature_detected!("avx")
                && std::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// `a b + c`, rounded once ([`Fused`]) or twice ([`Split`]): the one
/// primitive the exponential and the logarithm are written over, so the
/// machine's [`fused`] decides their rounding and every backend follows.
pub trait MulAdd {
    fn mul_add(a: f64, b: f64, c: f64) -> f64;
}

/// The fused multiply-add.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub struct Fused;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl MulAdd for Fused {
    #[inline(always)]
    fn mul_add(a: f64, b: f64, c: f64) -> f64 {
        a.mul_add(b, c)
    }
}

/// A product and a sum, each rounded.
pub struct Split;
impl MulAdd for Split {
    #[inline(always)]
    fn mul_add(a: f64, b: f64, c: f64) -> f64 {
        a * b + c
    }
}

/// The exponential's and the logarithm's constants and steps, for a backend
/// that computes them in its own instructions (it must take the same
/// steps).
pub mod exp_ln {
    pub use super::{exp_core, exp_with, ln_core, ln_with};
    pub use super::{EXP_INDEX_MASK, EXP_SHIFT, EXP_UNBIAS, LN_EXP_MASK, LN_INDEX_MASK, LN_SHIFT};
    pub use crate::constants::{
        EXP_AT_LIMIT, EXP_INV_LN2_N, EXP_LIMIT, EXP_LN2_HI_N, EXP_LN2_LO_N, EXP_POLY,
        EXP_SCALE_BIAS, EXP_TABLE, EXP_TABLE_BITS, EXP_UNDERFLOW, F64_EXP_BIAS, F64_FRAC_BITS,
        LN_AT_FLOOR, LN_FLOOR, LN_LN2_HI, LN_LN2_LO, LN_OFF, LN_POLY, LN_TABLE, LN_TABLE_BITS,
        MAGIC,
    };
}

/// `exp`'s table index: the low bits of `k`.
pub const EXP_INDEX_MASK: u64 = (1 << EXP_TABLE_BITS) - 1;
/// The shift that takes `k` to `k / N` in the exponent field.
pub const EXP_SHIFT: u32 = F64_FRAC_BITS - EXP_TABLE_BITS;
/// `2^-EXP_SCALE_BIAS`.
pub const EXP_UNBIAS: f64 =
    f64::from_bits(((F64_EXP_BIAS - EXP_SCALE_BIAS) as u64) << F64_FRAC_BITS);
/// `ln`'s table index: the top fraction bits of `z`, shifted down by
/// [`LN_SHIFT`].
pub const LN_INDEX_MASK: u64 = (1 << LN_TABLE_BITS) - 1;
/// See [`LN_INDEX_MASK`].
pub const LN_SHIFT: u32 = F64_FRAC_BITS - LN_TABLE_BITS;
/// The sign and exponent bits of an `f64`.
pub const LN_EXP_MASK: u64 = u64::MAX << F64_FRAC_BITS;

/// `exp(x)` for `x <= EXP_LIMIT` (and NaN), multiply-adds by `M`: `x = k
/// ln 2 / N + r`, `exp(x) = 2^(k / N) exp(r)`. Zero below the subnormals.
#[inline(always)]
pub fn exp_with<M: MulAdd>(x: f64) -> f64 {
    let t = M::mul_add(x, EXP_INV_LN2_N, MAGIC);
    let kd = t - MAGIC;
    let ki = t.to_bits();
    let r = M::mul_add(kd, -EXP_LN2_HI_N, x);
    let r = M::mul_add(kd, -EXP_LN2_LO_N, r);
    let j = (ki & EXP_INDEX_MASK) as usize;
    let tail = f64::from_bits(EXP_TABLE[2 * j]);
    let scale = f64::from_bits(EXP_TABLE[2 * j + 1].wrapping_add(ki << EXP_SHIFT));
    let [c2, c3, c4, c5] = EXP_POLY;
    let r2 = r * r;
    let p23 = M::mul_add(r, c3, c2);
    let p45 = M::mul_add(r, c5, c4);
    let p = M::mul_add(r2, p23, tail + r);
    let p = M::mul_add(r2 * r2, p45, p);
    let v = M::mul_add(scale, p, scale) * EXP_UNBIAS;
    let v = if x < EXP_UNDERFLOW { 0.0 } else { v };
    if x.is_nan() {
        x
    } else {
        v
    }
}

/// `ln(x)` for normal `x > 0` (and `+inf`, NaN), multiply-adds by `M`: `x
/// = 2^k z`, `log x = k ln 2 + logc + log1p(r)` with `r = (z - c) invc`.
#[inline(always)]
pub fn ln_with<M: MulAdd>(x: f64) -> f64 {
    let ix = x.to_bits();
    let tmp = ix.wrapping_sub(LN_OFF);
    let i = ((tmp >> LN_SHIFT) & LN_INDEX_MASK) as usize;
    let kd = ((tmp as i64) >> F64_FRAC_BITS) as f64;
    let z = f64::from_bits(ix.wrapping_sub(tmp & LN_EXP_MASK));
    let [invc, c, logc_hi, logc_lo] = LN_TABLE[i];
    let r = (z - c) * invc;
    let w = M::mul_add(kd, LN_LN2_HI, logc_hi);
    let hi = w + r;
    let lo = (w - hi) + r;
    let lo = lo + M::mul_add(kd, LN_LN2_LO, logc_lo);
    let a = LN_POLY;
    let r2 = r * r;
    let q01 = M::mul_add(r, a[1], a[0]);
    let q23 = M::mul_add(r, a[3], a[2]);
    let q45 = M::mul_add(r, a[5], a[4]);
    let q03 = M::mul_add(r2, q23, q01);
    let q46 = M::mul_add(r2, a[6], q45);
    let p = M::mul_add(r2 * r2, q46, q03);
    let t = M::mul_add(r2, -0.5, lo);
    let t = M::mul_add(r2 * r, p, t);
    let v = t + hi;
    if x.is_nan() || x == f64::INFINITY {
        x
    } else {
        v
    }
}

#[cfg(target_arch = "x86_64")]
mod fma {
    use super::{exp_with, ln_with, Fused};

    #[target_feature(enable = "fma")]
    pub(super) unsafe fn exp(x: f64) -> f64 {
        exp_with::<Fused>(x)
    }
    #[target_feature(enable = "fma")]
    pub(super) unsafe fn ln(x: f64) -> f64 {
        ln_with::<Fused>(x)
    }
}

/// [`exp_with`] in this machine's rounding ([`fused`]).
#[inline]
pub fn exp_core(x: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if fused() {
        // SAFETY: `fused` holds only where the CPU has FMA.
        return unsafe { fma::exp(x) };
    }
    #[cfg(target_arch = "aarch64")]
    if fused() {
        return exp_with::<Fused>(x);
    }
    exp_with::<Split>(x)
}

/// [`ln_with`] in this machine's rounding ([`fused`]).
#[inline]
pub fn ln_core(x: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if fused() {
        // SAFETY: `fused` holds only where the CPU has FMA.
        return unsafe { fma::ln(x) };
    }
    #[cfg(target_arch = "aarch64")]
    if fused() {
        return ln_with::<Fused>(x);
    }
    ln_with::<Split>(x)
}

/// The guarded exponential of [`unary_f64`]: linear above [`EXP_LIMIT`].
#[inline(always)]
pub fn exp_guarded(x: f64) -> f64 {
    let tail = EXP_AT_LIMIT * (1.0 + (x - EXP_LIMIT));
    if x > EXP_LIMIT {
        tail
    } else {
        exp_core(x)
    }
}

/// The guarded logarithm of [`unary_f64`]: `ln(LN_FLOOR)` at or below
/// [`LN_FLOOR`], a NaN argument staying NaN.
#[inline(always)]
pub fn ln_guarded(x: f64) -> f64 {
    if x <= LN_FLOOR {
        LN_AT_FLOOR
    } else {
        ln_core(x)
    }
}

/// `x^n` for an integer `n`, the reference every evaluator takes: square
/// and multiply over `|n|` from the low bit, then one division for a
/// negative exponent. `n = 2` is exactly `x * x` and `n = -1` exactly
/// `1 / x`, the forms native code emits inline. Not the platform's `powi`,
/// whose rounding is unspecified and differs between targets (a runtime
/// `x.powi(-1)` is not `1 / x` on every one).
pub fn powi_f64(x: f64, n: i32) -> f64 {
    let mut e = n.unsigned_abs();
    let mut base = x;
    let mut acc: Option<f64> = None;
    while e > 0 {
        if e & 1 == 1 {
            acc = Some(match acc {
                None => base,
                Some(a) => a * base,
            });
        }
        e >>= 1;
        if e > 0 {
            base *= base;
        }
    }
    let r = acc.unwrap_or(1.0);
    if n < 0 {
        1.0 / r
    } else {
        r
    }
}

/// Evaluate a unary op on a real argument. Single source of truth shared by the
/// arena evaluator ([`crate::eval`](mod@crate::eval)) and the compiled tape ([`crate::tape`]).
///
/// `exp`/`ln`/`sqrt` are domain-guarded (limited exponential, log floor, and
/// `sqrt` of a negative argument clamped to 0): on an out-of-range intermediate
/// Newton iterate these return a finite, smoothly-extrapolated value instead of
/// `inf`/`NaN`, so a single out-of-range internal node cannot poison the residual
/// and Jacobian. Every in-range argument evaluates identically to the bare op.
/// The derivative rules in [`crate::autodiff`] mirror these guards exactly.
pub fn unary_f64(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Exp => exp_guarded(x),
        UnaryOp::Ln => ln_guarded(x),
        UnaryOp::Sqrt => {
            if x <= 0.0 {
                0.0
            } else {
                x.sqrt()
            }
        }
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Sinh => x.sinh(),
        UnaryOp::Cosh => x.cosh(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Atan => x.atan(),
        UnaryOp::Floor => x.floor(),
        UnaryOp::Tan => libm::tan(x),
        UnaryOp::Log10 => libm::log10(x),
        UnaryOp::Log2 => libm::log2(x),
        UnaryOp::Log1p => libm::log1p(x),
        UnaryOp::Expm1 => libm::expm1(x),
        UnaryOp::Cbrt => libm::cbrt(x),
        UnaryOp::Abs => x.abs(),
        UnaryOp::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                x
            }
        }
        UnaryOp::Ceil => x.ceil(),
        UnaryOp::Round => libm::round(x),
        UnaryOp::Trunc => x.trunc(),
        UnaryOp::Asin => libm::asin(x),
        UnaryOp::Acos => libm::acos(x),
        UnaryOp::Asinh => libm::asinh(x),
        UnaryOp::Acosh => libm::acosh(x),
        UnaryOp::Atanh => libm::atanh(x),
        UnaryOp::Erf => libm::erf(x),
        UnaryOp::Erfc => libm::erfc(x),
        UnaryOp::Lgamma => libm::lgamma(x),
        UnaryOp::Tgamma => libm::tgamma(x),
        UnaryOp::Digamma => digamma(x),
        UnaryOp::Trigamma => trigamma(x),
        UnaryOp::RandUniform => rand_uniform(x),
    }
}

/// Evaluate a [`CmpOp`] on two ordered arguments. Single source of truth shared
/// by the arena evaluator, the complex evaluator, the compiled tape (all `f64`)
/// and the constant-folding interner (exact `BigRational`), so a `Cmp` node's
/// `1.0`/`0.0` result agrees across every backend.
pub fn cmp_bool<T: PartialOrd>(op: CmpOp, x: T, y: T) -> bool {
    match op {
        CmpOp::Gt => x > y,
        CmpOp::Ge => x >= y,
        CmpOp::Lt => x < y,
        CmpOp::Le => x <= y,
        CmpOp::Eq => x == y,
        CmpOp::Ne => x != y,
    }
}

/// The inner product in the fold of [`reduce_slice`] over the products.
pub fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    crate::simd::dot(a, b)
}

/// A dense matrix-vector product: `out[i] = dot(a[i*n..(i+1)*n], x)` for
/// `m` rows, each row the fold of [`dot_slice`], so a fused product is
/// bit-identical to its rows as separate dots.
pub fn gemv(a: &[f64], x: &[f64], m: usize, n: usize, out: &mut [f64]) {
    crate::simd::gemv(a, x, m, n, out)
}

/// [`gemv`] with its rows folded per `codes` against `c` (see
/// [`gemm_fold`]).
pub fn gemv_fold(
    a: &[f64],
    x: &[f64],
    m: usize,
    n: usize,
    c: Option<&[f64]>,
    codes: &[u32],
    out: &mut [f64],
) {
    crate::simd::gemv(a, x, m, n, out);
    fold_in_place(codes, c, out);
}

/// A dense matrix-matrix product against rows: `out[i*n + j] =
/// dot(a[i*k..], b[j*k..])` for `m` rows of `a` and `n` rows of `b`, each
/// of `k` (the right factor by columns, each contiguous). Every entry is
/// the fold of [`dot_slice`], so a fused product is bit-identical to its
/// entries as separate dots.
pub fn gemm(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, out: &mut [f64]) {
    crate::simd::gemm(a, b, m, k, n, out)
}

/// [`gemm`] with its entries folded per `codes` (see [`crate::tape::Fold`])
/// against the accumulator `c`: an entry folded with an operand is folded
/// as it is stored; a self fold (against another entry's product) is
/// folded after every product is stored, the products it reads being those
/// of entries that stay plain.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fold(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    c: Option<&[f64]>,
    codes: &[u32],
    out: &mut [f64],
) {
    crate::simd::gemm_fold(a, b, m, k, n, c, codes, out)
}

pub(crate) fn fold_in_place(codes: &[u32], c: Option<&[f64]>, out: &mut [f64]) {
    for i in 0..out.len() {
        let f = Fold(codes[i]);
        if f.is_plain() {
            continue;
        }
        let acc = if f.is_self() {
            out[f.self_index()]
        } else {
            c.map_or(out[i], |c| c[i])
        };
        out[i] = f.fold(acc, out[i]);
    }
}

/// `x` with `A x = b` for the `n` by `n` matrix `a` (row-major): Gaussian
/// elimination with partial pivoting (the first row of the largest
/// magnitude), panel-blocked past [`crate::constants::LU_UNBLOCKED_MAX`]
/// unknowns. The order of its operations is part of the contract.
pub fn solve(a: &[f64], b: &[f64], n: usize, out: &mut [f64]) {
    solve_many(a, b, n, 1, out)
}

/// [`solve`] for `k` right-hand sides at once: `b` holds `k` vectors of
/// `n` back to back, `out` receives the `k` solutions the same way. One
/// factorization serves every right-hand side; each solution is bit-identical
/// to its own [`solve`] (the pivots depend on `a` alone, and every
/// right-hand side column runs through the same updates in the same order).
/// Allocates its scratch; [`solve_many_into`] takes it from the caller.
pub fn solve_many(a: &[f64], b: &[f64], n: usize, k: usize, out: &mut [f64]) {
    let mut scratch = vec![0.0; solve_scratch_len(n, k)];
    solve_many_into(a, b, n, k, out, &mut scratch)
}

/// [`solve_many`] over a `scratch` of at least [`solve_scratch_len`]
/// values, allocating nothing.
pub fn solve_many_into(
    a: &[f64],
    b: &[f64],
    n: usize,
    k: usize,
    out: &mut [f64],
    scratch: &mut [f64],
) {
    crate::simd::solve_many(a, b, n, k, out, &mut scratch[..solve_scratch_len(n, k)])
}

/// `count` independent systems of one shape: `a` holds their `n` by `n`
/// matrices back to back, `b` their `k` right-hand sides each (as
/// [`solve_many`] takes them), `out` their solutions the same way. Each
/// solution is bit-identical to its system's own [`solve_many_into`]; the
/// systems share the `scratch` of one.
#[allow(clippy::too_many_arguments)]
pub fn solve_batch_into(
    a: &[f64],
    b: &[f64],
    n: usize,
    k: usize,
    count: usize,
    out: &mut [f64],
    scratch: &mut [f64],
) {
    let scratch = &mut scratch[..solve_scratch_len(n, k)];
    crate::simd::solve_batch(a, b, n, k, count, out, scratch)
}

pub(crate) fn lu_panel(n: usize) -> usize {
    if n <= LU_UNBLOCKED_MAX {
        0
    } else if n < LU_PANEL_SWITCH {
        LU_PANEL_SMALL
    } else {
        LU_PANEL_LARGE
    }
}

/// The scratch a solve of `n` unknowns and `k` right-hand sides works in:
/// the augmented matrix, and past [`crate::constants::LU_UNBLOCKED_MAX`]
/// the panel's buffers for the trailing update.
pub fn solve_scratch_len(n: usize, k: usize) -> usize {
    let (w, nb) = (n + k, lu_panel(n));
    n * w
        + if nb > 0 {
            nb * w + LU_TRAILING_ROWS * nb + LU_TRAILING_ROWS * w
        } else {
            0
        }
}

pub(crate) fn solve_parts<'s>(
    scratch: &'s mut [f64],
    a: &[f64],
    b: &[f64],
    n: usize,
    k: usize,
) -> [&'s mut [f64]; 4] {
    let (w, nb) = (n + k, lu_panel(n));
    let (m, rest) = scratch.split_at_mut(n * w);
    for i in 0..n {
        m[i * w..i * w + n].copy_from_slice(&a[i * n..(i + 1) * n]);
        for c in 0..k {
            m[i * w + n + c] = b[c * n + i];
        }
    }
    let (ut, rest) = rest.split_at_mut(nb * w);
    let (lrows, prod) = rest.split_at_mut(LU_TRAILING_ROWS * nb);
    [m, ut, lrows, prod]
}
