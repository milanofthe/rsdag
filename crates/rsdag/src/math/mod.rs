//! The elementary functions every backend evaluates, as rsdag's own code:
//! IEEE additions, multiplications, divisions and integer bit operations in
//! a fixed order, no fused multiply-add, no platform library. A value is
//! therefore the same bits on every platform (AArch64, x86-64, wasm32) and
//! in every backend (the interpreter calls these functions, the native
//! code calls or reproduces them).
//!
//! - [`exp`]: table-driven, 128 points of `2^(i/128)` with their tails and
//!   a degree-5 polynomial on `|r| <= ln2/256`.
//! - [`ln`]: the reduction to `[sqrt(2)/2, sqrt(2)]` and the rational form
//!   of FreeBSD msun's `e_log.c` (Sun Microsystems, notice below).
//! - [`sinh`], [`cosh`], [`tanh`]: on [`exp`], and on `libm::expm1` near
//!   zero where `exp` would cancel.
//! - [`powi`]: square and multiply over the exponent's bits, then one
//!   reciprocal for a negative exponent.
//!
//! The functions without a kernel here (`sin`, `cos`, `atan`, `tan`, the
//! inverse and special functions) are the `libm` crate's, which is Rust
//! code too and so as deterministic.
//!
//! `ln` is derived from FreeBSD `/usr/src/lib/msun/src/e_log.c`:
//!
//! > Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//! > Developed at SunSoft, a Sun Microsystems, Inc. business. Permission to
//! > use, copy, modify, and distribute this software is freely granted,
//! > provided that this notice is preserved.

#[cfg(feature = "complex")]
pub mod complex;
mod table;

/// The top 12 bits of `x` (sign and exponent).
#[inline]
fn top12(x: f64) -> u32 {
    (x.to_bits() >> 52) as u32
}

/// `0x1.8p52`: added to a value below `2^51` in magnitude it rounds it to
/// an integer that the low bits of the sum then hold, in two's complement.
const SHIFT: f64 = 6755399441055744.0;
const C2: f64 = 1.0 / 2.0;
const C3: f64 = 1.0 / 6.0;
const C4: f64 = 1.0 / 24.0;
const C5: f64 = 1.0 / 120.0;

/// `e^x`. `x = k ln2/128 + r` with `|r| <= ln2/256`, `k = 128 q + i`; then
/// `e^x = 2^q 2^(i/128) e^r`, the table giving `2^(i/128)` as a double and
/// its tail, the polynomial `e^r - 1` to far below an ulp.
#[inline]
pub fn exp(x: f64) -> f64 {
    exp_pow2(x, 0)
}

/// `e^x 2^s`, the power of two applied to the table scale, so exactly: the
/// result rounds once, wherever `e^x` alone would overflow.
#[inline]
fn exp_pow2(x: f64, s: i64) -> f64 {
    let mut abstop = top12(x) & 0x7ff;
    if abstop.wrapping_sub(top12(f64::from_bits(0x3c9 << 52)))
        >= top12(512.0).wrapping_sub(top12(f64::from_bits(0x3c9 << 52)))
    {
        if abstop.wrapping_sub(top12(f64::from_bits(0x3c9 << 52))) >= 0x8000_0000 {
            // |x| < 2^-54: 1 + x rounds to what e^x rounds to.
            return 1.0 + x;
        }
        if abstop >= top12(1024.0) {
            if x == f64::NEG_INFINITY {
                return 0.0;
            }
            if abstop >= top12(f64::INFINITY) {
                return 1.0 + x; // NaN, +inf
            }
            return if x.is_sign_negative() {
                0.0
            } else {
                f64::INFINITY
            };
        }
        // 512 <= |x| < 1024: the scale may leave the normal range.
        abstop = 0;
    }
    let z = table::INV_LN2_N * x;
    let kd = z + SHIFT;
    let ki = kd.to_bits();
    let kd = kd - SHIFT;
    let r = x + kd * table::NEG_LN2_HI_N + kd * table::NEG_LN2_LO_N;
    let idx = 2 * (ki % 128) as usize;
    let top = (ki << 45).wrapping_add((s as u64) << 52);
    let tail = f64::from_bits(table::EXP[idx]);
    let sbits = table::EXP[idx + 1].wrapping_add(top);
    let r2 = r * r;
    let tmp = tail + r + r2 * (C2 + r * C3) + r2 * r2 * (C4 + r * C5);
    if abstop == 0 {
        return exp_scaled(tmp, sbits, ki);
    }
    let scale = f64::from_bits(sbits);
    scale + scale * tmp
}

/// `scale (1 + tmp)` when the exponent of `scale` over- or underflowed:
/// scaled into range, computed, scaled back; the subnormal result rounded
/// once, not twice.
#[cold]
fn exp_scaled(tmp: f64, sbits: u64, ki: u64) -> f64 {
    if ki & 0x8000_0000 == 0 {
        // k > 0: the exponent overflowed by at most 460.
        let scale = f64::from_bits(sbits.wrapping_sub(1009 << 52));
        return f64::from_bits(0x7f0 << 52) * (scale + scale * tmp);
    }
    // k < 0: the result may be subnormal.
    let scale = f64::from_bits(sbits.wrapping_add(1022 << 52));
    let mut y = scale + scale * tmp;
    if y < 1.0 {
        let lo = scale - y + scale * tmp;
        let hi = 1.0 + y;
        let lo = 1.0 - hi + y + lo;
        y = (hi + lo) - 1.0;
        if y == 0.0 {
            y = 0.0;
        }
    }
    f64::from_bits(0x001 << 52) * y
}

/// msun `e_log.c`, by bit pattern.
const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);
const LG1: f64 = f64::from_bits(0x3fe5_5555_5555_5593);
const LG2: f64 = f64::from_bits(0x3fd9_9999_9997_fa04);
const LG3: f64 = f64::from_bits(0x3fd2_4924_9422_9359);
const LG4: f64 = f64::from_bits(0x3fcc_71c5_1d8e_78af);
const LG5: f64 = f64::from_bits(0x3fc7_4664_96cb_03de);
const LG6: f64 = f64::from_bits(0x3fc3_9a09_d078_c69f);
const LG7: f64 = f64::from_bits(0x3fc2_f112_df3e_5244);

/// `ln x`. `x = 2^k (1 + f)` with `1 + f` in `[sqrt(2)/2, sqrt(2)]`, then
/// `ln(1 + f) = f - f^2/2 + s (f^2/2 + R(s^2))`, `s = f/(2 + f)`.
#[inline]
pub fn ln(x: f64) -> f64 {
    let mut x = x;
    let mut ui = x.to_bits();
    let mut hx = (ui >> 32) as u32;
    let mut k: i32 = 0;
    if hx < 0x0010_0000 || hx >> 31 != 0 {
        if ui << 1 == 0 {
            return f64::NEG_INFINITY;
        }
        if hx >> 31 != 0 {
            return f64::NAN;
        }
        // Subnormal: scale up by 2^54.
        k -= 54;
        x *= f64::from_bits(0x435 << 52);
        ui = x.to_bits();
        hx = (ui >> 32) as u32;
    } else if hx >= 0x7ff0_0000 {
        return x;
    } else if hx == 0x3ff0_0000 && ui << 32 == 0 {
        return 0.0;
    }
    hx += 0x3ff0_0000 - 0x3fe6_a09e;
    k += (hx >> 20) as i32 - 0x3ff;
    hx = (hx & 0x000f_ffff) + 0x3fe6_a09e;
    ui = ((hx as u64) << 32) | (ui & 0xffff_ffff);
    x = f64::from_bits(ui);
    let f = x - 1.0;
    let hfsq = 0.5 * f * f;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    let dk = k as f64;
    s * (hfsq + r) + dk * LN2_LO - hfsq + f + dk * LN2_HI
}

/// Below this `|x|`, `sinh`, `cosh` and `tanh` go through `expm1`: `e^x`
/// and `e^-x` are too close for their difference.
const NEAR_ZERO: f64 = 0.5;

/// `e^a / 2`, finite wherever the result is.
#[inline]
fn half_exp(a: f64) -> f64 {
    if a < 1024.0 {
        exp_pow2(a, -1)
    } else {
        f64::INFINITY
    }
}

/// `sinh x`.
#[inline]
pub fn sinh(x: f64) -> f64 {
    let a = x.abs();
    // Below 1, e^a - e^-a still cancels a bit: the expm1 form holds there.
    let r = if a < 1.0 {
        if a < f64::from_bits(0x3e3 << 52) {
            return x; // |x| < 2^-28
        }
        let t = libm::expm1(a);
        0.5 * (2.0 * t - t * t / (t + 1.0))
    } else if a < 22.0 {
        let e = exp(a);
        0.5 * e - 0.5 / e
    } else {
        half_exp(a)
    };
    r.copysign(x)
}

/// `cosh x`.
#[inline]
pub fn cosh(x: f64) -> f64 {
    let a = x.abs();
    if a.is_nan() {
        return a;
    }
    if a < NEAR_ZERO {
        if a < f64::from_bits(0x3e3 << 52) {
            return 1.0;
        }
        let t = libm::expm1(a);
        return 1.0 + t * t / (2.0 * (1.0 + t));
    }
    if a < 22.0 {
        let e = exp(a);
        return 0.5 * e + 0.5 / e;
    }
    half_exp(a)
}

/// `tanh x`.
#[inline]
pub fn tanh(x: f64) -> f64 {
    let a = x.abs();
    if a.is_nan() {
        return x;
    }
    let r = if a < NEAR_ZERO {
        if a < f64::from_bits(0x3e3 << 52) {
            return x;
        }
        let t = libm::expm1(-2.0 * a);
        -t / (t + 2.0)
    } else if a < 22.0 {
        1.0 - 2.0 / (exp(2.0 * a) + 1.0)
    } else {
        1.0
    };
    r.copysign(x)
}

/// `x^n`: square and multiply over the bits of `|n|`, low bit first, and
/// one reciprocal for a negative `n` (the order compiler-rt's `__powidf2`
/// uses, fixed here rather than left to the toolchain).
#[inline]
pub fn powi(x: f64, n: i32) -> f64 {
    let mut b = n.unsigned_abs();
    let mut a = x;
    let mut r = 1.0;
    loop {
        if b & 1 != 0 {
            r *= a;
        }
        b >>= 1;
        if b == 0 {
            break;
        }
        a *= a;
    }
    if n < 0 {
        1.0 / r
    } else {
        r
    }
}
