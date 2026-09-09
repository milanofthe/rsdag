//! The execution scalar: what a program computes in.
//!
//! A [`Tape`](crate::tape::Tape) is lowered from a graph once; it can then be
//! evaluated in any `Scalar`: `f64` (the reference, and the only type the JIT
//! and the C backend emit), `f32`, or `Complex<f64>` for frequency-domain
//! work. Every type provides the reference implementations of the unary and
//! binary functions and of the comparisons, so a tape means the same thing
//! for every backend of a given `T`.
//!
//! Comparisons and selects need real predicates: for a complex `T` they act
//! on the real part, ordered ops (`Floor`, `Sign`, `Ceil`, `Round`, `Trunc`,
//! `Min`, `Max`) and the special real functions act on the real part and
//! return a real value.

use num_complex::Complex64;

use crate::node::{binary_f64, cmp_bool, unary_f64, BinOp, CmpOp, ReduceOp, UnaryOp};

pub trait Scalar: Copy + Send + Sync + std::fmt::Debug + 'static {
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(x: f64) -> Self;
    /// Not-a-number, for a missing input.
    fn nan() -> Self;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    /// `self * b + c` with one rounding where the type has a fused
    /// multiply-add (`f64`, `f32`); a complex product has no single
    /// rounding, so it is the multiply and the add.
    fn mul_add(self, b: Self, c: Self) -> Self;
    fn neg(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn unary(op: UnaryOp, x: Self) -> Self;
    fn binary(op: BinOp, x: Self, y: Self) -> Self;
    /// `1` or `0` from a comparison of the real parts.
    fn cmp(op: CmpOp, x: Self, y: Self) -> Self;
    /// The select predicate: nonzero real part.
    fn is_true(self) -> bool;
    fn min(self, o: Self) -> Self;
    fn max(self, o: Self) -> Self;
}

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f64(x: f64) -> Self {
        x
    }
    fn nan() -> Self {
        f64::NAN
    }
    fn add(self, o: Self) -> Self {
        self + o
    }
    fn sub(self, o: Self) -> Self {
        self - o
    }
    fn mul(self, o: Self) -> Self {
        self * o
    }
    fn mul_add(self, b: Self, c: Self) -> Self {
        f64::mul_add(self, b, c)
    }
    fn neg(self) -> Self {
        -self
    }
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    fn unary(op: UnaryOp, x: Self) -> Self {
        unary_f64(op, x)
    }
    fn binary(op: BinOp, x: Self, y: Self) -> Self {
        binary_f64(op, x, y)
    }
    fn cmp(op: CmpOp, x: Self, y: Self) -> Self {
        if cmp_bool(op, x, y) {
            1.0
        } else {
            0.0
        }
    }
    fn is_true(self) -> bool {
        self != 0.0
    }
    fn min(self, o: Self) -> Self {
        ReduceOp::Min.combine(self, o)
    }
    fn max(self, o: Self) -> Self {
        ReduceOp::Max.combine(self, o)
    }
}

impl Scalar for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn nan() -> Self {
        f32::NAN
    }
    fn add(self, o: Self) -> Self {
        self + o
    }
    fn sub(self, o: Self) -> Self {
        self - o
    }
    fn mul(self, o: Self) -> Self {
        self * o
    }
    fn mul_add(self, b: Self, c: Self) -> Self {
        f32::mul_add(self, b, c)
    }
    fn neg(self) -> Self {
        -self
    }
    fn powi(self, n: i32) -> Self {
        f32::powi(self, n)
    }
    /// Single precision goes through the double reference and rounds once:
    /// the same guards, one rounding, no second set of algorithms.
    fn unary(op: UnaryOp, x: Self) -> Self {
        unary_f64(op, x as f64) as f32
    }
    fn binary(op: BinOp, x: Self, y: Self) -> Self {
        binary_f64(op, x as f64, y as f64) as f32
    }
    fn cmp(op: CmpOp, x: Self, y: Self) -> Self {
        if cmp_bool(op, x, y) {
            1.0
        } else {
            0.0
        }
    }
    fn is_true(self) -> bool {
        self != 0.0
    }
    fn min(self, o: Self) -> Self {
        ReduceOp::Min.combine(self as f64, o as f64) as f32
    }
    fn max(self, o: Self) -> Self {
        ReduceOp::Max.combine(self as f64, o as f64) as f32
    }
}

impl Scalar for Complex64 {
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }
    fn from_f64(x: f64) -> Self {
        Complex64::new(x, 0.0)
    }
    fn nan() -> Self {
        Complex64::new(f64::NAN, f64::NAN)
    }
    fn add(self, o: Self) -> Self {
        self + o
    }
    fn sub(self, o: Self) -> Self {
        self - o
    }
    fn mul(self, o: Self) -> Self {
        self * o
    }
    fn mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }
    fn neg(self) -> Self {
        -self
    }
    fn powi(self, n: i32) -> Self {
        Complex64::powi(&self, n)
    }
    fn unary(op: UnaryOp, x: Self) -> Self {
        match op {
            UnaryOp::Exp => x.exp(),
            UnaryOp::Ln => x.ln(),
            UnaryOp::Sqrt => x.sqrt(),
            UnaryOp::Sin => x.sin(),
            UnaryOp::Cos => x.cos(),
            UnaryOp::Sinh => x.sinh(),
            UnaryOp::Cosh => x.cosh(),
            UnaryOp::Tanh => x.tanh(),
            UnaryOp::Atan => x.atan(),
            UnaryOp::Tan => x.tan(),
            UnaryOp::Log10 => x.ln() / std::f64::consts::LN_10,
            UnaryOp::Log2 => x.ln() / std::f64::consts::LN_2,
            UnaryOp::Log1p => (x + 1.0).ln(),
            UnaryOp::Expm1 => x.exp() - 1.0,
            UnaryOp::Cbrt => x.powf(1.0 / 3.0),
            UnaryOp::Abs => Complex64::new(x.norm(), 0.0),
            UnaryOp::Asin => x.asin(),
            UnaryOp::Acos => x.acos(),
            UnaryOp::Asinh => x.asinh(),
            UnaryOp::Acosh => x.acosh(),
            UnaryOp::Atanh => x.atanh(),
            UnaryOp::Floor
            | UnaryOp::Sign
            | UnaryOp::Ceil
            | UnaryOp::Round
            | UnaryOp::Trunc
            | UnaryOp::Erf
            | UnaryOp::Erfc
            | UnaryOp::Lgamma
            | UnaryOp::Tgamma
            | UnaryOp::Digamma
            | UnaryOp::Trigamma
            | UnaryOp::RandUniform => Complex64::new(unary_f64(op, x.re), 0.0),
        }
    }
    fn binary(op: BinOp, x: Self, y: Self) -> Self {
        match op {
            BinOp::Powf => x.powc(y),
            BinOp::Mod | BinOp::Atan2 | BinOp::Hypot => {
                Complex64::new(binary_f64(op, x.re, y.re), 0.0)
            }
        }
    }
    fn cmp(op: CmpOp, x: Self, y: Self) -> Self {
        if cmp_bool(op, x.re, y.re) {
            Self::one()
        } else {
            Self::zero()
        }
    }
    fn is_true(self) -> bool {
        self.re != 0.0
    }
    fn min(self, o: Self) -> Self {
        if o.re < self.re {
            o
        } else {
            self
        }
    }
    fn max(self, o: Self) -> Self {
        if o.re > self.re {
            o
        } else {
            self
        }
    }
}

/// The reduction of a slice in `T`, with the same fixed order as the `f64`
/// reference (`reduce_slice`): four accumulators above `REDUCE_SIMD_MIN`.
pub fn reduce_slice_t<T: Scalar>(op: ReduceOp, xs: &[T]) -> T {
    use crate::node::REDUCE_SIMD_MIN;
    match op {
        ReduceOp::Sum if xs.len() >= REDUCE_SIMD_MIN => {
            let mut a = [T::zero(); 4];
            let ch = xs.len() / 4;
            for c in 0..ch {
                for l in 0..4 {
                    a[l] = a[l].add(xs[4 * c + l]);
                }
            }
            let mut acc = (a[0].add(a[1])).add(a[2].add(a[3]));
            for &x in &xs[ch * 4..] {
                acc = acc.add(x);
            }
            acc
        }
        ReduceOp::Product if xs.len() >= REDUCE_SIMD_MIN => {
            let mut a = [T::one(); 4];
            let ch = xs.len() / 4;
            for c in 0..ch {
                for l in 0..4 {
                    a[l] = a[l].mul(xs[4 * c + l]);
                }
            }
            let mut acc = (a[0].mul(a[1])).mul(a[2].mul(a[3]));
            for &x in &xs[ch * 4..] {
                acc = acc.mul(x);
            }
            acc
        }
        ReduceOp::Sum => xs.iter().fold(T::zero(), |acc, &x| acc.add(x)),
        ReduceOp::Product => xs.iter().fold(T::one(), |acc, &x| acc.mul(x)),
        ReduceOp::Min => {
            let mut it = xs.iter();
            match it.next() {
                None => T::from_f64(f64::INFINITY),
                Some(&f) => it.fold(f, |acc, &x| acc.min(x)),
            }
        }
        ReduceOp::Max => {
            let mut it = xs.iter();
            match it.next() {
                None => T::from_f64(f64::NEG_INFINITY),
                Some(&f) => it.fold(f, |acc, &x| acc.max(x)),
            }
        }
    }
}

/// The inner product in `T`, same order as `dot_slice`.
pub fn dot_slice_t<T: Scalar>(a: &[T], b: &[T]) -> T {
    use crate::node::REDUCE_SIMD_MIN;
    if a.len() >= REDUCE_SIMD_MIN {
        let mut acc = [T::zero(); 4];
        let ch = a.len() / 4;
        for c in 0..ch {
            for l in 0..4 {
                acc[l] = acc[l].add(a[4 * c + l].mul(b[4 * c + l]));
            }
        }
        let mut s = (acc[0].add(acc[1])).add(acc[2].add(acc[3]));
        for k in ch * 4..a.len() {
            s = s.add(a[k].mul(b[k]));
        }
        s
    } else {
        let mut s = T::zero();
        for (&x, &y) in a.iter().zip(b.iter()) {
            s = s.add(x.mul(y));
        }
        s
    }
}
