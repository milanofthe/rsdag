//! How expressions print ([`crate::display`], [`crate::dot`]). Text only.

pub(crate) const INFIX_MAX: usize = 4096;

pub(crate) const INFIX_OP_BYTES: usize = 8;

pub(crate) const LABEL_INT_MAX: f64 = 1e15;
const _: () = assert!(LABEL_INT_MAX < (1u64 << f64::MANTISSA_DIGITS) as f64);

pub(crate) const LABEL_FIXED_MIN: f64 = 1e-3;
pub(crate) const LABEL_FIXED_MAX: f64 = 1e5;

pub(crate) const LABEL_EXACT_MAX: usize = 8;
