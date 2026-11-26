//! How expressions print ([`crate::display`], [`crate::dot`]). Text only.

/// An expression whose estimated infix form has at most this many bytes
/// prints as one infix string. A larger one prints as a listing with one
/// line per node, because a DAG printed as a tree grows exponentially with
/// its sharing.
pub(crate) const INFIX_MAX: usize = 4096;

/// The bytes that an operator adds to an infix form, approximately, in the
/// estimate that is compared against [`INFIX_MAX`].
pub(crate) const INFIX_OP_BYTES: usize = 8;

/// An integer number label of a magnitude below this prints as an integer.
/// An `f64` holds every integer up to `2^53`, which is about `9e15`.
pub(crate) const LABEL_INT_MAX: f64 = 1e15;
const _: () = assert!(LABEL_INT_MAX < (1u64 << f64::MANTISSA_DIGITS) as f64);

/// A number label of a magnitude from this up to, but not including,
/// [`LABEL_FIXED_MAX`] prints in fixed notation with four decimal places and
/// without trailing zeros. Outside this range it prints in exponent form.
pub(crate) const LABEL_FIXED_MIN: f64 = 1e-3;
/// See [`LABEL_FIXED_MIN`].
pub(crate) const LABEL_FIXED_MAX: f64 = 1e5;

/// An exact constant whose rendering is a fraction of at most this many
/// characters prints as that fraction. Any other exact constant prints as
/// a number.
pub(crate) const LABEL_EXACT_MAX: usize = 8;
