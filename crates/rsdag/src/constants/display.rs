//! How expressions print ([`crate::display`], [`crate::dot`]). Text only.

/// Expressions whose infix form stays under this many bytes print as one
/// infix string; a larger one prints as a listing, one line per node (a DAG
/// printed as a tree grows exponentially with its sharing).
pub(crate) const INFIX_MAX: usize = 4096;

/// Bytes an operator adds to an infix form, about, in the estimate against
/// [`INFIX_MAX`].
pub(crate) const INFIX_OP_BYTES: usize = 8;

/// A number label: an integer below this prints as one (an `f64` holds
/// every integer up to `2^53`, about `9e15`).
pub(crate) const LABEL_INT_MAX: f64 = 1e15;

/// A number label of a magnitude from this up to [`LABEL_FIXED_MAX`]
/// prints in fixed notation, four significant digits; beyond, in exponent
/// form.
pub(crate) const LABEL_FIXED_MIN: f64 = 1e-3;
/// See [`LABEL_FIXED_MIN`].
pub(crate) const LABEL_FIXED_MAX: f64 = 1e5;

/// An exact constant prints as its fraction up to this many characters,
/// else as a number.
pub(crate) const LABEL_EXACT_MAX: usize = 8;
