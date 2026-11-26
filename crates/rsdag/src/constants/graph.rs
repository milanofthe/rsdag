//! The graph's analyses and differentiation ([`crate::graph`],
//! [`crate::autodiff`]).

pub(crate) const BITS_MAX: usize = 4096;

/// Rows of this many touched unknowns and more are differentiated in
/// reverse mode, one adjoint sweep for the whole row; below it, forward
/// sweeps per unknown, shared by every row that touches it, are cheaper.
/// The two modes build a derivative as different expressions, so moving
/// the threshold can move a derivative's value by rounding.
pub const REVERSE_MIN_TOUCHED: usize = 16;

pub(crate) const CBRT_DEGREE: i64 = 3;

pub(crate) const FORWARD_STACK_CAPACITY: usize = 64;

pub(crate) const NODE_BYTES: usize = 16;
