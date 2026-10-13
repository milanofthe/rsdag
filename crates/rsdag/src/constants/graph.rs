//! The graph's analyses and differentiation ([`crate::graph`],
//! [`crate::autodiff`]).

/// The widest universe a set of the graph's data-flow analyses keeps as a
/// bitset; a wider one is a sorted list. Time and memory only.
pub(crate) const BITS_MAX: usize = 4096;

/// Rows of this many touched unknowns and more are differentiated in
/// reverse mode, one adjoint sweep for the whole row; below it, forward
/// sweeps per unknown, shared by every row that touches it, are cheaper.
/// The two modes build a derivative as different expressions, so moving
/// the threshold can move a derivative's value by rounding.
pub const REVERSE_MIN_TOUCHED: usize = 16;
