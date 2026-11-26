//! The graph's analyses and differentiation ([`crate::graph`],
//! [`crate::autodiff`]).

/// The widest universe for which a set of the graph's data-flow analyses
/// is a bitset. A set over a wider universe is a sorted list. The value
/// affects time and memory only.
pub(crate) const BITS_MAX: usize = 4096;

/// Rows of this many touched unknowns and more are differentiated in
/// reverse mode, one adjoint sweep for the whole row; below it, forward
/// sweeps per unknown, shared by every row that touches it, are cheaper.
/// The two modes build a derivative as different expressions, so moving
/// the threshold can move a derivative's value by rounding.
pub const REVERSE_MIN_TOUCHED: usize = 16;

/// The factor 3 in the derivative of the cube root,
/// `d cbrt(a) = 1 / (3 cbrt(a)^2)`.
pub(crate) const CBRT_DEGREE: i64 = 3;

/// The first capacity of the stack of forward differentiation
/// (`autodiff::forward`). The stack grows as needed.
pub(crate) const FORWARD_STACK_CAPACITY: usize = 64;

/// The size of a [`Node`](crate::node::Node) in bytes, which `node.rs`
/// asserts.
pub(crate) const NODE_BYTES: usize = 16;
