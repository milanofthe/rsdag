//! The tape compiler's and the dense kernels' thresholds ([`crate::tape`],
//! [`crate::semantics`]): what fuses, batches or runs as a region. Time
//! only: a fused or batched form computes its parts' bits.

/// Row dots against one vector fuse into a `Gemv` from this many rows on.
pub(crate) const GEMV_MIN_ROWS: usize = 2;

/// Ops a piece of a region takes on, about (whole connected parts of the
/// run): as many as a native chunk.
pub(crate) const PIECE_OPS: usize = 1024;

/// A run of ops without a call becomes a region (see
/// [`Region`](crate::Region)) from this many ops on: two pieces.
pub(crate) const REGION_MIN_OPS: usize = 2 * PIECE_OPS;

/// Ops a call of a bundle without a body
/// ([`ExternBundle::body`](crate::ExternBundle::body)) counts as, per
/// instance.
pub(crate) const OPAQUE_OPS: u64 = 1000;

/// A dense kernel's operand stretch of this many single values or more,
/// none placed yet, gets consecutive slots, so the kernel reads it in
/// place; a shorter one is read where its values are.
pub(crate) const PLACE_MIN_RUN: usize = 4;

/// The order maintenance of the scheduler (`tape::topo`, after Bender et
/// al.) relabels a range of `2^i` labels once it holds fewer than this to
/// the `i`th elements (their `2 / T` with `T = 1.5`): the allowance shrinks
/// with the range's size. The order is the same whatever the labels.
pub(crate) const ORDER_RELABEL_GROWTH: f64 = 4.0 / 3.0;

/// Outputs a block of [`write_outputs`](crate::tape::write_outputs)
/// copies, from which on a large program copies its outputs in blocks on
/// the installed pool.
pub(crate) const OUTPUT_BLOCK: usize = 8192;

/// Systems of at most this many unknowns and right-hand sides solve side
/// by side in a batch, four to a vector
/// ([`solve_batch_into`](crate::semantics::solve_batch_into) in `f64`);
/// the tape batches solves of these shapes only, larger ones gain nothing
/// from it.
pub const SOLVE_BATCH_MAX_N: usize = 16;
/// See [`SOLVE_BATCH_MAX_N`].
pub const SOLVE_BATCH_MAX_K: usize = 4;
