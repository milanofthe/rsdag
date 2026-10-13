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

/// Rows a dense matrix-vector product takes at a time, each with its own
/// [`FOLD_LANES`](crate::constants::FOLD_LANES) accumulators: sixteen
/// independent chains for the core. Every row's fold is the dot's.
pub(crate) const GEMV_ROW_BLOCK: usize = 4;

/// Rows of the left factor a dense matrix-matrix product takes at a time,
/// against [`GEMM_COL_BLOCK`] of the right one: thirty-two independent
/// accumulators, six operand rows in the near cache. Every entry's fold is
/// the dot's.
pub(crate) const GEMM_ROW_BLOCK: usize = 4;
/// See [`GEMM_ROW_BLOCK`]; the kernels are written out for two.
pub(crate) const GEMM_COL_BLOCK: usize = 2;

/// Rows of the trailing matrix the blocked elimination updates at a time,
/// one product against the panel's rows; each entry is the same dot
/// whatever the block.
pub(crate) const LU_TRAILING_ROWS: usize = 4;

/// Systems of at most this many unknowns and right-hand sides solve side
/// by side in a batch, four to a vector
/// ([`solve_batch_into`](crate::semantics::solve_batch_into) in `f64`);
/// the tape batches solves of these shapes only, larger ones gain nothing
/// from it.
pub const SOLVE_BATCH_MAX_N: usize = 16;
/// See [`SOLVE_BATCH_MAX_N`].
pub const SOLVE_BATCH_MAX_K: usize = 4;
