//! The tape compiler's and the dense kernels' thresholds ([`crate::tape`],
//! [`crate::semantics`]): what fuses, batches or runs as a region. Time
//! only: a fused or batched form computes its parts' bits.

/// Row dots against one vector fuse into a `Gemv` from this many rows on.
pub(crate) const GEMV_MIN_ROWS: usize = 2;

/// Ops per function a native backend emits (`rsdag-jit` re-exports it as
/// its `CHUNK_OPS`). A chunk boundary costs nothing but the call, so the
/// size is a matter of parallel build granularity.
pub const CHUNK_OPS: usize = 1024;

/// The ops that a piece of a region takes on, approximately. A piece is a
/// set of whole connected parts of the run, and its size is that of a
/// native chunk.
pub(crate) const PIECE_OPS: usize = CHUNK_OPS;

/// A run of ops without a call becomes a region (see
/// [`Region`](crate::Region)) from this many ops on, which is two pieces.
pub(crate) const REGION_MIN_OPS: usize = 2 * PIECE_OPS;

/// The ops that a call of a bundle without a body
/// ([`ExternBundle::body`](crate::ExternBundle::body)) counts as, per
/// instance.
pub(crate) const OPAQUE_OPS: u64 = 1000;

/// A stretch of a dense kernel's operands of this many single values or
/// more, none of them placed yet, gets consecutive slots, so that the kernel
/// reads it in place. A shorter stretch is read where its values are.
pub(crate) const PLACE_MIN_RUN: usize = 4;

/// The parameter `T` of the order maintenance of the scheduler
/// (`tape::topo`), which follows Bender et al.
const ORDER_MAINTENANCE_T: f64 = 1.5;
/// The order maintenance relabels a range of `2^i` labels once the range
/// holds fewer elements than this constant to the power `i`. The constant
/// is `2 / T` with [`ORDER_MAINTENANCE_T`], so the allowed share of
/// occupied labels, `T^-i`, shrinks with the size of the range. The order
/// is the same whatever the labels are.
pub(crate) const ORDER_RELABEL_GROWTH: f64 = 2.0 / ORDER_MAINTENANCE_T;

/// The outputs that one block of
/// [`write_outputs`](crate::tape::write_outputs) copies. A program with more
/// outputs than this copies them in blocks on the installed pool.
pub(crate) const OUTPUT_BLOCK: usize = 8192;

/// The rows that a dense matrix-vector product takes at a time, each with
/// its own [`FOLD_LANES`](crate::constants::FOLD_LANES) accumulators. This
/// gives the core sixteen independent chains. Every row folds as its dot
/// product does.
pub(crate) const GEMV_ROW_BLOCK: usize = 4;

/// The rows of the left factor that a dense matrix-matrix product takes at
/// a time, against [`GEMM_COL_BLOCK`] columns of the right factor. This
/// gives thirty-two independent accumulators and keeps six operand rows in
/// the near cache. Every entry folds as its dot product does.
pub(crate) const GEMM_ROW_BLOCK: usize = 4;
/// See [`GEMM_ROW_BLOCK`]. The kernels are written out for two.
pub(crate) const GEMM_COL_BLOCK: usize = 2;

/// The rows of the trailing matrix that the blocked elimination updates at
/// a time, as one product against the rows of the panel. Each entry is the
/// same dot product whatever the block.
pub(crate) const LU_TRAILING_ROWS: usize = 4;

/// Systems of at most this many unknowns and right-hand sides solve side
/// by side in a batch, four to a vector
/// ([`solve_batch_into`](crate::semantics::solve_batch_into) in `f64`);
/// the tape batches solves of these shapes only, larger ones gain nothing
/// from it.
pub const SOLVE_BATCH_MAX_N: usize = 16;
/// See [`SOLVE_BATCH_MAX_N`].
pub const SOLVE_BATCH_MAX_K: usize = 4;

/// The instances that a batched call of an interpreted body runs side by
/// side in the lanes of one pass over its ops (`Tape::main_batch`). The
/// call runs blocks of sixteen while more than eight instances are left,
/// and the spare lanes of the last such block stay idle. Fewer instances
/// run in a block of eight, four or two, and a single one runs alone. A
/// pass costs the dispatch of each op once for the whole block, and the
/// narrower blocks keep the slots of the lanes in the near caches.
/// Measured on a Ryzen 9 9900X as the time per instance in the main phases
/// relative to one instance at a time, a PSP103 body of nine instances
/// takes 0.15 in one block of sixteen, a BSIM4 body of four instances takes
/// 0.17 in a block of four, and of two instances 0.31 in a block of two. A
/// block of sixty-four is slower than a block of sixteen.
pub(crate) const LANES_WIDE: usize = 16;
/// See [`LANES_WIDE`].
pub(crate) const LANES_MID: usize = 8;
/// See [`LANES_WIDE`].
pub(crate) const LANES_NARROW: usize = 4;
/// See [`LANES_WIDE`].
pub(crate) const LANES_PAIR: usize = 2;

/// Calls a macro with the sizes that the fixed eliminations are
/// instantiated for, `1` to [`SOLVE_BATCH_MAX_N`].
macro_rules! for_each_fixed_solve_size {
    ($m:ident) => {
        $m!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
    };
}
pub(crate) use for_each_fixed_solve_size;

macro_rules! count_sizes {
    ($($n:literal)*) => {
        [$($n),*].len()
    };
}
const _: () = assert!(for_each_fixed_solve_size!(count_sizes) == SOLVE_BATCH_MAX_N);
