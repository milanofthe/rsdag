//! The tape compiler's and the dense kernels' thresholds ([`crate::tape`],
//! [`crate::semantics`]): what fuses, batches or runs as a region. Time
//! only: a fused or batched form computes its parts' bits.

pub(crate) const GEMV_MIN_ROWS: usize = 2;

/// Ops per function a native backend emits (`rsdag-jit` re-exports it as
/// its `CHUNK_OPS`). A chunk boundary costs nothing but the call, so the
/// size is a matter of parallel build granularity.
pub const CHUNK_OPS: usize = 1024;

pub(crate) const PIECE_OPS: usize = CHUNK_OPS;

pub(crate) const REGION_MIN_OPS: usize = 2 * PIECE_OPS;

pub(crate) const OPAQUE_OPS: u64 = 1000;

pub(crate) const PLACE_MIN_RUN: usize = 4;

const ORDER_MAINTENANCE_T: f64 = 1.5;
pub(crate) const ORDER_RELABEL_GROWTH: f64 = 2.0 / ORDER_MAINTENANCE_T;

pub(crate) const OUTPUT_BLOCK: usize = 8192;

pub(crate) const GEMV_ROW_BLOCK: usize = 4;

pub(crate) const GEMM_ROW_BLOCK: usize = 4;
pub(crate) const GEMM_COL_BLOCK: usize = 2;

pub(crate) const LU_TRAILING_ROWS: usize = 4;

/// Systems of at most this many unknowns and right-hand sides solve side
/// by side in a batch, four to a vector
/// ([`solve_batch_into`](crate::semantics::solve_batch_into) in `f64`);
/// the tape batches solves of these shapes only, larger ones gain nothing
/// from it.
pub const SOLVE_BATCH_MAX_N: usize = 16;
/// See [`SOLVE_BATCH_MAX_N`].
pub const SOLVE_BATCH_MAX_K: usize = 4;

pub(crate) const LANES_WIDE: usize = 16;
pub(crate) const LANES_MID: usize = 8;
pub(crate) const LANES_NARROW: usize = 4;
pub(crate) const LANES_PAIR: usize = 2;

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
