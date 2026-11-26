//! Every number the native backend's behavior depends on, in one place.
//! What the code computes is rsdag's contract (`rsdag::constants::contract`,
//! the steps of `rsdag::semantics::exp_ln`); these decide how it is
//! emitted and run, and change time alone. The instruction encodings,
//! register numbers and ABI frames stay with their encoders (`x86_64.rs`,
//! `aarch64.rs`).

const AVX_REGISTER_BYTES: usize = 32;

pub(crate) const MAX_LANES: usize = AVX_REGISTER_BYTES / std::mem::size_of::<f64>();

pub(crate) const LANE_HOST_CALL_COST_OPS: usize = 6;

pub(crate) const LANE_MOVE_COST_OPS: usize = 2;

pub(crate) const HOST_GATHER_MIN_ARGS: usize = 64;

pub(crate) const POWI_INLINE_MAX_EXPONENT: u32 = 64;

pub(crate) const INLINE_MINMAX_MAX_TERMS: usize = 16;

pub(crate) const HOST_PRODUCT_MIN_ROWS: u32 = 8;

pub(crate) const CODE_ALIGN_BYTES: usize = 16;

pub(crate) const FLOAT_REG_NUMBERS: usize = 32;

pub(crate) const KERNEL_OPERANDS_A_XB_ACCUMULATOR: usize = 3;

pub(crate) const LANE_HOST_CALL_TWO_ARGUMENTS_AND_RESULT: usize = 3;

pub(crate) const FORCED_LANES_SCALAR_COST_WITHOUT_OVERFLOW: usize = usize::MAX / 16;

pub(crate) const OPTIONS_KEY_CHUNK_OPS_SHIFT: u32 = 32;

pub(crate) const OPTIONS_KEY_LANES_SHIFT: u32 = 62;

pub(crate) const OPTIONS_KEY_FREE_BIT_FOR_BACKGROUND: u32 = 61;

#[cfg(target_arch = "aarch64")]
pub(crate) const PREFETCH_EVERY_BYTES: usize = 128;

#[cfg(target_arch = "aarch64")]
pub(crate) const PREFETCH_AHEAD_BYTES: usize = 2048;
