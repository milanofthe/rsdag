//! Every number the native backend's behavior depends on, in one place.
//! What the code computes is rsdag's contract (`rsdag::constants::contract`,
//! the steps of `rsdag::semantics::exp_ln`); these decide how it is
//! emitted and run, and change time alone. The instruction encodings,
//! register numbers and ABI frames stay with their encoders (`x86_64.rs`,
//! `aarch64.rs`).

/// Lanes the lane code runs at most: the four `f64` of an AVX register.
pub(crate) const MAX_LANES: usize = 4;

/// What a host call costs, in ops, against an op of the code (in lane
/// code's costs, `lane_costs`); measured on SANE's device bodies.
pub(crate) const LANE_CALL: usize = 6;

/// What moving a value into or out of the lanes costs, in ops (as
/// [`LANE_CALL`]).
pub(crate) const LANE_MOVE: usize = 2;

/// Calls with this many arguments over all their instances and more have
/// the host gather them from a table; narrower ones gather in the code,
/// whose size would otherwise grow with every argument.
pub(crate) const GATHER_TABLE: usize = 64;

/// Integer powers up to this exponent are emitted as their products,
/// larger ones call the host. The same products either way
/// (`rsdag::semantics::powi_t`); independent of rsdag's
/// `POWF_AS_POWI_MAX`, which decides what is an integer power at all.
pub(crate) const POWI_INLINE: u32 = 64;

/// A min or max over at most this many terms is a chain of instructions
/// where the ISA has one (`Isa::MINMAX`); longer ones go through the host
/// routine, whose call costs about as much as this many terms.
pub(crate) const INLINE_MINMAX_MAX: usize = 16;

/// A product kernel of fewer rows than this runs as its entries' dots,
/// inline in the code: there the host call costs more than the arithmetic
/// (measured: a six by six product inline in two thirds of the host's
/// time, an eight by eight one on the host in two thirds of the inline).
pub(crate) const INLINE_ROWS: u32 = 8;

/// Instances a parallel task of a batch takes at most, however few the
/// threads: the tasks stay small enough to balance.
pub(crate) const TASK_MAX_INSTANCES: usize = 4096;

/// Bytes each emitted chunk is aligned to in the code mapping.
pub(crate) const CODE_ALIGN: usize = 16;

/// Register numbers the floating-point registers take, `0..32` (AArch64's
/// `d0` to `d31`; x86-64's `xmm0` to `xmm15` among them).
pub(crate) const REG_NUMBERS: usize = 32;

/// AArch64 code asks for the code ahead every this many bytes emitted
/// (`prfm plil2keep`): a large program streams through the front end once,
/// so that is a third off a 7M-op program and nothing on a small one.
#[cfg(target_arch = "aarch64")]
pub(crate) const PREFETCH_EVERY: usize = 128;

/// How far ahead [`PREFETCH_EVERY`]'s prefetch reaches, in bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) const PREFETCH_AHEAD: usize = 2048;
