//! Every number the native backend's behavior depends on, in one place.
//! What the code computes is rsdag's contract (`rsdag::constants::contract`,
//! the steps of `rsdag::semantics::exp_ln`); these decide how it is
//! emitted and run, and change time alone. The instruction encodings,
//! register numbers and ABI frames stay with their encoders (`x86_64.rs`,
//! `aarch64.rs`).

/// The bytes of an AVX register.
const AVX_REGISTER_BYTES: usize = 32;

/// The most lanes that lane code runs, the four `f64` of an AVX register.
pub(crate) const MAX_LANES: usize = AVX_REGISTER_BYTES / std::mem::size_of::<f64>();

/// The cost of a host call in lane code, in ops of the code, as
/// `lane_costs` counts it. The value was measured on SANE's device bodies.
pub(crate) const LANE_HOST_CALL_COST_OPS: usize = 6;

/// The cost of moving one value into or out of the lanes, in ops, in the
/// same unit and from the same measurement as [`LANE_HOST_CALL_COST_OPS`].
pub(crate) const LANE_MOVE_COST_OPS: usize = 2;

/// A call with this many arguments or more over all its instances has the
/// host gather them from a table. A call with fewer arguments gathers them
/// in the code, because the size of that code grows with every argument.
pub(crate) const HOST_GATHER_MIN_ARGS: usize = 64;

/// Integer powers up to this exponent in magnitude are emitted as their
/// products, and larger ones call the host. Both compute the same products
/// (`rsdag::semantics::powi_f64`). The limit is independent of rsdag's
/// `POWF_AS_POWI_MAX`, which decides what is an integer power at all.
pub(crate) const POWI_INLINE_MAX_EXPONENT: u32 = 64;

/// A min or max over at most this many terms is a chain of instructions
/// where the ISA has them (`Isa::MINMAX`). A longer one calls the host
/// routine, whose call costs about as much as this many terms.
pub(crate) const INLINE_MINMAX_MAX_TERMS: usize = 16;

/// A product kernel of fewer rows than this runs inline in the code as the
/// dots of its entries, and a kernel of this many rows or more calls the
/// host. Below this size the host call costs more than the arithmetic. In
/// the measurement, a six by six product ran inline in two thirds of the
/// time of the host call, and an eight by eight product ran on the host in
/// two thirds of the time of the inline code.
pub(crate) const HOST_PRODUCT_MIN_ROWS: u32 = 8;

/// The bytes to which each emitted chunk is aligned in the code mapping.
pub(crate) const CODE_ALIGN_BYTES: usize = 16;

/// The register numbers of the floating-point registers, `0..32`. These
/// are AArch64's `d0` to `d31`, and x86-64's `xmm0` to `xmm15` are among
/// them.
pub(crate) const FLOAT_REG_NUMBERS: usize = 32;

/// The operands whose places a dense kernel's descriptor (`host::KernelDesc`)
/// holds: `a`, then `x` or `b`, then the accumulator.
pub(crate) const KERNEL_OPERANDS_A_XB_ACCUMULATOR: usize = 3;

/// The values per lane that the gather area of lane code holds for a host
/// call: two arguments and the result.
pub(crate) const LANE_HOST_CALL_TWO_ARGUMENTS_AND_RESULT: usize = 3;

/// The cost of the scalar code per instance when the lanes are forced, so
/// that the scalar code is never the cheaper way. The division leaves
/// headroom for the multiples of the lane count.
pub(crate) const FORCED_LANES_SCALAR_COST_WITHOUT_OVERFLOW: usize = usize::MAX / 16;

/// The bit at which the chunk size starts in the options key of a bundle's
/// native form (`options_key`). The key holds the chunk size, the batch
/// mode in the low bits, and the lane mode.
pub(crate) const OPTIONS_KEY_CHUNK_OPS_SHIFT: u32 = 32;

/// The bit at which the lane mode starts in the options key, see
/// [`OPTIONS_KEY_CHUNK_OPS_SHIFT`].
pub(crate) const OPTIONS_KEY_LANES_SHIFT: u32 = 62;

/// A bit of the options key that `options_key` leaves free. It holds
/// whether the work goes to the background, so that the form of a bundle
/// that the program runs is keyed by its code options and by where its work
/// goes.
pub(crate) const OPTIONS_KEY_FREE_BIT_FOR_BACKGROUND: u32 = 61;

/// AArch64 code prefetches the code ahead (`prfm plil2keep`) every this
/// many bytes emitted. A large program streams through the front end once,
/// so the prefetch takes a third off a 7M-op program and nothing off a
/// small one.
#[cfg(target_arch = "aarch64")]
pub(crate) const PREFETCH_EVERY_BYTES: usize = 128;

/// How far ahead the prefetch of [`PREFETCH_EVERY_BYTES`] reaches, in bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) const PREFETCH_AHEAD_BYTES: usize = 2048;
