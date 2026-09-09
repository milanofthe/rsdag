//! Chunked Cranelift JIT backend for the evaluation [`Tape`](rsdag::Tape).
//!
//! The tape is the evaluation IR; [`Tape::eval`](rsdag::Tape::eval) is the interpreting
//! backend and this crate is the native one. The previous JIT lowered a tape to
//! *one* Cranelift function and died on BSIM4-amplifier tapes (~500k ops):
//! register allocation on a single huge straight-line function is superlinear,
//! Cranelift ground for tens of seconds and aborted. The fix is structural and
//! cheap because tape slots are **memory-based SSA**: values live in a
//! caller-owned work array, so the op stream can be cut into chunks of
//! [`CHUNK_OPS`] instructions, each lowered to its own small function
//! `fn(work, inputs, ctx)`. A value crossing a chunk boundary simply stays in
//! the work array -- the interpreter's storage model *is* the spill model --
//! and the chunks compile independently (in parallel, on the rayon pool).
//!
//! Bit-exactness is a hard invariant: the same IEEE operation sequence as the
//! interpreter (no fast-math, no FMA contraction -- `MulAdd` lowers to `fmul`
//! + `fadd`), and every transcendental routes through the *same*
//! [`unary_f64`](rsdag::node::unary_f64) host trampolines, so the domain guards (limexp, ln-floor,
//! sqrt-clamp) hold identically. `tests/parity.rs` fuzzes arena == tape ==
//! chunked-JIT to the bit.

/// Instructions per compiled function for the solver's outer tapes (residual
/// and Jacobian over the whole circuit). Bounds Cranelift's superlinear
/// register allocation (the failure mode of the unchunked predecessor).
/// Measured on the pressure-scheduled 364k-op step tape of an 18-transistor
/// PSP103 ring: 1024 compiles ~5x faster than 16384 at equal or better native
/// eval speed -- the scheduler keeps the live set small, so a chunk boundary
/// costs almost nothing there; 256 loses eval speed to boundary traffic and
/// per-chunk fixed cost.
pub const CHUNK_OPS: usize = 1024;

/// Instructions per compiled function for device-bundle *bodies*: small (5k
/// to 30k ops), evaluated thousands of times per Newton step through the lane
/// backends, and compiled off the critical path. There the per-chunk fixed
/// cost (call, callee-saved vector registers, boundary stripes) is what
/// counts: on the same ring, 1024-op body chunks cost 9% of the whole
/// transient against 16384.
pub const CHUNK_OPS_BODY: usize = 16384;

/// Reasons a tape cannot be compiled (callers fall back to the interpreter).
#[derive(Debug)]
pub enum JitError {
    /// Cranelift module/codegen error.
    Codegen(String),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::Codegen(s) => write!(f, "cranelift codegen error: {s}"),
        }
    }
}
impl std::error::Error for JitError {}

mod host;
mod lanes;
mod scalar;

pub use lanes::{suggest_lanes, LaneTape, LANES, LANE_WIDTHS};
pub use scalar::ChunkedTape;
