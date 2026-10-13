//! The contract: values every evaluator (the arena sweep, the tape
//! interpreter, the native code) computes with, so that a program's
//! results are the same bit for bit whichever runs it. A change to one
//! changes results; `tests/contract_snapshot.rs` asserts each one's exact
//! value, so a change takes an edit there too, in the same commit.

/// Argument above which `exp` is linearly extrapolated (a "limited exponential",
/// SPICE `limexp`). `exp` overflows f64 near 709; compact models routinely drive
/// the argument far past that on intermediate Newton iterates (out-of-range
/// internal-node guesses), producing `inf`/`NaN` that poison the whole solve.
/// Linearizing above this threshold keeps the value and its (constant) derivative
/// finite with ample headroom against overflow, while leaving every physical
/// evaluation -- exp arguments are O(10) at any real operating point -- bit-for-bit
/// unchanged. 80 is the classic SPICE `limexp` knee. A consumer may limit
/// earlier on its own (SANE linearizes junction exponentials at 40, its
/// `EXP_VCRIT`); this one holds for every `exp`. See
/// [`unary_f64`](crate::semantics::unary_f64).
pub const EXP_LIMIT: f64 = 80.0;

/// Argument at or below which `ln` is clamped (so `ln(x<=0)` returns a finite,
/// modest `ln(LN_FLOOR)` instead of `-inf`/`NaN` on an out-of-range Newton
/// iterate). A *clamp*, not a linear extrapolation: a steep extrapolation slope
/// (`1/LN_FLOOR`) would blow intermediate values up to `~1e30` and wreck
/// conditioning. In-range arguments are untouched. See
/// [`unary_f64`](crate::semantics::unary_f64).
pub const LN_FLOOR: f64 = 1e-30;

/// Arguments below this give `exp` exactly zero. The true value there is
/// less than half the smallest subnormal, so it rounds to zero anyway; the
/// cut makes it so for every such argument, where the scaling by `2^k` in
/// two halves of [`exp_core`](crate::semantics::exp_core) would leave the
/// exponent range.
pub const EXP_UNDERFLOW: f64 = -745.2;

/// The accumulators of the reference fold of sums, products and dot
/// products: term `i` goes to accumulator `i mod 4`, the accumulators merge
/// as `(a0 + a1) + (a2 + a3)`, then the tail adds in order. Every evaluator
/// folds so (a vector kernel holds one accumulator per lane, native code
/// one per register), which is what makes their results agree to the bit.
/// The merge is written out for four.
pub const FOLD_LANES: usize = 4;

/// A `Powf` whose exponent is a constant integer of at most this magnitude
/// is an integer power ([`Graph::pow_i`](crate::Graph::pow_i)): it
/// evaluates by multiplication ([`powi_t`](crate::semantics::powi_t)), not
/// by `pow`. The native backend's own bound on the powers it multiplies
/// out inline (`POWI_INLINE`) is a matter of code size, not this.
pub const POWF_AS_POWI_MAX: f64 = 64.0;

// The bits of an `f64` that the exponential and the logarithm take apart
// (fdlibm's reduction, `crate::semantics::exp_core` and `ln_core`), the
// same in every backend that computes them in its own instructions.

/// The exponent bias of an `f64`: a biased exponent of `1023` is `2^0`.
pub const F64_EXP_BIAS: i64 = 1023;
/// The fraction bits of an `f64`, below its exponent.
pub const F64_FRAC_BITS: u32 = 52;
/// The shift from an `f64`'s bits to its high word: the sign, the exponent
/// and the top [`F64_HIGH_FRAC_BITS`] of the fraction, the word `ln`
/// reduces its argument on.
pub const F64_HIGH_WORD: u32 = 32;
/// The fraction bits in the high word.
pub const F64_HIGH_FRAC_BITS: u32 = F64_FRAC_BITS - F64_HIGH_WORD;
/// The low word of an `f64`'s bits.
pub const F64_LOW_WORD_MASK: u64 = 0xffff_ffff;
/// The fraction bits of the high word.
pub const LN_HIGH_FRAC_MASK: i64 = 0x000f_ffff;
/// fdlibm's constant: added to the high word's fraction bits, it carries
/// into [`LN_HIGH_UNIT`] once they reach `0x6_a09c`, a mantissa of about
/// `sqrt(2)`; such a mantissa moves to the next exponent, halved, so it
/// lands in `[sqrt(2)/2, sqrt(2))`.
pub const LN_SQRT2_CARRY: i64 = 0x9_5f64;
/// The unit bit of the high word, one above its fraction bits: the carry
/// of [`LN_SQRT2_CARRY`].
pub const LN_HIGH_UNIT: i64 = 0x10_0000;
/// The high word of `1.0`; with the carry flipped in, the exponent of `1`
/// or of `1/2`, which the reduced mantissa takes.
pub const LN_HIGH_ONE: i64 = 0x3ff0_0000;

/// The column-panel widths of the blocked elimination in
/// [`solve_t`](crate::semantics::solve_t): the panel's own updates run one
/// element at a time, the trailing update as a product, and the balance
/// moves with the size (measured: 16 wins below about five hundred
/// unknowns, 32 above). A width is a constant of its instantiation so the
/// panel loops unroll. The panel decides the order of a solve's
/// operations, so it is part of the contract.
pub const LU_PANEL_SMALL: usize = 16;
/// See [`LU_PANEL_SMALL`].
pub const LU_PANEL_LARGE: usize = 32;
/// Systems of fewer unknowns than this use [`LU_PANEL_SMALL`].
pub const LU_PANEL_SWITCH: usize = 512;
/// Systems of at most this many unknowns are eliminated right-looking
/// without panels: each pivot updates the whole trailing matrix and the
/// right-hand sides row by row (one product and one difference per entry),
/// which at these sizes beats the panel machinery.
pub const LU_UNBLOCKED_MAX: usize = 64;

/// Largest exponent [`Field::powi`](crate::Field::powi) folds exactly for a
/// rational base other than `1` or `-1`; past it (the digits grow with the
/// exponent) a power of a constant stays a node and evaluates in floating
/// point.
pub const EXACT_POWI_MAX: u64 = 1024;

/// The current [`Module::version`](crate::Module::version). Bump it when the
/// meaning of an existing node changes; adding a variant at the end of an
/// enum does not need it, because an older reader fails on the unknown
/// discriminant anyway.
pub const MODULE_VERSION: u32 = 2;
