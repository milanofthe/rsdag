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
