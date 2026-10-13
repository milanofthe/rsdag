//! Parallel execution ([`crate::parallel`]): when a stage goes to the
//! pool, and how its workers wait. Time only: a stage computes the same
//! bits on any number of threads.

/// [`Parallel::min_ops`](crate::parallel::Parallel::min_ops) by default,
/// in ops (instances times their body's ops): about ten microseconds of
/// work. The Python bindings gate a native batch at another value of the
/// same unit (`PY_JIT_BATCH_MIN_OPS` in `rsdag-py`, `1 << 16`); both stay
/// as they are until a gate measured in time replaces them.
pub const MIN_OPS: usize = 20_000;

/// Pieces of work per worker a stage (and a native batch) is cut into
/// ([`piece_len`](crate::parallel::piece_len)): about four balance
/// instances of unequal cost, and a piece still carries enough work to
/// pay for its hand-off.
pub const PIECES_PER_WORKER: usize = 4;

/// How long [`Workers`](crate::parallel::Workers) keep waiting for work
/// before they sleep: longer than the serial stretch between a solver's
/// evaluations.
pub const SPIN: std::time::Duration = std::time::Duration::from_micros(300);

/// A thread waiting for work yields its core once every this many empty
/// looks, and spins between them: rarely enough to answer fast, often
/// enough to let another thread on the core run.
pub(crate) const SPIN_YIELD_EVERY: u32 = 64;
