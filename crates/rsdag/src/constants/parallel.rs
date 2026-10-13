//! Parallel execution ([`crate::parallel`]): when a stage goes to the
//! pool, and how its workers wait. Time only: a stage computes the same
//! bits on any number of threads.

/// [`Parallel::min_ops`](crate::parallel::Parallel::min_ops) by default,
/// in ops (instances times their body's ops): about ten microseconds of
/// work. The Python bindings gate a native batch at another value of the
/// same unit (`PY_JIT_BATCH_MIN_OPS` in `rsdag-py`, `1 << 16`); both stay
/// as they are until a gate measured in time replaces them.
pub const MIN_OPS: usize = 20_000;

/// How long [`Workers`](crate::parallel::Workers) keep waiting for work
/// before they sleep: longer than the serial stretch between a solver's
/// evaluations.
pub const SPIN: std::time::Duration = std::time::Duration::from_micros(300);
