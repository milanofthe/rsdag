//! Parallel execution ([`crate::parallel`], on [`rspool`]): what an op of
//! rsdag's code costs. Time only: a stage computes the same bits on any
//! number of threads.

/// The ns an op costs ([`Rate`](crate::parallel::Rate)): interpreted code
/// (a tape's own ops, an interpreted body) and native code. Machine: an
/// estimate, nothing measured at run time.
pub const INTERPRETED_NS_PER_OP: f32 = 5.0;
/// See [`INTERPRETED_NS_PER_OP`].
pub const NATIVE_NS_PER_OP: f32 = 0.5;
/// The ns a background native compile takes per op of a program and of the
/// bodies it calls ([`Policy::wait`](crate::Policy::wait)), a prior that
/// is never replaced. Measured on milan-pc (Ryzen 9900X, compile pool of
/// ten threads), kick to landing through `Adaptive`: ring_psp103 14.6 ms
/// for 42.5k ops (343 ns), Alfio_RAFFC 23.8 ms for 41.7k (571 ns),
/// chain1000 about 33 ms for 60.4k (546 ns). Time alone: higher waits less
/// for the native code, lower waits more.
pub const COMPILE_NS_PER_OP: f64 = 550.0;
/// The ns per value a block of a program's outputs costs (a gather from all
/// over the slots). Machine: an estimate.
pub const OUTPUT_NS_PER_VALUE: f32 = 1.0;
