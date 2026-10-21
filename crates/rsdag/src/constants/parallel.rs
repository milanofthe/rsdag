//! Parallel execution ([`crate::parallel`], on [`rspool`]): what an op of
//! rsdag's code costs before its runs are measured. Time only: a stage
//! computes the same bits on any number of threads.

/// The ns per op an op costs before any is measured
/// ([`Rate`](crate::parallel::Rate)): interpreted code (a tape's own ops,
/// an interpreted body) and native code. Machine: priors only, replaced by
/// what the runs measure.
pub const INTERPRETED_NS_PER_OP: f32 = 5.0;
/// See [`INTERPRETED_NS_PER_OP`].
pub const NATIVE_NS_PER_OP: f32 = 0.5;
/// The ns per value a block of a program's outputs costs before it is
/// measured (a gather from all over the slots). Machine: a prior.
pub const OUTPUT_NS_PER_VALUE: f32 = 1.0;
