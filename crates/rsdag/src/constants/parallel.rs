//! Parallel execution ([`crate::parallel`]): when a section goes to the
//! pool and on how many threads, and how its workers wait. Time only: a
//! section computes the same bits on any number of threads.
//!
//! No value here is a fact about one machine: what a section costs is
//! measured on the pool that runs it ([`Workers`](crate::parallel::Workers)
//! calibrates its [`Dispatch`](crate::parallel::Dispatch)), and what an op
//! costs is learned from the runs ([`Rate`](crate::parallel::Rate)). The
//! values below are priors before either is known, and margins.

/// A section goes wide only if that saves at least this fraction of its
/// serial time ([`Dispatch::width`](crate::parallel::Dispatch::width)):
/// margin for the scatter of the measured costs, so that a section near
/// break-even stays serial instead of flipping between the two.
pub const GAIN: f64 = 0.2;

/// What a section costs a pool that measures nothing
/// ([`Dispatch::assumed`](crate::parallel::Dispatch::assumed)), in ns:
/// this, plus [`ASSUMED_HELPER_NS`] per thread beyond the calling one.
/// Machine: a desktop CPU's wake and hand-off; a pool that knows better
/// says so through [`Pool::dispatch`](crate::parallel::Pool::dispatch).
pub const ASSUMED_SECTION_NS: f64 = 2_000.0;
/// See [`ASSUMED_SECTION_NS`].
pub const ASSUMED_HELPER_NS: f64 = 1_000.0;

/// The calibration of a pool's [`Dispatch`](crate::parallel::Dispatch):
/// sections timed per thread count (the median taken), after as many to
/// wake the workers. About a millisecond per thread count.
pub const CALIBRATION_ROUNDS: usize = 64;
/// Cache lines each piece of a calibration section writes, so that the
/// calibration pays the hand-over of a piece's output a real section pays.
pub const CALIBRATION_LINES: usize = 8;

/// The ns per op an op costs before any is measured
/// ([`Rate`](crate::parallel::Rate)): interpreted code (a tape's own ops,
/// an interpreted body) and native code. Machine: priors only, replaced
/// by what the runs measure.
pub const INTERPRETED_NS_PER_OP: f32 = 5.0;
/// See [`INTERPRETED_NS_PER_OP`].
pub const NATIVE_NS_PER_OP: f32 = 0.5;
/// The ns per value a block of a program's outputs costs before it is
/// measured (a gather from all over the slots). Machine: a prior.
pub const OUTPUT_NS_PER_VALUE: f32 = 1.0;

/// A [`Rate`](crate::parallel::Rate) learns from one run in this many, and
/// only from runs of at least [`RATE_MIN_OPS`] ops: two clock reads against
/// enough work that they do not show.
pub const RATE_SAMPLE_EVERY: u32 = 16;
/// See [`RATE_SAMPLE_EVERY`].
pub const RATE_MIN_OPS: f64 = 2_000.0;
/// The first runs of at least [`RATE_MIN_OPS`] ops of a
/// [`Rate`](crate::parallel::Rate)'s code run serially and are measured,
/// the first of them (cold caches) not counted: the rate a section's width
/// is decided on comes from runs of one thread, never from wide ones,
/// whose time mixes the work with what the hand-off cost.
pub const RATE_PROBES: u32 = 4;
/// A wide run (one in [`RATE_SAMPLE_EVERY`] is timed) that takes less than
/// this fraction of the time its rate predicts shows the rate stale (the
/// code got cheaper since: a body specialized, compiled): the rate probes
/// anew.
pub const RATE_STALE: f64 = 0.5;
/// A section that runs on `k` threads runs serially, measured, once in
/// this many times `k - 1` runs: its rate stays the code's as it is now,
/// for about a percent of the section's time.
pub const RATE_EXPLORE: u32 = 100;
/// The weight of a new measurement in a [`Rate`](crate::parallel::Rate)'s
/// running mean, toward less time and toward more: interruptions only ever
/// add time to a run, so the mean follows a faster run at once and a slower
/// one only as they keep coming.
pub const RATE_WEIGHT_DOWN: f32 = 0.5;
/// See [`RATE_WEIGHT_DOWN`].
pub const RATE_WEIGHT_UP: f32 = 0.1;

/// Pieces of work per thread a section is cut into
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
