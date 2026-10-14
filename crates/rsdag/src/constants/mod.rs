//! Every number rsdag's behavior depends on, in one place: the contract
//! every backend reproduces to the bit ([`contract`]), and the tuning
//! values, thresholds and limits, which decide how a program runs but not
//! what it computes.
//!
//! Each constant says what it is and in what unit, why it has its value,
//! and what a change would change: results (the contract), or time alone.
//! The modules that use one re-export it where it was before
//! (`rsdag::semantics::EXP_LIMIT`, `rsdag::parallel::MIN_OPS`, ...).
//! `tests/constants_guard.rs` keeps numbers from drifting back into the
//! code, `tests/contract_snapshot.rs` holds the contract to its bits.

pub mod contract;
mod display;
mod graph;
mod parallel;
mod specialize;
mod tape;

pub use contract::*;
pub(crate) use display::*;
pub use graph::*;
pub use parallel::*;
pub(crate) use specialize::*;
pub use tape::*;
