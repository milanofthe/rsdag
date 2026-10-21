//! Parallel execution of a program's independent calls, on the threads a
//! host lends ([`rspool`]).
//!
//! A tape groups consecutive calls that read nothing another one of them
//! writes into stages (see [`Tape::stages`](crate::Tape::stages)); every
//! instance of every call in a stage is a piece of work of its own. When a
//! [`Pool`] is lent on the calling thread ([`install`]), a stage runs on as
//! many of its threads as its work is worth ([`width`]): its ops at what an
//! op of its bodies costs (each body's [`Rate`](crate::ExternBundle::rate),
//! learned from the runs) against what a section costs the pool (its
//! [`Dispatch`], which [`Workers`] measure). A small stage stays on the
//! calling thread whatever the machine; a stage as wide as the pool leaves
//! the calls inside it their share, one thread. Each piece runs over scratch
//! of its worker's. Either way an instance computes exactly what the serial
//! loop computes, so the results are the same bit for bit, whatever the
//! thread count. Both backends, the interpreter and the native code, run
//! stages so.
//!
//! The pool, the section costs, the rates and the sections themselves are
//! [`rspool`]'s, re-exported here: a host that lends its workers to other
//! libraries too (a sparse solver between the evaluations) shares one set of
//! threads and one cost model with them.

pub use rspool::{
    current, dispatch, in_piece, install, install_lent, piece_len, run, run_rated, run_watched,
    run_width, threads, width, Dispatch, Go, Here, Lent, Piece, Place, Pool, Rate, Serial, Spawn,
    Task, Timing, Work, Workers, GAIN, PIECES_PER_WORKER, SPIN,
};
