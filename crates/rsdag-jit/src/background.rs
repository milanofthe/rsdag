//! The background compiler: the thread pool compile jobs (and a
//! consumer's own compiles) run on.
//!
//! Compiling is long work (a large program takes tens to hundreds of
//! milliseconds).
//! On the global rayon pool it would starve the latency-bound work that
//! runs there: a worker waiting in a `join` steals a compile chunk and
//! holds the caller's critical path for its duration. Compiles therefore
//! run on a pool of their own, half the machine by default, so a compile
//! burst never owns every core either.

use std::collections::BTreeSet;
use std::sync::{Condvar, Mutex, OnceLock};

/// The compile pool: half the cores (at least one).
pub fn pool() -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let n = (cores / 2).max(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("rsdag-jit-{i}"))
            .build()
            .expect("failed to build the compile pool")
    })
}

static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// Run `job` on the compile pool, alongside the jobs submitted before it:
/// the pool's threads take them as they come, so the compiles of
/// independent bodies land together rather than one after another. A job
/// that needs another's result is submitted by it, or does that work
/// itself. The submission costs the caller nothing measurable; a panic in
/// the job is caught and ends only the job.
pub fn submit(job: impl FnOnce() + Send + 'static) {
    let (lock, _) = &JOBS;
    let mut jobs = lock.lock().unwrap_or_else(|e| e.into_inner());
    let seq = jobs.next;
    jobs.next += 1;
    jobs.running.insert(seq);
    drop(jobs);
    pool().spawn(move || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
        let (lock, done) = &JOBS;
        lock.lock()
            .unwrap_or_else(|e| e.into_inner())
            .running
            .remove(&seq);
        done.notify_all();
    });
}

/// Wait until every job submitted before this call has run.
pub fn drain() {
    let (lock, done) = &JOBS;
    let jobs = lock.lock().unwrap_or_else(|e| e.into_inner());
    let before = jobs.next;
    let waiting = |j: &mut Jobs| j.running.first().is_some_and(|&s| s < before);
    drop(done.wait_while(jobs, waiting));
}

/// The jobs submitted, by number, and those not finished.
struct Jobs {
    next: u64,
    running: BTreeSet<u64>,
}

static JOBS: (Mutex<Jobs>, Condvar) = (
    Mutex::new(Jobs {
        next: 0,
        running: BTreeSet::new(),
    }),
    Condvar::new(),
);
