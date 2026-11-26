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
use std::sync::mpsc::{channel, Sender};
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
/// itself. The submission is one channel send, so it costs the caller
/// nothing measurable; a panic in the job is caught and ends only the job.
/// The job runs in the scope `background` under the rsprobe node current
/// where it was submitted.
pub fn submit(job: impl FnOnce() + Send + 'static) {
    let node = rsprobe::current();
    let job = move || {
        let _node = node.enter();
        let _s = rsprobe::scope!("background");
        job()
    };
    let (lock, _) = &JOBS;
    let mut jobs = lock.lock().unwrap_or_else(|e| e.into_inner());
    let seq = jobs.next_seq;
    jobs.next_seq += 1;
    jobs.unfinished.insert(seq);
    drop(jobs);
    let _ = queue_spawning_off_caller_path().send(Box::new(move || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
        let (lock, done) = &JOBS;
        lock.lock()
            .unwrap_or_else(|e| e.into_inner())
            .unfinished
            .remove(&seq);
        done.notify_all();
    }));
}

fn queue_spawning_off_caller_path() -> &'static Sender<Job> {
    static Q: OnceLock<Sender<Job>> = OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = channel::<Job>();
        let detached_so_process_can_exit = std::thread::Builder::new()
            .name("rsdag-jit-queue".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    pool().spawn(job);
                }
            })
            .expect("spawn the compile queue");
        drop(detached_so_process_can_exit);
        tx
    })
}

type Job = Box<dyn FnOnce() + Send>;

/// Wait until every job submitted before this call has run.
pub fn drain() {
    let (lock, done) = &JOBS;
    let jobs = lock.lock().unwrap_or_else(|e| e.into_inner());
    let before = jobs.next_seq;
    let waiting = |j: &mut Jobs| j.unfinished.first().is_some_and(|&s| s < before);
    drop(done.wait_while(jobs, waiting));
}

struct Jobs {
    next_seq: u64,
    unfinished: BTreeSet<u64>,
}

static JOBS: (Mutex<Jobs>, Condvar) = (
    Mutex::new(Jobs {
        next_seq: 0,
        unfinished: BTreeSet::new(),
    }),
    Condvar::new(),
);
