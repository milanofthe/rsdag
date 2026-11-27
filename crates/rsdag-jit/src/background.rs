//! The background compiler: the threads compile jobs run on, which also
//! emit the chunks of the compiles in those jobs.
//!
//! Compiling is long work (a large program takes tens to hundreds of
//! milliseconds).
//! On the global rayon pool it would starve the latency-bound work that
//! runs there: a worker waiting in a `join` steals a compile chunk and
//! holds the caller's critical path for its duration. Compiles therefore
//! run on threads of their own, half the machine by default, so a compile
//! burst never owns every core either.
//!
//! A thread runs one job at a time, start to end. The chunks of a compile
//! in a job (`map`) are claimed one by one by the thread that compiles and
//! by the threads that are idle, which take them before the next queued
//! job. A compiling thread whose chunks are all claimed waits only for the
//! chunks in flight: it never runs other work meanwhile, so the depth of its
//! stack does not depend on how many jobs are queued, and a compile is never
//! held up behind another job.
//!
//! A compile outside a job has a caller that waits for it: its chunks run
//! on the threads that caller holds, the pool lent to it or else its
//! current rayon pool, as wide as the chunks go.

use std::any::Any;
use std::cell::Cell;
use std::collections::{BTreeSet, VecDeque};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        (cores / 2).max(1)
    })
}

/// Run `job` on a compile thread, alongside the jobs submitted before it:
/// the threads take them in order as they come, so the compiles of
/// independent bodies land together rather than one after another. A job
/// that needs another's result is submitted by it, or does that work
/// itself. The submission is one push to the queue, so it costs the caller
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
    threads_started();
    let mut s = lock();
    let seq = s.next_seq;
    s.next_seq += 1;
    s.unfinished.insert(seq);
    s.queued.push_back((seq, Box::new(job)));
    drop(s);
    WORK.notify_one();
}

/// Wait until every job submitted before this call has run.
pub fn drain() {
    let s = lock();
    let before = s.next_seq;
    let waiting = |s: &mut State| s.unfinished.first().is_some_and(|&q| q < before);
    drop(DONE.wait_while(s, waiting));
}

pub(crate) fn map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    if n < 2 {
        return (0..n).map(f).collect();
    }
    if !COMPILE_THREAD.get() {
        return rspool::map::<rspool::rayon::Rayon, _>(n, rspool::Work::ns(f64::INFINITY), f);
    }
    let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let piece = |i: usize| {
        let r = f(i);
        *slots[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
    };
    let section = Section {
        next: AtomicUsize::new(0),
        n,
        piece: &piece,
        helpers: AtomicUsize::new(0),
        panic: Mutex::new(None),
    };
    let shared = Shared(std::ptr::from_ref(&section).cast::<Section<'static>>());
    let mut s = lock();
    s.sections.push(shared);
    let idle = s.idle;
    drop(s);
    match n > idle {
        true => WORK.notify_all(),
        false => (1..n).for_each(|_| WORK.notify_one()),
    }
    section.claim();
    let mut s = lock();
    s.sections.retain(|o| !std::ptr::eq(o.0, shared.0));
    let s = DONE.wait_while(s, |_| section.helpers.load(Ordering::Acquire) > 0);
    drop(s);
    let panic = section
        .panic
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    if let Some(p) = panic {
        resume_unwind(p);
    }
    (slots.into_iter())
        .filter_map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner()))
        .collect()
}

struct Section<'a> {
    next: AtomicUsize,
    n: usize,
    piece: &'a (dyn Fn(usize) + Sync),
    helpers: AtomicUsize,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Section<'_> {
    fn claim(&self) {
        let claimed = catch_unwind(AssertUnwindSafe(|| loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.n {
                break;
            }
            (self.piece)(i);
        }));
        if let Err(p) = claimed {
            self.next.store(self.n, Ordering::Relaxed);
            let mut first = self.panic.lock().unwrap_or_else(|e| e.into_inner());
            first.get_or_insert(p);
        }
    }

    fn open(&self) -> bool {
        self.next.load(Ordering::Relaxed) < self.n
    }
}

#[derive(Clone, Copy)]
struct Shared(*const Section<'static>);

// SAFETY: the section lives in the frame of `map`, which takes it out of
// the list and then waits for every helper that took it from there to
// leave it before it returns; its fields are `Sync`.
unsafe impl Send for Shared {}

fn threads_started() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        for i in 0..threads() {
            let detached_so_process_can_exit = std::thread::Builder::new()
                .name(format!("rsdag-jit-{i}"))
                .spawn(run)
                .expect("spawn the compile queue");
            drop(detached_so_process_can_exit);
        }
    });
}

enum Next {
    Help(Shared),
    Run(u64, Job),
}

thread_local! {
    static COMPILE_THREAD: Cell<bool> = const { Cell::new(false) };
}

fn run() {
    COMPILE_THREAD.set(true);
    loop {
        let next = {
            let mut s = lock();
            loop {
                // SAFETY: a section in the list is alive (see `Shared`).
                let open = s.sections.iter().find(|o| unsafe { (*o.0).open() });
                if let Some(&o) = open {
                    unsafe { (*o.0).helpers.fetch_add(1, Ordering::AcqRel) };
                    break Next::Help(o);
                }
                if let Some((seq, job)) = s.queued.pop_front() {
                    break Next::Run(seq, job);
                }
                s.idle += 1;
                s = WORK.wait(s).unwrap_or_else(|e| e.into_inner());
                s.idle -= 1;
            }
        };
        match next {
            Next::Help(o) => {
                // SAFETY: counted among its helpers, so its `map` waits.
                let section = unsafe { &*o.0 };
                section.claim();
                let s = lock();
                section.helpers.fetch_sub(1, Ordering::AcqRel);
                drop(s);
                DONE.notify_all();
            }
            Next::Run(seq, job) => {
                let _ = catch_unwind(AssertUnwindSafe(job));
                lock().unfinished.remove(&seq);
                DONE.notify_all();
            }
        }
    }
}

type Job = Box<dyn FnOnce() + Send>;

struct State {
    next_seq: u64,
    unfinished: BTreeSet<u64>,
    queued: VecDeque<(u64, Job)>,
    sections: Vec<Shared>,
    idle: usize,
}

fn lock() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

static STATE: Mutex<State> = Mutex::new(State {
    next_seq: 0,
    unfinished: BTreeSet::new(),
    queued: VecDeque::new(),
    sections: Vec::new(),
    idle: 0,
});
static WORK: Condvar = Condvar::new();
static DONE: Condvar = Condvar::new();

#[cfg(test)]
mod tests {
    use super::{drain, map, submit};

    const PIECES: usize = 1000;

    fn here_and_in_a_job(check: fn()) {
        check();
        let (done, checked) = std::sync::mpsc::channel();
        submit(move || {
            check();
            let _ = done.send(());
        });
        drain();
        assert!(checked.try_recv().is_ok(), "the check failed in a job");
    }

    #[test]
    fn map_keeps_the_order_of_its_pieces() {
        here_and_in_a_job(|| assert_eq!(map(PIECES, |i| i), (0..PIECES).collect::<Vec<_>>()));
    }

    #[test]
    fn map_raises_a_panic_of_a_piece_on_the_caller() {
        here_and_in_a_job(|| {
            let raised = std::panic::catch_unwind(|| map(PIECES, |i| assert!(i != PIECES / 2)));
            assert!(raised.is_err());
            assert_eq!(map(PIECES, |i| i).len(), PIECES);
        });
    }
}
