//! Parallel execution of a program's independent calls.
//!
//! A tape groups consecutive calls that read nothing another one of them
//! writes into stages (see [`Tape::stages`](crate::Tape::stages)); every
//! instance of every call in a stage is a piece of work of its own. When
//! a [`Pool`] is installed on the calling thread ([`install`]), a stage
//! runs on as many of its threads as its work is worth ([`width`]): its
//! time (its ops at what an op of its code costs, a [`Rate`] learned from
//! the runs) against what a section costs the pool (its [`Dispatch`],
//! which [`Workers`] measure), so a small stage stays on the calling
//! thread whatever the machine. Each piece runs over scratch of its
//! worker's. Either way an instance computes exactly what the serial loop
//! computes, so the results are the same bit for bit, whatever the thread
//! count. Both backends, the interpreter and the native code, run stages
//! so.
//!
//! The host lends the pool: [`Workers`], this module's own, keeps its
//! workers awake across the short serial stretches between a solver's
//! evaluations (a factorization, a step decision) and lets the calling
//! thread take a share, so a stage costs no thread put to sleep and woken.
//! Work running on a pool, and a stage's calls, see no pool installed, so
//! a call inside a parallel stage runs its own stages serially.
//!
//! [`Workers`] steal work: a thread waiting for a section it started (a
//! `run`, a `join`, a `scope`) runs other queued work meanwhile, so the
//! sections a piece of work opens run on the pool too. Other libraries the
//! host calls between its evaluations (a sparse factorization) run their
//! own nested sections on the same workers through [`Pool::run`],
//! [`Pool::join`] and [`Pool::scope`].

use std::cell::RefCell;
use std::sync::Arc;

/// Workers to run independent pieces of work on.
pub trait Pool: Send + Sync {
    /// Number of workers.
    fn threads(&self) -> usize;
    /// Run `f(item)` for every `item` in `0..n`, returning when all ran.
    /// A piece may open sections of its own.
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync));
    /// [`run`](Self::run) on at most `width` threads, the calling one among
    /// them. The default runs on all.
    fn run_width(&self, n: usize, width: usize, f: &(dyn Fn(usize) + Sync)) {
        let _ = width;
        self.run(n, f)
    }
    /// What a section costs on this pool. The default assumes
    /// ([`Dispatch::assumed`]); a pool that can measure itself says so.
    fn dispatch(&self) -> Dispatch {
        Dispatch::assumed(self.threads())
    }
    /// Run `f` where handing work to the pool is cheapest: on one of its
    /// workers, for a pool whose hand-off from outside costs more (a
    /// thread put to sleep and woken). The default runs it here.
    fn enter(&self, f: &mut (dyn FnMut() + Send)) {
        f()
    }
    /// Run `a` and `b`, side by side where the pool can, returning when
    /// both ran. The default runs one after the other.
    fn join(&self, a: &mut (dyn FnMut() + Send), b: &mut (dyn FnMut() + Send)) {
        a();
        b();
    }
    /// Run `body` with a [`Spawn`] its tasks start tasks through, returning
    /// once every task ran. The default runs the tasks one after the other
    /// after `body`.
    fn scope<'s>(&self, body: &mut (dyn FnMut(&dyn Spawn<'s>) + Send)) {
        let queue = Serial(std::sync::Mutex::new(std::collections::VecDeque::new()));
        body(&queue);
        loop {
            let task = lock(&queue.0).pop_front();
            match task {
                Some(task) => task(&queue),
                None => return,
            }
        }
    }
}

/// What a scope's tasks are started through (see [`Pool::scope`]).
pub trait Spawn<'s>: Sync {
    /// Start `task`; its scope returns once it ran.
    fn spawn(&self, task: Box<dyn FnOnce(&dyn Spawn<'s>) + Send + 's>);
}

/// The tasks of a scope run one after the other.
type SerialTask<'s> = Box<dyn FnOnce(&dyn Spawn<'s>) + Send + 's>;
struct Serial<'s>(std::sync::Mutex<std::collections::VecDeque<SerialTask<'s>>>);

impl<'s> Spawn<'s> for Serial<'s> {
    fn spawn(&self, task: Box<dyn FnOnce(&dyn Spawn<'s>) + Send + 's>) {
        lock(&self.0).push_back(task);
    }
}

/// `m` locked: a lock is poisoned only by a panic its section carries on.
fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A pool and what a section costs on it.
#[derive(Clone)]
pub struct Parallel {
    pub pool: Arc<dyn Pool>,
    /// The pool's [`Pool::dispatch`], or another for a host that wants
    /// every section wide ([`Dispatch::free`]) or none ([`Dispatch::never`]).
    pub dispatch: Dispatch,
}

use crate::constants::{ASSUMED_HELPER_NS, ASSUMED_SECTION_NS};
use crate::constants::{CALIBRATION_LINES, CALIBRATION_ROUNDS};
pub use crate::constants::{GAIN, PIECES_PER_WORKER};
use crate::constants::{RATE_EXPLORE, RATE_MIN_OPS, RATE_PROBES, RATE_SAMPLE_EVERY, RATE_STALE};
use crate::constants::{RATE_WEIGHT_DOWN, RATE_WEIGHT_UP};

impl Parallel {
    /// `pool` with what a section costs on it.
    pub fn new(pool: Arc<dyn Pool>) -> Self {
        let dispatch = pool.dispatch();
        Parallel { pool, dispatch }
    }
}

/// What a section costs on a pool: `cost(k)` ns for a section on `k`
/// threads, the calling one among them. A section of `work` ns of serial
/// work in `p` equal pieces takes about `work * ceil(p / k) / p + cost(k)`
/// on `k` threads, and goes as wide as that is shortest ([`width`](
/// Self::width)).
#[derive(Clone, Debug)]
pub struct Dispatch {
    /// `cost[k]` for `k` threads; `cost[0]` and `cost[1]` are zero.
    cost: Arc<[f64]>,
}

impl Dispatch {
    /// The costs `cost[k]` of a section on `k` threads, for `k` up to
    /// `cost.len() - 1` (`cost[0]` and `cost[1]` are not read).
    pub fn new(cost: Vec<f64>) -> Self {
        let mut cost = cost;
        for c in cost.iter_mut().take(2) {
            *c = 0.0;
        }
        Dispatch { cost: cost.into() }
    }

    /// Sections cost nothing on `threads` threads: every section of two
    /// pieces or more goes wide (tests, and hosts that know better).
    pub fn free(threads: usize) -> Self {
        Dispatch::new(vec![0.0; threads.max(1) + 1])
    }

    /// No section goes wide.
    pub fn never() -> Self {
        Dispatch::new(vec![0.0; 2])
    }

    /// What a pool of `threads` threads that measures nothing costs:
    /// [`ASSUMED_SECTION_NS`] plus [`ASSUMED_HELPER_NS`] per helper.
    pub fn assumed(threads: usize) -> Self {
        let cost = (0..=threads.max(1))
            .map(|k| ASSUMED_SECTION_NS + ASSUMED_HELPER_NS * k.saturating_sub(1) as f64)
            .collect();
        Dispatch::new(cost)
    }

    /// The most threads a section is costed for.
    pub fn threads(&self) -> usize {
        self.cost.len().saturating_sub(1).max(1)
    }

    /// What a section costs on `k` threads, in ns.
    pub fn cost(&self, k: usize) -> f64 {
        if k < 2 {
            return 0.0;
        }
        self.cost.get(k).copied().unwrap_or(f64::INFINITY)
    }

    /// The threads a section of `work` ns in `pieces` pieces runs on: the
    /// count it takes the shortest on, if that saves at least [`GAIN`] of
    /// the serial time; one (serially) otherwise.
    pub fn width(&self, work: f64, pieces: usize) -> usize {
        if !(work > 0.0) || pieces < 2 {
            return 1;
        }
        let p = pieces as f64;
        let mut best = (1, work);
        for k in 2..=pieces.min(self.threads()) {
            let t = work * pieces.div_ceil(k) as f64 / p + self.cost[k];
            if t < best.1 {
                best = (k, t);
            }
        }
        if best.1 <= (1.0 - GAIN) * work {
            best.0
        } else {
            1
        }
    }
}

/// What an op of some code costs, in ns: a prior until runs are measured,
/// then their running mean. Code that runs on the pool carries one (a
/// function body, a backend's own ops), so a section's work is its ops at
/// its code's rate. Only serial runs are measured: the first
/// [`RATE_PROBES`] of at least [`RATE_MIN_OPS`] ops run serially for it
/// ([`probing`](Self::probing)), and after them one serial run in
/// [`RATE_SAMPLE_EVERY`].
#[derive(Debug)]
pub struct Rate {
    /// ns per op, as `f32` bits.
    ns: std::sync::atomic::AtomicU32,
    runs: std::sync::atomic::AtomicU32,
    /// Serial probes run.
    probes: std::sync::atomic::AtomicU32,
}

impl Rate {
    /// A rate of `prior` ns per op until runs are measured.
    pub const fn new(prior: f32) -> Self {
        Rate {
            ns: std::sync::atomic::AtomicU32::new(prior.to_bits()),
            runs: std::sync::atomic::AtomicU32::new(0),
            probes: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Whether a run of `ops` ops is to run serially, to be measured: the
    /// rate has not had its probes yet.
    pub fn probing(&self, ops: f64) -> bool {
        ops >= RATE_MIN_OPS && self.probes.load(std::sync::atomic::Ordering::Relaxed) < RATE_PROBES
    }

    /// Whether a run of `ops` ops that would go `k` wide is to run serially
    /// instead, measured: one in [`RATE_EXPLORE`] times `k - 1`.
    pub fn explore(&self, ops: f64, k: usize) -> bool {
        let every = RATE_EXPLORE.saturating_mul(k.saturating_sub(1).max(1) as u32);
        ops >= RATE_MIN_OPS
            && self
                .runs
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                .is_multiple_of(every)
    }

    /// The rate is stale: its next runs probe again (warm, all counted).
    pub fn reprobe(&self) {
        self.probes.store(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// ns per op.
    pub fn ns_per_op(&self) -> f64 {
        f32::from_bits(self.ns.load(std::sync::atomic::Ordering::Relaxed)) as f64
    }

    /// Whether to measure a serial run of `ops` ops (a probe, or one in
    /// [`RATE_SAMPLE_EVERY`]).
    pub fn sample(&self, ops: f64) -> bool {
        ops >= RATE_MIN_OPS
            && self
                .runs
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                .is_multiple_of(RATE_SAMPLE_EVERY)
    }

    /// A serial run of `ops` ops took `ns` ns.
    pub fn learn(&self, ns: f64, ops: f64) {
        if !(ns > 0.0 && ops > 0.0) {
            return;
        }
        use std::sync::atomic::Ordering::Relaxed;
        let seen = (ns / ops) as f32;
        let _ = self.ns.fetch_update(Relaxed, Relaxed, |b| {
            let old = f32::from_bits(b);
            let w = if seen < old {
                RATE_WEIGHT_DOWN
            } else {
                RATE_WEIGHT_UP
            };
            Some((old + w * (seen - old)).to_bits())
        });
    }

    /// `f`, a serial run of `ops` ops, measured if it is a probe or
    /// sampled. The first probe warms the caches and is not counted; the
    /// first counted one replaces the prior.
    pub fn time<R>(&self, ops: f64, f: impl FnOnce() -> R) -> R {
        use std::sync::atomic::Ordering::Relaxed;
        let probe = self.probing(ops).then(|| self.probes.fetch_add(1, Relaxed));
        if probe.is_none() && !self.sample(ops) {
            return f();
        }
        let t = std::time::Instant::now();
        let r = f();
        let ns = t.elapsed().as_nanos() as f64;
        match probe {
            Some(0) => {}
            Some(1) => self.ns.store(((ns / ops) as f32).to_bits(), Relaxed),
            _ => self.learn(ns, ops),
        }
        r
    }

    /// `f`, a serial run of `ops` ops, measured.
    pub fn measure<R>(&self, ops: f64, f: impl FnOnce() -> R) -> R {
        let t = std::time::Instant::now();
        let r = f();
        self.learn(t.elapsed().as_nanos() as f64, ops);
        r
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Parallel>> = const { RefCell::new(None) };
    /// This thread is running pieces of a stage: a stage inside runs serially.
    static IN_STAGE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with `p` installed: the programs `f` evaluates run their stages
/// on `p`'s pool. `f` runs where the pool hands work out cheapest
/// ([`Pool::enter`]: for a rayon pool, on one of its workers), so a solver
/// that evaluates its programs many times should run whole inside one
/// `install`, not install per evaluation. The previous installation comes
/// back when `f` returns (or unwinds).
pub fn install<R: Send>(p: Parallel, f: impl FnOnce() -> R + Send) -> R {
    struct Restore(Option<Parallel>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let p = self.0.take();
            CURRENT.with(|c| *c.borrow_mut() = p);
        }
    }
    let pool = p.pool.clone();
    let (mut f, mut r) = (Some(f), None);
    pool.enter(&mut || {
        let _restore = Restore(CURRENT.with(|c| c.replace(Some(p.clone()))));
        r = Some((f.take().expect("entered once"))());
    });
    r.expect("the pool ran it")
}

/// What is installed on this thread.
pub fn current() -> Option<Parallel> {
    CURRENT.with(|c| c.borrow().clone())
}

/// The threads `work` ns of work in `pieces` pieces runs on here: as many
/// as the installed pool's [`Dispatch`] makes it worth, one (serially)
/// outside a pool and inside a stage.
pub fn width(work: f64, pieces: usize) -> usize {
    if IN_STAGE.with(|f| f.get()) {
        return 1;
    }
    CURRENT.with(|c| {
        c.borrow().as_ref().map_or(1, |p| {
            p.dispatch.width(work, pieces).min(p.pool.threads()).max(1)
        })
    })
}

/// Run `n` pieces of `work` ns of work together: on as many threads of the
/// installed pool as it is worth ([`width`]), else serially on this
/// thread.
pub fn run(n: usize, work: f64, f: &(dyn Fn(usize) + Sync)) {
    run_width(n, width(work, n), f)
}

/// Run `n` pieces of `ops` ops of code at `rate` together, on as many
/// threads as they are worth ([`run`]); the rate learns from the serial
/// runs (its probes first, see [`Rate`]).
pub fn run_rated(n: usize, ops: f64, rate: &Rate, f: &(dyn Fn(usize) + Sync)) {
    let k = if rate.probing(ops) {
        1
    } else {
        width(ops * rate.ns_per_op(), n)
    };
    if k < 2 {
        return rate.time(ops, || (0..n).for_each(f));
    }
    if rate.explore(ops, k) {
        return rate.measure(ops, || (0..n).for_each(f));
    }
    if run_watched(n, k, ops * rate.ns_per_op(), rate.sample(ops), f) {
        rate.reprobe();
    }
}

/// [`run_width`] of `n` pieces of `work` ns on `k` threads, timed if
/// `timed`: whether it took less than [`RATE_STALE`] of the time predicted
/// for it (the rates its work was estimated at are stale).
pub fn run_watched(n: usize, k: usize, work: f64, timed: bool, f: &(dyn Fn(usize) + Sync)) -> bool {
    if !timed {
        run_width(n, k, f);
        return false;
    }
    let t = std::time::Instant::now();
    run_width(n, k, f);
    let spent = t.elapsed().as_nanos() as f64;
    let cost = CURRENT.with(|c| c.borrow().as_ref().map_or(0.0, |p| p.dispatch.cost(k)));
    let predicted = work * n.div_ceil(k.max(1)) as f64 / n.max(1) as f64 + cost;
    spent < RATE_STALE * predicted
}

/// Run `n` pieces of work on `k` threads of the installed pool (the calling
/// one among them), serially for `k < 2` or without a pool.
pub fn run_width(n: usize, k: usize, f: &(dyn Fn(usize) + Sync)) {
    let p = if n >= 2 && k >= 2 {
        CURRENT.with(|c| c.borrow().as_ref().map(|p| p.pool.clone()))
    } else {
        None
    };
    match p {
        Some(pool) => {
            struct Clear;
            impl Drop for Clear {
                fn drop(&mut self) {
                    IN_STAGE.with(|f| f.set(false));
                }
            }
            IN_STAGE.with(|f| f.set(true));
            let _clear = Clear;
            pool.run_width(n, k, f)
        }
        None => (0..n).for_each(f),
    }
}

/// Instances per piece of work for a stage of `n` instances on `k`
/// threads: about [`PIECES_PER_WORKER`] pieces per thread (balance over
/// instances of unequal cost), all in one piece on one.
pub fn block(n: usize, k: usize) -> usize {
    if k < 2 {
        return n.max(1);
    }
    piece_len(n, k)
}

/// Items per piece of `n` items over `threads` workers: about
/// [`PIECES_PER_WORKER`] pieces per worker, at least one item each.
pub fn piece_len(n: usize, threads: usize) -> usize {
    (n / (threads * PIECES_PER_WORKER)).max(1)
}

pub use crate::constants::SPIN;
use crate::constants::SPIN_YIELD_EVERY;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Condvar, Mutex};

/// A pool of threads that steal work: `threads - 1` workers and the thread
/// that starts a section, which takes part in it. A section runs on as
/// many of them as it asks for ([`Pool::run_width`]); what that costs is
/// measured once, when they start ([`Pool::dispatch`]). Every worker keeps a
/// queue of its own, the threads outside the pool share one; a thread
/// looking for work takes the newest of its own queue, else the oldest of
/// the outside one or another worker's. A thread waiting for its section to
/// finish keeps taking work meanwhile, so sections nest on the pool: what a
/// piece of work opens, the waiting threads run. After the work runs out
/// the workers keep looking for [`SPIN`] (as OpenMP's active wait), then
/// sleep until work comes; so the stages of a solve, a few tens of
/// microseconds apart, reach awake workers, and an idle pool costs nothing.
pub struct Workers {
    shared: Arc<Shared>,
    threads: usize,
    handles: Vec<std::thread::JoinHandle<()>>,
    /// What a section costs on them, measured when they start.
    dispatch: Dispatch,
}

/// A piece of work in a queue: what it runs on and how.
#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    run: unsafe fn(*const ()),
}

// SAFETY: a job's data is the frame or the box of the section that queued
// it, which outlives the job (see the sections).
unsafe impl Send for Job {}

impl Job {
    /// # Safety
    /// Its section is waiting for it.
    unsafe fn execute(self) {
        unsafe { (self.run)(self.data) }
    }
}

/// What the workers and the threads starting sections share.
struct Shared {
    /// Queue `0` is the outside threads', queue `k` worker `k`'s.
    queues: Vec<Mutex<VecDeque<Job>>>,
    /// Jobs in all the queues.
    queued: AtomicUsize,
    shutdown: AtomicBool,
    spin: std::time::Duration,
    /// Sleeping workers, waiting on `wake` for work.
    sleepers: AtomicUsize,
    sleep: Mutex<()>,
    wake: Condvar,
}

thread_local! {
    /// The pool this thread works for and its queue there (`None` outside
    /// any pool).
    static WORKER: std::cell::Cell<Option<(*const Shared, usize)>> =
        const { std::cell::Cell::new(None) };
}

impl Shared {
    /// This thread's queue in this pool: its own for a worker, else the
    /// outside threads'.
    fn own(&self) -> usize {
        match WORKER.with(|w| w.get()) {
            Some((p, k)) if std::ptr::eq(p, self) => k,
            _ => 0,
        }
    }

    fn push(&self, job: Job) {
        lock(&self.queues[self.own()]).push_back(job);
        self.queued.fetch_add(1, SeqCst);
        // Queued, then look: a worker going to sleep either is seen here or
        // sees the job (both sequentially consistent); it waits holding the
        // lock from its look on, so the notice reaches it.
        if self.sleepers.load(SeqCst) > 0 {
            let _g = lock(&self.sleep);
            self.wake.notify_one();
        }
    }

    /// Take `job` back from this thread's queue if nobody took it yet.
    fn retract(&self, job: Job) -> bool {
        let mut q = lock(&self.queues[self.own()]);
        let at = q.iter().rposition(|j| std::ptr::eq(j.data, job.data));
        let found = at.and_then(|i| q.remove(i)).is_some();
        drop(q);
        if found {
            self.queued.fetch_sub(1, SeqCst);
        }
        found
    }

    /// A queued job: the newest of this thread's queue, else the oldest of
    /// another.
    fn find(&self) -> Option<Job> {
        if self.queued.load(SeqCst) == 0 {
            return None;
        }
        let own = self.own();
        let n = self.queues.len();
        let mine = lock(&self.queues[own]).pop_back();
        let job =
            mine.or_else(|| (1..n).find_map(|d| lock(&self.queues[(own + d) % n]).pop_front()))?;
        self.queued.fetch_sub(1, SeqCst);
        Some(job)
    }

    /// Run queued work until `done` holds.
    fn wait_until(&self, done: impl Fn() -> bool) {
        let mut idle = 0u32;
        while !done() {
            match self.find() {
                // SAFETY: a queued job's section waits for it.
                Some(job) => unsafe { job.execute() },
                None => {
                    idle += 1;
                    if idle.is_multiple_of(SPIN_YIELD_EVERY) {
                        std::thread::yield_now();
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }
        }
    }

    /// A worker's life: run work while there is some, wait for more awake
    /// for `spin`, then asleep.
    fn work(&self, k: usize) {
        WORKER.with(|w| w.set(Some((self as *const Shared, k))));
        let mut since = std::time::Instant::now();
        let mut idle = 0u32;
        loop {
            if self.shutdown.load(SeqCst) {
                return;
            }
            if let Some(job) = self.find() {
                // SAFETY: a queued job's section waits for it.
                unsafe { job.execute() };
                since = std::time::Instant::now();
                continue;
            }
            if since.elapsed() < self.spin {
                idle += 1;
                if idle.is_multiple_of(SPIN_YIELD_EVERY) {
                    std::thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
                continue;
            }
            // Announce, then look (see `push`).
            let mut g = lock(&self.sleep);
            self.sleepers.fetch_add(1, SeqCst);
            while self.queued.load(SeqCst) == 0 && !self.shutdown.load(SeqCst) {
                g = self.wake.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            self.sleepers.fetch_sub(1, SeqCst);
            drop(g);
            since = std::time::Instant::now();
        }
    }
}

/// A panic out of a job, carried to the thread that waits for it.
type Panic = Mutex<Option<Box<dyn std::any::Any + Send>>>;

fn catch(slot: &Panic, f: impl FnOnce()) {
    if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        lock(slot).get_or_insert(p);
    }
}

fn resume(slot: Panic) {
    if let Some(p) = slot.into_inner().unwrap_or_else(|e| e.into_inner()) {
        std::panic::resume_unwind(p);
    }
}

/// A `run` in progress, on the frame of the thread that started it: its
/// pieces, taken in order, and its helpers queued for other threads, each
/// taking pieces too.
struct RunJob<'a> {
    f: &'a (dyn Fn(usize) + Sync),
    n: usize,
    next: AtomicUsize,
    done: AtomicUsize,
    /// Helpers queued and not yet finished or taken back.
    out: AtomicUsize,
    panic: Panic,
}

impl RunJob<'_> {
    fn take(&self) {
        loop {
            let i = self.next.fetch_add(1, SeqCst);
            if i >= self.n {
                return;
            }
            catch(&self.panic, || (self.f)(i));
            self.done.fetch_add(1, SeqCst);
        }
    }

    /// # Safety
    /// `data` is a `RunJob` whose `out` counts this helper.
    unsafe fn help(data: *const ()) {
        let job = unsafe { &*(data as *const RunJob<'_>) };
        job.take();
        // The last touch: the starting thread may return from here on.
        job.out.fetch_sub(1, SeqCst);
    }
}

/// One side of a `join`, on the frame of the thread that started it.
struct SideJob<'a> {
    f: Mutex<Option<&'a mut (dyn FnMut() + Send)>>,
    done: AtomicBool,
    panic: Panic,
}

impl SideJob<'_> {
    /// # Safety
    /// `data` is a `SideJob` its `join` waits for.
    unsafe fn run(data: *const ()) {
        let job = unsafe { &*(data as *const SideJob<'_>) };
        if let Some(f) = lock(&job.f).take() {
            catch(&job.panic, f);
        }
        // The last touch: the joining thread may return from here on.
        job.done.store(true, SeqCst);
    }
}

/// A `scope` in progress: its tasks not yet finished.
struct ScopeJob<'p> {
    shared: &'p Shared,
    open: AtomicUsize,
    panic: Panic,
}

/// A task of a scope, boxed, with its scope.
struct TaskJob<'s> {
    scope: *const ScopeJob<'s>,
    task: Box<dyn FnOnce(&dyn Spawn<'s>) + Send + 's>,
}

impl<'s> Spawn<'s> for ScopeJob<'s> {
    fn spawn(&self, task: Box<dyn FnOnce(&dyn Spawn<'s>) + Send + 's>) {
        self.open.fetch_add(1, SeqCst);
        let boxed = Box::new(TaskJob {
            scope: self as *const ScopeJob<'s>,
            task,
        });
        self.shared.push(Job {
            data: Box::into_raw(boxed) as *const (),
            run: TaskJob::run,
        });
    }
}

impl TaskJob<'_> {
    /// # Safety
    /// `data` is a boxed `TaskJob` whose scope waits for it.
    unsafe fn run(data: *const ()) {
        let TaskJob { scope, task } = *unsafe { Box::from_raw(data as *mut TaskJob<'_>) };
        let scope = unsafe { &*scope };
        catch(&scope.panic, || task(scope));
        // The last touch: the scope may return from here on.
        scope.open.fetch_sub(1, SeqCst);
    }
}

impl Workers {
    /// A pool of `threads` threads (the caller one of them), waiting
    /// [`SPIN`] for work before they sleep.
    pub fn new(threads: usize) -> Self {
        Self::with_spin(threads, SPIN)
    }

    /// [`new`](Self::new) with the workers waiting `spin` for work.
    pub fn with_spin(threads: usize, spin: std::time::Duration) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            queues: (0..threads).map(|_| Mutex::new(VecDeque::new())).collect(),
            queued: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            spin,
            sleepers: AtomicUsize::new(0),
            sleep: Mutex::new(()),
            wake: Condvar::new(),
        });
        let handles = (1..threads)
            .map(|k| {
                let sh = shared.clone();
                std::thread::Builder::new()
                    .name(format!("rsdag-worker-{k}"))
                    .spawn(move || sh.work(k))
                    .expect("a worker thread")
            })
            .collect();
        let mut w = Workers {
            shared,
            threads,
            handles,
            dispatch: Dispatch::never(),
        };
        let measured = Dispatch::measure(threads, &|k, piece| w.run_width(k, k, piece));
        w.dispatch = measured;
        w
    }
}

impl Dispatch {
    /// What a section costs on a pool of `threads` threads, measured:
    /// `run(k, piece)` runs `piece(i)` for `i` in `0..k` on `k` threads.
    /// Per thread count the median of [`CALIBRATION_ROUNDS`] sections of as
    /// many pieces, each writing [`CALIBRATION_LINES`] cache lines (the
    /// hand-over of a piece's output a real section pays), less what one
    /// piece costs alone; never less than with fewer threads. Run after as
    /// many sections on all threads, so it finds the workers awake as a
    /// solver's sections do. About a millisecond per thread count.
    pub fn measure(threads: usize, run: &dyn Fn(usize, &(dyn Fn(usize) + Sync))) -> Dispatch {
        if threads < 2 {
            return Dispatch::never();
        }
        #[repr(align(64))]
        struct Line(std::sync::atomic::AtomicU64);
        let lines: Vec<Line> = (0..threads * CALIBRATION_LINES)
            .map(|_| Line(std::sync::atomic::AtomicU64::new(0)))
            .collect();
        let piece = |i: usize| {
            for l in &lines[i * CALIBRATION_LINES..(i + 1) * CALIBRATION_LINES] {
                l.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        };
        let median = |f: &dyn Fn()| {
            let mut t: Vec<f64> = (0..CALIBRATION_ROUNDS)
                .map(|_| {
                    let s = std::time::Instant::now();
                    f();
                    s.elapsed().as_nanos() as f64
                })
                .collect();
            t.sort_by(f64::total_cmp);
            t[t.len() / 2]
        };
        for _ in 0..CALIBRATION_ROUNDS {
            run(threads, &piece);
        }
        let one = median(&|| piece(0));
        let mut cost = vec![0.0; threads + 1];
        for k in 2..=threads {
            let c = median(&|| run(k, &piece)) - one;
            cost[k] = c.max(cost[k - 1]);
        }
        Dispatch::new(cost)
    }
}

impl Pool for Workers {
    fn threads(&self) -> usize {
        self.threads
    }

    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        self.run_width(n, self.threads, f)
    }

    fn dispatch(&self) -> Dispatch {
        self.dispatch.clone()
    }

    fn run_width(&self, n: usize, width: usize, f: &(dyn Fn(usize) + Sync)) {
        let width = width.min(self.threads);
        if width < 2 || n < 2 {
            (0..n).for_each(f);
            return;
        }
        let sh = &*self.shared;
        let job = RunJob {
            f,
            n,
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            out: AtomicUsize::new(0),
            panic: Mutex::new(None),
        };
        let helper = Job {
            data: &job as *const RunJob<'_> as *const (),
            run: RunJob::help,
        };
        let helpers = (n - 1).min(width - 1);
        job.out.store(helpers, SeqCst);
        for _ in 0..helpers {
            sh.push(helper);
        }
        job.take();
        // Every piece is taken: the helpers nobody started come back.
        while job.out.load(SeqCst) > 0 && sh.retract(helper) {
            job.out.fetch_sub(1, SeqCst);
        }
        sh.wait_until(|| job.done.load(SeqCst) == n && job.out.load(SeqCst) == 0);
        resume(job.panic);
    }

    fn join(&self, a: &mut (dyn FnMut() + Send), b: &mut (dyn FnMut() + Send)) {
        if self.threads < 2 {
            a();
            b();
            return;
        }
        let sh = &*self.shared;
        let side = SideJob {
            f: Mutex::new(Some(b)),
            done: AtomicBool::new(false),
            panic: Mutex::new(None),
        };
        let job = Job {
            data: &side as *const SideJob<'_> as *const (),
            run: SideJob::run,
        };
        sh.push(job);
        let first: Panic = Mutex::new(None);
        catch(&first, a);
        if sh.retract(job) {
            // SAFETY: taken back, so this thread alone runs it.
            unsafe { job.execute() };
        } else {
            sh.wait_until(|| side.done.load(SeqCst));
        }
        resume(first);
        resume(side.panic);
    }

    fn scope<'s>(&self, body: &mut (dyn FnMut(&dyn Spawn<'s>) + Send)) {
        let scope = ScopeJob {
            shared: &self.shared,
            open: AtomicUsize::new(0),
            panic: Mutex::new(None),
        };
        // SAFETY: the tasks borrow nothing that outlives the scope beyond
        // `'s`, and the scope returns only once every task ran.
        let spawn: &ScopeJob<'s> =
            unsafe { std::mem::transmute::<&ScopeJob<'_>, &ScopeJob<'s>>(&scope) };
        catch(&scope.panic, || body(spawn));
        self.shared.wait_until(|| scope.open.load(SeqCst) == 0);
        resume(scope.panic);
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, SeqCst);
        {
            let _g = lock(&self.shared.sleep);
            self.shared.wake.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}
