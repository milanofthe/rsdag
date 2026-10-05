//! Parallel execution of a program's independent calls.
//!
//! A tape groups consecutive calls that read nothing another one of them
//! writes into stages (see [`Tape::stages`](crate::Tape::stages)); every
//! instance of every call in a stage is a piece of work of its own. When
//! a [`Pool`] is installed on the calling thread ([`install`]) and a stage
//! carries at least [`Parallel::min_ops`] ops, its instances run on the
//! pool, each over scratch of its worker's; otherwise one after the other.
//! Either way an instance computes exactly what the serial loop computes,
//! so the results are the same bit for bit, whatever the thread count.
//! Both backends, the interpreter and the native code, run stages so.
//!
//! The core brings no pool of its own: the host lends one, the same it
//! runs its other parallel work on, so nothing oversubscribes the machine.
//! With the `rayon` feature every [`rayon::ThreadPool`] is a [`Pool`]. Work
//! the pool runs sees no pool installed, so a call inside a parallel stage
//! runs its own stages serially.

use std::cell::RefCell;
use std::sync::Arc;

/// Workers to run independent pieces of work on.
pub trait Pool: Send + Sync {
    /// Number of workers.
    fn threads(&self) -> usize;
    /// Run `f(item)` for every `item` in `0..n`, returning when all ran.
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync));
    /// Run `f` where handing work to the pool is cheapest: on one of its
    /// workers, for a pool whose hand-off from outside costs more (a
    /// thread put to sleep and woken). The default runs it here.
    fn enter(&self, f: &mut (dyn FnMut() + Send)) {
        f()
    }
}

/// A pool and from what size of stage on it is worth the hand-off.
#[derive(Clone)]
pub struct Parallel {
    pub pool: Arc<dyn Pool>,
    /// Ops a stage carries (instances times their body's ops) from which on
    /// it runs on the pool; smaller stages run serially.
    pub min_ops: usize,
}

/// [`Parallel::min_ops`] by default: about ten microseconds of work.
pub const MIN_OPS: usize = 20_000;

impl Parallel {
    pub fn new(pool: Arc<dyn Pool>) -> Self {
        Parallel {
            pool,
            min_ops: MIN_OPS,
        }
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Parallel>> = const { RefCell::new(None) };
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

/// Whether work of `ops` ops runs on the installed pool: there is one, with
/// at least two workers, and the work reaches its [`Parallel::min_ops`].
pub fn worth(ops: usize) -> bool {
    CURRENT.with(|c| {
        c.borrow()
            .as_ref()
            .is_some_and(|p| ops >= p.min_ops && p.pool.threads() >= 2)
    })
}

/// Run `n` pieces of work of `ops` ops together: on the installed pool when
/// there is one and the work is worth it, else serially on this thread.
pub fn run(n: usize, ops: usize, f: &(dyn Fn(usize) + Sync)) {
    let p = if n >= 2 {
        CURRENT.with(|c| {
            c.borrow()
                .as_ref()
                .filter(|p| ops >= p.min_ops && p.pool.threads() >= 2)
                .map(|p| p.pool.clone())
        })
    } else {
        None
    };
    match p {
        Some(pool) => pool.run(n, f),
        None => (0..n).for_each(f),
    }
}

/// Instances per piece of work for `n` instances of a stage: about four
/// pieces per worker of the installed pool, one instance each when there
/// is none.
pub fn block(n: usize) -> usize {
    let threads = CURRENT.with(|c| c.borrow().as_ref().map_or(0, |p| p.pool.threads()));
    if threads < 2 {
        return n.max(1);
    }
    (n / (threads * 4)).max(1)
}

/// Scratch of values of type `T` for the work this thread runs, kept between
/// calls: a buffer per thread and type, grown on demand.
pub fn with_scratch<T: Copy + 'static, R>(len: usize, zero: T, f: impl FnOnce(&mut [T]) -> R) -> R {
    use std::any::{Any, TypeId};
    thread_local! {
        static BUFS: RefCell<Vec<(TypeId, Box<dyn Any>)>> = const { RefCell::new(Vec::new()) };
    }
    // Taken out for the call, so a body that runs a stage of its own (on
    // this thread, serially) takes a buffer of its own.
    let mut buf: Box<Vec<T>> = BUFS.with(|b| {
        let mut b = b.borrow_mut();
        let at = b.iter().position(|(t, _)| *t == TypeId::of::<T>());
        at.and_then(|i| b.swap_remove(i).1.downcast::<Vec<T>>().ok())
            .unwrap_or_default()
    });
    if buf.len() < len {
        buf.resize(len, zero);
    }
    let r = f(&mut buf[..len]);
    BUFS.with(|b| b.borrow_mut().push((TypeId::of::<T>(), buf)));
    r
}

#[cfg(feature = "rayon")]
impl Pool for rayon::ThreadPool {
    fn threads(&self) -> usize {
        self.current_num_threads()
    }
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        use rayon::prelude::*;
        self.install(|| (0..n).into_par_iter().for_each(f));
    }
    fn enter(&self, f: &mut (dyn FnMut() + Send)) {
        self.install(f)
    }
}
