//! Where a consumer plugs in its logging and its clock.
//!
//! rsdag has no logging dependency and no opinion about time: SANE routes
//! diagnostics through its own logger and runs on wasm where
//! `std::time::Instant` does not exist. So both are traits with a no-op
//! default, set once per process. What rsdag reports through them is small
//! and cheap to ignore -- the stage timings of a compile, at
//! [`Level::Debug`] -- and a consumer that never installs anything pays a
//! relaxed atomic load per report.
//!
//! ```
//! use rsdag::hooks::{self, Level, Log};
//!
//! struct Stderr;
//! impl Log for Stderr {
//!     fn log(&self, level: Level, message: &str) {
//!         eprintln!("[{level:?}] {message}");
//!     }
//! }
//! static SINK: Stderr = Stderr;
//! hooks::set_log(&SINK);
//! ```

use std::sync::atomic::{AtomicPtr, Ordering};

/// Severity of a report.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    Debug,
    Info,
    Warn,
}

/// A sink for reports. The default discards them.
pub trait Log: Sync {
    fn log(&self, level: Level, message: &str);
}

/// A monotonic clock in nanoseconds. The default reads
/// `std::time::Instant`; a wasm host installs its own.
pub trait Clock: Sync {
    fn now_ns(&self) -> u64;
}

struct Silent;
impl Log for Silent {
    fn log(&self, _: Level, _: &str) {}
}

struct Std;
impl Clock for Std {
    fn now_ns(&self) -> u64 {
        use std::sync::OnceLock;
        use std::time::Instant;
        static START: OnceLock<Instant> = OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

static SILENT: Silent = Silent;
static STD: Std = Std;
static LOG: AtomicPtr<&'static dyn Log> = AtomicPtr::new(std::ptr::null_mut());
static CLOCK: AtomicPtr<&'static dyn Clock> = AtomicPtr::new(std::ptr::null_mut());

/// Install the process-wide sink. A `&'static` because a report can come
/// from any thread at any time; leak a `Box` for a sink built at runtime.
pub fn set_log(sink: &'static dyn Log) {
    let cell: &'static mut &'static dyn Log = Box::leak(Box::new(sink));
    LOG.store(cell, Ordering::Release);
}

/// Install the process-wide clock.
pub fn set_clock(clock: &'static dyn Clock) {
    let cell: &'static mut &'static dyn Clock = Box::leak(Box::new(clock));
    CLOCK.store(cell, Ordering::Release);
}

fn log_sink() -> &'static dyn Log {
    let p = LOG.load(Ordering::Acquire);
    if p.is_null() {
        &SILENT
    } else {
        // SAFETY: only `set_log` stores here, and it stores a leaked
        // `&'static` that is never freed.
        unsafe { *p }
    }
}

fn clock() -> &'static dyn Clock {
    let p = CLOCK.load(Ordering::Acquire);
    if p.is_null() {
        &STD
    } else {
        // SAFETY: as in `log_sink`.
        unsafe { *p }
    }
}

/// Report through the installed sink.
pub fn log(level: Level, message: &str) {
    log_sink().log(level, message)
}

/// The installed clock, in nanoseconds since an arbitrary origin.
pub fn now_ns() -> u64 {
    clock().now_ns()
}

/// Run `f` and report how long it took, at [`Level::Debug`], as
/// `"<what>: <nanoseconds> ns"`.
pub fn timed<T>(what: &str, f: impl FnOnce() -> T) -> T {
    let t0 = now_ns();
    let out = f();
    let dt = now_ns().saturating_sub(t0);
    log(Level::Debug, &format!("{what}: {dt} ns"));
    out
}
