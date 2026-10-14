//! Calls as every executor runs them: the tape interpreter and the native
//! code's host routine hand a call's description to [`run_call`], and a
//! stage of independent calls to [`run_stage`], so gathering, phases,
//! batches and stages have one implementation.
//!
//! A call works in three regions of its program's work buffer: the slots
//! (its operands, outputs and instance states), the gather area (its
//! arguments, group-major at the offset the tape gave it, apart from the
//! other calls of its stage) and the scratch its bundle works in.
//!
//! A stateful call's states are a block of the slots ([`state_block`]): a
//! row per instance, then the row each instance reads. Instances whose pure
//! arguments are the same, bit for bit, read one row: the prolog computes
//! it once and the main phases share it, so a circuit of many instances of
//! one model card and geometry keeps one state per kind of device in its
//! caches instead of one per instance. The results are those of every
//! instance on its own row: a prolog is a function of the pure arguments.

use std::hash::Hasher;

use rustc_hash::{FxHashMap as HashMap, FxHasher};

use crate::extern_fn::ExternBundle;
use crate::scalar::Scalar;

use super::input_index;

/// Which entry of its bundle a call runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Everything, per instance.
    Whole,
    /// The main phase over the instances' states.
    Main,
    /// The instances' states from their pure arguments.
    Prolog,
}

/// Slots of the state block of a stateful call over `n_groups` instances of
/// `state_len` state values: a row per instance, then each instance's row
/// (see the module docs).
pub fn state_block(n_groups: u32, state_len: u32) -> u32 {
    n_groups * (state_len + 1)
}

/// A call over `n_groups` instances of `bundle`.
pub struct Call<'a> {
    pub bundle: &'a dyn ExternBundle,
    pub phase: Phase,
    pub n_groups: usize,
    /// Arguments per instance: the bundle's, or a prolog's pure ones.
    pub n_args: usize,
    /// Outputs per instance: the bundle's, or a prolog's state block.
    pub n_out: usize,
    /// The operands, `n_groups` times `n_in` (a slot, or a tagged input);
    /// `None` when the arguments were gathered already.
    pub operands: Option<&'a [u32]>,
    /// The argument each of an instance's `n_in` operands is, when an
    /// instance gathers only some (see [`Op::Call`](super::Op::Call));
    /// `None` for all, in order.
    pub places: Option<&'a [u32]>,
    /// Where the arguments are in the gather area, the outputs and (a main
    /// phase) the states among the slots.
    pub args: usize,
    pub out: usize,
    pub state: usize,
    /// Ops it carries: its instances times its body's.
    pub ops: u64,
}

/// A region of the work buffer the pieces of work of a stage share.
#[derive(Clone, Copy)]
pub struct Shared<T>(*mut T, usize);

// SAFETY: the pieces of work of a stage access disjoint parts of a region,
// or read the same ones; see `plan_stages`.
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Sync> Sync for Shared<T> {}

impl<T: Copy> Shared<T> {
    pub fn new(region: &mut [T]) -> Self {
        Shared(region.as_mut_ptr(), region.len())
    }

    /// # Safety
    /// `ptr` holds `len` values, and nothing else accesses them in a way
    /// the calls run over it do not allow (see [`run_call`]).
    pub unsafe fn from_raw(ptr: *mut T, len: usize) -> Self {
        Shared(ptr, len)
    }

    /// # Safety
    /// No piece of work writes slot `k` while this runs.
    pub(super) unsafe fn get(self, k: usize) -> T {
        assert!(k < self.1);
        unsafe { *self.0.add(k) }
    }

    /// # Safety
    /// No piece of work writes `k` while another reads it.
    pub(super) unsafe fn set(self, k: usize, v: T) {
        assert!(k < self.1);
        unsafe { *self.0.add(k) = v }
    }

    /// # Safety
    /// No piece of work writes `at..at + n` while the slice lives.
    unsafe fn slice<'a>(self, at: usize, n: usize) -> &'a [T] {
        assert!(at + n <= self.1);
        unsafe { std::slice::from_raw_parts(self.0.add(at), n) }
    }

    /// # Safety
    /// No other piece of work reads or writes `at..at + n` while the slice
    /// lives.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice_mut<'a>(self, at: usize, n: usize) -> &'a mut [T] {
        assert!(at + n <= self.1);
        unsafe { std::slice::from_raw_parts_mut(self.0.add(at), n) }
    }
}

/// Instances `groups` of `c`: their operands gathered into their arguments,
/// then its bundle's phase over them, batched for two or more. `scratch`
/// holds the bundle's [`work_len`](ExternBundle::work_len). A prolog over
/// all its instances first finds the rows they share ([`share_rows`]); a
/// part of one runs on the rows found before.
///
/// # Safety
/// `slots` and `gather` are the program's regions; nothing else accesses
/// the instances' arguments, outputs and states while this runs, and
/// nothing writes what they read.
pub unsafe fn run_call<T: Scalar>(
    c: &Call,
    slots: Shared<T>,
    gather: Shared<T>,
    inputs: &[T],
    groups: std::ops::Range<usize>,
    scratch: &mut [T],
) {
    if c.phase == Phase::Prolog && groups == (0..c.n_groups) {
        unsafe { share_rows(c, slots, gather, inputs) };
    }
    unsafe { run_part(c, slots, gather, inputs, groups, scratch) }
}

/// The row of every instance of the prolog `c`, after its rows: the first
/// instance with its pure arguments, bit for bit (see the module docs). A
/// scalar without a value's bits ([`Scalar::bits`]) keeps every instance
/// on its own row.
///
/// # Safety
/// As [`run_call`]; the pure arguments are the operands, or gathered
/// already where the call has none.
pub unsafe fn share_rows<T: Scalar>(c: &Call, slots: Shared<T>, gather: Shared<T>, inputs: &[T]) {
    let (n, na) = (c.n_groups, c.n_args);
    let at = c.out + n * c.n_out;
    let arg = |g: usize, j: usize| match c.operands {
        Some(ops) => match input_index(ops[g * na + j]) {
            Some(i) => inputs.get(i as usize).copied().unwrap_or(T::nan()),
            None => unsafe { slots.get(ops[g * na + j] as usize) },
        },
        None => unsafe { gather.get(c.args + g * na + j) },
    };
    let same = |a: usize, b: usize| (0..na).all(|j| arg(a, j).bits() == arg(b, j).bits());
    let mut first: HashMap<u64, usize> = HashMap::default();
    for g in 0..n {
        let mut h = FxHasher::default();
        let mut identified = true;
        for j in 0..na {
            match arg(g, j).bits() {
                Some(b) => h.write_u64(b),
                None => identified = false,
            }
        }
        let row = match identified {
            // a hash both rows give, their values compared: a collision
            // keeps its own row
            true => match first.entry(h.finish()) {
                std::collections::hash_map::Entry::Occupied(e) if same(*e.get(), g) => *e.get(),
                std::collections::hash_map::Entry::Occupied(_) => g,
                std::collections::hash_map::Entry::Vacant(e) => *e.insert(g),
            },
            false => g,
        };
        unsafe { slots.set(at + g, T::from_f64(row as f64)) };
    }
}

/// [`run_call`] on the rows [`share_rows`] found.
///
/// # Safety
/// As [`run_call`].
unsafe fn run_part<T: Scalar>(
    c: &Call,
    slots: Shared<T>,
    gather: Shared<T>,
    inputs: &[T],
    groups: std::ops::Range<usize>,
    scratch: &mut [T],
) {
    let (na, no, b) = (c.n_args, c.n_out, c.bundle);
    let (g0, ng) = (groups.start, groups.len());
    let shape = match c.phase {
        Phase::Prolog => b.state_len(),
        _ => b.n_outputs(),
    };
    assert_eq!(shape, no, "bundle changed since compile");
    if let Some(ops) = c.operands {
        let ni = ops.len() / c.n_groups.max(1);
        let value = |k: u32| match input_index(k) {
            Some(i) => inputs.get(i as usize).copied().unwrap_or(T::nan()),
            None => unsafe { slots.get(k as usize) },
        };
        for g in groups.clone() {
            let (from, to) = (&ops[g * ni..(g + 1) * ni], c.args + g * na);
            match c.places {
                None => from
                    .iter()
                    .enumerate()
                    .for_each(|(j, &k)| unsafe { gather.set(to + j, value(k)) }),
                Some(p) => from
                    .iter()
                    .zip(p)
                    .for_each(|(&k, &j)| unsafe { gather.set(to + j as usize, value(k)) }),
            }
        }
    }
    let args = unsafe { gather.slice(c.args + g0 * na, ng * na) };
    let out = unsafe { slots.slice_mut(c.out + g0 * no, ng * no) };
    match c.phase {
        Phase::Whole if ng >= 2 => T::call_bundle_batch(b, args, ng, na, out),
        Phase::Whole => T::call_bundle_whole(b, args, scratch, out),
        Phase::Main => {
            let (n, sl) = (c.n_groups, b.state_len());
            let states = unsafe { slots.slice(c.state, n * sl) };
            let rows = unsafe { slots.slice(c.state + n * sl + g0, ng) };
            T::call_bundle_main_batch(b, args, states, rows, ng, na, scratch, out);
        }
        Phase::Prolog => {
            // the instances whose row is their own: the others read theirs
            let rows = unsafe { slots.slice(c.out + c.n_groups * no + g0, ng) };
            crate::scratch::with(|own: &mut Vec<u32>| {
                own.clear();
                own.extend(
                    (0..ng)
                        .filter(|&k| rows[k].to_f64() as usize == g0 + k)
                        .map(|k| k as u32),
                );
                T::call_bundle_prolog_batch(b, args, ng, na, own, scratch, out)
            })
        }
    }
}

/// A stage of `n` independent calls, the `k`th of them `call(k)`: on as
/// many threads of the installed pool as its work is worth
/// ([`crate::parallel::width`]; each call's ops at its bundle's
/// [`rate`](ExternBundle::rate)), every block of instances over scratch of
/// its thread's, else one call after the other. Either way the results are
/// the serial ones, and nothing is allocated.
///
/// # Safety
/// As [`run_call`], for every call; and the calls of the stage read nothing
/// another one writes, their arguments apart in the gather area.
pub unsafe fn run_stage<'a, T: Scalar>(
    n: usize,
    call: impl Fn(usize) -> Call<'a> + Sync,
    slots: Shared<T>,
    gather: Shared<T>,
    inputs: &[T],
) {
    let work_of = |c: &Call| c.ops as f64 * c.bundle.rate().ns_per_op();
    let work: f64 = (0..n).map(|k| work_of(&call(k))).sum();
    let groups: usize = (0..n).map(|k| call(k).n_groups).sum();
    // the rows the prologs' instances share, before any part of them runs
    for k in 0..n {
        let c = call(k);
        if c.phase == Phase::Prolog {
            unsafe { share_rows(&c, slots, gather, inputs) };
        }
    }
    let run_groups = |c: &Call, groups: std::ops::Range<usize>| {
        crate::scratch::with_len(c.bundle.work_len(), T::zero(), |s| unsafe {
            run_part(c, slots, gather, inputs, groups, s)
        })
    };
    let width = crate::parallel::width(work, groups);
    if width < 2 {
        for k in 0..n {
            let c = call(k);
            c.bundle
                .rate()
                .time(c.ops as f64, || run_groups(&c, 0..c.n_groups));
        }
        return;
    }
    // Pieces of `bs` instances, call after call; the `it`th found by
    // counting through the calls (a stage has few).
    let bs = crate::parallel::block(groups, width);
    let pieces = |c: &Call| c.n_groups.div_ceil(bs);
    let run = |mut it: usize| {
        let mut k = 0;
        let c = loop {
            let c = call(k);
            if it < pieces(&c) {
                break c;
            }
            it -= pieces(&c);
            k += 1;
        };
        let g0 = it * bs;
        run_groups(&c, g0..(g0 + bs).min(c.n_groups));
    };
    let total = (0..n).map(|k| pieces(&call(k))).sum();
    // Measured as a whole, now and then: the threads' time less the
    // section's cost, shared out over the calls as estimated, so a stage
    // that went wide on a rate too high learns its way back.
    let ops: f64 = (0..n).map(|k| call(k).ops as f64).sum();
    if !call(0).bundle.rate().sample(ops) {
        crate::parallel::run_width(total, width, &run);
        return;
    }
    let t = std::time::Instant::now();
    crate::parallel::run_width(total, width, &run);
    let spent = t.elapsed().as_nanos() as f64 - crate::parallel::section_cost(width);
    let serial = spent.max(0.0) * width as f64;
    for k in 0..n {
        let c = call(k);
        c.bundle
            .rate()
            .learn(serial * work_of(&c) / work, c.ops as f64);
    }
}
