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
//! Where every instance reads one operand (a model card bound to all of
//! them) the values are the same by construction: the rows are found on
//! the other arguments alone, and only an instance that computes its row
//! gathers its arguments.
//!
//! A main phase run under a [`Latency`] keeps, per instance, the arguments
//! it last computed at and its outputs there, in a block of the slots
//! ([`cache_len`]): an instance whose arguments are within the tolerance
//! of those takes the kept outputs and runs nothing. A prolog voids the
//! block ([`void_cache`]), since the outputs depend on its states.

use std::hash::Hasher;

use rustc_hash::{FxHashMap as HashMap, FxHasher};

use crate::extern_fn::{ExternBundle, Instances, Rows};

use super::input_index;

/// Which entry of its bundle a call runs, in the order a body's phases
/// run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// The instances' states from their pure arguments.
    Prolog,
    /// The main phase over the instances' states.
    Main,
    /// Everything, per instance.
    Whole,
}

/// Slots of the state block of a stateful call over `n_groups` instances of
/// `state_len` state values: a row per instance, then each instance's row
/// (see the module docs).
pub fn state_block(n_groups: u32, state_len: u32) -> u32 {
    n_groups * (state_len + 1)
}

/// A call over `n_groups` instances of `bundle`.
#[derive(Clone, Copy)]
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
    /// A prolog's arguments in which its instances can differ, ascending
    /// (see [`Op::CallProlog`](super::Op::CallProlog)); `None` for a call.
    pub own: Option<&'a [u32]>,
    /// Where the arguments are in the gather area, the outputs and (a main
    /// phase) the states among the slots.
    pub args: usize,
    pub out: usize,
    pub state: usize,
    /// Ops it carries: its instances times its body's.
    pub ops: u64,
    /// Where a main phase keeps its instances' arguments and outputs
    /// ([`cache_len`]), and when an instance takes them ([`Latency`]);
    /// `None` for one that computes every instance.
    pub cache: Option<(usize, Latency)>,
}

/// When an instance of a call keeps the outputs it computed before: every
/// argument it reads within `atol + rtol |a|` of the value `a` it was
/// computed at. With both zero, only arguments the same bit for bit; the
/// outputs are then those of the computation. Otherwise an output is off by
/// at most its derivatives times the tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Latency {
    pub atol: f64,
    pub rtol: f64,
}

impl Latency {
    /// Whether `v` is within the tolerance of `at`.
    #[inline]
    fn holds(&self, v: f64, at: f64) -> bool {
        v.to_bits() == at.to_bits() || (v - at).abs() <= self.atol + self.rtol * at.abs()
    }
}

/// Slots of the cache of a main phase over `n_groups` instances that read
/// `n_reads` arguments and write `n_out` outputs: a row per instance,
/// whether it computed at the last run, then the arguments and the outputs
/// it computed at.
pub fn cache_len(n_groups: usize, n_reads: usize, n_out: usize) -> usize {
    n_groups * (1 + n_reads + n_out)
}

/// The cache block `block` of a work buffer voided: every instance computes
/// at its next run (a NaN argument is within no tolerance).
pub fn void_cache(work: &mut [f64], block: std::ops::Range<usize>) {
    work[block].fill(f64::NAN);
}

/// A region of the work buffer the pieces of work of a stage share.
#[derive(Clone, Copy)]
pub struct Shared(*mut f64, usize);

// SAFETY: the pieces of work of a stage access disjoint parts of a region,
// or read the same ones; see `plan_stages`.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    pub fn new(region: &mut [f64]) -> Self {
        Shared(region.as_mut_ptr(), region.len())
    }

    /// # Safety
    /// `ptr` holds `len` values, and nothing else accesses them in a way
    /// the calls run over it do not allow (see [`run_call`]).
    pub unsafe fn from_raw(ptr: *mut f64, len: usize) -> Self {
        Shared(ptr, len)
    }

    /// The region's first value, for a piece of work that accesses it
    /// as [`get`](Self::get) and [`set`](Self::set) would.
    pub(super) fn ptr(self) -> *mut f64 {
        self.0
    }

    /// # Safety
    /// No piece of work writes slot `k` while this runs.
    pub(super) unsafe fn get(self, k: usize) -> f64 {
        assert!(k < self.1);
        unsafe { *self.0.add(k) }
    }

    /// # Safety
    /// No piece of work writes `k` while another reads it.
    pub(super) unsafe fn set(self, k: usize, v: f64) {
        assert!(k < self.1);
        unsafe { *self.0.add(k) = v }
    }

    /// # Safety
    /// No piece of work writes `at..at + n` while the slice lives.
    unsafe fn slice<'a>(self, at: usize, n: usize) -> &'a [f64] {
        assert!(at + n <= self.1);
        unsafe { std::slice::from_raw_parts(self.0.add(at), n) }
    }

    /// # Safety
    /// No other piece of work reads or writes `at..at + n` while the slice
    /// lives.
    #[allow(clippy::mut_from_ref)]
    pub(super) unsafe fn slice_mut<'a>(self, at: usize, n: usize) -> &'a mut [f64] {
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
pub unsafe fn run_call(
    c: &Call,
    slots: Shared,
    gather: Shared,
    inputs: &[f64],
    groups: std::ops::Range<usize>,
    scratch: &mut [f64],
) {
    if c.phase == Phase::Prolog && groups == (0..c.n_groups) {
        unsafe { share_rows(c, slots, gather, inputs) };
    }
    unsafe { run_part(c, slots, gather, inputs, groups, scratch) };
}

/// The row of every instance of the prolog `c`, after its rows: the first
/// instance with its pure arguments, bit for bit (see the module docs),
/// told apart by the arguments it owns; with none, every instance reads
/// the first row.
///
/// # Safety
/// As [`run_call`]; the pure arguments are the operands, or gathered
/// already where the call has none.
pub unsafe fn share_rows(c: &Call, slots: Shared, gather: Shared, inputs: &[f64]) {
    let (n, na) = (c.n_groups, c.n_args);
    let at = c.out + n * c.n_out;
    let own = c
        .own
        .expect("a prolog names the arguments its instances own");
    let arg = |g: usize, j: usize| match c.operands {
        Some(ops) => match input_index(ops[g * na + j]) {
            Some(i) => inputs.get(i as usize).copied().unwrap_or(f64::NAN),
            None => unsafe { slots.get(ops[g * na + j] as usize) },
        },
        None => unsafe { gather.get(c.args + g * na + j) },
    };
    let set = |g: usize, row: usize| unsafe { slots.set(at + g, row as f64) };
    // The arguments every instance shares are the first one's.
    if own.is_empty() {
        (0..n).for_each(|g| set(g, 0));
        return;
    }
    let hash = |g: usize| {
        let mut h = FxHasher::default();
        for &j in own {
            h.write_u64(arg(g, j as usize).to_bits());
        }
        Some(h.finish())
    };
    let same = |a: usize, b: usize| {
        own.iter()
            .all(|&j| arg(a, j as usize).to_bits() == arg(b, j as usize).to_bits())
    };
    find_rows(n, hash, same, set);
}

/// The row of each of `n` instances, handed to `set`: the first instance
/// with its key. `hash` is an instance's key hashed (`None`: a value
/// without bits, its own row), `same` whether two instances' keys are
/// equal; a hash two keys give, their values unequal, keeps its own row.
pub(crate) fn find_rows(
    n: usize,
    hash: impl Fn(usize) -> Option<u64>,
    same: impl Fn(usize, usize) -> bool,
    mut set: impl FnMut(usize, usize),
) {
    crate::scratch::with(|first: &mut HashMap<u64, usize>| {
        first.clear();
        for g in 0..n {
            let row = match hash(g) {
                Some(h) => match first.entry(h) {
                    std::collections::hash_map::Entry::Occupied(e) if same(*e.get(), g) => *e.get(),
                    std::collections::hash_map::Entry::Occupied(_) => g,
                    std::collections::hash_map::Entry::Vacant(e) => *e.insert(g),
                },
                None => g,
            };
            set(g, row);
        }
    })
}

/// [`run_call`] on the rows [`share_rows`] found; returns the instances
/// that computed (a main phase's cache holds the others).
///
/// # Safety
/// As [`run_call`].
unsafe fn run_part(
    c: &Call,
    slots: Shared,
    gather: Shared,
    inputs: &[f64],
    groups: std::ops::Range<usize>,
    scratch: &mut [f64],
) -> usize {
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
            Some(i) => inputs.get(i as usize).copied().unwrap_or(f64::NAN),
            None => unsafe { slots.get(k as usize) },
        };
        // a prolog's instance that reads another's row computes nothing
        let rows = c.out + c.n_groups * no;
        let computes =
            |g: &usize| c.phase != Phase::Prolog || unsafe { slots.get(rows + g) } as usize == *g;
        for g in groups.clone().filter(computes) {
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
        Phase::Whole if ng >= 2 => {
            b.call_batch(args, ng, na, out);
            ng
        }
        Phase::Whole => {
            b.call_into(args, scratch, out);
            ng
        }
        Phase::Main => {
            let (n, sl) = (c.n_groups, b.state_len());
            let states = unsafe { slots.slice(c.state, n * sl) };
            let rows = unsafe { slots.slice(c.state + n * sl + g0, ng) };
            let at = Instances::first(ng, na, sl).with_rows(Rows::At(rows));
            let main = |at: &Instances, out: &mut [f64], scratch: &mut [f64]| match at.len() {
                0 => {}
                1 => {
                    let (g, r) = (at.at(0), at.row(at.at(0)));
                    let o = &mut out[g * no..(g + 1) * no];
                    b.main_into(&args[g * na..][..na], &states[r * sl..][..sl], scratch, o)
                }
                _ => b.main_batch(args, states, out, at),
            };
            let Some((cache, latency)) = c.cache else {
                main(&at, out, scratch);
                return ng;
            };
            // The instances whose arguments moved past the tolerance since
            // they computed; the others take their kept outputs.
            let reads: &[u32] = c.places.unwrap_or(&[]);
            let nr = if c.places.is_some() { reads.len() } else { na };
            let w = 1 + nr + no;
            let kept = unsafe { slots.slice_mut(cache + g0 * w, ng * w) };
            let arg = |g: usize, k: usize| match c.places {
                Some(_) => args[g * na + reads[k] as usize],
                None => args[g * na + k],
            };
            crate::scratch::with(|moved: &mut Vec<u32>| {
                moved.clear();
                for (g, row) in kept.chunks_exact_mut(w).enumerate() {
                    let still = (0..nr).all(|k| latency.holds(arg(g, k), row[1 + k]));
                    if still {
                        row[0] = 0.0;
                        out[g * no..(g + 1) * no].copy_from_slice(&row[1 + nr..]);
                    } else {
                        row[0] = 1.0;
                        (0..nr).for_each(|k| row[1 + k] = arg(g, k));
                        moved.push(g as u32);
                    }
                }
                rsprobe::add!(DEBUG, "instances kept", ng - moved.len());
                main(&at.listed(moved), out, scratch);
                for &g in moved.iter() {
                    let g = g as usize;
                    kept[g * w + 1 + nr..(g + 1) * w].copy_from_slice(&out[g * no..(g + 1) * no]);
                }
                moved.len()
            })
        }
        Phase::Prolog => {
            // the instances whose row is their own: the others read theirs
            let rows = unsafe { slots.slice(c.out + c.n_groups * no + g0, ng) };
            crate::scratch::with(|own: &mut Vec<u32>| {
                own.clear();
                own.extend(
                    (0..ng)
                        .filter(|&k| rows[k] as usize == g0 + k)
                        .map(|k| k as u32),
                );
                if ng >= 2 {
                    let at = Instances::first(ng, na, b.state_len());
                    match own.len() == ng {
                        true => b.prolog_batch(args, out, &at),
                        false => b.prolog_batch(args, out, &at.listed(own)),
                    }
                } else if ng == 1 && !own.is_empty() {
                    b.prolog_into(args, scratch, out);
                }
                own.len()
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
pub unsafe fn run_stage<'a>(
    n: usize,
    call: impl Fn(usize) -> Call<'a> + Sync,
    slots: Shared,
    gather: Shared,
    inputs: &[f64],
) {
    // A call that keeps outputs counts the ops of the instances that
    // computed at its last run.
    let ops_of = |c: &Call| match c.cache {
        Some((cache, _)) if c.n_groups > 0 => {
            let nr = c.places.map_or(c.n_args, <[u32]>::len);
            let w = 1 + nr + c.n_out;
            let ran = (0..c.n_groups)
                .filter(|&g| unsafe { slots.get(cache + g * w) } != 0.0)
                .count();
            c.ops as f64 * ran as f64 / c.n_groups as f64
        }
        _ => c.ops as f64,
    };
    let work_of = |c: &Call| ops_of(c) * c.bundle.rate().ns_per_op();
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
        crate::scratch::with_len(c.bundle.work_len(), 0.0, |s| unsafe {
            run_part(c, slots, gather, inputs, groups, s)
        })
    };
    // The stage runs as wide as its runs on the first call's rate were
    // fastest for work of its size (`rspool::Work`); a serial run is
    // measured as a whole for that choice, and call by call for the ns per
    // op of each body.
    let ops: f64 = (0..n).map(|k| ops_of(&call(k))).sum();
    let decide = rspool::Work::estimated(work, ops, call(0).bundle.rate());
    let width = match decide.go::<rspool::Here>(groups) {
        rspool::Go::Wide(k) => k,
        rspool::Go::Here => {
            let whole = decide.serial_timed();
            for k in 0..n {
                let c = call(k);
                let rate = c.bundle.rate();
                let timing = rate.begin(ops_of(&c), false);
                let ran = run_groups(&c, 0..c.n_groups);
                if let Some(t) = timing {
                    rate.end(t, c.ops as f64 * ran as f64 / c.n_groups.max(1) as f64);
                }
            }
            if let Some(t) = whole {
                decide.record_serial(t);
            }
            return;
        }
    };
    // Pieces of `bs` instances, call after call; the `it`th found by
    // counting through the calls (a stage has few).
    let bs = crate::parallel::piece_len(groups, width);
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
    decide.wide::<rspool::Here>(total, width, &run);
}

#[cfg(test)]
mod tests {
    use super::find_rows;
    use std::hash::Hasher;

    /// The rows of `keys` keyed on the positions `on`, hashed by `hash`.
    fn rows(keys: &[[f64; 3]], on: &[usize], hash: impl Fn(&[u64]) -> u64) -> Vec<usize> {
        let bits = |g: usize| {
            on.iter()
                .map(|&j| keys[g][j].to_bits())
                .collect::<Vec<u64>>()
        };
        let mut out = vec![usize::MAX; keys.len()];
        find_rows(
            keys.len(),
            |g| Some(hash(&bits(g))),
            |a, b| bits(a) == bits(b),
            |g, r| out[g] = r,
        );
        out
    }

    #[test]
    fn rows_on_the_owned_arguments_are_the_rows_on_all() {
        // Position 0 and 2 the same for all instances, 1 their own: equal
        // and unequal instances, 0.0 and -0.0 apart. With a hash every key
        // gives (every unequal pair collides) and with a real one, keying
        // the owned position finds the rows keying all of them finds.
        let qs = [0.2, 0.3, 0.2, 0.2, 0.0, -0.0, 0.3, 0.7, 0.2, 0.0];
        let keys: Vec<[f64; 3]> = qs.iter().map(|&q| [1.5, q, -2.0]).collect();
        let fx = |b: &[u64]| {
            let mut h = rustc_hash::FxHasher::default();
            b.iter().for_each(|&x| h.write_u64(x));
            h.finish()
        };
        let collide = |_: &[u64]| 7u64;
        let real = rows(&keys, &[0, 1, 2], fx);
        assert_eq!(real, [0, 1, 0, 0, 4, 5, 1, 7, 0, 4]);
        assert_eq!(rows(&keys, &[1], fx), real);
        // every instance unequal to the first keeps its own row
        let colliding = rows(&keys, &[0, 1, 2], collide);
        assert_eq!(colliding, [0, 1, 0, 0, 4, 5, 6, 7, 0, 9]);
        assert_eq!(rows(&keys, &[1], collide), colliding);
    }
}
