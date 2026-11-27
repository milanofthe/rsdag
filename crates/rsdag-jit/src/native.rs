//! The native backend: the op stream emitted straight to machine code.
//!
//! The tape already knows what a code generator needs: it is straight-line
//! code over a work array of slots, with every value's lifetime computed.
//! So this backend does the least a compiler can do: for each op, operands
//! come from a small register cache or a load from the work array, one
//! instruction computes, and the result stays in the cache. It is written
//! to its slot in the work array only when it has to be: on eviction, at a
//! host call (which clobbers the caller-saved part of the cache) and at the
//! chunk's end, and then only if some later op still reads it, which the
//! liveness knows. A value that dies inside the chunk never touches
//! memory; nothing is ever spilled anywhere but where the interpreter keeps
//! it anyway. Compile time is linear in the op count, and the chunks are
//! emitted in parallel on the compile threads ([`crate::background`]) into
//! one executable mapping.
//!
//! A large program is executed once per evaluation, straight through, so
//! its cost is instruction fetch: the bytes per op. That is why nothing is
//! stored that need not be, why the frequent host routines sit in
//! callee-saved registers, and why the AArch64 side addresses the work
//! array through a moving window.
//!
//! Everything that is not one instruction goes to a host routine: the
//! transcendentals, `Min`/`Max` reductions, bundle calls and dense kernels,
//! the last three with their operands gathered into a scratch area at the
//! end of the work array, so no chunk ever touches the stack beyond its own
//! frame. The code reads the tape's ops, slots and call plan as they are;
//! the one lowering of its own is a small product kernel, emitted as the
//! dots of its entries.
//! Bit-exactness against the interpreter is the invariant: the same IEEE
//! operation sequence, reductions folded in the reference order, every
//! transcendental through the same host function.

use rsdag::node::{BinOp, CmpOp, ReduceOp, UnaryOp};
use rsdag::tape::calls::{Call, Phase};
use rsdag::tape::{input_index, read, Accum, Fold, Op, Operand, Src, INPUT};
use rsdag::{ExternBundle, Instances, Program, Tape};
use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::constants::{
    CODE_ALIGN_BYTES, FLOAT_REG_NUMBERS, FORCED_LANES_SCALAR_COST, HOST_GATHER_MIN_ARGS,
    KERNEL_OPERANDS_A_XB_ACCUMULATOR, LANE_HOST_CALL_COST_OPS, LANE_HOST_CALL_VALUES,
    LANE_MOVE_COST_OPS, MAX_LANES, OPTIONS_KEY_BACKGROUND_BIT, OPTIONS_KEY_CHUNK_OPS_SHIFT,
    OPTIONS_KEY_LANES_SHIFT,
};
use crate::host::{self, Bundles};
use crate::ops::{inline_product, minmax_inline, powi_inline, routine, Liveness};
use rsdag::constants::FOLD_LANES;

const F64_BYTES: usize = std::mem::size_of::<f64>();

static NATIVE_REGION_OP_RATE: rsdag::parallel::Rate =
    rsdag::parallel::Rate::new(rsdag::constants::NATIVE_NS_PER_OP);
use crate::isa::{Arg, Arith, Base, IArg, Isa, Round};
use crate::{Batch, JitError, Lanes, Options};

#[cfg(target_arch = "aarch64")]
type Arch = crate::aarch64::A64;
#[cfg(target_arch = "x86_64")]
type Arch = crate::x86_64::X64<1>;

type ChunkFn = unsafe extern "C" fn(*mut f64, *const f64, *const Layout);

/// A tape compiled to native code. Evaluation mirrors [`Tape`]: a
/// caller-owned work buffer, inputs padded with NaN, and the prolog/main
/// split of a specialized tape.
pub struct NativeTape {
    _code_the_chunks_point_into: Mapping,
    chunks: Vec<ChunkFn>,
    prolog_chunks: usize,
    _descs_the_code_points_to: Vec<(Vec<Call<'static>>, Vec<host::KernelDesc>)>,
    bundles: Bundles,
    _tables_the_code_points_to: Vec<Box<[u32]>>,
    outputs: Vec<u32>,
    layout: Box<Layout>,
    n_ops: usize,
    state_len: usize,
    lanes: usize,
    reads: [Vec<u32>; 2],
    phase_ops_and_host_calls: [(usize, usize); 2],
    regions: Vec<ChunkRegion>,
}

struct ChunkRegion {
    starts: Vec<usize>,
    hi: usize,
    ops: usize,
}

fn lane_gather_len<I: Isa>(op: &Op) -> usize {
    let l = I::LANES;
    match *op {
        _ if minmax_inline::<I>(op) => 0,
        Op::Reduce(ReduceOp::Min | ReduceOp::Max, _, n) => n as usize * l + n as usize + l,
        _ if routine::<I>(op).is_some() => LANE_HOST_CALL_VALUES * l,
        _ => 0,
    }
}

struct NativeBody {
    tape: NativeTape,
    lanes_widest_first: Vec<LaneCode>,
    scalar_cost_per_phase: [usize; 3],
    n_out: usize,
    pure: Vec<bool>,
    rank_among_pure: Vec<Option<usize>>,
    batch: Batch,
    rate: rsdag::parallel::Rate,
}

impl NativeBody {
    fn run_groups(
        &self,
        work: &mut Vec<f64>,
        args: &[f64],
        n_args: usize,
        out: &mut [f64],
        groups: std::ops::Range<usize>,
    ) {
        work.resize(self.tape.layout.total, 0.0);
        self.run_groups_into(work, args, n_args, out, groups);
    }

    // https://github.com/milanofthe/rsdag/issues/240
    fn run_groups_into(
        &self,
        work: &mut [f64],
        args: &[f64],
        n_args: usize,
        out: &mut [f64],
        groups: std::ops::Range<usize>,
    ) {
        let n_out = self.n_out;
        for g in groups {
            let ins = &args[g * n_args..(g + 1) * n_args];
            self.tape.run(0..self.tape.chunks.len(), ins, work);
            for (k, &slot) in self.tape.outputs[..n_out].iter().enumerate() {
                out[g * n_out + k] = read(ins, work, slot);
            }
        }
    }
}

fn chunked(
    tape: &Tape,
    ops: std::ops::Range<usize>,
    chunk_ops: usize,
    cuts: &[usize],
) -> Vec<std::ops::Range<usize>> {
    debug_assert!(cuts.is_sorted());
    let mut out = Vec::new();
    let mut start = ops.start;
    let mut cut = cuts.iter().copied().filter(|&c| c > ops.start).peekable();
    for k in ops.clone() {
        let inside = stage_of(tape, k).is_some_and(|st| k + 1 < st.hi as usize);
        while cut.peek().is_some_and(|&c| c <= k) {
            cut.next();
        }
        let before_cut = cut.peek() == Some(&(k + 1));
        if (k + 1 - start >= chunk_ops.max(1) || before_cut) && !inside {
            out.push(start..k + 1);
            start = k + 1;
        }
    }
    if start < ops.end {
        out.push(start..ops.end);
    }
    out
}

fn stage_of(tape: &Tape, i: usize) -> Option<&rsdag::tape::Stage> {
    let stages = tape.stages();
    stages
        .get(stages.partition_point(|st| st.hi as usize <= i))
        .filter(|st| st.lo as usize <= i)
}

const PHASES: usize = Phase::Whole as usize + 1;

struct LaneCode {
    tape: NativeTape,
    block_cost_per_phase: [usize; PHASES],
    inputs_moved_per_phase: [Vec<u32>; PHASES],
}

impl LaneCode {
    fn pays(&self, phase: usize, scalar_instance_cost: usize) -> bool {
        self.block_cost_per_phase[phase] < self.tape.lanes * scalar_instance_cost
    }
}

fn lane_costs(lt: &NativeTape, n_state: usize, n_out: usize) -> ([usize; 3], [usize; 3]) {
    let [(o0, h0), (o1, h1)] = lt.phase_ops_and_host_calls;
    let [r0, r1] = [lt.reads[0].len(), lt.reads[1].len()];
    let phases = [
        (o0, h0, r0 + n_state),
        (o1, h1, r1 + n_state + n_out),
        (o0 + o1, h0 + h1, r0 + r1 + n_out),
    ];
    let block = phases.map(|(ops, host, moved)| {
        ops - host + lt.lanes * (host * (1 + LANE_HOST_CALL_COST_OPS) + moved * LANE_MOVE_COST_OPS)
    });
    (block, phases.map(|(ops, _, _)| ops))
}

impl NativeBody {
    fn run_lanes_first(
        &self,
        phase: Phase,
        n: usize,
        mut run: impl FnMut(&LaneCode, std::ops::Range<usize>),
    ) -> usize {
        let (k, one) = (phase as usize, self.scalar_cost_per_phase[phase as usize]);
        let pays = |lc: &&LaneCode| lc.pays(k, one);
        let Some(widest) = self.lanes_widest_first.iter().find(pays) else {
            return 0;
        };
        let full = n / widest.tape.lanes * widest.tape.lanes;
        if full > 0 {
            run(widest, 0..full);
        }
        let rest = n - full;
        let mut best = [(0usize, None::<usize>); MAX_LANES];
        for m in 1..=rest {
            best[m] = (best[m - 1].0 + one, None);
            for (i, lc) in self
                .lanes_widest_first
                .iter()
                .enumerate()
                .filter(|(_, lc)| pays(lc))
            {
                let c = best[m.saturating_sub(lc.tape.lanes)].0 + lc.block_cost_per_phase[k];
                if c < best[m].0 {
                    best[m] = (c, Some(i));
                }
            }
        }
        let (mut g, mut m) = (full, rest);
        while m > 0 {
            match best[m].1 {
                Some(i) => {
                    let w = self.lanes_widest_first[i].tape.lanes.min(m);
                    run(&self.lanes_widest_first[i], g..g + w);
                    g += w;
                    m -= w;
                }
                None => m -= 1,
            }
        }
        g
    }

    #[allow(clippy::too_many_arguments)]
    fn run_lanes(
        &self,
        lc: &LaneCode,
        phase: Phase,
        args: &[f64],
        at: &Instances,
        block: std::ops::Range<usize>,
        states: Option<&[f64]>,
        mut states_out: Option<&mut [f64]>,
        mut out: Option<&mut [f64]>,
    ) {
        let lt = &lc.tape;
        let (l, sl, n_out) = (lt.lanes, self.tape.state_len, self.n_out);
        let (n_args, stride) = (at.n_args, at.stride);
        let n_in = lt.layout.n_inputs.max(n_args).max(self.pure.len());
        let moved = &lc.inputs_moved_per_phase[phase as usize];
        let range = match phase {
            Phase::Prolog => 0..lt.prolog_chunks,
            Phase::Main => lt.prolog_chunks..lt.chunks.len(),
            Phase::Whole => 0..lt.chunks.len(),
        };
        rsdag::scratch::with_len(lt.layout.total, 0.0, |work: &mut [f64]| {
            rsdag::scratch::with_len(n_in * l, f64::NAN, |ins: &mut [f64]| {
                let end = block.end;
                // https://github.com/milanofthe/rsdag/issues/240
                let mut state_rows_in_lanes: Option<[usize; MAX_LANES]> = None;
                for c in block.step_by(l) {
                    let mut rows = [usize::MAX; MAX_LANES];
                    for (lane, r) in rows.iter_mut().enumerate().take(l) {
                        *r = at.row(at.at((c + lane).min(end - 1)));
                    }
                    let fresh = phase != Phase::Main || state_rows_in_lanes != Some(rows);
                    if fresh {
                        state_rows_in_lanes = (phase == Phase::Main).then_some(rows);
                    }
                    for lane in 0..l {
                        let g = at.at((c + lane).min(end - 1));
                        let a = &args[g * n_args..(g + 1) * n_args];
                        for &k in moved {
                            let k = k as usize;
                            ins[k * l + lane] = if phase == Phase::Prolog {
                                self.rank_among_pure
                                    .get(k)
                                    .copied()
                                    .flatten()
                                    .map_or(f64::NAN, |r| a[r])
                            } else {
                                a.get(k).copied().unwrap_or(f64::NAN)
                            };
                        }
                        if let Some(st) = states.filter(|_| fresh) {
                            let row = &st[at.row(g) * stride..][..sl];
                            for (s, &v) in row.iter().enumerate() {
                                work[s * l + lane] = v;
                            }
                        }
                    }
                    lt.run(range.clone(), ins, work);
                    for lane in 0..l.min(end - c) {
                        let g = at.at(c + lane);
                        if let Some(st) = states_out.as_deref_mut() {
                            let row = &mut st[at.row(g) * stride..][..sl];
                            for (s, v) in row.iter_mut().enumerate() {
                                *v = work[s * l + lane];
                            }
                        }
                        if let Some(o) = out.as_deref_mut() {
                            let a = &args[g * n_args..(g + 1) * n_args];
                            for (k, &slot) in lt.outputs[..n_out].iter().enumerate() {
                                o[g * n_out + k] = match input_index(slot) {
                                    Some(_) => read(a, work, slot),
                                    None => work[slot as usize * l + lane],
                                };
                            }
                        }
                    }
                }
            })
        });
    }
}

impl ExternBundle for NativeBody {
    fn n_outputs(&self) -> usize {
        self.n_out
    }
    fn rate(&self) -> &rsdag::parallel::Rate {
        &self.rate
    }
    fn work_len(&self) -> usize {
        self.tape.layout.total + self.pure.len()
    }
    fn batch_lanes(&self) -> usize {
        self.lanes_widest_first
            .first()
            .map_or(1, |lc| lc.tape.lanes)
    }
    fn call_into(&self, args: &[f64], work: &mut [f64], out: &mut [f64]) {
        let w = &mut work[..self.tape.layout.total];
        self.run_groups_into(w, args, args.len(), out, 0..1);
    }
    fn state_len(&self) -> usize {
        self.tape.state_len
    }
    fn pure_args(&self) -> &[bool] {
        &self.pure
    }
    fn prolog_into(&self, pure: &[f64], work: &mut [f64], state: &mut [f64]) {
        let (w, a) = work.split_at_mut(self.tape.layout.total);
        let a = &mut a[..self.pure.len()];
        rsdag::extern_fn::spread_pure(&self.pure, pure, a);
        self.tape.run(0..self.tape.prolog_chunks, a, w);
        let sl = self.tape.state_len;
        state[..sl].copy_from_slice(&w[..sl]);
    }
    fn main_into(&self, args: &[f64], state: &[f64], work: &mut [f64], out: &mut [f64]) {
        let w = &mut work[..self.tape.layout.total];
        let sl = self.tape.state_len;
        w[..sl].copy_from_slice(&state[..sl]);
        self.tape
            .run(self.tape.prolog_chunks..self.tape.chunks.len(), args, w);
        for (k, &slot) in self.tape.outputs[..self.n_out].iter().enumerate() {
            out[k] = read(args, w, slot);
        }
    }
    fn prolog_batch(&self, pure: &[f64], states: &mut [f64], at: &Instances) {
        let k0 = self.run_lanes_first(Phase::Prolog, at.len(), |lc, r| {
            self.run_lanes(
                lc,
                Phase::Prolog,
                pure,
                at,
                r,
                None,
                Some(&mut *states),
                None,
            )
        });
        let (na, sl, st) = (at.n_args, self.tape.state_len, at.stride);
        rsdag::scratch::with_len(self.work_len(), 0.0, |w| {
            for g in (k0..at.len()).map(|k| at.at(k)) {
                let (p, r) = (&pure[g * na..(g + 1) * na], at.row(g));
                self.prolog_into(p, w, &mut states[r * st..r * st + sl]);
            }
        });
    }
    fn main_batch(&self, args: &[f64], states: &[f64], out: &mut [f64], at: &Instances) {
        let k0 = self.run_lanes_first(Phase::Main, at.len(), |lc, r| {
            self.run_lanes(
                lc,
                Phase::Main,
                args,
                at,
                r,
                Some(states),
                None,
                Some(&mut *out),
            )
        });
        let (na, sl, st, no) = (at.n_args, self.tape.state_len, at.stride, self.n_out);
        rsdag::scratch::with_len(self.work_len(), 0.0, |w| {
            for g in (k0..at.len()).map(|k| at.at(k)) {
                let (a, r) = (&args[g * na..(g + 1) * na], at.row(g));
                let s = &states[r * st..r * st + sl];
                self.main_into(a, s, w, &mut out[g * no..(g + 1) * no]);
            }
        });
    }
    fn call_batch(&self, args: &[f64], n_groups: usize, n_args: usize, out: &mut [f64]) {
        if self.batch == Batch::Serial {
            let at = Instances::first(n_groups, n_args, 0);
            let g0 = self.run_lanes_first(Phase::Whole, n_groups, |lc, r| {
                self.run_lanes(lc, Phase::Whole, args, &at, r, None, None, Some(&mut *out))
            });
            rsdag::scratch::with::<Vec<f64>, _>(|work| {
                self.run_groups(work, args, n_args, out, g0..n_groups)
            });
            return;
        }
        let work = match self.batch {
            Batch::Always => rspool::Work::ns(f64::INFINITY),
            _ => rspool::Work::of((n_groups * self.tape.n_ops) as f64, &self.rate),
        };
        if self.n_out == 0 {
            rsdag::scratch::with::<Vec<f64>, _>(|work| {
                self.run_groups(work, args, n_args, out, 0..n_groups)
            });
            return;
        }
        rspool::chunks_mut::<rspool::rayon::Rayon, _>(out, self.n_out, work, |g0, dst| {
            let g1 = g0 + dst.len() / self.n_out;
            let ins = &args[g0 * n_args..g1 * n_args];
            rsdag::scratch::with::<Vec<f64>, _>(|work| {
                self.run_groups(work, ins, n_args, dst, 0..g1 - g0)
            });
        });
    }
}

fn native_body(
    body: &Tape,
    pure: &[bool],
    no: usize,
    opts: &Options,
) -> Result<Arc<dyn ExternBundle>, JitError> {
    let _s = rsprobe::scope!("body");
    let sl = body.state_len();
    let lanes = match opts.lanes {
        Lanes::Never => Vec::new(),
        _ => NativeTape::compile_lanes_widest_first(body),
    };
    let mut scalar = [FORCED_LANES_SCALAR_COST; PHASES];
    let lanes: Vec<LaneCode> = lanes
        .into_iter()
        .map(|lt| {
            let (block, one) = lane_costs(&lt, sl, no);
            if opts.lanes == Lanes::Auto {
                scalar = one;
            }
            let mut whole = [lt.reads[0].as_slice(), lt.reads[1].as_slice()].concat();
            whole.sort_unstable();
            whole.dedup();
            let reads = [lt.reads[0].clone(), lt.reads[1].clone(), whole];
            LaneCode {
                tape: lt,
                block_cost_per_phase: block,
                inputs_moved_per_phase: reads,
            }
        })
        .collect();
    let lanes = lanes
        .into_iter()
        .filter(|lc| (0..PHASES).any(|k| lc.pays(k, scalar[k])))
        .collect::<Vec<_>>();
    debug_assert!(lanes.windows(2).all(|w| w[0].tape.lanes > w[1].tape.lanes));
    rsprobe::add!("lane forms", lanes.len());
    Ok(Arc::new(NativeBody {
        tape: NativeTape::compile_opts(body, opts, &[])?,
        lanes_widest_first: lanes,
        scalar_cost_per_phase: scalar,
        n_out: no,
        pure: pure.to_vec(),
        rank_among_pure: pure
            .iter()
            .scan(0, |next, &p| {
                *next += usize::from(p);
                Some(p.then(|| *next - 1))
            })
            .collect(),
        batch: opts.batch,
        rate: rsdag::parallel::Rate::new(rsdag::constants::NATIVE_NS_PER_OP),
    }))
}

fn options_key(opts: &Options) -> u64 {
    let batch = match opts.batch {
        Batch::Serial => 0,
        Batch::Parallel => 1,
        Batch::Always => 2,
    };
    let lanes = match opts.lanes {
        Lanes::Auto => 0,
        Lanes::Always => 1,
        Lanes::Never => 2,
    };
    ((opts.chunk_ops as u64) << OPTIONS_KEY_CHUNK_OPS_SHIFT)
        ^ batch
        ^ (lanes << OPTIONS_KEY_LANES_SHIFT)
}

#[derive(Clone, Copy)]
pub(crate) struct Layout {
    pub(crate) gather_start: usize,
    pub(crate) scratch_start: usize,
    pub(crate) scratch_len: usize,
    total: usize,
    pub(crate) n_inputs: usize,
}

struct Emitted {
    bytes: Vec<u8>,
    calls: Vec<Call<'static>>,
    kernels: Vec<host::KernelDesc>,
    tables: Vec<Box<[u32]>>,
}

// SAFETY: the mapping is immutable after `Mapping::new`, so calling the
// code from any thread is sound.
unsafe impl Send for Mapping {}
// SAFETY: as for `Send`, nothing writes the mapping after `Mapping::new`.
unsafe impl Sync for Mapping {}

impl NativeTape {
    pub fn compile(tape: &Tape) -> Result<NativeTape, JitError> {
        Self::compile_opts(tape, &Options::default(), &[])
    }

    /// Compile as `opts` says, keeping the slots `live` written to the work
    /// array at the end of the program, as the outputs are: what a consumer
    /// that reads a specialized tape's prolog guards from `work` after
    /// [`eval_prolog_into`](rsdag::Program::eval_prolog_into) passes
    /// ([`rsdag::SpecializedTape::prolog_guards`]).
    pub fn compile_opts(tape: &Tape, opts: &Options, live: &[u32]) -> Result<NativeTape, JitError> {
        Self::compile_isa::<Arch>(tape, opts, live)
    }

    fn compile_lanes_widest_first(tape: &Tape) -> Vec<NativeTape> {
        #[cfg(target_arch = "x86_64")]
        {
            use crate::x86_64::X64;
            let opts = Options::default();
            let wide = crate::x86_64::features()
                .avx2
                .then(|| Self::compile_isa::<X64<MAX_LANES>>(tape, &opts, &[]));
            let two = Some(Self::compile_isa::<X64<2>>(tape, &opts, &[]));
            wide.into_iter().chain(two).filter_map(Result::ok).collect()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = tape;
            Vec::new()
        }
    }

    fn compile_isa<A: Isa>(
        tape: &Tape,
        opts: &Options,
        live: &[u32],
    ) -> Result<NativeTape, JitError> {
        let _s = rsprobe::scope!("native");
        let chunk_ops = opts.chunk_ops;
        if !cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
            return Err(JitError::Unsupported);
        }
        let (n, split) = (tape.n_ops(), tape.prolog_len());
        let ops = tape.ops();
        let lanes = A::LANES;
        let kernel = |op: &Op| {
            matches!(op, Op::Gemv { .. } | Op::Gemm { .. } | Op::Solve { .. })
                && !inline_product(op)
        };
        if lanes > 1
            && ops
                .iter()
                .any(|op| matches!(op, Op::Call { .. } | Op::CallProlog { .. }) || kernel(op))
        {
            return Err(JitError::Unsupported);
        }
        let o = *opts;
        let backend = rsdag::BodyBackend {
            compile: Some(Arc::new(move |t: &Tape, pure: &[bool], n_out: usize| {
                native_body(t, pure, n_out, &o).ok()
            })),
            key: options_key(opts),
            submit: opts
                .background
                .then(|| -> rsdag::Submit { Arc::new(crate::background::submit) }),
            variants: None,
        };
        let bundles: Result<Bundles, JitError> = {
            let _s = rsprobe::scope!("bundles");
            tape.bundles()
                .iter()
                .map(|b| {
                    let make = || -> Result<Arc<dyn ExternBundle>, JitError> {
                        if let Some(v) = b.with_backend(&backend) {
                            return Ok(v);
                        }
                        match b.body() {
                            Some(body) => native_body(body, b.pure_args(), b.n_outputs(), opts),
                            None => Ok(b.clone()),
                        }
                    };
                    let key = options_key(opts)
                        ^ (u64::from(opts.background) << OPTIONS_KEY_BACKGROUND_BIT);
                    match b.backend_cache() {
                        Some(cache) => cache.get_or_try_insert(key, make),
                        None => make(),
                    }
                })
                .collect()
        };
        let bundles = bundles?;
        let reads = [tape.reads(0..split), tape.reads(split..n)];
        let n_inputs = (reads.iter().flatten().copied())
            .chain(tape.outputs().iter().filter_map(|&o| input_index(o)))
            .map(|i| i as usize + 1)
            .max()
            .unwrap_or(0);
        let mut phase_ops_and_host_calls = [(0usize, 0usize); 2];
        for (k, op) in ops.iter().enumerate() {
            let p = &mut phase_ops_and_host_calls[usize::from(k >= split)];
            let lane_host_call =
                lanes > 1 && matches!(op, Op::Unary(UnaryOp::Exp | UnaryOp::Ln, _));
            p.0 += if lane_host_call {
                1 + LANE_HOST_CALL_COST_OPS
            } else {
                1
            };
            p.1 += usize::from(routine::<A>(op).is_some() && !lane_host_call);
        }
        let gather_len = match lanes {
            1 => tape.gather_len(),
            _ => ops.iter().map(lane_gather_len::<A>).max().unwrap_or(0),
        };
        let n_work = tape.n_slots();
        let liveness = Liveness::new(tape, tape.outputs().iter().chain(live).copied());
        let scratch_len = (bundles.iter().map(|b| b.work_len()))
            .chain([tape.lent()])
            .max()
            .unwrap_or(0);
        let layout = Box::new(Layout {
            gather_start: n_work * lanes,
            scratch_start: n_work * lanes + gather_len,
            scratch_len,
            total: (n_work * lanes + gather_len + scratch_len).max(1),
            n_inputs,
        });
        let mut cuts: Vec<usize> = (tape.regions().iter())
            .flat_map(|r| r.starts.iter().chain([&r.hi]))
            .map(|&k| k as usize)
            .collect();
        cuts.sort_unstable();
        let pro = chunked(tape, 0..split, chunk_ops, &cuts);
        let prolog_chunks = pro.len();
        let jobs: Vec<std::ops::Range<usize>> = pro
            .into_iter()
            .chain(chunked(tape, split..n, chunk_ops, &cuts))
            .collect();
        debug_assert!(jobs.iter().all(|r| r.end <= split || r.start >= split));
        let chunk_at =
            |k: u32| (jobs.binary_search_by_key(&(k as usize), |r| r.start)).unwrap_or(jobs.len());
        // https://github.com/milanofthe/rsdag/issues/240
        let serial = lanes > 1;
        let regions = (tape.regions().iter())
            .filter(|_| !serial)
            .map(|r| ChunkRegion {
                starts: r.starts.iter().map(|&k| chunk_at(k)).collect(),
                hi: chunk_at(r.hi),
                ops: r.ops() as usize,
            })
            .collect();
        let emitted: Result<Vec<Emitted>, JitError> = {
            let _s = rsprobe::scope!("emit");
            let node = rsprobe::current();
            crate::background::map(jobs.len(), |k| {
                let _node = node.enter();
                emit_chunk::<A>(tape, jobs[k].clone(), &layout, &liveness, &bundles)
            })
            .into_iter()
            .collect()
        };
        let emitted = emitted?;
        // https://github.com/milanofthe/rsdag/issues/240
        let size = emitted
            .iter()
            .map(|e| e.bytes.len() + CODE_ALIGN_BYTES - 1)
            .sum();
        let mut bytes: Vec<u8> = Vec::with_capacity(size);
        let mut offsets = Vec::with_capacity(emitted.len());
        let mut descs = Vec::with_capacity(emitted.len());
        let mut tables = Vec::new();
        for e in emitted {
            bytes.resize(bytes.len().next_multiple_of(CODE_ALIGN_BYTES), 0);
            offsets.push(bytes.len());
            bytes.extend_from_slice(&e.bytes);
            descs.push((e.calls, e.kernels));
            tables.extend(e.tables);
        }
        rsprobe::add!("ops", n);
        rsprobe::add!("chunks", offsets.len());
        rsprobe::add!("code bytes", bytes.len());
        let code = {
            let _s = rsprobe::scope!("map");
            Mapping::new(&bytes)?
        };
        let chunks = offsets
            .iter()
            .map(|&o| {
                // SAFETY: each offset is the entry of a function emitted for
                // the `ChunkFn` convention, inside the mapping, which the
                // tape keeps.
                unsafe { std::mem::transmute::<*mut u8, ChunkFn>(code.ptr.add(o)) }
            })
            .collect();
        Ok(NativeTape {
            _code_the_chunks_point_into: code,
            chunks,
            _descs_the_code_points_to: descs,
            prolog_chunks,
            bundles,
            _tables_the_code_points_to: tables,
            outputs: tape.outputs().to_vec(),
            layout,
            n_ops: n,
            state_len: tape.state_len(),
            lanes,
            reads,
            phase_ops_and_host_calls,
            regions,
        })
    }

    fn padded<R>(&self, inputs: &[f64], f: impl FnOnce(&[f64]) -> R) -> R {
        match inputs.len() < self.layout.n_inputs {
            true => rsdag::scratch::with(|buf: &mut Vec<f64>| {
                buf.clear();
                buf.extend_from_slice(inputs);
                buf.resize(self.layout.n_inputs, f64::NAN);
                f(buf)
            }),
            false => f(inputs),
        }
    }

    fn run(&self, range: std::ops::Range<usize>, inputs: &[f64], work: &mut [f64]) {
        assert!(
            work.len() >= self.layout.total && inputs.len() >= self.layout.n_inputs * self.lanes,
            "buffers not prepared by this tape"
        );
        let (wp, ip, lp) = (
            work.as_mut_ptr(),
            inputs.as_ptr(),
            &*self.layout as *const Layout,
        );
        let (work_addr_sendable, inputs_addr_sendable, layout_addr_sendable) =
            (wp as usize, ip as usize, lp as usize);
        let chunk = |c: usize| {
            let (w, i, l) = (
                work_addr_sendable as *mut f64,
                inputs_addr_sendable as *const f64,
                layout_addr_sendable as *const Layout,
            );
            // SAFETY: the buffers were checked above, the code and the
            // tables it addresses live as long as `self`; a region's pieces
            // write slots no other piece reads or writes.
            unsafe { self.chunks[c](w, i, l) };
            host::resume_panic();
        };
        let mut next = self.regions.partition_point(|r| r.starts[0] < range.start);
        let mut c = range.start;
        while c < range.end {
            if let Some(r) = self.regions.get(next).filter(|r| r.starts[0] == c) {
                next += 1;
                if r.hi <= range.end {
                    let _s = rsprobe::scope!(TRACE, "region");
                    let piece = |k: usize| {
                        let end = r.starts.get(k + 1).copied().unwrap_or(r.hi);
                        (r.starts[k]..end).for_each(chunk);
                    };
                    rsdag::parallel::run_rated(
                        r.starts.len(),
                        r.ops as f64,
                        &NATIVE_REGION_OP_RATE,
                        &piece,
                    );
                    c = r.hi;
                    continue;
                }
            }
            chunk(c);
            c += 1;
        }
    }

    /// The bundles the calls call, by index, each as this backend runs it;
    /// mirrors [`Tape::bundles`].
    pub fn bundles(&self) -> &[Arc<dyn ExternBundle>] {
        &self.bundles
    }

    fn eval_range(
        &self,
        range: std::ops::Range<usize>,
        inputs: &[f64],
        work: &mut [f64],
        out: Option<&mut [f64]>,
    ) {
        self.padded(inputs, |inputs| {
            self.run(range, inputs, work);
            if let Some(out) = out {
                assert_eq!(
                    out.len(),
                    self.outputs.len(),
                    "output buffer of the wrong size"
                );
                rsdag::tape::write_outputs(&self.outputs, inputs, work, out);
            }
        })
    }

    /// Evaluate many instances in parallel: `inputs` holds `n` input
    /// vectors of `stride` values back to back (NaN-padded when shorter
    /// than the program's), `out` receives the `n` output vectors back to
    /// back. Instances share nothing, so ranges of them run on as many
    /// threads as their ops are worth, of the lent pool or else the current
    /// rayon pool (the caller's `install`, else the global one), each over
    /// a work buffer of its thread's; this is what a batch of identical
    /// devices, a parameter sweep or an ensemble amounts to. The serial
    /// form is [`Program::eval_many_into`](rsdag::Program::eval_many_into).
    pub fn eval_many(&self, inputs: &[f64], stride: usize, out: &mut Vec<f64>) {
        let n = inputs.len().checked_div(stride).unwrap_or(0);
        let n_out = self.outputs.len();
        out.clear();
        out.resize(n * n_out, 0.0);
        if n == 0 || n_out == 0 {
            return;
        }
        let n_in = self.layout.n_inputs.max(stride);
        let work = rspool::Work::of((n * self.n_ops) as f64, &NATIVE_REGION_OP_RATE);
        rspool::chunks_mut::<rspool::rayon::Rayon, _>(out, n_out, work, |i, dst| {
            let ins = &inputs[i * stride..(i + dst.len() / n_out) * stride];
            rsdag::scratch::with::<Vec<f64>, _>(|work| {
                work.resize(self.layout.total, 0.0);
                if n_in == stride {
                    self.eval_many_into(ins, stride, work, dst);
                    return;
                }
                let mut padded = vec![f64::NAN; ins.len() / stride * n_in];
                for (p, i) in padded.chunks_exact_mut(n_in).zip(ins.chunks_exact(stride)) {
                    p[..stride].copy_from_slice(i);
                }
                self.eval_many_into(&padded, n_in, work, dst);
            });
        });
    }
}

struct Emitter<'a, I: Isa> {
    isa: I,
    tape: &'a Tape,
    bundles: &'a Bundles,
    layout: Layout,
    live: &'a Liveness,
    pos: u32,
    next_host_call: u32,
    slot_of_cache: Vec<Option<u32>>,
    dirty: Vec<bool>,
    death: Vec<u32>,
    cache_of_slot: FxHashMap<u32, usize>,
    cache_of_reg: [u8; FLOAT_REG_NUMBERS],
    pinned: Vec<bool>,
    reads_left_in_op: FxHashMap<u32, u32>,
    calls: Vec<Call<'static>>,
    kernels: Vec<host::KernelDesc>,
    tables: Vec<Box<[u32]>>,
}

impl<'a, I: Isa> Emitter<'a, I> {
    fn new(
        tape: &'a Tape,
        bundles: &'a Bundles,
        layout: Layout,
        live: &'a Liveness,
        hot: &[*const ()],
    ) -> Emitter<'a, I> {
        let mut index = [0u8; FLOAT_REG_NUMBERS];
        for (i, &r) in I::CACHE.iter().enumerate() {
            index[r as usize] = i as u8;
        }
        Emitter {
            isa: I::new(hot),
            tape,
            bundles,
            layout,
            live,
            pos: 0,
            next_host_call: u32::MAX,
            slot_of_cache: vec![None; I::CACHE.len()],
            dirty: vec![false; I::CACHE.len()],
            death: vec![0; I::CACHE.len()],
            cache_of_slot: Default::default(),
            cache_of_reg: index,
            pinned: vec![false; I::CACHE.len()],
            reads_left_in_op: FxHashMap::default(),
            calls: Vec::new(),
            kernels: Vec::new(),
            tables: Vec::new(),
        }
    }

    fn drop_index(&mut self, i: usize) {
        if let Some(s) = self.slot_of_cache[i].take() {
            let read_by_current_op_or_later = self.death[i] >= self.pos;
            if self.dirty[i] && read_by_current_op_or_later {
                self.isa.store(I::CACHE[i], Base::Work, Self::slot_off(s));
            }
            self.dirty[i] = false;
            self.cache_of_slot.remove(&s);
        }
    }
    fn flush(&mut self) {
        for i in 0..I::CACHE.len() {
            self.drop_index(i);
        }
    }
    fn fresh(&mut self) -> u8 {
        self.fresh_in(I::SAVED..I::CACHE.len())
    }
    fn fresh_for(&mut self, slot: u32) -> u8 {
        self.fresh_until(self.live.death(slot, self.pos, true))
    }
    // https://github.com/milanofthe/rsdag/issues/240
    fn fresh_until(&mut self, death: u32) -> u8 {
        let host_call_before_death = I::SAVED > 0 && self.next_host_call <= death;
        if host_call_before_death {
            self.fresh_in(0..I::SAVED)
        } else {
            self.fresh_in(I::SAVED..I::CACHE.len())
        }
    }
    fn fresh_in(&mut self, pool: std::ops::Range<usize>) -> u8 {
        let n = I::CACHE.len();
        let other = if pool.start == 0 {
            I::SAVED..n
        } else {
            0..I::SAVED
        };
        let take = |e: &mut Self, i: usize| {
            e.drop_index(i);
            e.pinned[i] = true;
            I::CACHE[i]
        };
        for i in pool.clone().chain(other) {
            let free = match self.slot_of_cache[i] {
                None => true,
                Some(s) => {
                    let input_reloads_from_inputs = input_index(s).is_some();
                    input_reloads_from_inputs || self.death[i] < self.pos
                }
            };
            if !self.pinned[i] && free {
                return take(self, i);
            }
        }
        let belady_victim = |e: &Self, r: std::ops::Range<usize>| {
            r.filter(|&i| !e.pinned[i]).max_by_key(|&i| {
                let next = e.slot_of_cache[i].map_or(u32::MAX, |s| e.live.next_read(s, e.pos));
                let clean_saves_a_store = !e.dirty[i];
                (next, clean_saves_a_store)
            })
        };
        match belady_victim(self, pool).or_else(|| belady_victim(self, 0..n)) {
            Some(i) => take(self, i),
            None => unreachable!("an op pins fewer than {n} registers"),
        }
    }
    fn pin(&mut self, r: u8) {
        self.pinned[self.cache_of_reg[r as usize] as usize] = true;
    }
    fn release_except(&mut self, keep: &[u8]) {
        self.pinned.iter_mut().for_each(|p| *p = false);
        keep.iter().for_each(|&r| self.pin(r));
    }
    fn slot_off(s: u32) -> usize {
        s as usize * F64_BYTES * I::LANES
    }
    fn get(&mut self, slot: u32) -> u8 {
        let r = match (self.cache_of_slot.get(&slot), input_index(slot)) {
            (Some(&i), _) => {
                self.pinned[i] = true;
                I::CACHE[i]
            }
            (None, Some(k)) => {
                let r = self.fresh();
                self.isa.load(r, Base::Inputs, Self::slot_off(k));
                self.bind(r, slot, 0);
                r
            }
            (None, None) => {
                let death = self.live.death(slot, self.pos, false);
                let r = self.fresh_until(death);
                self.isa.load(r, Base::Work, Self::slot_off(slot));
                self.bind(r, slot, death);
                r
            }
        };
        self.read_one(slot);
        r
    }
    fn read_one(&mut self, slot: u32) {
        if self.reads_left_in_op.is_empty() {
            return;
        }
        if let Some(n) = self.reads_left_in_op.get_mut(&slot) {
            *n -= 1;
            if *n == 0 {
                self.reads_left_in_op.remove(&slot);
                if let Some(i) = self.cache_of_slot.remove(&slot) {
                    self.slot_of_cache[i] = None;
                    self.dirty[i] = false;
                }
            }
        }
    }
    fn invalidate(&mut self, dst: u32, n: u32) {
        for s in dst..dst + n {
            if let Some(i) = self.cache_of_slot.remove(&s) {
                self.slot_of_cache[i] = None;
                self.dirty[i] = false;
            }
        }
    }
    fn bind(&mut self, r: u8, slot: u32, death: u32) {
        if let Some(i) = self.cache_of_slot.remove(&slot) {
            self.slot_of_cache[i] = None;
            self.dirty[i] = false;
        }
        let i = self.cache_of_reg[r as usize] as usize;
        self.drop_index(i);
        self.slot_of_cache[i] = Some(slot);
        self.death[i] = death;
        self.cache_of_slot.insert(slot, i);
    }
    fn put(&mut self, slot: u32, r: u8) {
        let death = self.live.death(slot, self.pos, true);
        self.bind(r, slot, death);
        self.dirty[self.cache_of_reg[r as usize] as usize] = true;
    }
    fn fconst(&mut self, v: f64) -> u8 {
        let r = self.fresh();
        self.isa.fconst(r, v);
        r
    }
    fn call(&mut self, addr: *const (), args: &[Arg]) {
        for i in I::SAVED..I::CACHE.len() {
            self.drop_index(i);
        }
        self.isa.call(addr, args);
    }
    fn call_into(&mut self, dst: u32, addr: *const (), args: &[Arg]) {
        if I::LANES == 1 {
            self.call(addr, args);
            let r = self.fresh_for(dst);
            self.isa.mov(r, I::RESULT);
            self.put(dst, r);
            return;
        }
        let l = I::LANES;
        let base = self.layout.gather_start * F64_BYTES;
        let mut nf = 0;
        for a in args {
            if let Arg::F(r) = *a {
                self.isa.store(r, Base::Work, base + nf * l * F64_BYTES);
                nf += 1;
            }
        }
        let res = base + nf * l * F64_BYTES;
        self.call_lanes(dst, addr, res, |e, lane| {
            let mut j = 0;
            args.iter()
                .map(|a| match *a {
                    Arg::F(_) => {
                        let r = e.fresh();
                        e.isa
                            .load_lane(r, Base::Work, base + (j * l + lane) * F64_BYTES);
                        j += 1;
                        Arg::F(r)
                    }
                    Arg::I(i) => Arg::I(i),
                })
                .collect()
        });
    }
    fn call_lanes(
        &mut self,
        dst: u32,
        addr: *const (),
        res: usize,
        mut lane_args: impl FnMut(&mut Self, usize) -> Vec<Arg>,
    ) {
        for lane in 0..I::LANES {
            self.release_except(&[]);
            let args = lane_args(self, lane);
            self.call(addr, &args);
            self.isa
                .store_lane(I::RESULT, Base::Work, res + lane * F64_BYTES);
        }
        self.release_except(&[]);
        let r = self.fresh_for(dst);
        self.isa.load(r, Base::Work, res);
        self.put(dst, r);
    }
    fn gather(&mut self, slots: &[u32]) -> usize {
        self.gather_at(slots, None, 0)
    }
    fn gather_at(&mut self, slots: &[u32], places: Option<&[u32]>, at: usize) -> usize {
        debug_assert_eq!(I::LANES, 1, "lane code gathers lane by lane");
        let base = (self.layout.gather_start + at) * F64_BYTES;
        for (k, &s) in slots.iter().enumerate() {
            let r = self.get(s);
            let to = places.map_or(k, |p| p[k] as usize);
            self.isa.store(r, Base::Work, base + to * F64_BYTES);
            self.release_except(&[]);
        }
        base
    }

    fn publish(&mut self, slots: &[u32]) {
        for s in slots {
            if let Some(&i) = self.cache_of_slot.get(s) {
                if self.dirty[i] {
                    self.isa.store(I::CACHE[i], Base::Work, Self::slot_off(*s));
                    self.dirty[i] = false;
                }
            }
        }
    }

    fn dense_arg(&mut self, o: Operand, len: u32, at: &mut usize) -> host::Place {
        let (base, off) = match o {
            Operand::Inputs(k) => (host::Place::INPUTS, k as usize * F64_BYTES),
            Operand::Run(s) => {
                for slot in s..s + len {
                    if let Some(&i) = self.cache_of_slot.get(&slot) {
                        self.drop_index(i);
                    }
                }
                (host::Place::WORK, s as usize * F64_BYTES)
            }
            Operand::Slots(v) => {
                let p = self.gather_at(v, None, *at);
                *at += v.len();
                (host::Place::WORK, p)
            }
        };
        host::Place {
            base,
            byte_offset: off as u64,
        }
    }

    fn keep(&mut self, v: &[u32]) -> &'static [u32] {
        let t: Box<[u32]> = v.into();
        // SAFETY: the box goes with the code (`NativeTape::_tables`), and
        // its contents never move.
        let r = unsafe { std::slice::from_raw_parts(t.as_ptr(), t.len()) };
        self.tables.push(t);
        r
    }

    fn op(&mut self, i: usize) {
        self.op_inner(i);
        self.release_except(&[]);
    }

    fn op_inner(&mut self, i: usize) {
        let t = self.tape;
        let (op, dst) = (t.ops()[i], t.dst(i));
        match op {
            Op::Const(v) => {
                let r = self.fconst(v);
                self.put(dst, r);
            }
            Op::Add(a, b) => self.bin2(Arith::Add, dst, a, b),
            Op::Sub(a, b) => self.bin2(Arith::Sub, dst, a, b),
            Op::Mul(a, b) => self.bin2(Arith::Mul, dst, a, b),
            Op::MulAdd(a, b, c) => {
                let x = self.get(a);
                let m = self.fresh();
                self.arith_rm(Arith::Mul, m, x, b);
                let r = self.fresh_for(dst);
                self.arith_rm(Arith::Add, r, m, c);
                self.put(dst, r);
            }
            Op::Neg(a) => self.neg(dst, a),
            Op::Powi(a, n) => match (n, powi_inline(n)) {
                (-1, _) => {
                    let x = self.get(a);
                    let one = self.fconst(1.0);
                    let r = self.fresh_for(dst);
                    self.isa.arith(Arith::Div, r, one, x);
                    self.put(dst, r);
                }
                (2, _) => self.bin2(Arith::Mul, dst, a, a),
                (_, Some(magnitude)) => {
                    let mut base = self.get(a);
                    let low = magnitude.trailing_zeros();
                    for _ in 0..low {
                        let r = self.fresh();
                        self.isa.arith(Arith::Mul, r, base, base);
                        base = r;
                        self.release_except(&[base]);
                    }
                    let mut acc = base;
                    let mut e = magnitude.get() >> low;
                    loop {
                        e >>= 1;
                        if e > 0 {
                            let r = self.fresh();
                            self.isa.arith(Arith::Mul, r, base, base);
                            base = r;
                        }
                        self.release_except(&[base, acc]);
                        if e == 0 {
                            break;
                        }
                        if e & 1 == 1 {
                            let r = self.fresh();
                            self.isa.arith(Arith::Mul, r, acc, base);
                            acc = r;
                        }
                    }
                    let p = acc;
                    let r = self.fresh_for(dst);
                    if n < 0 {
                        let one = self.fconst(1.0);
                        self.isa.arith(Arith::Div, r, one, p);
                    } else {
                        self.isa.mov(r, p);
                    }
                    self.put(dst, r);
                }
                (_, None) => {
                    let x = self.get(a);
                    self.call_into(
                        dst,
                        host::h_powi as *const (),
                        &[Arg::F(x), Arg::I(IArg::Imm(n as i64 as u64))],
                    );
                }
            },
            Op::Unary(uop, a) => self.unary(dst, uop, a),
            Op::Binary(bop, a, b) => {
                let (x, y) = (self.get(a), self.get(b));
                let code = Arg::I(IArg::Imm(BinOp::code(bop) as u64));
                self.call_into(
                    dst,
                    host::h_binary as *const (),
                    &[code, Arg::F(x), Arg::F(y)],
                );
            }
            Op::Cmp(cop, a, b) => {
                let (x, y) = (self.get(a), self.get(b));
                let one = self.fconst(1.0);
                let zero = self.fconst(0.0);
                let r = self.fresh_for(dst);
                self.isa.cmp_select(cop, x, y, one, zero, r);
                self.put(dst, r);
            }
            Op::Select(c, a, b) => {
                let (cv, tv, ev) = (self.get(c), self.get(a), self.get(b));
                let r = self.fresh_for(dst);
                self.isa.select_nz(cv, tv, ev, r);
                self.put(dst, r);
            }
            Op::Reduce(rop, start, len) => {
                let inline = minmax_inline::<I>(&op);
                self.reduce(dst, rop, t.pool(start, len), inline)
            }
            Op::Dot(start, len) => {
                let r = self.fold(
                    Arith::Add,
                    0.0,
                    t.pool(start, len),
                    Some(t.pool(start + len, len)),
                );
                self.put(dst, r);
            }
            Op::Call { bundle, .. } | Op::CallProlog { bundle, .. } => self.call_op(i, bundle),
            Op::Gemv { a, x, m, n, acc } if inline_product(&op) => {
                self.product(dst, a, x, (m, n, 1), acc)
            }
            Op::Gemm { a, b, m, k, n, acc } if inline_product(&op) => {
                self.product(dst, a, b, (m, k, n), acc)
            }
            Op::Gemv { a, x, m, n, acc } => self.kernel(
                i,
                host::KernelDesc::GEMV,
                (m, 0, n),
                &[(a, m * n), (x, n)],
                acc,
            ),
            Op::Gemm { a, b, m, k, n, acc } => self.kernel(
                i,
                host::KernelDesc::GEMM,
                (m, k, n),
                &[(a, m * k), (b, n * k)],
                acc,
            ),
            Op::Solve { a, b, n, k, count } => {
                let srcs = [(a, count * n * n), (b, count * n * k)];
                self.kernel(
                    i,
                    host::KernelDesc::SOLVE_M_SYSTEMS_K_RHS,
                    (count, k, n),
                    &srcs,
                    None,
                )
            }
        }
    }

    fn reduce(&mut self, dst: u32, rop: ReduceOp, args: &[u32], inline: bool) {
        match rop {
            ReduceOp::Sum => {
                let r = self.fold(Arith::Add, 0.0, args, None);
                self.put(dst, r);
            }
            ReduceOp::Product => {
                let r = self.fold(Arith::Mul, 1.0, args, None);
                self.put(dst, r);
            }
            ReduceOp::Min | ReduceOp::Max if inline => {
                let first = self.get(args[0]);
                let mut acc = self.fresh();
                self.isa.mov(acc, first);
                for &k in &args[1..] {
                    let t = self.get(k);
                    let r = self.fresh();
                    self.isa.minmax(rop, r, acc, t);
                    acc = r;
                    self.release_except(&[acc]);
                }
                self.put(dst, acc);
            }
            ReduceOp::Min | ReduceOp::Max if I::LANES > 1 => {
                let (l, m) = (I::LANES, args.len());
                let base = self.layout.gather_start * F64_BYTES;
                for (j, &a) in args.iter().enumerate() {
                    let r = self.get(a);
                    self.isa.store(r, Base::Work, base + j * l * F64_BYTES);
                    self.release_except(&[]);
                }
                let (one_lane_terms, results) =
                    (base + m * l * F64_BYTES, base + (m * l + m) * F64_BYTES);
                self.call_lanes(dst, host::h_reduce as *const (), results, |e, lane| {
                    for j in 0..m {
                        let t = e.fresh();
                        e.isa
                            .load_lane(t, Base::Work, base + (j * l + lane) * F64_BYTES);
                        e.isa
                            .store_lane(t, Base::Work, one_lane_terms + j * F64_BYTES);
                        e.release_except(&[]);
                    }
                    vec![
                        Arg::I(IArg::Imm(host::reduce_code(rop))),
                        Arg::I(IArg::WorkAddr(one_lane_terms)),
                        Arg::I(IArg::Imm(m as u64)),
                    ]
                });
            }
            ReduceOp::Min | ReduceOp::Max => {
                let at = self.gather(args);
                let args = [
                    Arg::I(IArg::Imm(host::reduce_code(rop))),
                    Arg::I(IArg::WorkAddr(at)),
                    Arg::I(IArg::Imm(args.len() as u64)),
                ];
                self.call_into(dst, host::h_reduce as *const (), &args);
            }
        }
    }

    fn call_op(&mut self, i: usize, bundle: u32) {
        let t = self.tape;
        let c = t.call(i);
        let operands = c.operands.expect("a call of the tape names its operands");
        let host_gathers_from_table = operands.len() >= HOST_GATHER_MIN_ARGS;
        if host_gathers_from_table {
            self.publish(operands);
        } else {
            let arg_index_of_operand: Option<Vec<u32>> = c.places.map(|p| {
                (0..operands.len())
                    .map(|k| ((k / p.len()) * c.n_args) as u32 + p[k % p.len()])
                    .collect()
            });
            self.gather_at(operands, arg_index_of_operand.as_deref(), c.args);
        }
        // SAFETY: the bundles go with the code (`NativeTape::bundles`), and
        // what an `Arc` holds never moves.
        let b: &'static dyn ExternBundle =
            unsafe { &*(&*self.bundles[bundle as usize] as *const dyn ExternBundle) };
        let call = Call {
            bundle: b,
            phase: c.phase,
            n_groups: c.n_groups,
            n_args: c.n_args,
            n_out: c.n_out,
            operands: host_gathers_from_table.then(|| self.keep(operands)),
            places: c
                .places
                .filter(|_| host_gathers_from_table)
                .map(|p| self.keep(p)),
            own: c.own.map(|o| self.keep(o)),
            args: c.args,
            out: c.out,
            state: c.state,
            ops: c.ops,
        };
        let d = self.calls.len();
        assert!(d < self.calls.capacity(), "calls counted before emission");
        self.calls.push(call);
        let at = |e: &Self, k: usize| Arg::I(IArg::Imm(&e.calls[k] as *const Call as u64));
        let (work, inputs) = (Arg::I(IArg::WorkAddr(0)), Arg::I(IArg::InputAddr(0)));
        match stage_of(t, i) {
            None => {
                let args = [Arg::I(IArg::Layout), at(self, d), work, inputs];
                self.call(host::h_call as *const (), &args);
                self.invalidate(t.dst(i), t.width(i));
            }
            Some(st) if i + 1 < st.hi as usize => {}
            Some(st) => {
                let n = (st.hi - st.lo) as usize;
                let first = at(self, d + 1 - n);
                let args = [
                    Arg::I(IArg::Layout),
                    first,
                    Arg::I(IArg::Imm(n as u64)),
                    work,
                    inputs,
                ];
                self.call(host::h_stage as *const (), &args);
                for j in st.lo as usize..st.hi as usize {
                    self.invalidate(t.dst(j), t.width(j));
                }
            }
        }
    }

    fn kernel(
        &mut self,
        i: usize,
        kind: u64,
        (m, k, n): (u32, u32, u32),
        srcs: &[(Src, u32)],
        acc: Option<Accum>,
    ) {
        let t = self.tape;
        let (dst, width) = (t.dst(i), t.width(i));
        let c = acc.and_then(|f| f.c).map(|c| (c, width));
        let mut at = 0usize;
        let mut operands = [host::Place::NONE; KERNEL_OPERANDS_A_XB_ACCUMULATOR];
        for (p, &(src, len)) in operands.iter_mut().zip(srcs.iter().chain(&c)) {
            *p = self.dense_arg(t.operand(src, len), len, &mut at);
        }
        let codes = acc.map_or(0, |f| self.keep(t.pool(f.codes, width)).as_ptr() as u64);
        let desc = host::KernelDesc {
            kind,
            m: m as u64,
            k: k as u64,
            n: n as u64,
            operands_a_xb_accumulator: operands,
            fold_codes_addr_or_zero: codes,
            out_byte_offset: dst as u64 * F64_BYTES as u64,
        };
        let d = self.kernels.len();
        assert!(
            d < self.kernels.capacity(),
            "kernel descriptors counted before emission"
        );
        self.kernels.push(desc);
        let ptr = &self.kernels[d] as *const host::KernelDesc as u64;
        let args = [
            Arg::I(IArg::Layout),
            Arg::I(IArg::Imm(ptr)),
            Arg::I(IArg::WorkAddr(0)),
            Arg::I(IArg::InputAddr(0)),
        ];
        self.call(host::h_kernel as *const (), &args);
        self.invalidate(dst, width);
    }

    fn product(
        &mut self,
        dst: u32,
        a: Src,
        b: Src,
        (m, k, n): (u32, u32, u32),
        acc: Option<Accum>,
    ) {
        let t = self.tape;
        let width = m * n;
        let (a, b) = (t.operand(a, m * k), t.operand(b, n * k));
        let c = acc.and_then(|f| f.c).map(|c| t.operand(c, width));
        let codes = acc.map_or(&[][..], |f| t.pool(f.codes, width));
        let elem = |o: Operand, e: u32| match o {
            Operand::Inputs(i) => (i + e) | INPUT,
            Operand::Run(s) => s + e,
            Operand::Slots(v) => v[e as usize],
        };
        let mut reads: Vec<(u32, u32)> = (0..m * k).map(|e| (elem(a, e), n)).collect();
        reads.extend((0..n * k).map(|e| (elem(b, e), m)));
        if let Some(c) = c {
            let folds = codes.iter().enumerate();
            reads.extend(
                folds
                    .filter(|(_, &f)| Fold(f).reads_operand())
                    .map(|(e, _)| (elem(c, e as u32), 1)),
            );
        }
        for (s, n) in reads {
            if input_index(s).is_none() && self.live.death(s, self.pos, false) == self.pos {
                *self.reads_left_in_op.entry(s).or_default() += n;
            }
        }
        for e in 0..width {
            let (i, j) = (e / n, e % n);
            let row: Vec<u32> = (0..k).map(|l| elem(a, i * k + l)).collect();
            let col: Vec<u32> = (0..k).map(|l| elem(b, j * k + l)).collect();
            let r = self.fold(Arith::Add, 0.0, &row, Some(&col));
            self.put(dst + e, r);
            self.release_except(&[]);
        }
        for (e, &code) in codes.iter().enumerate() {
            let (f, d) = (Fold(code), dst + e as u32);
            let c = if f.is_self() {
                dst + f.self_index() as u32
            } else {
                c.map_or(d, |c| elem(c, e as u32))
            };
            match code & Fold::KIND {
                0 => {}
                1 => self.bin2(Arith::Sub, d, c, d),
                2 => self.bin2(Arith::Add, d, c, d),
                _ => self.neg(d, d),
            }
            self.release_except(&[]);
        }
        debug_assert!(
            self.reads_left_in_op.is_empty(),
            "every read of a product counted"
        );
    }

    fn neg(&mut self, dst: u32, a: u32) {
        let x = self.get(a);
        let r = self.fresh_for(dst);
        self.isa.neg(r, x);
        self.put(dst, r);
    }

    fn bin2(&mut self, op: Arith, dst: u32, a: u32, b: u32) {
        let x = self.get(a);
        let r = self.fresh_for(dst);
        self.arith_rm(op, r, x, b);
        self.put(dst, r);
    }

    fn arith_rm(&mut self, op: Arith, d: u8, a: u8, slot: u32) {
        if I::MEM_OPERANDS
            && input_index(slot).is_none()
            && !self.cache_of_slot.contains_key(&slot)
            && self.live.death(slot, self.pos, false) == self.pos
        {
            self.isa
                .arith_mem(op, d, a, Base::Work, Self::slot_off(slot));
            self.read_one(slot);
            return;
        }
        let b = self.get(slot);
        self.isa.arith(op, d, a, b);
    }

    fn unary(&mut self, dst: u32, uop: UnaryOp, a: u32) {
        let x = self.get(a);
        let inline = match uop {
            UnaryOp::Sqrt => {
                let r = self.fresh_for(dst);
                let zero = self.fconst(0.0);
                let s = self.fresh();
                self.isa.sqrt(s, x);
                self.isa.cmp_select(CmpOp::Le, x, zero, zero, s, r);
                Some(r)
            }
            UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Trunc if I::round_inline() => {
                let mode = match uop {
                    UnaryOp::Floor => Round::Floor,
                    UnaryOp::Ceil => Round::Ceil,
                    _ => Round::Trunc,
                };
                let r = self.fresh_for(dst);
                self.isa.round(mode, r, x);
                Some(r)
            }
            UnaryOp::Abs => {
                let r = self.fresh_for(dst);
                self.isa.abs(r, x);
                Some(r)
            }
            UnaryOp::Exp | UnaryOp::Ln => {
                let r = self.fresh_for(dst);
                let t = std::array::from_fn(|_| self.fresh());
                self.isa.exp_ln(uop == UnaryOp::Ln, r, x, t);
                Some(r)
            }
            UnaryOp::Sign => {
                let r = self.fresh_for(dst);
                let zero = self.fconst(0.0);
                let one = self.fconst(1.0);
                let minus = self.fconst(-1.0);
                let t = self.fresh();
                self.isa.cmp_select(CmpOp::Lt, x, zero, minus, x, t);
                self.isa.cmp_select(CmpOp::Gt, x, zero, one, t, r);
                Some(r)
            }
            _ => None,
        };
        match inline {
            Some(r) => self.put(dst, r),
            None => {
                let (addr, code) = host::unary_addr(uop);
                let args: Vec<Arg> = code
                    .into_iter()
                    .map(|c| Arg::I(IArg::Imm(c as u64)))
                    .chain([Arg::F(x)])
                    .collect();
                self.call_into(dst, addr, &args);
            }
        }
    }

    // https://github.com/milanofthe/rsdag/issues/240
    fn fold(&mut self, op: Arith, ident: f64, a: &[u32], b: Option<&[u32]>) -> u8 {
        let n = a.len();
        let ch = n / FOLD_LANES;
        let mut s = if ch == 0 {
            self.fconst(ident)
        } else {
            let acc: [u8; FOLD_LANES] = std::array::from_fn(|_| self.fconst(ident));
            for c in 0..ch {
                for (k, &ak) in acc.iter().enumerate() {
                    self.term(op, ak, ak, a, b, FOLD_LANES * c + k);
                    self.release_except(&acc);
                }
            }
            let [a0, a1, a2, a3] = acc;
            let l = self.fresh();
            self.isa.arith(op, l, a0, a1);
            let r = self.fresh();
            self.isa.arith(op, r, a2, a3);
            let s = self.fresh();
            self.isa.arith(op, s, l, r);
            s
        };
        for k in ch * FOLD_LANES..n {
            let s2 = self.fresh();
            self.term(op, s2, s, a, b, k);
            s = s2;
            self.release_except(&[s]);
        }
        s
    }

    fn term(&mut self, op: Arith, d: u8, acc: u8, a: &[u32], b: Option<&[u32]>, k: usize) {
        match b {
            None => self.arith_rm(op, d, acc, a[k]),
            Some(bb) => {
                let x = self.get(a[k]);
                let p = self.fresh();
                self.arith_rm(Arith::Mul, p, x, bb[k]);
                self.isa.arith(op, d, acc, p);
            }
        }
    }
}

fn hot_routines_most_frequent_first<I: Isa>(ops: &[Op]) -> Vec<*const ()> {
    const MIN_CALLS_WORTH_A_PROLOGUE_LOAD: usize = 2;
    let mut count: Vec<(*const (), usize)> = Vec::new();
    for h in ops.iter().filter_map(routine::<I>) {
        match count.iter_mut().find(|(a, _)| *a == h) {
            Some(c) => c.1 += 1,
            None => count.push((h, 1)),
        }
    }
    count.retain(|&(_, n)| n >= MIN_CALLS_WORTH_A_PROLOGUE_LOAD);
    count.sort_by_key(|a| std::cmp::Reverse(a.1));
    count.into_iter().map(|(a, _)| a).collect()
}

fn emit_chunk<A: Isa>(
    tape: &Tape,
    range: std::ops::Range<usize>,
    layout: &Layout,
    live: &Liveness,
    bundles: &Bundles,
) -> Result<Emitted, JitError> {
    let ops = &tape.ops()[range.clone()];
    let hot = hot_routines_most_frequent_first::<A>(ops);
    let mut next_host_call_at_or_after = vec![u32::MAX; ops.len() + 1];
    for k in (0..ops.len()).rev() {
        next_host_call_at_or_after[k] = match routine::<A>(&ops[k]) {
            Some(_) => (range.start + k) as u32,
            None => next_host_call_at_or_after[k + 1],
        };
    }
    let mut e: Emitter<A> = Emitter::new(tape, bundles, *layout, live, &hot);
    let count = |f: fn(&Op) -> bool| ops.iter().filter(|op| f(op)).count();
    e.calls = Vec::with_capacity(count(|op| {
        matches!(op, Op::Call { .. } | Op::CallProlog { .. })
    }));
    e.kernels = Vec::with_capacity(count(|op| {
        matches!(op, Op::Gemv { .. } | Op::Gemm { .. } | Op::Solve { .. }) && !inline_product(op)
    }));
    e.isa.prologue();
    for (k, i) in range.clone().enumerate() {
        e.pos = i as u32;
        e.next_host_call = next_host_call_at_or_after[k];
        e.op(i);
    }
    // https://github.com/milanofthe/rsdag/issues/240
    e.pos = range.end as u32;
    e.flush();
    e.isa.epilogue();
    Ok(Emitted {
        calls: std::mem::take(&mut e.calls),
        kernels: std::mem::take(&mut e.kernels),
        tables: std::mem::take(&mut e.tables),
        bytes: e.isa.finish()?,
    })
}

struct Mapping {
    ptr: *mut u8,
    #[cfg(unix)]
    len: usize,
}

#[cfg(unix)]
impl Mapping {
    fn new(bytes: &[u8]) -> Result<Mapping, JitError> {
        let len = bytes.len().max(1);
        unsafe {
            // W^X per thread under `MAP_JIT`: https://developer.apple.com/documentation/apple-silicon/porting-just-in-time-compilers-to-apple-silicon
            #[cfg(target_os = "macos")]
            let (flags, prot) = (
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            );
            #[cfg(not(target_os = "macos"))]
            let (flags, prot) = (
                libc::MAP_PRIVATE | libc::MAP_ANON,
                libc::PROT_READ | libc::PROT_WRITE,
            );
            let ptr = libc::mmap(std::ptr::null_mut(), len, prot, flags, -1, 0);
            if ptr == libc::MAP_FAILED {
                return Err(JitError::Codegen("mmap of executable memory failed".into()));
            }
            let ptr = ptr as *mut u8;
            let map = Mapping { ptr, len };
            #[cfg(target_os = "macos")]
            pthread_jit_write_protect_np(0);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            #[cfg(target_os = "macos")]
            {
                pthread_jit_write_protect_np(1);
                sys_icache_invalidate(ptr as *mut libc::c_void, bytes.len());
            }
            #[cfg(not(target_os = "macos"))]
            {
                let rx = libc::PROT_READ | libc::PROT_EXEC;
                if libc::mprotect(ptr as *mut libc::c_void, len, rx) != 0 {
                    return Err(JitError::Codegen(
                        "mprotect of the code to read and execute failed".into(),
                    ));
                }
                __clear_cache(
                    ptr as *mut libc::c_char,
                    ptr.add(bytes.len()) as *mut libc::c_char,
                );
            }
            Ok(map)
        }
    }
}

#[cfg(unix)]
impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_jit_write_protect_np(enabled: libc::c_int);
    fn sys_icache_invalidate(start: *mut libc::c_void, len: libc::size_t);
}
#[cfg(all(unix, not(target_os = "macos")))]
extern "C" {
    fn __clear_cache(start: *mut libc::c_char, end: *mut libc::c_char);
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn VirtualAlloc(addr: *mut u8, size: usize, kind: u32, protect: u32) -> *mut u8;
    fn VirtualFree(addr: *mut u8, size: usize, kind: u32) -> i32;
    fn VirtualProtect(addr: *mut u8, size: usize, protect: u32, old: *mut u32) -> i32;
    fn GetCurrentProcess() -> isize;
    fn FlushInstructionCache(process: isize, addr: *const u8, size: usize) -> i32;
}

#[cfg(windows)]
impl Mapping {
    fn new(bytes: &[u8]) -> Result<Mapping, JitError> {
        const MEM_COMMIT_RESERVE: u32 = 0x1000 | 0x2000;
        const PAGE_READWRITE: u32 = 0x04;
        const PAGE_EXECUTE_READ: u32 = 0x20;
        let len = bytes.len().max(1);
        unsafe {
            let ptr = VirtualAlloc(
                std::ptr::null_mut(),
                len,
                MEM_COMMIT_RESERVE,
                PAGE_READWRITE,
            );
            if ptr.is_null() {
                return Err(JitError::Codegen(
                    "VirtualAlloc of executable memory failed".into(),
                ));
            }
            let map = Mapping { ptr };
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            let mut old = 0u32;
            if VirtualProtect(ptr, len, PAGE_EXECUTE_READ, &mut old) == 0 {
                return Err(JitError::Codegen(
                    "VirtualProtect of the code to read and execute failed".into(),
                ));
            }
            FlushInstructionCache(GetCurrentProcess(), ptr, bytes.len());
            Ok(map)
        }
    }
}

#[cfg(windows)]
impl Drop for Mapping {
    fn drop(&mut self) {
        const MEM_RELEASE: u32 = 0x8000;
        unsafe {
            VirtualFree(self.ptr, 0, MEM_RELEASE);
        }
    }
}

impl rsdag::Program for NativeTape {
    fn n_inputs(&self) -> usize {
        self.layout.n_inputs
    }
    fn work_len(&self) -> usize {
        self.layout.total
    }
    fn out_len(&self) -> usize {
        self.outputs.len()
    }
    fn state_len(&self) -> usize {
        self.state_len
    }
    fn eval_into(&self, inputs: &[f64], work: &mut [f64], out: &mut [f64]) {
        let _s = rsprobe::scope!(DEBUG, "eval");
        self.eval_range(0..self.chunks.len(), inputs, work, Some(out));
    }
    fn eval_prolog_into(&self, inputs: &[f64], work: &mut [f64]) {
        let _s = rsprobe::scope!(DEBUG, "prolog");
        self.eval_range(0..self.prolog_chunks, inputs, work, None);
    }
    fn eval_main_into(&self, inputs: &[f64], work: &mut [f64], out: &mut [f64]) {
        let _s = rsprobe::scope!(DEBUG, "main");
        self.eval_range(
            self.prolog_chunks..self.chunks.len(),
            inputs,
            work,
            Some(out),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_arch = "x86_64")]
    use crate::x86_64::X64;
    use rsdag::semantics::{exp_guarded, ln_guarded, EXP_LIMIT, LN_FLOOR};
    use rsdag::{Graph, Node, F64};

    const N: usize = 64;

    fn tape_exp_and_ln_of_each_input() -> Tape {
        let mut g: Graph<F64> = Graph::new();
        let (mut roots, mut syms) = (Vec::new(), Vec::new());
        for k in 0..N {
            let x = g.sym(&format!("x{k}"));
            let Node::Symbol(s) = *g.node(x) else {
                unreachable!()
            };
            roots.extend([g.exp(x), g.ln(x)]);
            syms.push(s);
        }
        Tape::compile(&g, &roots, &syms)
    }

    fn edge_values_and_random_bits() -> Vec<f64> {
        let mut v = vec![
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
            f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
            -745.2,
            -745.1,
            -745.3,
            EXP_LIMIT,
            EXP_LIMIT.next_up(),
            EXP_LIMIT.next_down(),
            LN_FLOOR,
            LN_FLOOR.next_up(),
            1.0,
            1.0f64.next_up(),
            1.0f64.next_down(),
        ];
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for k in 0..600_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            v.push(match k % 3 {
                0 => -800.0 + 900.0 * u,
                1 => f64::from_bits(s),
                _ => 0.5 + 1.5 * u,
            });
        }
        v
    }

    fn check<A: Isa>() {
        let t = tape_exp_and_ln_of_each_input();
        let code = NativeTape::compile_isa::<A>(&t, &Options::default(), &[]).expect("native code");
        let l = A::LANES;
        let mut work = vec![0.0; code.layout.total];
        let mut ins = vec![0.0; N * l];
        for block in edge_values_and_random_bits().chunks(N * l) {
            ins[..block.len()].copy_from_slice(block);
            code.run(0..code.chunks.len(), &ins, &mut work);
            for (i, &x) in block.iter().enumerate() {
                let (k, j) = (i / l, i % l);
                for (o, name, want) in [
                    (2 * k, "exp", exp_guarded(x)),
                    (2 * k + 1, "ln", ln_guarded(x)),
                ] {
                    let got = work[code.outputs[o] as usize * l + j];
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{l} lanes: {name}({x:e}) = {got:e}, the reference {want:e}"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn lane_code_stays_in_its_work_array() {
        let mut g: Graph<F64> = Graph::new();
        let x = g.sym("x");
        let Node::Symbol(s) = *g.node(x) else {
            unreachable!()
        };
        let roots = [g.floor(x)];
        let t = Tape::compile(&g, &roots, &[s]);
        let code = NativeTape::compile_isa::<X64<2>>(&t, &Options::default(), &[]).expect("lanes");
        let total = code.layout.total;
        let mark = f64::from_bits(0x7FF8_0000_DEAD_BEEF);
        let mut work = vec![mark; total + 8];
        code.run(0..code.chunks.len(), &[2.5, -1.5], &mut work);
        let o = code.outputs[0] as usize * 2;
        assert_eq!(work[o..o + 2], [2.0, -2.0]);
        assert!(
            work[total..].iter().all(|v| v.to_bits() == mark.to_bits()),
            "lane code wrote past its work array of {total}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exp_and_ln_inline_are_the_reference_bit_for_bit() {
        check::<X64<1>>();
        check::<X64<2>>();
        if crate::x86_64::features().avx2 {
            check::<X64<4>>();
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn exp_and_ln_inline_are_the_reference_bit_for_bit() {
        check::<crate::aarch64::A64>();
    }
}
