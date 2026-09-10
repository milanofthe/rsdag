//! Compiled flat evaluator (tape) for fast, repeated numeric evaluation.
//!
//! A [`Tape`] is the reachable sub-DAG of a set of root expressions, lowered to
//! a flat instruction list over compact slots (ascending `ExprId` is already a
//! topological order, since a hash-consed node has a larger id than its
//! children). Symbols become positional inputs (no per-call hashing), and the
//! work buffer is caller-owned and reused -- so a Newton loop over a large
//! circuit neither re-hashes symbols nor reallocates a huge scratch each step.

use std::sync::Arc;

use crate::extern_fn::ExternBundle;
use crate::node::{
    binary_f64, cmp_bool, dot_slice, reduce_slice, unary_f64, BinOp, CmpOp, ReduceOp, UnaryOp,
};

#[derive(Clone, Copy, Debug)]
enum Op {
    Const(f64),
    Input(u32),
    Add(u32, u32),
    Mul(u32, u32),
    /// `a*b + c` as one instruction -- a fused *dispatch*, not a fused
    /// *rounding*: the two IEEE operations are evaluated exactly as the
    /// separate `Mul` + `Add` would be (no `mul_add` contraction), so tape
    /// results stay bit-identical to the arena sweep. Emitted by `compile`
    /// when an `Add` consumes a single-use `Mul`.
    MulAdd(u32, u32, u32),
    /// `a - b`, from an `Add` consuming a single-use `Neg` (IEEE subtraction
    /// is exactly addition of the negation, so this is bit-preserving too).
    Sub(u32, u32),
    Neg(u32),
    Powi(u32, i32),
    Unary(UnaryOp, u32),
    Binary(BinOp, u32, u32),
    Cmp(CmpOp, u32, u32),
    Select(u32, u32, u32),
    /// Reduction over `arg_pool[start .. start+len]`.
    Reduce(ReduceOp, u32, u32),
    /// Inner product of `arg_pool[start .. start+len]` (a) and the `len` slots
    /// that follow it (b).
    Dot(u32, u32),
    /// Evaluate the bundle at `bundles[bidx]` once, gathering its arguments from
    /// `arg_pool[start .. start+len]`, and write all its outputs into the
    /// bundle-scratch region starting at `base`. Emitted once per distinct
    /// (bundle, argument) group; its work slot is a never-read sink.
    BundleCall(u32, u32, u32, u32),
    /// Evaluate the bundle at `bundles[bidx]` for ALL of its argument groups at
    /// once (instance batching): `batches[tidx]` describes the group-major
    /// argument table in `arg_pool` and the consecutive scratch regions the
    /// outputs land in. Emitted instead of the per-group `BundleCall`s when
    /// every group's arguments are plain inputs/constants (hoistable to the
    /// first call site); the backend may evaluate the groups as SIMD lanes.
    BundleBatch(u32, u32),
    /// Read one already-computed bundle output from `bundle_scratch[idx]`.
    BundlePick(u32),
}

/// Layout of one batched bundle call: `n_groups` argument groups of `n_args`
/// slots each at `arg_pool[start ..]` (group-major); group `g`'s outputs go to
/// `bundle_scratch[base0 + g*n_out ..]`.
#[derive(Clone, Copy, Debug)]
pub struct BatchTable {
    pub start: u32,
    pub n_args: u32,
    pub n_groups: u32,
    pub base0: u32,
    pub n_out: u32,
}

mod compile;
mod specialize;

pub use specialize::SpecializedTape;

/// A compiled evaluator for a fixed set of root expressions over named inputs.
///
/// Each instruction writes to a `dst` work slot and reads from slots produced
/// earlier. Slots are reused once their value is dead (liveness-driven, LIFO
/// free list), so the work buffer is the live-set width, not the node count --
/// a deep accumulator chain of `n` nodes needs a handful of slots, not `n`.
pub struct Tape {
    ops: Vec<Op>,
    /// Destination work slot for each op (parallel to `ops`).
    dst: Vec<u32>,
    /// Number of `Select` ops (the length of a choice trace).
    n_selects: usize,
    /// Flat operand-slot pool for variadic ops (Reduce / Dot).
    arg_pool: Vec<u32>,
    outputs: Vec<u32>,
    n_work: usize,
    /// Widest gather any variadic op (Reduce / Dot / bundle call) needs, so `eval`
    /// can carve the scratch region out of the tail of the caller's `work`
    /// buffer instead of allocating one per call.
    max_args: usize,
    /// Multi-output bodies for `BundleCall` ops, indexed by their `bidx`.
    bundles: Vec<Arc<dyn ExternBundle>>,
    /// Total width of the persistent bundle-output scratch region (sum of every
    /// distinct group's output count), carved from the tail of `work` in `eval`.
    bundle_scratch_len: usize,
    /// Argument tables for `BundleBatch` ops.
    batches: Vec<BatchTable>,
    /// Widest flat argument gather any `BundleBatch` needs
    /// (`max n_groups*n_args`), carved from the tail of `work` like `max_args`.
    batch_args_len: usize,
    /// Instruction count of the parameter-pure prolog prefix (0 = no split; see
    /// [`compile_split`](Self::compile_split)).
    prolog_ops: usize,
}

/// A backend that lowers a [`Tape`]'s instruction stream: the seam every
/// code generator sits on.
///
/// The tape is the evaluation IR. [`Tape::eval`] is the interpreting backend,
/// [`Tape::lower`] drives any other one -- a printer, an alternative
/// evaluator, or a generator emitting C, Verilog or a GPU kernel. Each method
/// receives the destination work slot `dst` and the operand slots the op
/// reads; a backend keeps its own slot-to-value map (the value of a slot is
/// whatever its most recent writer produced -- the tape's liveness guarantees
/// a slot is never reused while a value it holds is still needed, so reading
/// the current occupant is always the intended SSA value).
///
/// Slots are `u32` indices into a work array of width [`Tape::n_work`];
/// outputs are read from the slots in [`Tape::outputs`]. A program with a
/// parameter-pure prefix (see [`Tape::compile_split`]) exposes it as the
/// first [`Tape::prolog_len`] ops, which a generator emits as a separate
/// function it can call once per parameter binding.
///
/// # What a generated backend must reproduce
///
/// rsdag's guarantee is that every backend computes the same IEEE operation
/// sequence, so a generator that wants to stay inside it has to match five
/// things. All five are available as data or as reference code in this
/// crate, which is what makes a generator a walk over this trait rather than
/// a re-derivation:
///
/// - **The domain guards.** [`crate::node::unary_f64`] is the reference for
///   every unary op, including the `exp` cap at [`crate::node::EXP_LIMIT`],
///   the `ln` floor at [`crate::node::LN_FLOOR`] and the `sqrt` clamp. An
///   unguarded `exp` diverges on the first out-of-range Newton iterate.
/// - **The op names.** [`crate::UnaryOp::c_fn`] and [`crate::BinOp::c_fn`]
///   give the conventional C callee per op, and `name()` the spelling for
///   any other target; a guarded op names an `rsdag_` helper the generator
///   supplies from the reference above.
/// - **The special functions.** [`crate::node::digamma`],
///   [`crate::node::trigamma`] and [`crate::node::rand_uniform`] are defined
///   here, not taken from a platform library, so a generator ports these
///   exact series.
/// - **The fold orders.** [`crate::node::reduce_slice`] and
///   [`crate::node::dot_slice`] fold with four accumulators merged as
///   `(a0 + a1) + (a2 + a3)`, then the tail in order. A different
///   association gives different bits.
/// - **No contraction.** [`TapeVisitor::mul_add`] is a fused *dispatch*, not
///   a fused rounding: it rounds the product and the sum separately. A C
///   generator therefore compiles with `-ffp-contract=off`, and no backend
///   emits a hardware FMA for it.
pub trait TapeVisitor {
    fn constant(&mut self, dst: u32, v: f64);
    /// `inputs[k]` (or the not-an-input sentinel `u32::MAX`).
    fn input(&mut self, dst: u32, k: u32);
    fn add(&mut self, dst: u32, a: u32, b: u32);
    fn mul(&mut self, dst: u32, a: u32, b: u32);
    /// `a*b + c` as one dispatch (unfused rounding; see `Op::MulAdd`).
    fn mul_add(&mut self, dst: u32, a: u32, b: u32, c: u32);
    /// `a - b` (from a fused `Add(Neg)`).
    fn sub(&mut self, dst: u32, a: u32, b: u32);
    fn neg(&mut self, dst: u32, a: u32);
    fn powi(&mut self, dst: u32, a: u32, n: i32);
    fn unary(&mut self, dst: u32, op: UnaryOp, a: u32);
    fn binary(&mut self, dst: u32, op: BinOp, a: u32, b: u32);
    fn cmp(&mut self, dst: u32, op: CmpOp, a: u32, b: u32);
    fn select(&mut self, dst: u32, c: u32, t: u32, e: u32);
    fn reduce(&mut self, dst: u32, op: ReduceOp, args: &[u32]);
    fn dot(&mut self, dst: u32, a: &[u32], b: &[u32]);
    /// Evaluate a bundle once, writing its outputs to the bundle-scratch region
    /// at `scratch_base` (width [`Tape::bundle_scratch_len`]).
    fn bundle_call(&mut self, b: &Arc<dyn ExternBundle>, args: &[u32], scratch_base: u32);
    /// Evaluate a bundle for `n_groups` argument groups at once (group-major
    /// `args`, `n_groups * n_args` slots); group `g`'s outputs go to
    /// `bundle_scratch[base0 + g*n_outputs ..]`.
    fn bundle_batch(
        &mut self,
        b: &Arc<dyn ExternBundle>,
        args: &[u32],
        n_groups: u32,
        n_args: u32,
        base0: u32,
    );
    /// Copy one already-computed bundle output from `bundle_scratch[idx]`.
    fn bundle_pick(&mut self, dst: u32, idx: u32);
}

/// Observer of `Select` decisions during evaluation. The no-op sink keeps the
/// plain [`Tape::eval`] monomorphization free of any tracing cost.
trait TraceSink {
    fn select(&mut self, taken: bool);
}

struct NoTrace;
impl TraceSink for NoTrace {
    #[inline(always)]
    fn select(&mut self, _taken: bool) {}
}

/// Records one byte per `Select` in op order (`1` = then-arm taken).
struct RecordTrace<'a>(&'a mut Vec<u8>);
impl TraceSink for RecordTrace<'_> {
    #[inline(always)]
    fn select(&mut self, taken: bool) {
        self.0.push(taken as u8);
    }
}

impl Tape {
    /// Number of work slots needed by [`eval`](Self::eval).
    pub fn n_slots(&self) -> usize {
        self.n_work
    }

    /// Number of outputs (= number of roots).
    pub fn n_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Human-readable instruction listing (diagnostics): one line per op with
    /// its destination slot, the prolog boundary marked.
    pub fn dump(&self) -> String {
        let mut out = String::new();
        for (i, op) in self.ops.iter().enumerate() {
            if i == self.prolog_ops && self.prolog_ops > 0 {
                out.push_str("---- main ----\n");
            }
            out.push_str(&format!("{i:5}: s{} <- {:?}\n", self.dst[i], op));
        }
        out.push_str(&format!("outputs {:?}\n", self.outputs));
        out
    }

    /// Number of `Select` ops in the tape (the length of a choice trace).
    pub fn n_selects(&self) -> usize {
        self.n_selects
    }

    /// Evaluate the tape. `work` is resized to `n_slots()` and reused; outputs
    /// are written into `out` (resized to `n_outputs()`).
    pub fn eval(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        self.eval_impl(inputs, work, out, &mut NoTrace);
    }

    /// [`eval`](Self::eval), additionally recording each `Select`'s taken arm
    /// into `choices` (cleared first; one entry per `Select` in op order, `1` =
    /// then-arm). The trace feeds [`specialize`](Self::specialize).
    pub fn eval_traced(
        &self,
        inputs: &[f64],
        work: &mut Vec<f64>,
        out: &mut Vec<f64>,
        choices: &mut Vec<u8>,
    ) {
        choices.clear();
        choices.reserve(self.n_selects);
        self.eval_impl(inputs, work, out, &mut RecordTrace(choices));
    }

    #[inline(always)]
    fn eval_impl<S: TraceSink>(
        &self,
        inputs: &[f64],
        work: &mut Vec<f64>,
        out: &mut Vec<f64>,
        sink: &mut S,
    ) {
        work.clear();
        work.resize(
            self.n_work + self.max_args + self.bundle_scratch_len + self.batch_args_len,
            0.0,
        );
        self.run_range(inputs, work, 0, self.ops.len(), sink);
        out.clear();
        let w = &work[..self.n_work];
        out.extend(self.outputs.iter().map(|&o| w[o as usize]));
    }

    /// Instruction count of the parameter-pure prolog (0 when compiled without
    /// [`compile_split`](Self::compile_split)).
    /// Evaluate the whole tape in another execution scalar (`f32`,
    /// `Complex<f64>`): constants convert from their `f64` lowering, every
    /// op goes through [`Scalar`](crate::scalar::Scalar)'s reference
    /// implementation for `T`. Tapes with bundle calls are `f64` only.
    pub fn eval_typed<T: crate::scalar::Scalar>(
        &self,
        inputs: &[T],
        work: &mut Vec<T>,
        out: &mut Vec<T>,
    ) {
        use crate::scalar::{dot_slice_t, reduce_slice_t};
        assert!(
            self.bundles.is_empty(),
            "eval_typed: tapes with bundle calls evaluate in f64 only"
        );
        work.clear();
        work.resize(self.n_work + self.max_args, T::zero());
        let (w, scratch) = work.split_at_mut(self.n_work);
        for i in 0..self.ops.len() {
            let g = |k: u32| w[k as usize];
            let v = match self.ops[i] {
                Op::Const(v) => T::from_f64(v),
                Op::Input(k) => inputs.get(k as usize).copied().unwrap_or(T::nan()),
                Op::Add(a, b) => g(a).add(g(b)),
                Op::Mul(a, b) => g(a).mul(g(b)),
                Op::MulAdd(a, b, c) => g(a).mul(g(b)).add(g(c)),
                Op::Sub(a, b) => g(a).sub(g(b)),
                Op::Neg(a) => g(a).neg(),
                Op::Powi(a, n) => g(a).powi(n),
                Op::Unary(op, a) => T::unary(op, g(a)),
                Op::Binary(op, a, b) => T::binary(op, g(a), g(b)),
                Op::Cmp(op, a, b) => T::cmp(op, g(a), g(b)),
                Op::Select(c, t, e) => {
                    if g(c).is_true() {
                        g(t)
                    } else {
                        g(e)
                    }
                }
                Op::Reduce(op, start, len) => {
                    for k in 0..len {
                        scratch[k as usize] = g(self.arg_pool[(start + k) as usize]);
                    }
                    reduce_slice_t(op, &scratch[..len as usize])
                }
                Op::Dot(start, len) => {
                    for k in 0..2 * len {
                        scratch[k as usize] = g(self.arg_pool[(start + k) as usize]);
                    }
                    dot_slice_t(
                        &scratch[..len as usize],
                        &scratch[len as usize..2 * len as usize],
                    )
                }
                Op::BundleCall(..) | Op::BundleBatch(..) | Op::BundlePick(..) => {
                    unreachable!("no bundles")
                }
            };
            w[self.dst[i] as usize] = v;
        }
        out.clear();
        out.extend(self.outputs.iter().map(|&o| w[o as usize]));
    }

    pub fn prolog_len(&self) -> usize {
        self.prolog_ops
    }

    /// Evaluate the parameter-pure prolog into `work` (sized/cleared here).
    /// A Newton loop calls this once per parameter binding, then
    /// [`eval_main`](Self::eval_main) per iteration over the *same* buffer.
    pub fn eval_prolog(&self, inputs: &[f64], work: &mut Vec<f64>) {
        work.clear();
        work.resize(
            self.n_work + self.max_args + self.bundle_scratch_len + self.batch_args_len,
            0.0,
        );
        self.run_range(inputs, work, 0, self.prolog_ops, &mut NoTrace);
    }

    /// Evaluate the main phase over a `work` buffer prepared by
    /// [`eval_prolog`](Self::eval_prolog) (prolog results are pinned slots, so
    /// repeated main passes may not clear or resize the buffer).
    pub fn eval_main(&self, inputs: &[f64], work: &mut [f64], out: &mut Vec<f64>) {
        assert_eq!(
            work.len(),
            self.n_work + self.max_args + self.bundle_scratch_len + self.batch_args_len,
            "eval_main requires a work buffer prepared by eval_prolog"
        );
        self.run_range(inputs, work, self.prolog_ops, self.ops.len(), &mut NoTrace);
        out.clear();
        let w = &work[..self.n_work];
        out.extend(self.outputs.iter().map(|&o| w[o as usize]));
    }

    /// Execute ops `lo..hi` over a fully-sized work buffer, feeding each
    /// `Select` decision to `sink` (the no-op sink costs nothing).
    fn run_range<S: TraceSink>(
        &self,
        inputs: &[f64],
        work: &mut [f64],
        lo: usize,
        hi: usize,
        sink: &mut S,
    ) {
        // Carve two scratch regions out of the tail of `work` so no buffer is
        // allocated per call: `scratch` is the transient variadic/bundle-arg
        // gather, `bscratch` the persistent bundle-output region (written by a
        // `BundleCall`, read by its `BundlePick`s later -- possibly in a later
        // main pass, which is why it sits in the caller's persistent buffer).
        let (work, rest) = work.split_at_mut(self.n_work);
        let (scratch, rest) = rest.split_at_mut(self.max_args);
        let (bscratch, batch_scratch) = rest.split_at_mut(self.bundle_scratch_len);
        // SAFETY: every slot index in the tape is `< n_work` by construction in
        // `compile` (each `dst`, op operand, `arg_pool` entry and output slot is a
        // freshly allocated or reused work slot), the variadic gathers stay within
        // the `max_args`-wide `scratch`, the bundle region within `bscratch`, and
        // `ops.len() == dst.len()`. So the data-dependent indices below (which the
        // compiler cannot prove in bounds, unlike the `0..len` loop counters) are
        // always valid, and the bounds checks they would otherwise cost per op in
        // this hot inner loop are elided.
        unsafe {
            for i in lo..hi {
                let g = |k: u32| *work.get_unchecked(k as usize);
                let v = match *self.ops.get_unchecked(i) {
                    Op::Const(v) => v,
                    Op::Input(k) => inputs.get(k as usize).copied().unwrap_or(f64::NAN),
                    Op::Add(a, b) => g(a) + g(b),
                    Op::Mul(a, b) => g(a) * g(b),
                    // separate mul + add on purpose (no `mul_add` contraction):
                    // the superinstruction fuses the dispatch, not the rounding
                    Op::MulAdd(a, b, c) => g(a) * g(b) + g(c),
                    Op::Sub(a, b) => g(a) - g(b),
                    Op::Neg(a) => -g(a),
                    Op::Powi(a, n) => g(a).powi(n),
                    Op::Unary(op, a) => unary_f64(op, g(a)),
                    Op::Binary(op, a, b) => binary_f64(op, g(a), g(b)),
                    Op::Cmp(op, a, b) => {
                        if cmp_bool(op, g(a), g(b)) {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Op::Select(c, t, e) => {
                        let taken = g(c) != 0.0;
                        sink.select(taken);
                        if taken {
                            g(t)
                        } else {
                            g(e)
                        }
                    }
                    Op::Reduce(op, start, len) => {
                        for k in 0..len {
                            *scratch.get_unchecked_mut(k as usize) =
                                g(*self.arg_pool.get_unchecked((start + k) as usize));
                        }
                        reduce_slice(op, scratch.get_unchecked(..len as usize))
                    }
                    Op::Dot(start, len) => {
                        for k in 0..2 * len {
                            *scratch.get_unchecked_mut(k as usize) =
                                g(*self.arg_pool.get_unchecked((start + k) as usize));
                        }
                        dot_slice(
                            scratch.get_unchecked(..len as usize),
                            scratch.get_unchecked(len as usize..2 * len as usize),
                        )
                    }
                    Op::BundleCall(bidx, start, len, base) => {
                        for k in 0..len {
                            *scratch.get_unchecked_mut(k as usize) =
                                g(*self.arg_pool.get_unchecked((start + k) as usize));
                        }
                        let b = self.bundles.get_unchecked(bidx as usize);
                        let n = b.n_outputs();
                        b.call(
                            scratch.get_unchecked(..len as usize),
                            bscratch.get_unchecked_mut(base as usize..base as usize + n),
                        );
                        0.0 // written to the never-read sink slot
                    }
                    Op::BundleBatch(bidx, tidx) => {
                        let t = *self.batches.get_unchecked(tidx as usize);
                        let flat = (t.n_groups * t.n_args) as usize;
                        for k in 0..flat {
                            *batch_scratch.get_unchecked_mut(k) =
                                g(*self.arg_pool.get_unchecked((t.start as usize) + k));
                        }
                        let b = self.bundles.get_unchecked(bidx as usize);
                        let w = (t.n_groups * t.n_out) as usize;
                        b.call_batch(
                            batch_scratch.get_unchecked(..flat),
                            t.n_groups as usize,
                            t.n_args as usize,
                            bscratch.get_unchecked_mut(t.base0 as usize..t.base0 as usize + w),
                        );
                        0.0 // sink
                    }
                    Op::BundlePick(idx) => *bscratch.get_unchecked(idx as usize),
                };
                *work.get_unchecked_mut(*self.dst.get_unchecked(i) as usize) = v;
            }
        }
    }

    /// Evaluate the tape on `L` independent input sets at once (structure-of-arrays
    /// batching). Slot `s` holds an `[f64; L]` lane vector; the pointwise ops
    /// (`Add`/`Mul`/`Neg`/`Cmp`/`Select`/...) are written as length-`L` loops the
    /// compiler auto-vectorises into one SIMD instruction, so one tape walk does
    /// `L` evaluations. Transcendentals (`Unary`), variadic reductions and extern
    /// device callbacks have no SIMD form and fall back to per-lane scalar.
    ///
    /// `inputs[k]` is the lane vector for input `k`; `out` is filled with one lane
    /// vector per tape output. This is the batched kernel for parameter sweeps,
    /// Monte Carlo and finite-difference / directional sensitivity, and the SoA
    /// layout a GPU backend would dispatch over.
    ///
    /// Runtime dispatch: on x86 with AVX2+FMA the length-`L` pointwise loops are
    /// compiled as 256-bit vector ops (one `vmulpd`/`vaddpd` for `L = 4` f64 rather
    /// than the two SSE2 halves the portable baseline emits), roughly doubling batch
    /// throughput. The AVX2 body is bit-identical to the scalar fallback: the tape's
    /// mul and add are separate ops through distinct slots, so no fused-multiply
    /// contraction (and no rounding change) occurs — FMA only widens, never fuses.
    /// On aarch64 no dispatch is needed: NEON is part of the baseline target, so
    /// the same generic lane loops vectorise directly (measured ~2.1–2.6x per-lane
    /// over scalar `eval` on an Apple M3; see `examples/batch_bench.rs`).
    pub fn eval_batch<const L: usize>(
        &self,
        inputs: &[[f64; L]],
        work: &mut Vec<[f64; L]>,
        out: &mut Vec<[f64; L]>,
    ) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: the target features are confirmed present at runtime just
                // above; `eval_batch_avx2` requires nothing else.
                unsafe {
                    self.eval_batch_avx2::<L>(inputs, work, out);
                }
                return;
            }
        }
        self.eval_batch_inner::<L>(inputs, work, out);
    }

    /// AVX2+FMA copy of [`eval_batch_inner`](Self::eval_batch_inner). The
    /// `#[inline(always)]` inner body is codegen'd with these features enabled so the
    /// SoA lane loops vectorise to 256-bit ops. Guarded by runtime detection in
    /// [`eval_batch`](Self::eval_batch).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn eval_batch_avx2<const L: usize>(
        &self,
        inputs: &[[f64; L]],
        work: &mut Vec<[f64; L]>,
        out: &mut Vec<[f64; L]>,
    ) {
        self.eval_batch_inner::<L>(inputs, work, out);
    }

    #[inline(always)]
    fn eval_batch_inner<const L: usize>(
        &self,
        inputs: &[[f64; L]],
        work: &mut Vec<[f64; L]>,
        out: &mut Vec<[f64; L]>,
    ) {
        work.clear();
        work.resize(
            self.n_work + self.max_args + self.bundle_scratch_len + self.batch_args_len,
            [0.0; L],
        );
        let (work, rest) = work.split_at_mut(self.n_work);
        // Only the bundle-output region is carved here; variadic/extern args are
        // gathered per lane into the scalar `sc` buffer below.
        let (_args, bscratch) = rest.split_at_mut(self.max_args);
        // Scalar gather buffer for the per-lane fallback ops (reduce/dot/extern).
        let mut sc = vec![0.0f64; self.max_args.max(1)];
        // SAFETY: identical invariant to [`eval`] -- every slot index (op operand,
        // `dst`, `arg_pool` entry, output) is `< n_work` and the gathers stay
        // within `sc`/`bscratch` by construction in `compile`, so the
        // data-dependent indexing is in bounds and its per-op bounds checks (paid
        // once per operand per op in this hot loop) are elided. The `0..L` /
        // `0..len` counters keep their provable checks.
        unsafe {
            for i in 0..self.ops.len() {
                let w = |k: u32| *work.get_unchecked(k as usize);
                let mut v = [0.0f64; L];
                match *self.ops.get_unchecked(i) {
                    Op::Const(c) => v = [c; L],
                    Op::Input(k) => v = inputs.get(k as usize).copied().unwrap_or([f64::NAN; L]),
                    Op::Add(a, b) => {
                        let (wa, wb) = (w(a), w(b));
                        for l in 0..L {
                            v[l] = wa[l] + wb[l];
                        }
                    }
                    Op::Mul(a, b) => {
                        let (wa, wb) = (w(a), w(b));
                        for l in 0..L {
                            v[l] = wa[l] * wb[l];
                        }
                    }
                    // Written as separate mul + add on purpose (no `mul_add`):
                    // the superinstruction fuses the dispatch, never the rounding.
                    Op::MulAdd(a, b, c) => {
                        let (wa, wb, wc) = (w(a), w(b), w(c));
                        for l in 0..L {
                            v[l] = wa[l] * wb[l] + wc[l];
                        }
                    }
                    Op::Sub(a, b) => {
                        let (wa, wb) = (w(a), w(b));
                        for l in 0..L {
                            v[l] = wa[l] - wb[l];
                        }
                    }
                    Op::Neg(a) => {
                        let wa = w(a);
                        for l in 0..L {
                            v[l] = -wa[l];
                        }
                    }
                    Op::Powi(a, n) => {
                        let wa = w(a);
                        for l in 0..L {
                            v[l] = wa[l].powi(n);
                        }
                    }
                    Op::Unary(op, a) => {
                        let wa = w(a);
                        for l in 0..L {
                            v[l] = unary_f64(op, wa[l]);
                        }
                    }
                    Op::Binary(op, a, b) => {
                        let (wa, wb) = (w(a), w(b));
                        for l in 0..L {
                            v[l] = binary_f64(op, wa[l], wb[l]);
                        }
                    }
                    Op::Cmp(op, a, b) => {
                        let (wa, wb) = (w(a), w(b));
                        for l in 0..L {
                            v[l] = if cmp_bool(op, wa[l], wb[l]) { 1.0 } else { 0.0 };
                        }
                    }
                    Op::Select(c, t, e) => {
                        let (wc, wt, we) = (w(c), w(t), w(e));
                        for l in 0..L {
                            v[l] = if wc[l] != 0.0 { wt[l] } else { we[l] };
                        }
                    }
                    Op::Reduce(op, start, len) => {
                        for l in 0..L {
                            for k in 0..len {
                                *sc.get_unchecked_mut(k as usize) =
                                    w(*self.arg_pool.get_unchecked((start + k) as usize))[l];
                            }
                            v[l] = reduce_slice(op, sc.get_unchecked(..len as usize));
                        }
                    }
                    Op::Dot(start, len) => {
                        for l in 0..L {
                            for k in 0..2 * len {
                                *sc.get_unchecked_mut(k as usize) =
                                    w(*self.arg_pool.get_unchecked((start + k) as usize))[l];
                            }
                            v[l] = dot_slice(
                                sc.get_unchecked(..len as usize),
                                sc.get_unchecked(len as usize..2 * len as usize),
                            );
                        }
                    }
                    Op::BundleCall(bidx, start, len, base) => {
                        let b = self.bundles.get_unchecked(bidx as usize);
                        let n = b.n_outputs();
                        let mut obuf = vec![0.0f64; n];
                        for l in 0..L {
                            for k in 0..len {
                                *sc.get_unchecked_mut(k as usize) =
                                    w(*self.arg_pool.get_unchecked((start + k) as usize))[l];
                            }
                            b.call(sc.get_unchecked(..len as usize), &mut obuf);
                            for (j, &o) in obuf.iter().enumerate() {
                                bscratch.get_unchecked_mut(base as usize + j)[l] = o;
                            }
                        }
                        v = [0.0; L]; // sink slot
                    }
                    Op::BundleBatch(bidx, tidx) => {
                        let t = *self.batches.get_unchecked(tidx as usize);
                        let b = self.bundles.get_unchecked(bidx as usize);
                        let (ng, na, no) =
                            (t.n_groups as usize, t.n_args as usize, t.n_out as usize);
                        let mut abuf = vec![0.0f64; ng * na];
                        let mut obuf = vec![0.0f64; ng * no];
                        for l in 0..L {
                            for k in 0..ng * na {
                                abuf[k] =
                                    w(*self.arg_pool.get_unchecked((t.start as usize) + k))[l];
                            }
                            b.call_batch(&abuf, ng, na, &mut obuf);
                            for (j, &o) in obuf.iter().enumerate() {
                                bscratch.get_unchecked_mut(t.base0 as usize + j)[l] = o;
                            }
                        }
                        v = [0.0; L]; // sink
                    }
                    Op::BundlePick(idx) => v = *bscratch.get_unchecked(idx as usize),
                }
                *work.get_unchecked_mut(*self.dst.get_unchecked(i) as usize) = v;
            }
        }
        out.clear();
        out.extend(
            self.outputs
                .iter()
                .map(|&o| unsafe { *work.get_unchecked(o as usize) }),
        );
    }

    /// Work-array width an alternative backend must provide (slot count).
    pub fn n_work(&self) -> usize {
        self.n_work
    }

    /// The work slots holding the outputs, in root order.
    pub fn outputs(&self) -> &[u32] {
        &self.outputs
    }

    /// Width of the persistent bundle-output scratch region a backend must hold.
    pub fn bundle_scratch_len(&self) -> usize {
        self.bundle_scratch_len
    }

    /// Drive `v` over the instruction stream in order (the lowering counterpart
    /// of [`eval`](Self::eval)). The interpreter and any alternative backend thus
    /// consume the exact same program, so their results agree.
    pub fn lower(&self, v: &mut dyn TapeVisitor) {
        for i in 0..self.ops.len() {
            let dst = self.dst[i];
            let ap = |s: u32, l: u32| &self.arg_pool[s as usize..(s + l) as usize];
            match self.ops[i] {
                Op::Const(c) => v.constant(dst, c),
                Op::Input(k) => v.input(dst, k),
                Op::Add(a, b) => v.add(dst, a, b),
                Op::Mul(a, b) => v.mul(dst, a, b),
                Op::MulAdd(a, b, c) => v.mul_add(dst, a, b, c),
                Op::Sub(a, b) => v.sub(dst, a, b),
                Op::Neg(a) => v.neg(dst, a),
                Op::Powi(a, n) => v.powi(dst, a, n),
                Op::Unary(op, a) => v.unary(dst, op, a),
                Op::Binary(op, a, b) => v.binary(dst, op, a, b),
                Op::Cmp(op, a, b) => v.cmp(dst, op, a, b),
                Op::Select(c, t, e) => v.select(dst, c, t, e),
                Op::Reduce(op, s, l) => v.reduce(dst, op, ap(s, l)),
                Op::Dot(s, l) => v.dot(dst, ap(s, l), ap(s + l, l)),
                Op::BundleCall(b, s, l, base) => {
                    v.bundle_call(&self.bundles[b as usize], ap(s, l), base)
                }
                Op::BundleBatch(b, t) => {
                    let tbl = self.batches[t as usize];
                    v.bundle_batch(
                        &self.bundles[b as usize],
                        ap(tbl.start, tbl.n_groups * tbl.n_args),
                        tbl.n_groups,
                        tbl.n_args,
                        tbl.base0,
                    );
                }
                Op::BundlePick(idx) => v.bundle_pick(dst, idx),
            }
        }
    }

    /// Number of instructions (diagnostics; compare against a
    /// [`SpecializedTape::n_ops`] for the shrink factor).
    pub fn n_ops(&self) -> usize {
        self.ops.len()
    }
}
