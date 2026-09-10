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
use crate::node::{BinOp, CmpOp, ReduceOp, UnaryOp};
use crate::scalar::Scalar;

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
    /// A dense matrix-vector product, table `gemvs[t]`: `m` rows of `n`
    /// against a vector of `n`, the rows written to `bundle_scratch[base..]`
    /// and read back by `BundlePick`s. Fused by `compile` from the rows'
    /// `Dot`s, and bit-identical to them.
    Gemv(u32),
}

/// Where a dense operand's `len` values live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    /// Consecutive inputs from index `k`: a parameter matrix or a state
    /// vector, read in place.
    Inputs(u32),
    /// Slots listed in the arg pool from `start`, gathered.
    Slots(u32),
}

/// One fused matrix-vector product.
#[derive(Clone, Copy, Debug)]
pub struct GemvTable {
    pub a: Src,
    pub x: Src,
    pub m: u32,
    pub n: u32,
    /// First bundle-scratch index of the `m` results.
    pub base: u32,
}

/// A dense operand as a backend sees it.
#[derive(Clone, Copy, Debug)]
pub enum Operand<'a> {
    Inputs(u32),
    Slots(&'a [u32]),
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
    /// The fused matrix-vector products (see [`Op::Gemv`]).
    gemvs: Vec<GemvTable>,
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
/// - **The domain guards.** [`crate::semantics::unary_f64`] is the reference for
///   every unary op, including the `exp` cap at [`crate::semantics::EXP_LIMIT`],
///   the `ln` floor at [`crate::semantics::LN_FLOOR`] and the `sqrt` clamp. An
///   unguarded `exp` diverges on the first out-of-range Newton iterate.
/// - **The op names.** [`crate::UnaryOp::c_fn`] and [`crate::BinOp::c_fn`]
///   give the conventional C callee per op, and `name()` the spelling for
///   any other target; a guarded op names an `rsdag_` helper the generator
///   supplies from the reference above.
/// - **The special functions.** [`crate::semantics::digamma`],
///   [`crate::semantics::trigamma`] and [`crate::semantics::rand_uniform`] are defined
///   here, not taken from a platform library, so a generator ports these
///   exact series.
/// - **The fold orders.** [`crate::semantics::reduce_slice`] and
///   [`crate::semantics::dot_slice`] fold with four accumulators merged as
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
    /// A dense matrix-vector product: `m` rows of `n` in `a` against `x`,
    /// results to `bundle_scratch[base .. base + m]`, each row the fold of
    /// [`dot`](Self::dot) (see [`crate::semantics::gemv_t`]).
    fn gemv(&mut self, a: Operand<'_>, x: Operand<'_>, m: u32, n: u32, base: u32);
}

/// Observer of `Select` decisions during evaluation: [`NoTrace`] costs
/// nothing, a `Vec<u8>` records one byte per `Select` in op order (`1` =
/// then-arm taken), the trace [`Tape::specialize`] takes.
pub trait TraceSink {
    fn select(&mut self, taken: bool);
}

pub struct NoTrace;
impl TraceSink for NoTrace {
    #[inline(always)]
    fn select(&mut self, _taken: bool) {}
}

impl TraceSink for Vec<u8> {
    #[inline(always)]
    fn select(&mut self, taken: bool) {
        self.push(taken as u8);
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

    /// Evaluate the tape in any execution scalar (`f64`, `f32`,
    /// `Complex<f64>`): constants convert from their `f64` lowering, every
    /// op goes through [`Scalar`]'s reference arithmetic for `T`, so a value
    /// cannot depend on which scalar computed it beyond the scalar itself.
    /// `work` is resized and reused; `out` receives one value per output.
    pub fn eval<T: Scalar>(&self, inputs: &[T], work: &mut Vec<T>, out: &mut Vec<T>) {
        self.eval_with(inputs, work, out, &mut NoTrace);
    }

    /// [`eval`](Self::eval) with every `Select` decision reported to `sink`;
    /// a `Vec<u8>` sink is the choice trace [`specialize`](Self::specialize)
    /// takes (the caller clears it first).
    pub fn eval_with<T: Scalar, S: TraceSink>(
        &self,
        inputs: &[T],
        work: &mut Vec<T>,
        out: &mut Vec<T>,
        sink: &mut S,
    ) {
        work.clear();
        work.resize(self.buffer_len(), T::zero());
        self.run_range(inputs, work, 0, self.ops.len(), sink);
        out.clear();
        let w = &work[..self.n_work];
        out.extend(self.outputs.iter().map(|&o| w[o as usize]));
    }

    /// The work buffer: the slots, then the gather scratch, the bundle
    /// outputs and the batch arguments.
    fn buffer_len(&self) -> usize {
        self.n_work + self.max_args + self.bundle_scratch_len + self.batch_args_len
    }

    /// Instruction count of the parameter-pure prolog (0 when compiled without
    /// [`compile_split`](Self::compile_split)).
    pub fn prolog_len(&self) -> usize {
        self.prolog_ops
    }

    /// Evaluate the parameter-pure prolog into `work` (sized/cleared here).
    /// A Newton loop calls this once per parameter binding, then
    /// [`eval_main`](Self::eval_main) per iteration over the *same* buffer.
    pub fn eval_prolog<T: Scalar>(&self, inputs: &[T], work: &mut Vec<T>) {
        work.clear();
        work.resize(self.buffer_len(), T::zero());
        self.run_range(inputs, work, 0, self.prolog_ops, &mut NoTrace);
    }

    /// Evaluate the main phase over a `work` buffer prepared by
    /// [`eval_prolog`](Self::eval_prolog) (prolog results are pinned slots, so
    /// repeated main passes may not clear or resize the buffer).
    pub fn eval_main<T: Scalar>(&self, inputs: &[T], work: &mut [T], out: &mut Vec<T>) {
        assert_eq!(
            work.len(),
            self.buffer_len(),
            "eval_main requires a work buffer prepared by eval_prolog"
        );
        self.run_range(inputs, work, self.prolog_ops, self.ops.len(), &mut NoTrace);
        out.clear();
        let w = &work[..self.n_work];
        out.extend(self.outputs.iter().map(|&o| w[o as usize]));
    }

    /// Execute ops `lo..hi` over a fully-sized work buffer, feeding each
    /// `Select` decision to `sink` (the no-op sink costs nothing).
    fn run_range<T: Scalar, S: TraceSink>(
        &self,
        inputs: &[T],
        work: &mut [T],
        lo: usize,
        hi: usize,
        sink: &mut S,
    ) {
        use crate::semantics::{dot_slice_t, reduce_slice_t};
        // Two scratch regions at the tail of `work`, so nothing is allocated
        // per call: the transient gather for variadic and bundle arguments,
        // the persistent bundle-output region (written by a `BundleCall`,
        // read by its `BundlePick`s later, possibly in a later main pass),
        // and the batch argument block.
        let (work, rest) = work.split_at_mut(self.n_work);
        let (scratch, rest) = rest.split_at_mut(self.max_args);
        let (bscratch, batch_scratch) = rest.split_at_mut(self.bundle_scratch_len);
        for i in lo..hi {
            let g = |k: u32| work[k as usize];
            let v = match self.ops[i] {
                Op::Const(v) => T::from_f64(v),
                Op::Input(k) => inputs.get(k as usize).copied().unwrap_or(T::nan()),
                Op::Add(a, b) => g(a).add(g(b)),
                Op::Mul(a, b) => g(a).mul(g(b)),
                // The product and the sum round separately: a fused
                // dispatch, not a fused rounding.
                Op::MulAdd(a, b, c) => g(a).mul(g(b)).add(g(c)),
                Op::Sub(a, b) => g(a).sub(g(b)),
                Op::Neg(a) => g(a).neg(),
                Op::Powi(a, n) => g(a).powi(n),
                Op::Unary(op, a) => T::unary(op, g(a)),
                Op::Binary(op, a, b) => T::binary(op, g(a), g(b)),
                Op::Cmp(op, a, b) => T::cmp(op, g(a), g(b)),
                Op::Select(c, t, e) => {
                    let taken = g(c).is_true();
                    sink.select(taken);
                    if taken {
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
                Op::BundleCall(bidx, start, len, base) => {
                    for k in 0..len {
                        scratch[k as usize] = g(self.arg_pool[(start + k) as usize]);
                    }
                    let b = &*self.bundles[bidx as usize];
                    let n = b.n_outputs();
                    T::call_bundle(
                        b,
                        &scratch[..len as usize],
                        &mut bscratch[base as usize..base as usize + n],
                    );
                    T::zero() // written to the never-read sink slot
                }
                Op::BundleBatch(bidx, tidx) => {
                    let t = self.batches[tidx as usize];
                    let flat = (t.n_groups * t.n_args) as usize;
                    for k in 0..flat {
                        batch_scratch[k] = g(self.arg_pool[(t.start as usize) + k]);
                    }
                    let b = &*self.bundles[bidx as usize];
                    let w = (t.n_groups * t.n_out) as usize;
                    T::call_bundle_batch(
                        b,
                        &batch_scratch[..flat],
                        t.n_groups as usize,
                        t.n_args as usize,
                        &mut bscratch[t.base0 as usize..t.base0 as usize + w],
                    );
                    T::zero() // sink
                }
                Op::BundlePick(idx) => bscratch[idx as usize],
                Op::Gemv(t) => {
                    let tb = self.gemvs[t as usize];
                    let (m, n) = (tb.m as usize, tb.n as usize);
                    // A dense operand in the inputs is read in place when the
                    // inputs reach; otherwise, and for slots, it is gathered.
                    let mut at = 0usize;
                    let mut place =
                        |src: Src, len: usize, scratch: &mut [T]| -> std::ops::Range<usize> {
                            let r = at..at + len;
                            match src {
                                Src::Inputs(k) if inputs.len() >= k as usize + len => {
                                    return usize::MAX..k as usize
                                }
                                Src::Inputs(k) => {
                                    for j in 0..len {
                                        scratch[at + j] =
                                            inputs.get(k as usize + j).copied().unwrap_or(T::nan());
                                    }
                                }
                                Src::Slots(start) => {
                                    for j in 0..len {
                                        scratch[at + j] =
                                            work[self.arg_pool[start as usize + j] as usize];
                                    }
                                }
                            }
                            at += len;
                            r
                        };
                    let ra = place(tb.a, m * n, scratch);
                    let rx = place(tb.x, n, scratch);
                    let a: &[T] = if ra.start == usize::MAX {
                        &inputs[ra.end..ra.end + m * n]
                    } else {
                        &scratch[ra]
                    };
                    let x: &[T] = if rx.start == usize::MAX {
                        &inputs[rx.end..rx.end + n]
                    } else {
                        &scratch[rx]
                    };
                    crate::semantics::gemv_t(
                        a,
                        x,
                        m,
                        n,
                        &mut bscratch[tb.base as usize..tb.base as usize + m],
                    );
                    T::zero() // sink
                }
            };
            work[self.dst[i] as usize] = v;
        }
    }

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
                Op::Gemv(t) => {
                    let tb = self.gemvs[t as usize];
                    let operand = |src: Src, len: u32| match src {
                        Src::Inputs(k) => Operand::Inputs(k),
                        Src::Slots(start) => Operand::Slots(ap(start, len)),
                    };
                    v.gemv(
                        operand(tb.a, tb.m * tb.n),
                        operand(tb.x, tb.n),
                        tb.m,
                        tb.n,
                        tb.base,
                    );
                }
            }
        }
    }

    /// Number of instructions (diagnostics; compare against a
    /// [`SpecializedTape::n_ops`] for the shrink factor).
    pub fn n_ops(&self) -> usize {
        self.ops.len()
    }
}
