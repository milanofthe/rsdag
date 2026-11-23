//! Compiled flat evaluator (tape) for fast, repeated numeric evaluation.
//!
//! A [`Tape`] is the reachable sub-DAG of a set of root expressions, lowered
//! to a flat instruction list over compact slots. Symbols become positional
//! inputs, read where they are used (an operand with the input tag names an
//! input directly, no copy into a slot), and the work buffer is caller-owned
//! and reused, so a Newton loop over a large circuit neither re-hashes
//! symbols nor reallocates a scratch each step.
//!
//! An instruction writes one slot, or, for a kernel (a bundle call, a batch
//! of calls, a matrix-vector product, a dense solve), a block of consecutive
//! slots starting at its `dst`. Every value is a slot; there is no second
//! storage.
//!
//! # What a backend must reproduce
//!
//! - **The domain guards.** [`crate::semantics::unary_f64`] is the reference
//!   for every unary op, including the `exp` cap at
//!   [`crate::semantics::EXP_LIMIT`] and the `ln` floor at
//!   [`crate::semantics::LN_FLOOR`]; a backend that calls its own `exp`
//!   diverges from the arena where the guards bite.
//! - **The special functions.** [`crate::semantics::digamma`],
//!   [`crate::semantics::trigamma`] and [`crate::semantics::rand_uniform`]
//!   are defined there, not taken from a platform library.
//! - **The fold orders.** [`crate::semantics::reduce_slice`] and
//!   [`crate::semantics::dot_slice`] fold with four accumulators merged as
//!   `(a0 + a1) + (a2 + a3)`, then the tail in order; a kernel is those
//!   folds row by row.
//! - **No contraction.** [`Op::MulAdd`] is a fused *dispatch*, not
//!   a fused rounding: it rounds the product and the sum separately.

use std::sync::Arc;

use crate::extern_fn::{ExternBundle, Instances};
use crate::node::{BinOp, CmpOp, ReduceOp, UnaryOp};

/// The tag bit of an operand that names an input rather than a slot.
pub const INPUT: u32 = 1 << 31;

/// The `state` of a call that keeps none: the bundle runs whole.
pub const NO_STATE: u32 = u32::MAX;

/// [`Op::Call::reads`] of a call whose operands are its arguments in order.
pub const ALL_ARGS: u32 = u32::MAX;

/// The input index of a tagged operand.
#[inline]
pub fn input_index(k: u32) -> Option<u32> {
    (k & INPUT != 0).then_some(k & !INPUT)
}

/// One instruction. Operands are slots, or inputs when tagged with
/// [`INPUT`]; every instruction writes its `dst` slot, a kernel the block
/// of slots from `dst` on.
#[derive(Clone, Copy, Debug)]
pub enum Op {
    Const(f64),
    Add(u32, u32),
    Mul(u32, u32),
    /// `a*b + c` as one instruction, a fused *dispatch* with two roundings,
    /// so results stay bit-identical to the arena. Emitted by `compile`
    /// when an `Add` consumes a single-use `Mul`.
    MulAdd(u32, u32, u32),
    /// `a - b`, from an `Add` consuming a single-use `Neg`.
    Sub(u32, u32),
    Neg(u32),
    Powi(u32, i32),
    Unary(UnaryOp, u32),
    Binary(BinOp, u32, u32),
    Cmp(CmpOp, u32, u32),
    Select(u32, u32, u32),
    /// Reduction over `arg_pool[start .. start+len]`.
    Reduce(ReduceOp, u32, u32),
    /// Inner product of `arg_pool[start .. start+len]` and the `len` that
    /// follow.
    Dot(u32, u32),
    /// `bundles[b]` on `n_groups` argument groups of `n_args`, group `g`'s
    /// `n_out` outputs to `dst + g*n_out ..`; with a `state` slot (not
    /// [`NO_STATE`]), the bundle's main phase over group `g`'s instance
    /// state at `state + g*state_len`. The operands are `n_in` per group,
    /// group-major at `arg_pool[start ..]`: the arguments in order, or with
    /// `reads` (not [`ALL_ARGS`]) the ones at the positions
    /// `arg_pool[reads .. reads+n_in]`, the only ones a main phase reads
    /// (see [`Tape::main_reads`]); the others are left unset. The
    /// arguments are gathered at `args` in the gather area (see
    /// [`calls`]).
    Call {
        bundle: u32,
        start: u32,
        n_groups: u32,
        n_args: u32,
        n_in: u32,
        reads: u32,
        n_out: u32,
        state: u32,
        args: u32,
    },
    /// The prolog of `bundles[b]` for `n_groups` instances, their pure
    /// arguments group-major at `arg_pool[start ..]`, `n_pure` per group:
    /// instance `g`'s state to `dst + g*state_len ..`. The pure arguments
    /// in which the instances can differ are the `n_own` positions at
    /// `arg_pool[own ..]`; at every other position all instances read one
    /// operand, so only these decide which instances share a state row
    /// (see [`calls::share_rows`]).
    CallProlog {
        bundle: u32,
        start: u32,
        n_groups: u32,
        n_pure: u32,
        own: u32,
        n_own: u32,
        args: u32,
    },
    /// A matrix-vector product: `m` rows of `n` in `a` against `x`, the rows
    /// to `dst .. dst+m`, each row the fold of `Dot`; with `acc`, each row
    /// folded per [`Fold`] with its accumulator entry.
    Gemv {
        a: Src,
        x: Src,
        m: u32,
        n: u32,
        acc: Option<Accum>,
    },
    /// A matrix-matrix product: `m` rows of `k` in `a` against `n` rows of
    /// `k` in `b` (the right factor by columns), entry `(i, j)` to
    /// `dst + i*n + j`, each entry the fold of `Dot`; with `acc`, each
    /// entry folded per [`Fold`] with its accumulator entry.
    Gemm {
        a: Src,
        b: Src,
        m: u32,
        k: u32,
        n: u32,
        acc: Option<Accum>,
    },
    /// The dense solves of `count` systems of one shape, each `k`
    /// right-hand sides against its `n` by `n` matrix: `a` holds the
    /// matrices back to back, `b` each system's `k` vectors of `n` back to
    /// back, the solutions go to `dst .. dst + count*n*k` the same way (see
    /// [`crate::semantics::solve_batch_into`]).
    Solve {
        a: Src,
        b: Src,
        n: u32,
        k: u32,
        count: u32,
    },
}

/// How a product kernel folds one of its entries `d` after the fold of
/// its dot: kept as is, `c - d`, `c + d` or `-d`, one rounding, as the
/// `Sub`, `Add` or `Neg` it replaces; `c` is the entry's accumulator, an
/// operand of the kernel or, for a self fold, another entry's product. A
/// kernel with `acc` carries one code per entry in the arg pool
/// (`acc.1 ..`) and the accumulator operand at `acc.0` (a plain or self
/// fold's entry there is a placeholder).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fold(pub u32);

impl Fold {
    pub const PLAIN: Fold = Fold(0);
    pub const NEG: Fold = Fold(3);
    /// `c - d` with `c` the accumulator operand.
    pub const SUB: Fold = Fold(1);
    /// `c + d` with `c` the accumulator operand.
    pub const ADD: Fold = Fold(2);
    /// The bits of a code that say which fold: [`PLAIN`](Self::PLAIN),
    /// [`SUB`](Self::SUB), [`ADD`](Self::ADD) or [`NEG`](Self::NEG).
    pub const KIND: u32 = 3;
    /// The bit of a self fold, whose accumulator is another entry's product.
    pub const SELF: u32 = 4;
    /// Where a self fold's code keeps that entry's index.
    pub const SELF_SHIFT: u32 = 3;
    pub fn is_plain(self) -> bool {
        self.0 & Self::KIND == 0
    }
    pub fn is_self(self) -> bool {
        self.0 & Self::SELF != 0
    }
    /// The entry whose product is the accumulator of a self fold.
    pub fn self_index(self) -> usize {
        (self.0 >> Self::SELF_SHIFT) as usize
    }
    /// The product `d` folded against the accumulator `a` (unused by a
    /// plain or negating fold).
    #[inline]
    pub fn fold(self, a: f64, d: f64) -> f64 {
        match self.0 & Self::KIND {
            0 => d,
            1 => a - d,
            2 => a + d,
            _ => -d,
        }
    }
    /// Whether the fold reads the accumulator operand.
    pub fn reads_operand(self) -> bool {
        matches!(self.0 & Self::KIND, 1 | 2) && !self.is_self()
    }
}

/// A product kernel's folds: the fold code per entry from `codes` in the
/// arg pool, and the accumulator operand when any code reads one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Accum {
    pub c: Option<Src>,
    pub codes: u32,
}

/// Where a kernel's dense operand lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    /// Consecutive inputs from index `k`, read in place.
    Inputs(u32),
    /// Operands listed in the arg pool from `start`, gathered.
    Pool(u32),
    /// Consecutive work slots from `s`, read in place.
    Slots(u32),
}

/// A dense operand as a backend sees it.
#[derive(Clone, Copy, Debug)]
pub enum Operand<'a> {
    /// Consecutive inputs from `k`.
    Inputs(u32),
    /// Consecutive work slots from `s`.
    Run(u32),
    /// Slots to gather.
    Slots(&'a [u32]),
}

/// What the interpreter runs an op as: the single-value ops by kind, the
/// unary ones with their function, everything else a kernel
/// [`Tape::run_range`] runs itself.
#[derive(Clone, Copy, Debug)]
enum Code {
    /// The value's bits in `a` (low) and `b` (high).
    Const,
    Add,
    Mul,
    MulAdd,
    Sub,
    Neg,
    /// The exponent in `b`.
    Powi,
    Unary(UnaryOp),
    Binary(BinOp),
    Cmp(CmpOp),
    Select,
    /// Over `arg_pool[a .. a + b]`.
    Reduce(ReduceOp),
    /// Of `arg_pool[a .. a + b]` and the `b` that follow.
    Dot,
    Kernel,
}

/// An op as the interpreter runs it: its code, its operands `a`, `b`, `c`
/// as in the op (a slot, or an input tagged with [`INPUT`]) and its `dst`
/// slot `d`, twenty bytes where the op takes up to forty-eight.
#[derive(Clone, Copy, Debug)]
struct Ins {
    code: Code,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

/// The tape as the interpreter runs it: one [`Ins`] per op, the inputs it
/// reads and whether it has kernels.
#[derive(Clone, Debug, Default)]
pub(crate) struct Interp {
    ins: Vec<Ins>,
    /// One past the highest input any op or output reads: the inputs an
    /// evaluation holds, a missing one read as `NaN`.
    reach: usize,
    /// Whether an op is a kernel; a tape without one runs instances side
    /// by side.
    kernels: bool,
    /// The inputs the main phase reads, ascending (without kernels).
    main_reads: Vec<u32>,
}

impl Interp {
    /// The ops `ops` writing `dst`, their lists in `pool`, over `n_work`
    /// slots, the first `prolog_ops` the prolog, read from `outputs`. Every
    /// operand is checked here against the slots and against
    /// [`reach`](Self::reach), so the interpreter reads them unchecked.
    pub(crate) fn new(
        ops: &[Op],
        dst: &[u32],
        pool: &[u32],
        outputs: &[u32],
        n_work: usize,
        prolog_ops: usize,
    ) -> Self {
        let (mut reach, mut main_reads) = (0usize, Vec::new());
        // whether the operands checked are read by the main phase
        let main = std::cell::Cell::new(true);
        let mut check = |k: u32| match input_index(k) {
            Some(i) => {
                reach = reach.max(i as usize + 1);
                if main.get() {
                    main_reads.push(i);
                }
            }
            None => assert!((k as usize) < n_work, "operand past the slots"),
        };
        outputs.iter().for_each(|&k| check(k));
        let mut kernels = false;
        let ins = ops
            .iter()
            .zip(dst)
            .enumerate()
            .map(|(i, (op, &d))| {
                main.set(i >= prolog_ops);
                let ins = |code, a, b, c| Ins { code, a, b, c, d };
                let x = match *op {
                    Op::Const(v) => {
                        let bits = v.to_bits();
                        ins(Code::Const, bits as u32, (bits >> u32::BITS) as u32, 0)
                    }
                    Op::Add(a, b) => ins(Code::Add, a, b, 0),
                    Op::Mul(a, b) => ins(Code::Mul, a, b, 0),
                    Op::MulAdd(a, b, c) => ins(Code::MulAdd, a, b, c),
                    Op::Sub(a, b) => ins(Code::Sub, a, b, 0),
                    Op::Neg(a) => ins(Code::Neg, a, 0, 0),
                    Op::Powi(a, n) => ins(Code::Powi, a, n as u32, 0),
                    Op::Unary(u, a) => ins(Code::Unary(u), a, 0, 0),
                    Op::Binary(f, a, b) => ins(Code::Binary(f), a, b, 0),
                    Op::Cmp(f, a, b) => ins(Code::Cmp(f), a, b, 0),
                    Op::Select(c, t, e) => ins(Code::Select, c, t, e),
                    Op::Reduce(r, s, l) => ins(Code::Reduce(r), s, l, 0),
                    Op::Dot(s, l) => ins(Code::Dot, s, l, 0),
                    _ => {
                        kernels = true;
                        return ins(Code::Kernel, 0, 0, 0);
                    }
                };
                assert!((d as usize) < n_work, "destination past the slots");
                match x.code {
                    Code::Const => {}
                    Code::Reduce(_) => pool[x.a as usize..(x.a + x.b) as usize]
                        .iter()
                        .for_each(|&k| check(k)),
                    Code::Dot => pool[x.a as usize..(x.a + 2 * x.b) as usize]
                        .iter()
                        .for_each(|&k| check(k)),
                    Code::Neg | Code::Powi | Code::Unary(_) => check(x.a),
                    Code::MulAdd | Code::Select => [x.a, x.b, x.c].into_iter().for_each(&mut check),
                    _ => [x.a, x.b].into_iter().for_each(&mut check),
                }
                x
            })
            .collect();
        main_reads.sort_unstable();
        main_reads.dedup();
        Interp {
            ins,
            reach,
            kernels,
            main_reads,
        }
    }
}

/// A lane of a block of instances (see [`Tape::instances`]): instance
/// `l`'s slot `k` at `slots[k * b + l]`, its input `i` at
/// `inputs[i * b + l]`.
pub(crate) struct Lane<'a> {
    slots: &'a mut [f64],
    inputs: &'a mut [f64],
    l: usize,
    b: usize,
}

impl Lane<'_> {
    fn set_slot(&mut self, k: usize, v: f64) {
        self.slots[k * self.b + self.l] = v;
    }
    fn set_input(&mut self, i: usize, v: f64) {
        self.inputs[i * self.b + self.l] = v;
    }
    fn slot(&self, k: usize) -> f64 {
        self.slots[k * self.b + self.l]
    }
    /// The value of an operand: a slot, or an input when tagged.
    fn get(&self, k: u32) -> f64 {
        match input_index(k) {
            Some(i) => self.inputs[i as usize * self.b + self.l],
            None => self.slot(k as usize),
        }
    }
}

pub mod calls;
mod compile;
mod specialize;
mod topo;

use calls::Shared;
pub use compile::Lowered;
pub use specialize::{Guards, ParamSelects, SpecializedTape};

/// Consecutive calls of a tape that read nothing another of them writes:
/// every instance of every call in it is a piece of work of its own, run
/// together on the installed pool ([`crate::parallel`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage {
    /// The calls are ops `lo..hi`.
    pub lo: u32,
    pub hi: u32,
    /// Instances over all its calls.
    pub instances: u32,
    /// Ops it carries: per call, its instances times its body's ops.
    pub ops: u64,
}

/// Consecutive ops of a tape without a call, a dense kernel or a gather
/// (a min or max), in pieces that read nothing another piece writes and
/// write no slot another one reads or writes: each piece runs on its own,
/// in order, and the pieces together on the installed pool
/// ([`crate::parallel`]), as the stages run their calls. What a program
/// assembles from its calls' results (the sums of a circuit's Jacobian
/// entries, its rows) comes in many small parts no other part reads, and
/// runs so alongside.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    /// Piece `k` is ops `starts[k]..starts[k + 1]`, the last one up to
    /// `hi`.
    pub starts: Vec<u32>,
    pub hi: u32,
}

impl Region {
    /// The first op.
    pub fn lo(&self) -> u32 {
        self.starts[0]
    }

    /// Its ops.
    pub fn ops(&self) -> u64 {
        u64::from(self.hi - self.lo())
    }

    /// The ops of piece `k`.
    pub fn piece(&self, k: usize) -> std::ops::Range<usize> {
        let end = self.starts.get(k + 1).copied().unwrap_or(self.hi);
        self.starts[k] as usize..end as usize
    }
}

use crate::constants::OPAQUE_OPS;

/// Ops one instance of `call` runs in its bundle's body: the prolog for a
/// prolog, the main phase for a call over a state, all of it for a call
/// without one. A bundle without a body counts 1000 ops.
pub fn call_ops(bundle: &dyn ExternBundle, call: &Op) -> u64 {
    let Some(t) = bundle.body() else {
        return OPAQUE_OPS;
    };
    let (all, prolog) = (t.n_ops() as u64, t.prolog_len() as u64);
    match *call {
        Op::CallProlog { .. } => prolog.max(1),
        Op::Call { state, .. } if state != NO_STATE => (all - prolog).max(1),
        _ => all.max(1),
    }
}

/// The stages of an op stream (see [`Stage`]): each maximal run of
/// consecutive calls, within the prolog or within the main phase, in which
/// no call reads or overwrites a slot another one writes, and which holds
/// at least two instances.
fn plan_stages(
    ops: &[Op],
    dst: &[u32],
    pool: &[u32],
    bundles: &[Arc<dyn ExternBundle>],
    prolog_ops: usize,
) -> Vec<Stage> {
    // Per call: the slots it reads, the slots it writes, instances, ops.
    let call = |i: usize| -> Option<(Vec<(u32, u32)>, (u32, u32), u32, u64)> {
        let d = dst[i];
        let slots = |start: u32, len: u32| {
            pool[start as usize..(start + len) as usize]
                .iter()
                .filter(|&&k| input_index(k).is_none())
                .map(|&k| (k, k + 1))
                .collect::<Vec<_>>()
        };
        let (bundle, mut reads, width, ng) = match ops[i] {
            Op::Call {
                bundle,
                start,
                n_groups,
                n_in,
                n_out,
                state,
                ..
            } => {
                let mut r = slots(start, n_groups * n_in);
                if state != NO_STATE {
                    let sl = bundles[bundle as usize].state_len() as u32;
                    r.push((state, state + calls::state_block(n_groups, sl)));
                }
                (bundle, r, n_groups * n_out, n_groups)
            }
            Op::CallProlog {
                bundle,
                start,
                n_groups,
                n_pure,
                ..
            } => {
                let sl = bundles[bundle as usize].state_len() as u32;
                (
                    bundle,
                    slots(start, n_groups * n_pure),
                    calls::state_block(n_groups, sl),
                    n_groups,
                )
            }
            _ => return None,
        };
        reads.sort_unstable();
        let per = call_ops(&*bundles[bundle as usize], &ops[i]);
        Some((reads, (d, d + width), ng, ng as u64 * per))
    };
    let overlaps = |a: (u32, u32), b: (u32, u32)| a.0 < b.1 && b.0 < a.1;
    let mut stages = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        let Some(first) = call(i) else {
            i += 1;
            continue;
        };
        let end = if i < prolog_ops {
            prolog_ops
        } else {
            ops.len()
        };
        let (mut reads, mut writes) = (first.0, vec![first.1]);
        let (mut instances, mut work) = (first.2, first.3);
        let mut hi = i + 1;
        while hi < end {
            let Some((r, w, ng, o)) = call(hi) else {
                break;
            };
            let clash = r.iter().any(|&a| writes.iter().any(|&b| overlaps(a, b)))
                || writes.iter().any(|&b| overlaps(w, b))
                || reads.iter().any(|&a| overlaps(a, w));
            if clash {
                break;
            }
            reads.extend(r);
            writes.push(w);
            instances += ng;
            work += o;
            hi += 1;
        }
        if instances >= 2 {
            stages.push(Stage {
                lo: i as u32,
                hi: hi as u32,
                instances,
                ops: work,
            });
        }
        i = hi;
    }
    stages
}

/// A compiled evaluator for a set of expression roots.
pub struct Tape {
    ops: Vec<Op>,
    /// The ops as the interpreter runs them.
    interp: Interp,
    /// Destination slot of each op (the first of a kernel's block).
    dst: Vec<u32>,
    n_selects: usize,
    arg_pool: Vec<u32>,
    /// Output operands, one per root (a slot, or a tagged input).
    outputs: Vec<u32>,
    n_work: usize,
    /// Widest gather any variadic op or kernel needs.
    max_args: usize,
    /// The widest scratch a called bundle ([`ExternBundle::work_len`]) or a
    /// dense solve ([`crate::semantics::solve_scratch_len`]) works in, lent
    /// from the tail of the work buffer, after the gather.
    lent: usize,
    bundles: Vec<Arc<dyn ExternBundle>>,
    /// Instruction count of the parameter-pure prolog prefix (0 = no split;
    /// see [`compile_split`](Self::compile_split)).
    prolog_ops: usize,
    /// The prolog's results the main phase reads: `work[..state_len]`
    /// (see [`state_len`](Self::state_len)).
    state_len: usize,
    /// The inputs it was compiled over (`input_syms.len()`).
    n_inputs: usize,
    /// Its independent calls, by op range (see [`Stage`]).
    stages: Vec<Stage>,
    /// Its independent pieces of ops without a call (see [`Region`]).
    regions: Vec<Region>,
}

/// A compiled program as a solver drives it, whichever backend runs it:
/// buffers the caller owns and sizes from [`work_len`](Self::work_len) and
/// [`out_len`](Self::out_len), the prolog/main split, and the state layout
/// of [`state_len`](Self::state_len), so a prolog one backend ran serves
/// another's main phase. [`Tape`] implements it, and so does the native
/// code. Nothing is allocated and nothing returned: `out` may point
/// anywhere (a factorization's value array, a right-hand side, a numpy
/// array). An input past the end of `inputs` reads as `NaN`.
pub trait Program: Send + Sync {
    /// Inputs the program reads.
    fn n_inputs(&self) -> usize;
    /// The length `work` has at least: the slots, then the gather area and
    /// the scratch lent to called bodies and dense solves.
    fn work_len(&self) -> usize;
    /// Number of outputs, the length `out` has.
    fn out_len(&self) -> usize;
    /// The values the prolog leaves for the main phase are
    /// `work[..state_len]`: everything a later main phase needs of a prolog
    /// run, so an instance's prolog result is saved and restored as this
    /// prefix. The layout is the tape's, shared by every backend; `0`
    /// without a split.
    fn state_len(&self) -> usize;
    /// Everything, prolog and main.
    fn eval_into(&self, inputs: &[f64], work: &mut [f64], out: &mut [f64]);
    /// The parameter-pure prolog, into `work`: once per parameter binding,
    /// then the main phase per iteration over the same buffer.
    fn eval_prolog_into(&self, inputs: &[f64], work: &mut [f64]);
    /// The main phase over a `work` whose state a prolog left.
    fn eval_main_into(&self, inputs: &[f64], work: &mut [f64], out: &mut [f64]);
    /// [`eval_into`](Program::eval_into) over `Vec`s: `work` grown to
    /// [`work_len`](Self::work_len) (never shrunk or cleared, so a state a
    /// prolog left stays) and `out` sized to [`out_len`](Self::out_len).
    /// The convenience where an allocation on the first call does not
    /// matter; [`eval_prolog`](Self::eval_prolog) and
    /// [`eval_main`](Self::eval_main) are the phases so.
    fn eval(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        fit(self.work_len(), work, Some((self.out_len(), &mut *out)));
        self.eval_into(inputs, work, out);
    }
    /// [`eval_prolog_into`](Program::eval_prolog_into) over a `Vec` (see
    /// [`eval`](Self::eval)).
    fn eval_prolog(&self, inputs: &[f64], work: &mut Vec<f64>) {
        fit(self.work_len(), work, None);
        self.eval_prolog_into(inputs, work);
    }
    /// [`eval_main_into`](Program::eval_main_into) over `Vec`s (see
    /// [`eval`](Self::eval)).
    fn eval_main(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        fit(self.work_len(), work, Some((self.out_len(), &mut *out)));
        self.eval_main_into(inputs, work, out);
    }
    /// Instances of `stride` inputs back to back (`stride` at least
    /// [`n_inputs`](Self::n_inputs)), their outputs back to back into `out`,
    /// one after the other over one `work`. Running parts of a batch in
    /// parallel is the caller's choice: split `inputs` and `out` alike.
    fn eval_many_into(&self, inputs: &[f64], stride: usize, work: &mut [f64], out: &mut [f64]) {
        let n_out = self.out_len();
        if stride == 0 || n_out == 0 {
            return;
        }
        assert_eq!(
            out.len() / n_out,
            inputs.len() / stride,
            "one output vector per input vector"
        );
        for (ins, dst) in inputs.chunks_exact(stride).zip(out.chunks_exact_mut(n_out)) {
            self.eval_into(ins, work, dst);
        }
    }
}

/// `work` grown to `work_len`, `out` sized.
fn fit(work_len: usize, work: &mut Vec<f64>, out: Option<(usize, &mut Vec<f64>)>) {
    if work.len() < work_len {
        work.resize(work_len, 0.0);
    }
    if let Some((n, out)) = out {
        out.resize(n, 0.0);
    }
}

impl Program for Tape {
    fn n_inputs(&self) -> usize {
        self.n_inputs
    }
    fn work_len(&self) -> usize {
        self.n_work + self.max_args + self.lent
    }
    fn out_len(&self) -> usize {
        self.outputs.len()
    }
    fn state_len(&self) -> usize {
        self.state_len
    }
    fn eval_into(&self, inputs: &[f64], work: &mut [f64], out: &mut [f64]) {
        let _s = rsprobe::scope!(DEBUG, "eval");
        self.run_with(0, inputs, work, out, &mut NoTrace)
    }
    fn eval_prolog_into(&self, inputs: &[f64], work: &mut [f64]) {
        let _s = rsprobe::scope!(DEBUG, "prolog");
        assert!(work.len() >= self.work_len(), "work buffer too short");
        self.run_range(inputs, work, 0, self.prolog_ops, &mut NoTrace);
    }
    fn eval_main_into(&self, inputs: &[f64], work: &mut [f64], out: &mut [f64]) {
        let _s = rsprobe::scope!(DEBUG, "main");
        self.eval_main_with(inputs, work, out, &mut NoTrace)
    }
}

/// Observer of `Select` decisions during evaluation: [`NoTrace`] costs
/// nothing, a `Vec<u8>` records one byte per `Select` in op order (`1` =
/// then-arm taken), the trace [`Tape::specialize`] takes.
pub trait TraceSink {
    /// Whether it records anything: a trace is in op order, so a program
    /// observed so runs its regions' pieces one after the other.
    const RECORDS: bool = true;
    fn select(&mut self, taken: bool);
}

pub struct NoTrace;
impl TraceSink for NoTrace {
    const RECORDS: bool = false;
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
    /// Number of work slots an evaluation writes (without the gather area).
    pub fn n_slots(&self) -> usize {
        self.n_work
    }

    /// Number of outputs (= number of roots).
    pub fn n_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Human-readable instruction listing (diagnostics): one line per op
    /// with its destination slot, the prolog boundary marked; an operand
    /// `i7` is input 7.
    pub fn dump(&self) -> String {
        let name = |k: u32| match input_index(k) {
            Some(i) => format!("i{i}"),
            None => format!("s{k}"),
        };
        let list = |start: u32, len: u32| -> String {
            let names: Vec<String> = self.pool(start, len).iter().map(|&k| name(k)).collect();
            names.join(", ")
        };
        let src = |s: Src, len: u32| match s {
            Src::Inputs(k) => format!("i{k}..i{}", k + len),
            Src::Pool(start) => format!("[{}]", list(start, len)),
            Src::Slots(s) => format!("s{s}..s{}", s + len),
        };
        let mut out = String::new();
        for (i, op) in self.ops.iter().enumerate() {
            if i == self.prolog_ops && self.prolog_ops > 0 {
                out.push_str("---- main ----\n");
            }
            let text = match *op {
                Op::Const(v) => format!("Const({v})"),
                Op::Add(a, b) => format!("Add({}, {})", name(a), name(b)),
                Op::Mul(a, b) => format!("Mul({}, {})", name(a), name(b)),
                Op::MulAdd(a, b, c) => format!("MulAdd({}, {}, {})", name(a), name(b), name(c)),
                Op::Sub(a, b) => format!("Sub({}, {})", name(a), name(b)),
                Op::Neg(a) => format!("Neg({})", name(a)),
                Op::Powi(a, n) => format!("Powi({}, {n})", name(a)),
                Op::Unary(op, a) => format!("Unary({op:?}, {})", name(a)),
                Op::Binary(op, a, b) => format!("Binary({op:?}, {}, {})", name(a), name(b)),
                Op::Cmp(op, a, b) => format!("Cmp({op:?}, {}, {})", name(a), name(b)),
                Op::Select(c, t, e) => format!("Select({}, {}, {})", name(c), name(t), name(e)),
                Op::Reduce(op, s, l) => format!("Reduce({op:?}, [{}])", list(s, l)),
                Op::Dot(s, l) => format!("Dot([{}], [{}])", list(s, l), list(s + l, l)),
                Op::Call {
                    bundle,
                    start,
                    n_groups: 1,
                    n_in,
                    n_out,
                    state,
                    ..
                } => format!(
                    "Call(b{bundle}, [{}]{}) -> {n_out}",
                    list(start, n_in),
                    state_text(state)
                ),
                Op::Call {
                    bundle,
                    start,
                    n_groups,
                    n_in,
                    n_out,
                    state,
                    ..
                } => format!(
                    "CallBatch(b{bundle}, {n_groups} x [{}]{}) -> {n_groups} x {n_out}",
                    list(start, n_groups * n_in),
                    state_text(state)
                ),
                Op::CallProlog {
                    bundle,
                    start,
                    n_groups,
                    n_pure,
                    own,
                    n_own,
                    ..
                } => format!(
                    "CallProlog(b{bundle}, {n_groups} x [{}], own {:?}) -> {n_groups} x {}",
                    list(start, n_groups * n_pure),
                    self.pool(own, n_own),
                    self.bundles[bundle as usize].state_len()
                ),
                Op::Gemv { a, x, m, n, acc } => {
                    format!(
                        "Gemv({m}x{n} {}, {}){} -> {m}",
                        src(a, m * n),
                        src(x, n),
                        acc_text(&self.arg_pool, acc, m)
                    )
                }
                Op::Gemm { a, b, m, k, n, acc } => {
                    format!(
                        "Gemm({m}x{k} {}, {n}x{k} {}){} -> {}",
                        src(a, m * k),
                        src(b, n * k),
                        acc_text(&self.arg_pool, acc, m * n),
                        m * n
                    )
                }
                Op::Solve {
                    a,
                    b,
                    n,
                    k: 1,
                    count: 1,
                } => {
                    format!("Solve({n}x{n} {}, {}) -> {n}", src(a, n * n), src(b, n))
                }
                Op::Solve { a, b, n, k, count } if count > 1 => {
                    format!(
                        "SolveBatch({count} of {n}x{n} {}, {k} rhs {}) -> {}",
                        src(a, count * n * n),
                        src(b, count * n * k),
                        count * n * k
                    )
                }
                Op::Solve { a, b, n, k, .. } => {
                    format!(
                        "SolveMany({n}x{n} {}, {k} rhs {}) -> {}",
                        src(a, n * n),
                        src(b, n * k),
                        n * k
                    )
                }
            };
            out.push_str(&format!("{i:5}: s{} <- {text}\n", self.dst[i]));
        }
        let outs: Vec<String> = self.outputs.iter().map(|&k| name(k)).collect();
        out.push_str(&format!("outputs [{}]\n", outs.join(", ")));
        out
    }

    /// Number of `Select` ops in the tape (the length of a choice trace).
    pub fn n_selects(&self) -> usize {
        self.n_selects
    }

    /// [`eval`](Program::eval) with every `Select` decision reported to
    /// `sink`; a `Vec<u8>` sink is the choice trace
    /// [`specialize`](Self::specialize) takes (the caller clears it first).
    pub fn eval_with<S: TraceSink>(
        &self,
        inputs: &[f64],
        work: &mut Vec<f64>,
        out: &mut Vec<f64>,
        sink: &mut S,
    ) {
        fit(self.work_len(), work, Some((self.out_len(), &mut *out)));
        self.run_with(0, inputs, work, out, sink)
    }

    /// [`eval_main_into`](Program::eval_main_into), each `Select` of the
    /// main phase reporting its choice to `sink` in stream order: the trace
    /// [`specialize_main`](Self::specialize_main) shortens against.
    pub fn eval_main_with<S: TraceSink>(
        &self,
        inputs: &[f64],
        work: &mut [f64],
        out: &mut [f64],
        sink: &mut S,
    ) {
        self.run_with(self.prolog_ops, inputs, work, out, sink)
    }

    /// Ops `lo..` and the outputs.
    fn run_with<S: TraceSink>(
        &self,
        lo: usize,
        inputs: &[f64],
        work: &mut [f64],
        out: &mut [f64],
        sink: &mut S,
    ) {
        assert!(work.len() >= self.work_len(), "work buffer too short");
        assert_eq!(out.len(), self.out_len(), "output buffer of the wrong size");
        self.run_range(inputs, work, lo, self.ops.len(), sink);
        write_outputs(&self.outputs, inputs, work, out);
    }

    /// Instruction count of the parameter-pure prolog (0 when compiled without
    /// [`compile_split`](Self::compile_split)).
    pub fn prolog_len(&self) -> usize {
        self.prolog_ops
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
        use crate::semantics::solve_batch_into;
        let reach = self.interp.reach;
        if inputs.len() < reach {
            // the ops read their inputs unchecked: the missing ones as NaN
            return crate::scratch::with(|all: &mut Vec<f64>| {
                all.clear();
                all.extend_from_slice(inputs);
                all.resize(reach, f64::NAN);
                self.run_range(all, work, lo, hi, sink)
            });
        }
        // The gather scratch at the tail of `work`, so nothing is allocated
        // per call.
        let (work, scratch) = work.split_at_mut(self.n_work);
        // The next stage and region at or after `lo`.
        let mut next = self.stages.partition_point(|st| (st.lo as usize) < lo);
        let mut next_region = self.regions.partition_point(|r| (r.lo() as usize) < lo);
        let mut i = lo;
        while i < hi {
            if let Some(r) = self
                .regions
                .get(next_region)
                .filter(|r| r.lo() as usize == i)
            {
                next_region += 1;
                if !S::RECORDS && r.hi as usize <= hi {
                    let _s = rsprobe::scope!(TRACE, "region");
                    let slots = Shared::new(work);
                    let piece = |k: usize| {
                        crate::scratch::with_len(self.max_args, 0.0, |s| {
                            let ops = r.piece(k);
                            // SAFETY: a region's pieces hold no kernel and
                            // write slots no other piece reads or writes
                            // (`emit`); the inputs reach `reach`.
                            unsafe {
                                let p = slots.ptr();
                                self.lanes::<NoTrace, 1>(ops, p, inputs.as_ptr(), s, &mut NoTrace)
                            };
                        })
                    };
                    crate::parallel::run_rated(
                        r.starts.len(),
                        r.ops() as f64,
                        &INTERPRETED,
                        &piece,
                    );
                    i = r.hi as usize;
                    continue;
                }
            }
            if let Some(st) = self.stages.get(next).filter(|st| st.lo as usize == i) {
                next += 1;
                // `run_stage` runs it on as many threads as it is worth,
                // one included
                if st.hi as usize <= hi {
                    let _s = rsprobe::scope!(TRACE, "stage");
                    let (slots, gather) = (
                        Shared::new(work),
                        Shared::new(&mut scratch[..self.max_args]),
                    );
                    // SAFETY: `plan_stages` admits a call to a stage only if
                    // it reads no slot another call of the stage writes and
                    // writes no slot another one reads or writes; their
                    // arguments are apart in the gather area.
                    let (lo, n) = (st.lo as usize, (st.hi - st.lo) as usize);
                    let call = |k| self.call(lo + k);
                    unsafe { calls::run_stage(n, call, slots, gather, inputs) };
                    i = st.hi as usize;
                    continue;
                }
            }
            if !matches!(self.interp.ins[i].code, Code::Kernel) {
                // up to the next kernel, region or stage
                let end = (self.regions.get(next_region).map(|r| r.lo() as usize))
                    .into_iter()
                    .chain(self.stages.get(next).map(|st| st.lo as usize))
                    .fold(hi, usize::min);
                // SAFETY: `work` holds the slots, the inputs reach `reach`.
                i = unsafe {
                    let p = work.as_mut_ptr();
                    self.lanes::<S, 1>(i..end, p, inputs.as_ptr(), scratch, sink)
                };
                continue;
            }
            let d = self.dst[i] as usize;
            match self.ops[i] {
                Op::Call { .. } | Op::CallProlog { .. } => {
                    let c = self.call(i);
                    let (gather, lent) = scratch.split_at_mut(self.max_args);
                    let (slots, gather) = (Shared::new(work), Shared::new(gather));
                    // SAFETY: one call at a time, over its own regions.
                    unsafe { calls::run_call(&c, slots, gather, inputs, 0..c.n_groups, lent) };
                }
                Op::Gemv { a, x, m, n, acc } => {
                    let (m, n) = (m as usize, n as usize);
                    let mut at = 0usize;
                    let pool = &self.arg_pool;
                    let ra = place_operand(inputs, work, scratch, pool, a, m * n, &mut at);
                    let rx = place_operand(inputs, work, scratch, pool, x, n, &mut at);
                    let rc = acc.map(|f| {
                        let c =
                            f.c.map(|c| place_operand(inputs, work, scratch, pool, c, m, &mut at));
                        (c, f.codes)
                    });
                    let base = work.as_mut_ptr();
                    let av: &[f64] = dense_slice(inputs, base, scratch, ra, m * n);
                    let xv: &[f64] = dense_slice(inputs, base, scratch, rx, n);
                    // The kernel's outputs are fresh slots: no operand, the
                    // accumulator included, lives where they go.
                    let reads = [
                        Some((ra, m * n)),
                        Some((rx, n)),
                        rc.and_then(|(c, _)| c.map(|c| (c, m))),
                    ];
                    let out = unsafe { out_block(base, work.len(), d, m, &reads) };
                    match rc {
                        None => crate::semantics::gemv(av, xv, m, n, out),
                        Some((rc, codes)) => {
                            let cv: Option<&[f64]> =
                                rc.map(|rc| dense_slice(inputs, base, scratch, rc, m));
                            let codes = &self.arg_pool[codes as usize..codes as usize + m];
                            crate::semantics::gemv_fold(av, xv, m, n, cv, codes, out);
                        }
                    }
                }
                Op::Gemm { a, b, m, k, n, acc } => {
                    let (m, k, n) = (m as usize, k as usize, n as usize);
                    let mut at = 0usize;
                    let pool = &self.arg_pool;
                    let ra = place_operand(inputs, work, scratch, pool, a, m * k, &mut at);
                    let rb = place_operand(inputs, work, scratch, pool, b, n * k, &mut at);
                    let rc = acc.map(|f| {
                        let c = f
                            .c
                            .map(|c| place_operand(inputs, work, scratch, pool, c, m * n, &mut at));
                        (c, f.codes)
                    });
                    let base = work.as_mut_ptr();
                    let av: &[f64] = dense_slice(inputs, base, scratch, ra, m * k);
                    let bv: &[f64] = dense_slice(inputs, base, scratch, rb, n * k);
                    let reads = [
                        Some((ra, m * k)),
                        Some((rb, n * k)),
                        rc.and_then(|(c, _)| c.map(|c| (c, m * n))),
                    ];
                    let out = unsafe { out_block(base, work.len(), d, m * n, &reads) };
                    match rc {
                        None => crate::semantics::gemm(av, bv, m, k, n, out),
                        Some((rc, codes)) => {
                            let cv: Option<&[f64]> =
                                rc.map(|rc| dense_slice(inputs, base, scratch, rc, m * n));
                            let codes = &self.arg_pool[codes as usize..codes as usize + m * n];
                            crate::semantics::gemm_fold(av, bv, m, k, n, cv, codes, out);
                        }
                    }
                }
                Op::Solve { a, b, n, k, count } => {
                    let (n, k, count) = (n as usize, k as usize, count as usize);
                    let (la, lb) = (count * n * n, count * n * k);
                    // The operands gathered at the front of the scratch, the
                    // solve working behind them.
                    let (scratch, lent) = scratch.split_at_mut(self.max_args);
                    let (ra, rb) =
                        dense_operands(inputs, work, scratch, &self.arg_pool, a, la, b, lb);
                    let base = work.as_mut_ptr();
                    let av: &[f64] = dense_slice(inputs, base, scratch, ra, la);
                    let bv: &[f64] = dense_slice(inputs, base, scratch, rb, lb);
                    let reads = [Some((ra, la)), Some((rb, lb)), None];
                    let out = unsafe { out_block(base, work.len(), d, lb, &reads) };
                    solve_batch_into(av, bv, n, k, count, out, lent);
                }
                _ => unreachable!("not a kernel"),
            }
            i += 1;
        }
    }

    /// Ops `ops` up to the first kernel, each over `B` instances side by
    /// side: lane `l` of slot `k` at `slots[k * B + l]`, of input `i` at
    /// `inputs[i * B + l]`. Every lane runs the ops as
    /// [`crate::semantics`] defines them, one after the other in stream order, so each instance
    /// gets the bits it gets alone; `scratch` holds the widest list. Returns
    /// the op it stopped at.
    ///
    /// # Safety
    ///
    /// `slots` holds `n_slots() * B` values and `inputs` `reach * B`;
    /// nothing else accesses the slots the ops write while this runs, nor
    /// writes those they read.
    unsafe fn lanes<S: TraceSink, const B: usize>(
        &self,
        ops: std::ops::Range<usize>,
        slots: *mut f64,
        inputs: *const f64,
        scratch: &mut [f64],
        sink: &mut S,
    ) -> usize {
        use crate::semantics::{
            binary_f64, cmp_bool, dot_slice, powi_f64, reduce_slice, unary_f64,
        };
        use std::array::from_fn;
        // `Interp::new` checked every operand against the slots and `reach`.
        let at = |k: u32| -> *const [f64; B] {
            match k & INPUT != 0 {
                true => unsafe { inputs.add((k & !INPUT) as usize * B) }.cast(),
                false => unsafe { slots.add(k as usize * B) }.cast(),
            }
        };
        let get = |k: u32| unsafe { *at(k) };
        let list = |start: u32, len: u32| &self.arg_pool[start as usize..(start + len) as usize];
        let (start, end) = (ops.start, ops.end);
        for (j, x) in self.interp.ins[start..end].iter().enumerate() {
            let v: [f64; B] = match x.code {
                Code::Kernel => return start + j,
                Code::Const => {
                    let bits = u64::from(x.a) | u64::from(x.b) << u32::BITS;
                    [f64::from_bits(bits); B]
                }
                Code::Add => {
                    let (a, b) = (get(x.a), get(x.b));
                    from_fn(|l| a[l] + b[l])
                }
                Code::Mul => {
                    let (a, b) = (get(x.a), get(x.b));
                    from_fn(|l| a[l] * b[l])
                }
                // The product and the sum round separately: a fused
                // dispatch, not a fused rounding.
                Code::MulAdd => {
                    let (a, b, c) = (get(x.a), get(x.b), get(x.c));
                    from_fn(|l| a[l] * b[l] + c[l])
                }
                Code::Sub => {
                    let (a, b) = (get(x.a), get(x.b));
                    from_fn(|l| a[l] - b[l])
                }
                Code::Neg => get(x.a).map(|a| -a),
                Code::Powi => get(x.a).map(|a| powi_f64(a, x.b as i32)),
                // the common ones with the function known where it inlines
                Code::Unary(UnaryOp::Exp) => get(x.a).map(|a| unary_f64(UnaryOp::Exp, a)),
                Code::Unary(UnaryOp::Ln) => get(x.a).map(|a| unary_f64(UnaryOp::Ln, a)),
                Code::Unary(f) => get(x.a).map(|a| unary_f64(f, a)),
                Code::Binary(f) => {
                    let (a, b) = (get(x.a), get(x.b));
                    from_fn(|l| binary_f64(f, a[l], b[l]))
                }
                Code::Cmp(f) => {
                    let (a, b) = (get(x.a), get(x.b));
                    from_fn(|l| if cmp_bool(f, a[l], b[l]) { 1.0 } else { 0.0 })
                }
                Code::Select => {
                    let (c, t, e) = (get(x.a), get(x.b), get(x.c));
                    from_fn(|l| {
                        let taken = c[l] != 0.0;
                        sink.select(taken);
                        if taken {
                            t[l]
                        } else {
                            e[l]
                        }
                    })
                }
                Code::Reduce(f) => {
                    let ks = list(x.a, x.b);
                    from_fn(|l| {
                        for (s, &k) in scratch.iter_mut().zip(ks) {
                            *s = unsafe { (*at(k))[l] };
                        }
                        reduce_slice(f, &scratch[..ks.len()])
                    })
                }
                Code::Dot => {
                    let (ks, n) = (list(x.a, 2 * x.b), x.b as usize);
                    from_fn(|l| {
                        for (s, &k) in scratch.iter_mut().zip(ks) {
                            *s = unsafe { (*at(k))[l] };
                        }
                        dot_slice(&scratch[..n], &scratch[n..2 * n])
                    })
                }
            };
            unsafe { *slots.add(x.d as usize * B).cast::<[f64; B]>() = v };
        }
        end
    }

    /// The whole tape for the instances `at`, their arguments in `args`,
    /// their first `n_out` outputs into `out`: each instance's as
    /// [`eval_into`](Program::eval_into) computes them, bit for bit, in blocks
    /// of instances side by side.
    pub fn eval_batch(&self, args: &[f64], out: &mut [f64], n_out: usize, at: &Instances) {
        self.outputs_batch(0..self.ops.len(), args, None, out, n_out, at);
    }

    /// The prolog for the instances `at`, their pure arguments in `pure`
    /// (`at.n_args` each, the inputs `flags` marks in order, the others
    /// `NaN`), each instance's state into its row of `states`: what
    /// [`eval_prolog_into`](Program::eval_prolog_into) leaves in
    /// `work[..state_len]`, bit for bit.
    pub fn prolog_batch(&self, flags: &[bool], pure: &[f64], states: &mut [f64], at: &Instances) {
        let (na, sl, st) = (at.n_args, self.state_len, at.stride);
        let fill = |k: usize, lane: &mut Lane| {
            let mut p = &pure[at.at(k) * na..(at.at(k) + 1) * na];
            for i in 0..lane.inputs.len() / lane.b {
                let v = match (flags.get(i), p.split_first()) {
                    (Some(true), Some((&v, rest))) => {
                        p = rest;
                        v
                    }
                    _ => f64::NAN,
                };
                lane.set_input(i, v);
            }
        };
        self.instances(at.len(), 0..self.prolog_ops, fill, |k, lane| {
            let r = at.row(at.at(k));
            for j in 0..sl {
                states[r * st + j] = lane.slot(j);
            }
        });
    }

    /// The main phase for the instances `at` over their states, their rows
    /// of `states` as a prolog left them, their arguments in `args`, their
    /// first `n_out` outputs into `out`: each instance's as
    /// [`eval_main_into`](Program::eval_main_into) computes them, bit for
    /// bit.
    pub fn main_batch(
        &self,
        args: &[f64],
        states: &[f64],
        out: &mut [f64],
        n_out: usize,
        at: &Instances,
    ) {
        let ops = self.prolog_ops..self.ops.len();
        self.outputs_batch(ops, args, Some(states), out, n_out, at);
    }

    /// [`eval_batch`](Self::eval_batch) or, over `states`,
    /// [`main_batch`](Self::main_batch), as the ops `ops`.
    fn outputs_batch(
        &self,
        ops: std::ops::Range<usize>,
        args: &[f64],
        states: Option<&[f64]>,
        out: &mut [f64],
        n_out: usize,
        at: &Instances,
    ) {
        let (na, sl, st) = (at.n_args, self.state_len, at.stride);
        let main = states.is_some() && !self.interp.kernels;
        let fill = |k: usize, lane: &mut Lane| {
            let g = at.at(k);
            if let Some(states) = states {
                let r = at.row(g);
                for j in 0..sl {
                    lane.set_slot(j, states[r * st + j]);
                }
            }
            let a = &args[g * na..(g + 1) * na];
            let arg = |i: usize| a.get(i).copied().unwrap_or(f64::NAN);
            match main {
                // the inputs the main phase reads
                true => (self.interp.main_reads.iter())
                    .for_each(|&i| lane.set_input(i as usize, arg(i as usize))),
                false => (0..lane.inputs.len() / lane.b).for_each(|i| lane.set_input(i, arg(i))),
            }
        };
        self.instances(at.len(), ops, fill, |k, lane| {
            let g = at.at(k);
            for (j, &o) in self.outputs[..n_out].iter().enumerate() {
                out[g * n_out + j] = lane.get(o);
            }
        });
    }

    /// Ops `ops` for `n` instances, in blocks of lanes as wide as the
    /// instances fill (see [`LANES_WIDE`](crate::constants::LANES_WIDE)),
    /// one by one where the tape has a kernel: `fill(k, lane)` lays out the
    /// `k`th instance's inputs and the slots it starts from, `drain(k,
    /// lane)` takes its results. A block's spare lanes run its last
    /// instance again.
    fn instances(
        &self,
        n: usize,
        ops: std::ops::Range<usize>,
        fill: impl Fn(usize, &mut Lane),
        mut drain: impl FnMut(usize, &Lane),
    ) {
        use crate::constants::{LANES_MID, LANES_NARROW, LANES_PAIR, LANES_WIDE};
        let mut k = 0;
        while k < n {
            let (r, f, d) = (ops.clone(), &fill, &mut drain);
            k += match n - k {
                _ if self.interp.kernels => self.block::<1>(k, 1, r, f, d),
                left if left > LANES_MID => {
                    self.block::<LANES_WIDE>(k, left.min(LANES_WIDE), r, f, d)
                }
                left if left > LANES_NARROW => self.block::<LANES_MID>(k, left, r, f, d),
                left if left > LANES_PAIR => self.block::<LANES_NARROW>(k, left, r, f, d),
                left if left > 1 => self.block::<LANES_PAIR>(k, left, r, f, d),
                _ => self.block::<1>(k, 1, r, f, d),
            };
        }
    }

    /// Instances `k0 .. k0 + live` of [`instances`](Self::instances) in a
    /// block of `B` lanes, over buffers of this thread's; one alone as
    /// [`run_range`](Self::run_range) runs the tape, kernels included.
    /// Returns `live`.
    fn block<const B: usize>(
        &self,
        k0: usize,
        live: usize,
        ops: std::ops::Range<usize>,
        fill: &impl Fn(usize, &mut Lane),
        drain: &mut impl FnMut(usize, &Lane),
    ) -> usize {
        // One instance: the work buffer as an evaluation has it, and every
        // input, which a kernel reads checked. A block: the lanes' slots,
        // the widest list, and the inputs the ops read.
        let (n_slots, n_inputs) = match B {
            1 => (self.work_len(), self.n_inputs.max(self.interp.reach)),
            _ => (self.n_work * B + self.max_args, self.interp.reach * B),
        };
        crate::scratch::with_len(n_slots + n_inputs, 0.0, |buf| {
            let (slots, inputs) = buf.split_at_mut(n_slots);
            for l in 0..B {
                let mut lane = Lane {
                    slots,
                    inputs,
                    l,
                    b: B,
                };
                fill(k0 + l.min(live - 1), &mut lane);
            }
            if B == 1 {
                self.run_range(inputs, slots, ops.start, ops.end, &mut NoTrace);
            } else {
                let (slots, scratch) = slots.split_at_mut(self.n_work * B);
                // SAFETY: the buffers hold `B` lanes of the slots and of
                // the inputs up to `reach`; the tape has no kernel.
                unsafe {
                    let (p, i) = (slots.as_mut_ptr(), inputs.as_ptr());
                    self.lanes::<NoTrace, B>(ops, p, i, scratch, &mut NoTrace)
                };
            }
            for l in 0..live {
                drain(
                    k0 + l,
                    &Lane {
                        slots,
                        inputs,
                        l,
                        b: B,
                    },
                );
            }
        });
        live
    }

    /// Op `i`, a call, as [`calls::run_call`] runs it.
    pub fn call(&self, i: usize) -> calls::Call<'_> {
        let op = &self.ops[i];
        let (bundle, start, n_groups, n_args, n_in, places, n_out, state, args, phase) = match *op {
            Op::Call {
                bundle,
                start,
                n_groups,
                n_args,
                n_in,
                reads,
                n_out,
                state,
                args,
            } => {
                let places = (reads != ALL_ARGS).then(|| self.pool(reads, n_in));
                let phase = match state {
                    NO_STATE => calls::Phase::Whole,
                    _ => calls::Phase::Main,
                };
                (
                    bundle, start, n_groups, n_args, n_in, places, n_out, state, args, phase,
                )
            }
            Op::CallProlog {
                bundle,
                start,
                n_groups,
                n_pure,
                args,
                ..
            } => {
                let sl = self.bundles[bundle as usize].state_len() as u32;
                (
                    bundle,
                    start,
                    n_groups,
                    n_pure,
                    n_pure,
                    None,
                    sl,
                    0,
                    args,
                    calls::Phase::Prolog,
                )
            }
            _ => panic!("op {i} is not a call"),
        };
        let own = match *op {
            Op::CallProlog { own, n_own, .. } => Some(self.pool(own, n_own)),
            _ => None,
        };
        let b = &*self.bundles[bundle as usize];
        calls::Call {
            bundle: b,
            phase,
            n_groups: n_groups as usize,
            n_args: n_args as usize,
            n_out: n_out as usize,
            operands: Some(self.pool(start, n_groups * n_in)),
            places,
            own,
            args: args as usize,
            out: self.dst[i] as usize,
            state: state as usize,
            ops: n_groups as u64 * call_ops(b, op),
        }
    }

    /// Output operands, one per root: a slot, or an input when tagged.
    pub fn outputs(&self) -> &[u32] {
        &self.outputs
    }

    /// The instruction stream, for a backend that lowers it (the native
    /// code, a printer, a code generator): op `i` writes from slot
    /// [`dst`](Self::dst)`(i)`, its lists are windows of [`pool`](Self::pool)
    /// and its bundle an index into [`bundles`](Self::bundles). A slot is
    /// never reused while a value it holds is still needed, so the current
    /// occupant of a slot is the value an operand means. The first
    /// [`prolog_len`](Self::prolog_len) ops are the prolog.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// The destination slot of op `i` (the first of a kernel's block).
    pub fn dst(&self, i: usize) -> u32 {
        self.dst[i]
    }

    /// The inputs the ops `ops` read, ascending.
    pub fn reads(&self, ops: std::ops::Range<usize>) -> Vec<u32> {
        let mut r: Vec<u32> = Vec::new();
        for i in ops {
            self.for_each_operand(i, |k| r.extend(input_index(k)));
        }
        r.sort_unstable();
        r.dedup();
        r
    }

    /// The inputs the main phase reads, ascending: what a call of this
    /// tape as a body needs per evaluation once its prolog ran.
    pub fn main_reads(&self) -> Vec<u32> {
        let mut r = self.reads(self.prolog_ops..self.ops.len());
        r.extend(self.outputs.iter().filter_map(|&o| input_index(o)));
        r.sort_unstable();
        r.dedup();
        r
    }

    /// The widest gather any op needs: the gather area of the work buffer,
    /// after the slots.
    pub fn gather_len(&self) -> usize {
        self.max_args
    }

    /// The scratch lent to a called bundle or a dense solve, after the
    /// gather area.
    pub fn lent(&self) -> usize {
        self.lent
    }

    /// The operand list `start .. start + len` of the pool.
    pub fn pool(&self, start: u32, len: u32) -> &[u32] {
        &self.arg_pool[start as usize..(start + len) as usize]
    }

    /// The bundles the calls call, by index.
    pub fn bundles(&self) -> &[Arc<dyn ExternBundle>] {
        &self.bundles
    }

    /// The bundles' [`forms_epoch`](ExternBundle::forms_epoch) together:
    /// when it moved since a prolog, running the prolog again may give a
    /// faster state (a function body's variant landed in the background).
    pub fn forms_epoch(&self) -> u64 {
        self.bundles
            .iter()
            .fold(0, |e, b| e.wrapping_add(b.forms_epoch()))
    }

    /// Run the tapes the bundles carry of their own as `backend` says (see
    /// [`ExternBundle::with_backend`]). A program calls leaf bodies only
    /// (a composite function is compiled into it as a template), so this
    /// reaches every body it runs.
    pub fn with_backend(&mut self, backend: &crate::extern_fn::BodyBackend) {
        for b in &mut self.bundles {
            if let Some(nb) = b.with_backend(backend) {
                *b = nb;
            }
        }
    }

    /// A kernel's dense operand of `len` values.
    pub fn operand(&self, src: Src, len: u32) -> Operand<'_> {
        match src {
            Src::Inputs(k) => Operand::Inputs(k),
            Src::Pool(start) => Operand::Slots(self.pool(start, len)),
            Src::Slots(s) => Operand::Run(s),
        }
    }

    /// Every operand op `i` reads, in read order: slots, or inputs tagged
    /// with [`INPUT`] (a kernel's run of inputs one by one), and last the
    /// state block of a call that keeps one, by its first slot.
    pub fn for_each_operand(&self, i: usize, mut f: impl FnMut(u32)) {
        let f = &mut f as &mut dyn FnMut(u32);
        let dense = |src: Src, len: u32, f: &mut dyn FnMut(u32)| match src {
            Src::Inputs(k) => (k..k + len).for_each(|j| f(j | INPUT)),
            Src::Pool(start) => self.pool(start, len).iter().for_each(|&k| f(k)),
            Src::Slots(s) => (s..s + len).for_each(f),
        };
        let acc = |acc: Option<Accum>, len: u32, f: &mut dyn FnMut(u32)| {
            if let Some(Accum { c: Some(c), .. }) = acc {
                dense(c, len, f);
            }
        };
        match self.ops[i] {
            Op::Const(_) => {}
            Op::Neg(a) | Op::Powi(a, _) | Op::Unary(_, a) => f(a),
            Op::Add(a, b)
            | Op::Mul(a, b)
            | Op::Sub(a, b)
            | Op::Cmp(_, a, b)
            | Op::Binary(_, a, b) => {
                f(a);
                f(b);
            }
            Op::MulAdd(a, b, c) | Op::Select(a, b, c) => {
                f(a);
                f(b);
                f(c);
            }
            Op::Reduce(_, start, len) => dense(Src::Pool(start), len, f),
            Op::Dot(start, len) => dense(Src::Pool(start), 2 * len, f),
            Op::Call {
                start,
                n_groups,
                n_in,
                state,
                ..
            } => {
                dense(Src::Pool(start), n_groups * n_in, f);
                if state != NO_STATE {
                    f(state);
                }
            }
            Op::CallProlog {
                start,
                n_groups,
                n_pure,
                ..
            } => dense(Src::Pool(start), n_groups * n_pure, f),
            Op::Gemv { a, x, m, n, acc: c } => {
                dense(a, m * n, f);
                dense(x, n, f);
                acc(c, m, f);
            }
            Op::Gemm {
                a,
                b,
                m,
                k,
                n,
                acc: c,
            } => {
                dense(a, m * k, f);
                dense(b, n * k, f);
                acc(c, m * n, f);
            }
            Op::Solve { a, b, n, k, count } => {
                dense(a, count * n * n, f);
                dense(b, count * n * k, f);
            }
        }
    }

    /// How many slots op `i` writes from its destination.
    pub fn width(&self, i: usize) -> u32 {
        match self.ops[i] {
            Op::Call {
                n_groups, n_out, ..
            } => n_groups * n_out,
            Op::CallProlog {
                bundle, n_groups, ..
            } => calls::state_block(n_groups, self.bundles[bundle as usize].state_len() as u32),
            Op::Gemv { m, .. } => m,
            Op::Gemm { m, n, .. } => m * n,
            Op::Solve { n, k, count, .. } => count * n * k,
            _ => 1,
        }
    }

    /// Number of instructions (diagnostics; compare against a
    /// [`SpecializedTape::n_ops`] for the shrink factor).
    pub fn n_ops(&self) -> usize {
        self.ops.len()
    }

    /// Its stages of independent calls, in op order (see [`Stage`]).
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Its regions of independent pieces, in op order (see [`Region`]).
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }
}

use crate::constants::OUTPUT_BLOCK;

/// What an op of the interpreter's own costs (a region's), learned from
/// the runs.
static INTERPRETED: crate::parallel::Rate =
    crate::parallel::Rate::new(crate::constants::INTERPRETED_NS_PER_OP);
/// What an output of a block costs to gather, learned from the runs.
static OUTPUTS: crate::parallel::Rate =
    crate::parallel::Rate::new(crate::constants::OUTPUT_NS_PER_VALUE);

/// The program's outputs, the operands `outputs` (slots of `work`, or
/// tagged inputs), into `out`: in blocks on the installed pool
/// ([`crate::parallel`]) when there are enough of them (a circuit's
/// Jacobian entries, gathered from all over the slots), one after the other
/// otherwise. Every backend writes its outputs so.
#[inline]
pub fn write_outputs(outputs: &[u32], inputs: &[f64], work: &[f64], out: &mut [f64]) {
    let n = outputs.len().min(out.len());
    let block = |o: &mut [f64], ks: &[u32]| {
        for (o, &k) in o.iter_mut().zip(ks) {
            *o = read(inputs, work, k);
        }
    };
    let pieces = n.div_ceil(OUTPUT_BLOCK);
    if pieces <= 1 {
        // one block: nothing to share out, nothing for the rate to learn
        return block(&mut out[..n], &outputs[..n]);
    }
    let dst = Shared::new(out);
    let piece = |p: usize| {
        let (lo, hi) = (p * OUTPUT_BLOCK, ((p + 1) * OUTPUT_BLOCK).min(n));
        // SAFETY: the pieces write disjoint outputs.
        block(unsafe { dst.slice_mut(lo, hi - lo) }, &outputs[lo..hi])
    };
    crate::parallel::run_rated(pieces, n as f64, &OUTPUTS, &piece);
}

/// The value of an operand: a slot, or an input when tagged (a missing
/// input is `NaN`).
#[inline]
pub fn read(inputs: &[f64], work: &[f64], k: u32) -> f64 {
    match input_index(k) {
        Some(i) => inputs.get(i as usize).copied().unwrap_or(f64::NAN),
        None => work[k as usize],
    }
}

/// A dense operand resolved for a kernel: in place in the inputs, or in
/// the scratch from an element on.
#[derive(Clone, Copy)]
enum Dense {
    Inputs(usize),
    Scratch(usize),
    /// A run of consecutive work slots, read in place.
    Work(usize),
}

/// The slice a resolved dense operand names, `len` values long. `work` is
/// the work array's base pointer, the one the kernel's output block is
/// derived from too (see [`out_block`]): a run read in place is disjoint
/// from that block (the allocator gives a kernel a fresh block), and both
/// come from one pointer, so neither borrow invalidates the other.
fn dense_slice<'a>(
    inputs: &'a [f64],
    work: *mut f64,
    scratch: &'a [f64],
    d: Dense,
    len: usize,
) -> &'a [f64] {
    match d {
        Dense::Inputs(k) => &inputs[k..k + len],
        Dense::Scratch(s) => &scratch[s..s + len],
        // SAFETY: `s .. s + len` lies within the work array (the tape's
        // slots) and no kernel writes it while it is read.
        Dense::Work(s) => unsafe { std::slice::from_raw_parts(work.add(s), len) },
    }
}

/// A kernel's output block `d .. d + len` of the work array at `work`
/// (`work_len` long), derived from the same pointer as its in-place reads
/// `reads`, which it must not overlap.
///
/// # Safety
///
/// `work` points to `work_len` initialized values that nothing else
/// borrows for the returned lifetime except the in-place reads, which lie
/// outside the block.
unsafe fn out_block<'a>(
    work: *mut f64,
    work_len: usize,
    d: usize,
    len: usize,
    reads: &[Option<(Dense, usize)>],
) -> &'a mut [f64] {
    assert!(
        d + len <= work_len,
        "kernel output block past the work array"
    );
    for &(r, rlen) in reads.iter().flatten() {
        if let Dense::Work(s) = r {
            assert!(
                s + rlen <= d || d + len <= s,
                "kernel reads its own output block"
            );
        }
    }
    std::slice::from_raw_parts_mut(work.add(d), len)
}

/// Where a dense operand is read from: in place (a run of inputs the inputs
/// reach, a run of work slots), or gathered into `scratch` from `*at` on,
/// which advances past it.
fn place_operand(
    inputs: &[f64],
    work: &[f64],
    scratch: &mut [f64],
    pool: &[u32],
    src: Src,
    len: usize,
    at: &mut usize,
) -> Dense {
    match src {
        Src::Inputs(k) if inputs.len() >= k as usize + len => Dense::Inputs(k as usize),
        Src::Inputs(k) => {
            for j in 0..len {
                scratch[*at + j] = inputs.get(k as usize + j).copied().unwrap_or(f64::NAN);
            }
            *at += len;
            Dense::Scratch(*at - len)
        }
        Src::Slots(s) => Dense::Work(s as usize),
        Src::Pool(start) => {
            let run = &pool[start as usize..start as usize + len];
            for j in 0..len {
                scratch[*at + j] = read(inputs, work, run[j]);
            }
            *at += len;
            Dense::Scratch(*at - len)
        }
    }
}

/// Resolve a kernel's two dense operands, `a` first, then `b` (see
/// [`place_operand`]).
#[allow(clippy::too_many_arguments)]
fn dense_operands(
    inputs: &[f64],
    work: &[f64],
    scratch: &mut [f64],
    pool: &[u32],
    a: Src,
    len_a: usize,
    b: Src,
    len_b: usize,
) -> (Dense, Dense) {
    let mut at = 0usize;
    let ra = place_operand(inputs, work, scratch, pool, a, len_a, &mut at);
    let rb = place_operand(inputs, work, scratch, pool, b, len_b, &mut at);
    (ra, rb)
}

/// The accumulator part of a kernel's dump line: the operand and the
/// fold codes.
fn state_text(state: u32) -> String {
    if state == NO_STATE {
        String::new()
    } else {
        format!(", state @{state}")
    }
}

fn acc_text(pool: &[u32], acc: Option<Accum>, len: u32) -> String {
    match acc {
        None => String::new(),
        Some(Accum { c, codes }) => {
            let codes = &pool[codes as usize..(codes + len) as usize];
            let text: Vec<String> = codes
                .iter()
                .map(|&code| {
                    let f = Fold(code);
                    let op = match code & Fold::KIND {
                        0 => "=",
                        1 => "-",
                        2 => "+",
                        _ => "neg",
                    };
                    if f.is_self() {
                        format!("{op}#{}", f.self_index())
                    } else {
                        op.to_string()
                    }
                })
                .collect();
            format!(
                " acc {} [{}]",
                match c {
                    Some(Src::Inputs(k)) => format!("i{k}..i{}", k + len),
                    Some(Src::Pool(start)) => format!("pool{start}[{len}]"),
                    Some(Src::Slots(s)) => format!("s{s}..s{}", s + len),
                    None => "-".to_string(),
                },
                text.join(",")
            )
        }
    }
}
