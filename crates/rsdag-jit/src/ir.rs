//! The op stream: the tape's ops as a flat vector the emitter walks,
//! their lists resolved, so the chunk partitioning and the parallel
//! per-chunk codegen work on owned data. Bundles are indexed as the tape
//! indexes them, through the table every chunk receives. Operands are the
//! tape's: slots, or inputs when tagged (see [`rsdag::tape::INPUT`]).

use rsdag::node::{BinOp, CmpOp, ReduceOp, UnaryOp};
use rsdag::tape::{input_index, Accum, Op, Operand, Src, NO_STATE};
use rsdag::Tape;

pub(crate) enum ROp {
    Const(u32, f64),
    Add(u32, u32, u32),
    Mul(u32, u32, u32),
    MulAdd(u32, u32, u32, u32),
    Sub(u32, u32, u32),
    Neg(u32, u32),
    Powi(u32, u32, i32),
    Unary(u32, UnaryOp, u32),
    Binary(u32, BinOp, u32, u32),
    Cmp(u32, CmpOp, u32, u32),
    Select(u32, u32, u32, u32),
    Reduce(u32, ReduceOp, Vec<u32>),
    Dot(u32, Vec<u32>, Vec<u32>),
    /// A bundle call of any form (see [`CallSite`]).
    Call(CallSite),
    Gemv {
        dst: u32,
        a: Dense,
        x: Dense,
        m: u32,
        n: u32,
        /// The accumulator operand, the fold codes and, once compiled,
        /// the address of the codes' table.
        acc: Option<(Option<Dense>, Vec<u32>, usize)>,
    },
    Gemm {
        dst: u32,
        a: Dense,
        b: Dense,
        m: u32,
        k: u32,
        n: u32,
        /// The accumulator operand, the fold codes and, once compiled,
        /// the address of the codes' table.
        acc: Option<(Option<Dense>, Vec<u32>, usize)>,
    },
    Solve {
        dst: u32,
        a: Dense,
        b: Dense,
        n: u32,
    },
    SolveMany {
        dst: u32,
        a: Dense,
        b: Dense,
        n: u32,
        k: u32,
    },
}

/// What a call computes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u64)]
pub(crate) enum CallKind {
    /// The bundle whole, per group.
    Whole = 0,
    /// The bundle's main phase over each group's state.
    Main = 1,
    /// The bundle's prolog over each group's pure arguments, its outputs
    /// the states.
    Prolog = 2,
}

/// One call site: `n_groups` instances of bundle `bundle` on `args`
/// (group-major, `n_args` each), `n_out` values per group from `dst` on;
/// for [`CallKind::Main`], group `g`'s state at `state + g * state_len`.
/// For a prolog, `n_args` counts pure arguments and `n_out` is the state
/// length.
pub(crate) struct CallSite {
    pub(crate) dst: u32,
    pub(crate) bundle: u32,
    pub(crate) args: Vec<u32>,
    pub(crate) n_groups: u32,
    pub(crate) n_args: u32,
    pub(crate) n_out: u32,
    pub(crate) kind: CallKind,
    pub(crate) state: u32,
    pub(crate) state_len: u32,
    /// Several instances through the bundle's own batch entry.
    pub(crate) batch: bool,
}

/// A dense operand: a run of inputs or of work slots read in place, or
/// slots gathered.
pub(crate) enum Dense {
    Inputs(u32),
    /// `len` work slots from `s`.
    Run(u32, u32),
    Slots(Vec<u32>),
}

impl Dense {
    /// Every work slot the operand reads.
    pub(crate) fn for_each_slot(&self, mut f: impl FnMut(u32)) {
        match *self {
            Dense::Inputs(_) => {}
            Dense::Run(s, len) => (s..s + len).for_each(f),
            Dense::Slots(ref v) => v.iter().for_each(|&k| f(k)),
        }
    }
    /// How many values it gathers.
    fn gathered(&self) -> usize {
        match self {
            Dense::Slots(v) => v.len(),
            _ => 0,
        }
    }
    fn of(o: Operand<'_>, len: u32) -> Dense {
        match o {
            Operand::Inputs(k) => Dense::Inputs(k),
            Operand::Run(s) => Dense::Run(s, len),
            Operand::Slots(s) => Dense::Slots(s.to_vec()),
        }
    }
}

impl ROp {
    /// The dense operands of a kernel, the accumulator last.
    fn dense(&self) -> impl Iterator<Item = &Dense> {
        let (a, b, c): (Option<&Dense>, Option<&Dense>, Option<&Dense>) = match self {
            ROp::Gemv { a, x, acc, .. } => {
                (Some(a), Some(x), acc.as_ref().and_then(|c| c.0.as_ref()))
            }
            ROp::Gemm { a, b, acc, .. } => {
                (Some(a), Some(b), acc.as_ref().and_then(|c| c.0.as_ref()))
            }
            ROp::Solve { a, b, .. } | ROp::SolveMany { a, b, .. } => (Some(a), Some(b), None),
            _ => (None, None, None),
        };
        a.into_iter().chain(b).chain(c)
    }
    /// Every operand the op reads: slots, and the tagged inputs a scalar
    /// op or a list names (a kernel's input run is read by address).
    pub(crate) fn for_each_operand(&self, mut f: impl FnMut(u32)) {
        match self {
            ROp::Const(..) => {}
            ROp::Neg(_, a) | ROp::Powi(_, a, _) | ROp::Unary(_, _, a) => f(*a),
            ROp::Add(_, a, b)
            | ROp::Mul(_, a, b)
            | ROp::Sub(_, a, b)
            | ROp::Cmp(_, _, a, b)
            | ROp::Binary(_, _, a, b) => {
                f(*a);
                f(*b);
            }
            ROp::MulAdd(_, a, b, c) | ROp::Select(_, a, b, c) => {
                f(*a);
                f(*b);
                f(*c);
            }
            ROp::Reduce(_, _, args) => args.iter().copied().for_each(f),
            ROp::Call(c) => c.args.iter().copied().for_each(f),
            ROp::Dot(_, a, b) => a.iter().chain(b).copied().for_each(f),
            _ => self.dense().for_each(|d| d.for_each_slot(&mut f)),
        }
    }
    /// Every work slot the op reads.
    pub(crate) fn for_each_read(&self, mut f: impl FnMut(u32)) {
        self.for_each_operand(|k| {
            if input_index(k).is_none() {
                f(k)
            }
        });
    }
    /// The slots the op writes: its destination, a kernel's block from it.
    pub(crate) fn writes(&self) -> std::ops::Range<u32> {
        let (dst, n) = match *self {
            ROp::Const(d, _)
            | ROp::Add(d, ..)
            | ROp::Mul(d, ..)
            | ROp::MulAdd(d, ..)
            | ROp::Sub(d, ..)
            | ROp::Neg(d, _)
            | ROp::Powi(d, ..)
            | ROp::Unary(d, ..)
            | ROp::Binary(d, ..)
            | ROp::Cmp(d, ..)
            | ROp::Select(d, ..)
            | ROp::Reduce(d, ..)
            | ROp::Dot(d, ..) => (d, 1),
            ROp::Call(ref c) => (c.dst, c.n_groups * c.n_out),
            ROp::Gemv { dst, m, .. } => (dst, m),
            ROp::Gemm { dst, m, n, .. } => (dst, m * n),
            ROp::Solve { dst, n, .. } => (dst, n),
            ROp::SolveMany { dst, n, k, .. } => (dst, n * k),
        };
        dst..dst + n
    }
    /// The host routine the op calls, if any.
    pub(crate) fn host(&self) -> Option<*const ()> {
        Some(match self {
            ROp::Unary(_, op, _) => match op {
                UnaryOp::Sqrt
                | UnaryOp::Floor
                | UnaryOp::Ceil
                | UnaryOp::Trunc
                | UnaryOp::Abs
                | UnaryOp::Sign => return None,
                _ => crate::host::unary_addr(*op).0,
            },
            ROp::Binary(..) => crate::host::h_binary as *const (),
            ROp::Powi(_, _, n) if *n != -1 && *n != 2 => crate::host::h_powi as *const (),
            ROp::Reduce(_, ReduceOp::Min | ReduceOp::Max, _) => crate::host::h_reduce as *const (),
            ROp::Call(..) => crate::host::h_call as *const (),
            ROp::Gemv { acc: None, .. } => crate::host::h_gemv as *const (),
            ROp::Gemv { .. } => crate::host::h_gemv_acc as *const (),
            ROp::Gemm { acc: None, .. } => crate::host::h_gemm as *const (),
            ROp::Gemm { .. } => crate::host::h_gemm_acc as *const (),
            ROp::Solve { .. } => crate::host::h_solve as *const (),
            ROp::SolveMany { .. } => crate::host::h_solve_many as *const (),
            _ => return None,
        })
    }
    /// How many values the op hands to a host routine through the gather
    /// area of the work array.
    pub(crate) fn gather_len(&self) -> usize {
        match self {
            ROp::Reduce(_, ReduceOp::Min | ReduceOp::Max, args) => args.len(),
            ROp::Call(c) => c.args.len(),
            _ => self.dense().map(Dense::gathered).sum(),
        }
    }
}

/// The op stream of a tape, its bundles indexed as the tape indexes them.
pub(crate) fn record(tape: &Tape) -> Vec<ROp> {
    let dense = |s: Src, len: u32| Dense::of(tape.operand(s, len), len);
    let acc = |a: Option<Accum>, len: u32| {
        a.map(|a| {
            (
                a.c.map(|c| dense(c, len)),
                tape.pool(a.codes, len).to_vec(),
                0,
            )
        })
    };
    (0..tape.ops().len())
        .map(|i| {
            let dst = tape.dst(i);
            match tape.ops()[i] {
                Op::Const(v) => ROp::Const(dst, v),
                Op::Add(a, b) => ROp::Add(dst, a, b),
                Op::Mul(a, b) => ROp::Mul(dst, a, b),
                Op::MulAdd(a, b, c) => ROp::MulAdd(dst, a, b, c),
                Op::Sub(a, b) => ROp::Sub(dst, a, b),
                Op::Neg(a) => ROp::Neg(dst, a),
                Op::Powi(a, n) => ROp::Powi(dst, a, n),
                Op::Unary(op, a) => ROp::Unary(dst, op, a),
                Op::Binary(op, a, b) => ROp::Binary(dst, op, a, b),
                Op::Cmp(op, a, b) => ROp::Cmp(dst, op, a, b),
                Op::Select(c, t, e) => ROp::Select(dst, c, t, e),
                Op::Reduce(op, s, l) => ROp::Reduce(dst, op, tape.pool(s, l).to_vec()),
                Op::Dot(s, l) => {
                    ROp::Dot(dst, tape.pool(s, l).to_vec(), tape.pool(s + l, l).to_vec())
                }
                Op::Call {
                    bundle,
                    start,
                    n_groups,
                    n_args,
                    n_out,
                    state,
                } => ROp::Call(CallSite {
                    dst,
                    bundle,
                    args: tape.pool(start, n_groups * n_args).to_vec(),
                    n_groups,
                    n_args,
                    n_out,
                    kind: if state == NO_STATE {
                        CallKind::Whole
                    } else {
                        CallKind::Main
                    },
                    state: if state == NO_STATE { 0 } else { state },
                    state_len: tape.bundles()[bundle as usize].state_len() as u32,
                    batch: n_groups > 1,
                }),
                Op::CallProlog {
                    bundle,
                    start,
                    n_groups,
                    n_pure,
                } => {
                    let state_len = tape.bundles()[bundle as usize].state_len() as u32;
                    ROp::Call(CallSite {
                        dst,
                        bundle,
                        args: tape.pool(start, n_groups * n_pure).to_vec(),
                        n_groups,
                        n_args: n_pure,
                        n_out: state_len,
                        kind: CallKind::Prolog,
                        state: dst,
                        state_len,
                        batch: n_groups > 1,
                    })
                }
                Op::Gemv { a, x, m, n, acc: c } => ROp::Gemv {
                    dst,
                    a: dense(a, m * n),
                    x: dense(x, n),
                    m,
                    n,
                    acc: acc(c, m),
                },
                Op::Gemm {
                    a,
                    b,
                    m,
                    k,
                    n,
                    acc: c,
                } => ROp::Gemm {
                    dst,
                    a: dense(a, m * k),
                    b: dense(b, n * k),
                    m,
                    k,
                    n,
                    acc: acc(c, m * n),
                },
                Op::Solve { a, b, n, k: 1 } => ROp::Solve {
                    dst,
                    a: dense(a, n * n),
                    b: dense(b, n),
                    n,
                },
                Op::Solve { a, b, n, k } => ROp::SolveMany {
                    dst,
                    a: dense(a, n * n),
                    b: dense(b, n * k),
                    n,
                    k,
                },
            }
        })
        .collect()
}

/// When each value dies: per slot, the ops that read it and the ops that
/// write it, in stream order, the outputs read after the program. Slots
/// are reused, so a slot holds many values in turn; the one it holds at an
/// op dies at its last read before the slot's next write (an op reads its
/// operands before it writes). The register cache asks this when it takes
/// a value, so a value is written back only if a later op reads it and
/// counts as dead as soon as nothing will.
pub(crate) struct Liveness {
    reads: Vec<u32>,
    read_start: Vec<u32>,
    writes: Vec<u32>,
    write_start: Vec<u32>,
}

impl Liveness {
    /// Over `ops` with `n_work` slots, the slots `outputs` read at the end.
    pub(crate) fn new(ops: &[ROp], n_work: usize, outputs: impl Iterator<Item = u32>) -> Liveness {
        let outputs: Vec<u32> = outputs.filter(|&s| input_index(s).is_none()).collect();
        // Two passes each, counting then filling, in stream order so the
        // positions of a slot come out sorted.
        let csr = |each: &dyn Fn(&mut dyn FnMut(u32, u32))| -> (Vec<u32>, Vec<u32>) {
            let mut start = vec![0u32; n_work + 1];
            each(&mut |s, _| start[s as usize + 1] += 1);
            for s in 0..n_work {
                start[s + 1] += start[s];
            }
            let mut fill = start.clone();
            let mut at = vec![0u32; start[n_work] as usize];
            each(&mut |s, pos| {
                at[fill[s as usize] as usize] = pos;
                fill[s as usize] += 1;
            });
            (at, start)
        };
        let (reads, read_start) = csr(&|f| {
            for (i, op) in ops.iter().enumerate() {
                op.for_each_read(|s| f(s, i as u32));
            }
            for &s in &outputs {
                f(s, u32::MAX);
            }
        });
        let (writes, write_start) = csr(&|f| {
            for (i, op) in ops.iter().enumerate() {
                op.writes().for_each(|s| f(s, i as u32));
            }
        });
        Liveness {
            reads,
            read_start,
            writes,
            write_start,
        }
    }

    /// The last op reading the value slot `s` holds when op `pos` writes it
    /// (`def`) or reads it; `pos` when nothing reads it after, `u32::MAX`
    /// when the program's outputs do.
    pub(crate) fn death(&self, s: u32, pos: u32, def: bool) -> u32 {
        let s = s as usize;
        let reads = &self.reads[self.read_start[s] as usize..self.read_start[s + 1] as usize];
        let writes = &self.writes[self.write_start[s] as usize..self.write_start[s + 1] as usize];
        // The value lives from `lo` to the next write, whose op still reads
        // it.
        let lo = if def { pos + 1 } else { pos };
        let end = writes
            .get(writes.partition_point(|&w| w < lo))
            .copied()
            .unwrap_or(u32::MAX);
        let from = reads.partition_point(|&r| r < lo);
        let to = reads.partition_point(|&r| r <= end);
        if to > from {
            reads[to - 1]
        } else {
            pos
        }
    }
}
