//! What the emitter asks of the tape's ops beyond what the tape states:
//! which of them the ISA runs inline and which call a host routine, and
//! when the value a slot holds dies.

use rsdag::node::{ReduceOp, UnaryOp};
use rsdag::tape::{input_index, Op};
use rsdag::Tape;

use crate::constants::{INLINE_MINMAX_MAX, INLINE_ROWS, POWI_INLINE};
use crate::isa::Isa;

/// Whether `x^n` is emitted inline (as `semantics::powi_t` multiplies,
/// product for product) rather than through the host.
pub(crate) fn powi_inline(n: i32) -> bool {
    n != 0 && n.unsigned_abs() <= POWI_INLINE
}

/// Whether `op` is a product kernel the code computes inline, as the dots
/// of its entries (see [`INLINE_ROWS`]).
pub(crate) fn inline_product(op: &Op) -> bool {
    matches!(*op, Op::Gemv { m, .. } | Op::Gemm { m, .. } if m < INLINE_ROWS)
}

/// Whether `op` is a min or max the ISA `I` runs as a chain of its
/// instructions (see [`INLINE_MINMAX_MAX`]), calling no host routine.
pub(crate) fn minmax_inline<I: Isa>(op: &Op) -> bool {
    matches!(*op, Op::Reduce(ReduceOp::Min | ReduceOp::Max, _, len)
        if I::MINMAX && len > 0 && len as usize <= INLINE_MINMAX_MAX)
}

/// The host routine `op` calls on the ISA `I`, if any.
pub(crate) fn routine<I: Isa>(op: &Op) -> Option<*const ()> {
    if minmax_inline::<I>(op) || inline_product(op) {
        return None;
    }
    Some(match *op {
        Op::Unary(op, _) => match op {
            UnaryOp::Sqrt
            | UnaryOp::Floor
            | UnaryOp::Ceil
            | UnaryOp::Trunc
            | UnaryOp::Abs
            | UnaryOp::Sign => return None,
            UnaryOp::Exp | UnaryOp::Ln if I::exp_ln_inline() => return None,
            _ => crate::host::unary_addr(op).0,
        },
        Op::Binary(..) => crate::host::h_binary as *const (),
        Op::Powi(_, n) if !powi_inline(n) => crate::host::h_powi as *const (),
        Op::Reduce(ReduceOp::Min | ReduceOp::Max, ..) => crate::host::h_reduce as *const (),
        Op::Call { .. } | Op::CallProlog { .. } => crate::host::h_call as *const (),
        Op::Gemv { .. } | Op::Gemm { .. } | Op::Solve { .. } => crate::host::h_kernel as *const (),
        _ => return None,
    })
}

/// When each value dies: per slot, the ops that read it and the ops that
/// write it, in tape order, the outputs read after the program. Slots
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
    /// Over the ops of `tape`, the slots `outputs` read at the end.
    pub(crate) fn new(tape: &Tape, outputs: impl Iterator<Item = u32>) -> Liveness {
        let n_work = tape.n_slots();
        let outputs: Vec<u32> = outputs.filter(|&s| input_index(s).is_none()).collect();
        // Two passes each, counting then filling, in tape order so the
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
            for i in 0..tape.n_ops() {
                tape.for_each_operand(i, |k| {
                    if input_index(k).is_none() {
                        f(k, i as u32)
                    }
                });
            }
            for &s in &outputs {
                f(s, u32::MAX);
            }
        });
        let (writes, write_start) = csr(&|f| {
            for i in 0..tape.n_ops() {
                let d = tape.dst(i);
                (d..d + tape.width(i)).for_each(|s| f(s, i as u32));
            }
        });
        Liveness {
            reads,
            read_start,
            writes,
            write_start,
        }
    }

    /// The first op at or after `pos` that reads slot `s` (`u32::MAX` for the
    /// outputs at the end, or none).
    pub(crate) fn next_read(&self, s: u32, pos: u32) -> u32 {
        let s = s as usize;
        let reads = &self.reads[self.read_start[s] as usize..self.read_start[s + 1] as usize];
        reads
            .get(reads.partition_point(|&r| r < pos))
            .copied()
            .unwrap_or(u32::MAX)
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
