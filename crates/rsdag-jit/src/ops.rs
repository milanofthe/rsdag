//! What the emitter asks of the tape's ops beyond what the tape states:
//! which of them the ISA runs inline and which call a host routine, and
//! when the value a slot holds dies.

use rsdag::node::{ReduceOp, UnaryOp};
use rsdag::tape::{input_index, Op};
use rsdag::Tape;

use crate::constants::{HOST_PRODUCT_MIN_ROWS, INLINE_MINMAX_MAX_TERMS, POWI_INLINE_MAX_EXPONENT};
use crate::isa::Isa;

pub(crate) fn powi_inline(n: i32) -> Option<std::num::NonZeroU32> {
    std::num::NonZeroU32::new(n.unsigned_abs()).filter(|e| e.get() <= POWI_INLINE_MAX_EXPONENT)
}

pub(crate) fn inline_product(op: &Op) -> bool {
    matches!(*op, Op::Gemv { m, .. } | Op::Gemm { m, .. } if m < HOST_PRODUCT_MIN_ROWS)
}

pub(crate) fn minmax_inline<I: Isa>(op: &Op) -> bool {
    matches!(*op, Op::Reduce(ReduceOp::Min | ReduceOp::Max, _, len)
        if I::MINMAX && len > 0 && len as usize <= INLINE_MINMAX_MAX_TERMS)
}

pub(crate) fn routine<I: Isa>(op: &Op) -> Option<*const ()> {
    if minmax_inline::<I>(op) || inline_product(op) {
        return None;
    }
    Some(match *op {
        Op::Unary(op, _) => match op {
            UnaryOp::Sqrt | UnaryOp::Abs | UnaryOp::Sign => return None,
            UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Trunc if I::round_inline() => return None,
            UnaryOp::Exp | UnaryOp::Ln => return None,
            _ => crate::host::unary_addr(op).0,
        },
        Op::Binary(..) => crate::host::h_binary as *const (),
        Op::Powi(_, n) if powi_inline(n).is_none() => crate::host::h_powi as *const (),
        Op::Reduce(ReduceOp::Min | ReduceOp::Max, ..) => crate::host::h_reduce as *const (),
        Op::Call { .. } | Op::CallProlog { .. } => crate::host::h_call as *const (),
        Op::Gemv { .. } | Op::Gemm { .. } | Op::Solve { .. } => crate::host::h_kernel as *const (),
        _ => return None,
    })
}

pub(crate) struct Liveness {
    reads: Vec<u32>,
    read_start: Vec<u32>,
    writes: Vec<u32>,
    write_start: Vec<u32>,
}

impl Liveness {
    pub(crate) fn new(tape: &Tape, outputs: impl Iterator<Item = u32>) -> Liveness {
        let n_work = tape.n_slots();
        let outputs: Vec<u32> = outputs.filter(|&s| input_index(s).is_none()).collect();
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
            debug_assert!(
                (0..n_work).all(|s| at[start[s] as usize..start[s + 1] as usize].is_sorted()),
                "positions per slot in tape order"
            );
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

    pub(crate) fn next_read(&self, s: u32, pos: u32) -> u32 {
        let s = s as usize;
        let reads = &self.reads[self.read_start[s] as usize..self.read_start[s + 1] as usize];
        reads
            .get(reads.partition_point(|&r| r < pos))
            .copied()
            .unwrap_or(u32::MAX)
    }

    pub(crate) fn death(&self, s: u32, pos: u32, def: bool) -> u32 {
        let s = s as usize;
        let reads = &self.reads[self.read_start[s] as usize..self.read_start[s + 1] as usize];
        let writes = &self.writes[self.write_start[s] as usize..self.write_start[s + 1] as usize];
        let first_read = if def { pos + 1 } else { pos };
        let overwriting_op_reads_first = writes
            .get(writes.partition_point(|&w| w < first_read))
            .copied()
            .unwrap_or(u32::MAX);
        let from = reads.partition_point(|&r| r < first_read);
        let to = reads.partition_point(|&r| r <= overwriting_op_reads_first);
        if to > from {
            reads[to - 1]
        } else {
            pos
        }
    }
}
