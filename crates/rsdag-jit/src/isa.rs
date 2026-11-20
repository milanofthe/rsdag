//! The instruction layer an architecture provides to the emitter.
//!
//! The emitter in [`native`](crate::native) is architecture-neutral: it keeps
//! the value cache, walks the op stream and decides what to compute in which
//! register. Everything that is a matter of encoding goes through this trait,
//! so a second architecture is one file of encodings and nothing else.
//!
//! Registers are the architecture's own float register numbers. A result
//! register handed to a method is never one of its operands, except for
//! [`arith`](Isa::arith), whose result may alias the first operand (the
//! folds accumulate in place).

use rsdag::node::{CmpOp, ReduceOp};

/// Which base pointer a memory operand is relative to.
#[derive(Clone, Copy)]
pub(crate) enum Base {
    /// The work array (slots, bundle scratch, gather area).
    Work,
    /// The input vector.
    Inputs,
}

/// An integer argument of a host call.
#[derive(Clone, Copy)]
pub(crate) enum IArg {
    Imm(u64),
    /// The address of a byte offset into the work array.
    WorkAddr(usize),
    /// The address of a byte offset into the input vector.
    InputAddr(usize),
    /// The layout pointer the chunk received.
    Layout,
}

/// One argument of a host call, in the routine's signature order: the
/// conventions differ in whether floats and integers are counted together.
#[derive(Clone, Copy)]
pub(crate) enum Arg {
    F(u8),
    I(IArg),
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Arith {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy)]
pub(crate) enum Round {
    Floor,
    Ceil,
    Trunc,
}

pub(crate) trait Isa {
    /// Values per register: `1` for scalar code, more for code that runs
    /// that many instances side by side, one per lane (a slot is then
    /// `LANES` consecutive values in the work array, an input likewise).
    const LANES: usize;
    /// The registers the value cache may use, in eviction order. The first
    /// `SAVED` of them survive a host call.
    const CACHE: &'static [u8];
    const SAVED: usize;
    /// Where a host call leaves its f64 result.
    const RESULT: u8;

    /// `hot` lists host routines the chunk calls often, most frequent first;
    /// an architecture keeps as many of them in callee-saved registers as it
    /// has to spare, so those calls need no address immediate.
    fn new(hot: &[*const ()]) -> Self;
    fn finish(self) -> Vec<u8>;

    /// Function entry: `fn(work: *mut f64, inputs: *const f64, layout: *const _)`
    /// in the platform's C convention, the three pointers kept in callee-saved
    /// registers for the chunk's lifetime.
    fn prologue(&mut self);
    fn epilogue(&mut self);

    /// A register's `LANES` values from or to memory.
    fn load(&mut self, r: u8, base: Base, off: usize);
    fn store(&mut self, r: u8, base: Base, off: usize);
    /// One value, into a register's first lane or from it (the scalar
    /// arguments and result of a host call made lane by lane).
    fn load_lane(&mut self, r: u8, base: Base, off: usize);
    fn store_lane(&mut self, r: u8, base: Base, off: usize);
    fn fconst(&mut self, r: u8, v: f64);
    fn mov(&mut self, d: u8, a: u8);

    /// `d = a op b`; `d` may alias `a`.
    fn arith(&mut self, op: Arith, d: u8, a: u8, b: u8);
    /// Whether [`arith_mem`](Self::arith_mem) is available.
    const MEM_OPERANDS: bool = false;
    /// `d = a op [base + off]`, the second operand read from memory; `d`
    /// may alias `a`.
    fn arith_mem(&mut self, _op: Arith, _d: u8, _a: u8, _base: Base, _off: usize) {
        unreachable!("no memory operands")
    }
    fn neg(&mut self, d: u8, a: u8);
    fn abs(&mut self, d: u8, a: u8);
    fn sqrt(&mut self, d: u8, a: u8);
    /// Whether this CPU runs [`round`](Self::round); without it the caller
    /// uses the host routine.
    fn round_inline() -> bool;
    fn round(&mut self, mode: Round, d: u8, a: u8);
    /// Whether [`minmax`](Self::minmax) is available; without it the
    /// caller uses the host routine.
    const MINMAX: bool;
    /// `d = min(a, b)` or `max(a, b)` with the reference's NaN rule (a NaN
    /// operand yields the other), as one instruction.
    fn minmax(&mut self, op: ReduceOp, d: u8, a: u8, b: u8);

    /// `d = (a op b) ? t : e`, every comparison with a NaN false except `Ne`.
    fn cmp_select(&mut self, op: CmpOp, a: u8, b: u8, t: u8, e: u8, d: u8);
    /// `d = (c != 0) ? t : e` (a NaN condition selects `t`).
    fn select_nz(&mut self, c: u8, t: u8, e: u8, d: u8);

    /// `d = exp(x)`, or `ln(x)` with `ln`, as the reference's guarded
    /// functions compute them, bit for bit, in the code itself: no call, so
    /// the cache keeps its values. `t` are five scratch registers, none of
    /// them `d` or `x`.
    fn exp_ln(&mut self, ln: bool, d: u8, x: u8, t: [u8; 5]);

    /// Call `addr` with `args` in signature order; an f64 result is in
    /// `RESULT`.
    fn call(&mut self, addr: *const (), args: &[Arg]);
}
