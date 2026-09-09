//! A direct machine-code emitter for the op stream, without Cranelift.
//!
//! Cranelift costs 4 to 15 us per op to compile, almost all of it IR
//! construction and register allocation, and that cost was negligible while
//! a program was evaluated thousands of times per build. With the solve in
//! the graph the programs are five times larger and are rebuilt per sparsity
//! pattern, so the compiler is on the clock again.
//!
//! The tape already knows what a code generator needs: it is straight-line
//! code over a work array of slots, with every value's lifetime computed.
//! So this backend does the least a compiler can do: for each op, operands
//! come from a small register cache or a load from the work array, one
//! instruction computes, and the result is stored back and cached. Storing
//! through means a chunk boundary needs no special handling, a host call
//! (which clobbers the caller-saved registers the cache lives in) just
//! empties the cache, and nothing is ever spilled anywhere but where the
//! interpreter keeps it anyway. Chunks are emitted in parallel.
//!
//! Only AArch64 for now, as the experiment that decides whether the
//! approach earns a second architecture. What it does not lower it refuses
//! (`Min`/`Max` reductions, bundle calls), so a caller falls back to the
//! Cranelift backend for those programs.

use rayon::prelude::*;
use rsdag::node::{BinOp, CmpOp, ReduceOp, UnaryOp, REDUCE_SIMD_MIN};
use rsdag::Tape;

use crate::host::host_addr;
use crate::scalar::{ROp, Recorder};
use crate::{JitError, CHUNK_OPS};

/// A tape compiled by the direct emitter. Same evaluation contract as
/// [`crate::ChunkedTape`]: caller-owned work buffer, inputs padded with NaN.
pub struct EmittedTape {
    chunks: Vec<Code>,
    n_work: usize,
    n_inputs: usize,
    outputs: Vec<u32>,
}

/// One executable chunk.
struct Code {
    map: Mapping,
    func: extern "C" fn(*mut f64, *const f64, *const u8),
}
// The mapping is immutable after `finish`, so calling the code from any
// thread is sound and the chunks can be built on a rayon pool.
unsafe impl Send for Code {}
unsafe impl Sync for Code {}

impl EmittedTape {
    pub fn compile(tape: &Tape) -> Result<EmittedTape, JitError> {
        Self::compile_with(tape, CHUNK_OPS)
    }

    pub fn compile_with(tape: &Tape, chunk_ops: usize) -> Result<EmittedTape, JitError> {
        let mut rec = Recorder::default();
        tape.lower(&mut rec);
        if !rec.bundles.is_empty() {
            return Err(JitError::Codegen(
                "the direct emitter does not lower bundle calls".into(),
            ));
        }
        let n_inputs = rec
            .ops
            .iter()
            .filter_map(|op| match op {
                ROp::Input(_, k) if *k != u32::MAX => Some(*k as usize + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let chunks: Result<Vec<Code>, JitError> = rec
            .ops
            .chunks(chunk_ops.max(1))
            .collect::<Vec<_>>()
            .par_iter()
            .map(|ops| emit_chunk(ops))
            .collect();
        Ok(EmittedTape {
            chunks: chunks?,
            n_work: tape.n_work(),
            n_inputs,
            outputs: tape.outputs().to_vec(),
        })
    }

    pub fn n_chunks(&self) -> usize {
        self.chunks.len()
    }

    pub fn eval(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        let padded: Vec<f64>;
        let ins: &[f64] = if inputs.len() < self.n_inputs {
            padded = {
                let mut v = inputs.to_vec();
                v.resize(self.n_inputs, f64::NAN);
                v
            };
            &padded
        } else {
            inputs
        };
        work.clear();
        work.resize(self.n_work.max(1), 0.0);
        for c in &self.chunks {
            (c.func)(work.as_mut_ptr(), ins.as_ptr(), std::ptr::null());
        }
        out.clear();
        out.extend(self.outputs.iter().map(|&s| work[s as usize]));
    }
}

// --- the AArch64 encoder ------------------------------------------------------
//
// Register plan (AAPCS64): x0 work, x1 inputs, x2 ctx on entry, kept in the
// callee-saved x19, x20, x21 across host calls; x9 to x16 scratch; d0 and
// d1 carry host-call arguments and d0 the result; d16 to d31 are the value
// cache, caller-saved, so a host call simply empties it.

const WORK: u32 = 19;
const INPUTS: u32 = 20;
const CACHE_LO: u32 = 16;
const CACHE_N: u32 = 16;

struct Emitter {
    code: Vec<u32>,
    /// Slot held by each cache register, if any.
    held: [Option<u32>; CACHE_N as usize],
    /// Cache register holding each slot, if any.
    reg_of: rustc_hash::FxHashMap<u32, u32>,
    /// Round-robin victim pointer.
    next: usize,
    /// Registers an op still needs: `fresh` never evicts these. Every
    /// operand `get` pins its register, every temporary is pinned by its
    /// maker, and the pins clear when the op is done.
    pinned: [bool; CACHE_N as usize],
}

// Condition codes after `fcmp`. The unordered case (a NaN) sets C and V and
// clears N and Z; these choices make every comparison with a NaN false
// except `ne`, which is how the interpreter's `cmp_bool` behaves.
const COND_EQ: u32 = 0x0;
const COND_NE: u32 = 0x1;
const COND_MI: u32 = 0x4; // lt
const COND_LS: u32 = 0x9; // le
const COND_GE: u32 = 0xA;
const COND_GT: u32 = 0xC;

impl Emitter {
    fn new() -> Emitter {
        Emitter {
            code: Vec::with_capacity(4096),
            held: [None; CACHE_N as usize],
            reg_of: Default::default(),
            next: 0,
            pinned: [false; CACHE_N as usize],
        }
    }
    fn w(&mut self, word: u32) {
        self.code.push(word);
    }

    // -- the cache --------------------------------------------------------

    /// A cache register to write a new value into: the next unpinned one
    /// round-robin, evicting whatever it held. The result is pinned, so a
    /// later `fresh` within the same op cannot take it back.
    fn fresh(&mut self) -> u32 {
        for _ in 0..CACHE_N as usize {
            let i = self.next;
            self.next = (self.next + 1) % CACHE_N as usize;
            if self.pinned[i] {
                continue;
            }
            if let Some(s) = self.held[i].take() {
                self.reg_of.remove(&s);
            }
            self.pinned[i] = true;
            return CACHE_LO + i as u32;
        }
        unreachable!("an op pins fewer than {CACHE_N} registers");
    }
    fn bind(&mut self, reg: u32, slot: u32) {
        if let Some(old_reg) = self.reg_of.remove(&slot) {
            self.held[(old_reg - CACHE_LO) as usize] = None;
        }
        let i = (reg - CACHE_LO) as usize;
        if let Some(old) = self.held[i].replace(slot) {
            self.reg_of.remove(&old);
        }
        self.reg_of.insert(slot, reg);
    }
    fn forget_all(&mut self) {
        self.held = [None; CACHE_N as usize];
        self.reg_of.clear();
    }
    fn unpin_all(&mut self) {
        self.pinned = [false; CACHE_N as usize];
    }
    /// The register holding `slot`, loading it from the work array if the
    /// cache does not have it; pinned for the rest of the op.
    fn get(&mut self, slot: u32) -> u32 {
        if let Some(&r) = self.reg_of.get(&slot) {
            self.pinned[(r - CACHE_LO) as usize] = true;
            return r;
        }
        let r = self.fresh();
        self.ldr(r, WORK, slot as usize * 8);
        self.bind(r, slot);
        r
    }
    /// Store `reg` as the value of `slot` and remember it.
    fn put(&mut self, slot: u32, reg: u32) {
        self.str(reg, WORK, slot as usize * 8);
        self.bind(reg, slot);
    }

    // -- instructions -----------------------------------------------------

    /// `ldr Dt, [Xn, #off]`, any offset.
    fn ldr(&mut self, dt: u32, xn: u32, off: usize) {
        if off % 8 == 0 && off / 8 < 4096 {
            self.w(0xFD40_0000 | (((off / 8) as u32) << 10) | (xn << 5) | dt);
        } else {
            self.mov_imm(9, off as u64);
            self.w(0xFC60_6800 | (9 << 16) | (xn << 5) | dt);
        }
    }
    /// `str Dt, [Xn, #off]`, any offset.
    fn str(&mut self, dt: u32, xn: u32, off: usize) {
        if off % 8 == 0 && off / 8 < 4096 {
            self.w(0xFD00_0000 | (((off / 8) as u32) << 10) | (xn << 5) | dt);
        } else {
            self.mov_imm(9, off as u64);
            self.w(0xFC20_6800 | (9 << 16) | (xn << 5) | dt);
        }
    }
    /// `movz`/`movk` a 64-bit immediate into `Xd`.
    fn mov_imm(&mut self, xd: u32, v: u64) {
        let mut first = true;
        for hw in 0..4u32 {
            let part = ((v >> (16 * hw)) & 0xFFFF) as u32;
            if part == 0 && !(first && hw == 3) {
                continue;
            }
            let op = if first { 0xD280_0000 } else { 0xF280_0000 };
            self.w(op | (hw << 21) | (part << 5) | xd);
            first = false;
        }
        if first {
            self.w(0xD280_0000 | xd); // movz xd, #0
        }
    }
    /// A double constant into a fresh cache register.
    fn fconst(&mut self, v: f64) -> u32 {
        let r = self.fresh();
        if v.to_bits() == 0 {
            self.w(0x9E67_0000 | (31 << 5) | r); // fmov Dr, xzr
        } else {
            self.mov_imm(9, v.to_bits());
            self.w(0x9E67_0000 | (9 << 5) | r); // fmov Dr, x9
        }
        r
    }
    fn fbin(&mut self, opc: u32, dd: u32, dn: u32, dm: u32) {
        self.w(opc | (dm << 16) | (dn << 5) | dd);
    }
    fn fun(&mut self, opc: u32, dd: u32, dn: u32) {
        self.w(opc | (dn << 5) | dd);
    }
    fn fcmp(&mut self, dn: u32, dm: u32) {
        self.w(0x1E60_2000 | (dm << 16) | (dn << 5));
    }
    fn fcmp_zero(&mut self, dn: u32) {
        self.w(0x1E60_2008 | (dn << 5));
    }
    fn fcsel(&mut self, dd: u32, dn: u32, dm: u32, cond: u32) {
        self.w(0x1E60_0C00 | (dm << 16) | (cond << 12) | (dn << 5) | dd);
    }
    /// `fmov Dd, Dn`.
    fn fmov(&mut self, dd: u32, dn: u32) {
        self.w(0x1E60_4000 | (dn << 5) | dd);
    }
    /// Call a host routine by address; the cache is gone afterwards.
    fn call(&mut self, addr: *const u8) {
        self.mov_imm(16, addr as u64);
        self.w(0xD63F_0000 | (16 << 5)); // blr x16
        self.forget_all();
    }

    // -- prologue / epilogue --------------------------------------------------

    fn prologue(&mut self) {
        self.w(0xA9BF_7BFD); // stp x29, x30, [sp, #-16]!
        self.w(0x9100_03FD); // mov x29, sp
        self.w(0xA9BF_53F3); // stp x19, x20, [sp, #-16]!
        self.w(0xA9BF_5BF5); // stp x21, x22, [sp, #-16]!
        self.w(0xAA00_03F3); // mov x19, x0
        self.w(0xAA01_03F4); // mov x20, x1
        self.w(0xAA02_03F5); // mov x21, x2
    }
    fn epilogue(&mut self) {
        self.w(0xA8C1_5BF5); // ldp x21, x22, [sp], #16
        self.w(0xA8C1_53F3); // ldp x19, x20, [sp], #16
        self.w(0xA8C1_7BFD); // ldp x29, x30, [sp], #16
        self.w(0xD65F_03C0); // ret
    }

    // -- ops -------------------------------------------------------------------

    fn op(&mut self, op: &ROp) -> Result<(), JitError> {
        let r = self.op_inner(op);
        self.unpin_all();
        r
    }

    fn op_inner(&mut self, op: &ROp) -> Result<(), JitError> {
        match *op {
            ROp::Const(dst, v) => {
                let r = self.fconst(v);
                self.put(dst, r);
            }
            ROp::Input(dst, k) => {
                let r = self.fresh();
                if k == u32::MAX {
                    self.mov_imm(9, f64::NAN.to_bits());
                    self.w(0x9E67_0000 | (9 << 5) | r);
                } else {
                    self.ldr(r, INPUTS, k as usize * 8);
                }
                self.put(dst, r);
            }
            ROp::Add(dst, a, b) => self.bin2(0x1E60_2800, dst, a, b),
            ROp::Sub(dst, a, b) => self.bin2(0x1E60_3800, dst, a, b),
            ROp::Mul(dst, a, b) => self.bin2(0x1E60_0800, dst, a, b),
            ROp::MulAdd(dst, a, b, c) => {
                // Two roundings, like the interpreter: fmul then fadd.
                let (x, y) = (self.get(a), self.get(b));
                let m = self.fresh();
                self.fbin(0x1E60_0800, m, x, y);
                let z = self.get(c);
                let r = self.fresh();
                self.fbin(0x1E60_2800, r, m, z);
                self.put(dst, r);
            }
            ROp::Fma(dst, a, b, c) => {
                let (x, y, z) = (self.get(a), self.get(b), self.get(c));
                let r = self.fresh();
                // fmadd Dd, Dn, Dm, Da = Da + Dn*Dm
                self.w(0x1F40_0000 | (y << 16) | (z << 10) | (x << 5) | r);
                self.put(dst, r);
            }
            ROp::Neg(dst, a) => self.un1(0x1E61_4000, dst, a),
            ROp::Powi(dst, a, n) => match n {
                -1 => {
                    let x = self.get(a);
                    let one = self.fconst(1.0);
                    let r = self.fresh();
                    self.fbin(0x1E60_1800, r, one, x);
                    self.put(dst, r);
                }
                2 => self.bin2(0x1E60_0800, dst, a, a),
                _ => {
                    let x = self.get(a);
                    self.fmov(0, x);
                    self.mov_imm(0, n as i64 as u64);
                    self.call(host_addr("h_powi"));
                    let r = self.fresh();
                    self.fmov(r, 0);
                    self.put(dst, r);
                }
            },
            ROp::Unary(dst, uop, a) => {
                let x = self.get(a);
                let r = self.fresh();
                match uop {
                    UnaryOp::Sqrt => {
                        // x > 0 ? sqrt(x) : 0, the reference's guard.
                        let zero = self.fconst(0.0);
                        let s = self.fresh();
                        self.fun(0x1E61_C000, s, x);
                        self.fcmp_zero(x);
                        self.fcsel(r, s, zero, COND_GT);
                    }
                    UnaryOp::Floor => self.fun(0x1E65_4000, r, x),
                    UnaryOp::Ceil => self.fun(0x1E64_C000, r, x),
                    UnaryOp::Trunc => self.fun(0x1E65_C000, r, x),
                    UnaryOp::Abs => self.fun(0x1E60_C000, r, x),
                    UnaryOp::Sign => {
                        let one = self.fconst(1.0);
                        let minus = self.fconst(-1.0);
                        let t = self.fresh();
                        self.fcmp_zero(x);
                        self.fcsel(t, minus, x, COND_MI);
                        self.fcmp_zero(x);
                        self.fcsel(r, one, t, COND_GT);
                    }
                    _ => {
                        self.fmov(0, x);
                        match crate::host::unary_sym(uop) {
                            Some(sym) => self.call(host_addr(sym)),
                            None => {
                                self.mov_imm(0, uop.code() as u64);
                                self.call(host_addr("h_unary_ext"));
                            }
                        }
                        let r2 = self.fresh();
                        self.fmov(r2, 0);
                        self.put(dst, r2);
                        return Ok(());
                    }
                }
                self.put(dst, r);
            }
            ROp::Binary(dst, bop, a, b) => {
                let (x, y) = (self.get(a), self.get(b));
                self.fmov(0, x);
                self.fmov(1, y);
                self.mov_imm(0, BinOp::code(bop) as u64);
                self.call(host_addr("h_binary"));
                let r = self.fresh();
                self.fmov(r, 0);
                self.put(dst, r);
            }
            ROp::Cmp(dst, cop, a, b) => {
                let (x, y) = (self.get(a), self.get(b));
                let one = self.fconst(1.0);
                let zero = self.fconst(0.0);
                let r = self.fresh();
                self.fcmp(x, y);
                let cond = match cop {
                    CmpOp::Gt => COND_GT,
                    CmpOp::Ge => COND_GE,
                    CmpOp::Lt => COND_MI,
                    CmpOp::Le => COND_LS,
                    CmpOp::Eq => COND_EQ,
                    CmpOp::Ne => COND_NE,
                };
                self.fcsel(r, one, zero, cond);
                self.put(dst, r);
            }
            ROp::Select(dst, c, t, e) => {
                let (cv, tv, ev) = (self.get(c), self.get(t), self.get(e));
                let r = self.fresh();
                self.fcmp_zero(cv);
                self.fcsel(r, tv, ev, COND_NE);
                self.put(dst, r);
            }
            ROp::Reduce(dst, rop, ref args) => {
                let (opc, ident) = match rop {
                    ReduceOp::Sum => (0x1E60_2800u32, 0.0),
                    ReduceOp::Product => (0x1E60_0800u32, 1.0),
                    _ => {
                        return Err(JitError::Codegen(
                            "the direct emitter does not lower min/max reductions".into(),
                        ))
                    }
                };
                let r = self.fold(opc, ident, args, None);
                self.put(dst, r);
            }
            ROp::Dot(dst, ref a, ref b) => {
                let r = self.fold(0x1E60_2800, 0.0, a, Some(b));
                self.put(dst, r);
            }
            ROp::Bundle(..) | ROp::BundleBatch(..) | ROp::Pick(..) => {
                return Err(JitError::Codegen(
                    "the direct emitter does not lower bundle calls".into(),
                ))
            }
        }
        Ok(())
    }

    fn bin2(&mut self, opc: u32, dst: u32, a: u32, b: u32) {
        let (x, y) = (self.get(a), self.get(b));
        let r = self.fresh();
        self.fbin(opc, r, x, y);
        self.put(dst, r);
    }
    fn un1(&mut self, opc: u32, dst: u32, a: u32) {
        let x = self.get(a);
        let r = self.fresh();
        self.fun(opc, r, x);
        self.put(dst, r);
    }
    /// A reduction (or, with `b`, a dot) in the reference order: a left fold
    /// from the identity below `REDUCE_SIMD_MIN`, four accumulators above.
    /// Accumulators stay pinned across the terms; a term's registers are
    /// released once it is folded in, so a long list needs seven registers,
    /// not one per operand.
    fn fold(&mut self, opc: u32, ident: f64, a: &[u32], b: Option<&[u32]>) -> u32 {
        let n = a.len();
        if n >= REDUCE_SIMD_MIN {
            let acc: [u32; 4] = std::array::from_fn(|_| self.fconst(ident));
            let ch = n / 4;
            for c in 0..ch {
                for (k, &ak) in acc.iter().enumerate() {
                    let t = self.term(a, b, 4 * c + k);
                    self.fbin(opc, ak, ak, t);
                    self.release_except(&acc);
                }
            }
            let l = self.fresh();
            self.fbin(opc, l, acc[0], acc[1]);
            let r = self.fresh();
            self.fbin(opc, r, acc[2], acc[3]);
            let mut s = self.fresh();
            self.fbin(opc, s, l, r);
            for k in ch * 4..n {
                let t = self.term(a, b, k);
                let s2 = self.fresh();
                self.fbin(opc, s2, s, t);
                s = s2;
                self.release_except(&[s]);
            }
            s
        } else {
            let mut acc = self.fconst(ident);
            for k in 0..n {
                let t = self.term(a, b, k);
                let s = self.fresh();
                self.fbin(opc, s, acc, t);
                acc = s;
                self.release_except(&[acc]);
            }
            acc
        }
    }

    /// Term `k` of a fold: the operand, or the product for a dot.
    fn term(&mut self, a: &[u32], b: Option<&[u32]>, k: usize) -> u32 {
        match b {
            None => self.get(a[k]),
            Some(bb) => {
                let (x, y) = (self.get(a[k]), self.get(bb[k]));
                let p = self.fresh();
                self.fbin(0x1E60_0800, p, x, y);
                p
            }
        }
    }

    /// Unpin every register except `keep`, so the ring can reuse the
    /// operands of a term the fold has consumed.
    fn release_except(&mut self, keep: &[u32]) {
        self.pinned = [false; CACHE_N as usize];
        for &r in keep {
            self.pinned[(r - CACHE_LO) as usize] = true;
        }
    }
}

fn emit_chunk(ops: &[ROp]) -> Result<Code, JitError> {
    let mut e = Emitter::new();
    e.prologue();
    for op in ops {
        e.op(op)?;
    }
    e.epilogue();
    let map = Mapping::new(&e.code)?;
    let func: extern "C" fn(*mut f64, *const f64, *const u8) =
        unsafe { std::mem::transmute(map.ptr) };
    Ok(Code { map, func })
}

// --- executable memory -----------------------------------------------------------

struct Mapping {
    ptr: *mut u8,
    len: usize,
}

impl Mapping {
    fn new(code: &[u32]) -> Result<Mapping, JitError> {
        let len = (code.len() * 4).max(1);
        let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        unsafe {
            #[cfg(target_os = "macos")]
            let flags = libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT;
            #[cfg(not(target_os = "macos"))]
            let flags = libc::MAP_PRIVATE | libc::MAP_ANON;
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                flags,
                -1,
                0,
            );
            if ptr == libc::MAP_FAILED {
                return Err(JitError::Codegen("mmap of executable memory failed".into()));
            }
            let ptr = ptr as *mut u8;
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
                libc::mprotect(
                    ptr as *mut libc::c_void,
                    len,
                    libc::PROT_READ | libc::PROT_EXEC,
                );
                __clear_cache(
                    ptr as *mut libc::c_char,
                    ptr.add(bytes.len()) as *mut libc::c_char,
                );
            }
            Ok(Mapping { ptr, len })
        }
    }
}

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
#[cfg(not(target_os = "macos"))]
extern "C" {
    fn __clear_cache(start: *mut libc::c_char, end: *mut libc::c_char);
}
