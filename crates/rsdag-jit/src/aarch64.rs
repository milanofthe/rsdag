//! AArch64 encodings (AAPCS64, which Windows on ARM shares).
//!
//! `x19` work, `x20` inputs, `x21` layout; `x9` holds the current 32 KB
//! window of the work array, `x10` is scratch, `x16` the call target, and
//! `x22` to `x27` hold the chunk's frequent host routines; `d0` and `d1`
//! carry call arguments and `d0` the result. The cache is `d8` to `d15`,
//! callee-saved and so kept across host calls, then `d16` to `d31`.
//!
//! The exponential and the logarithm are instructions too (rsdag's own
//! algorithm, its multiply-adds `fmadd`, as `rsdag::semantics::fused` has
//! them here): `x11` to `x15` and `d0` to `d7`, outside the cache, are
//! their scratch, their constants and tables read from statics.
//!
//! The scaled 12-bit offset of `ldr` reaches 32 KB; beyond that the window
//! base in `x9` is moved only when an access leaves the current window, so
//! an access is one instruction almost always instead of four.

use crate::isa::*;
use rsdag::node::{CmpOp, ReduceOp};
use rsdag::semantics::exp_ln::*;

const WORK: u32 = 19;
const INPUTS: u32 = 20;
const LAYOUT: u32 = 21;

// Condition codes after `fcmp`. The unordered case sets C and V and clears N
// and Z; these choices make every comparison with a NaN false except `ne`.
const COND_EQ: u32 = 0x0;
const COND_NE: u32 = 0x1;
const COND_MI: u32 = 0x4; // lt
const COND_VS: u32 = 0x6; // unordered
const COND_LS: u32 = 0x9; // le
const COND_GE: u32 = 0xA;
const COND_GT: u32 = 0xC;

pub(crate) struct A64 {
    code: Vec<u32>,
    /// Byte offset of the work-array window `x9` currently points at.
    window: Option<usize>,
    /// Host routines held in `x22` to `x27`.
    hot: Vec<*const ()>,
}

const WINDOW: usize = 4096 * 8;
use crate::constants::{PREFETCH_AHEAD, PREFETCH_EVERY};
const HOT_LO: u32 = 22;
const HOT_N: usize = 6;

impl A64 {
    fn w(&mut self, word: u32) {
        self.code.push(word);
        // A large program streams through the front end once, so every
        // 128 bytes a `prfm plil2keep` asks for the code 2 KB ahead; that
        // is a third off a 7M-op program and nothing on a small one.
        if self.code.len().is_multiple_of(PREFETCH_EVERY / 4) {
            let imm19 = (PREFETCH_AHEAD / 4) as u32;
            self.code.push(0xD800_0000 | (imm19 << 5) | 0b01010);
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
            self.w(0xD280_0000 | xd);
        }
    }
    fn base(b: Base) -> u32 {
        match b {
            Base::Work => WORK,
            Base::Inputs => INPUTS,
        }
    }
    /// `ldr`/`str Dt, [Xn, #off]` for any offset; for the work array, through
    /// the window in `x9` when the offset is out of immediate reach.
    fn mem(&mut self, opc_imm: u32, opc_reg: u32, dt: u32, xn: u32, off: usize) {
        if off / 8 < 4096 {
            self.w(opc_imm | (((off / 8) as u32) << 10) | (xn << 5) | dt);
        } else if xn == WORK {
            let base = off - off % WINDOW;
            if self.window != Some(base) {
                if base < (1 << 24) {
                    // add x9, x19, #(base >> 12), lsl #12
                    self.w(0x9140_0000 | (((base >> 12) as u32) << 10) | (WORK << 5) | 9);
                } else {
                    self.mov_imm(10, base as u64);
                    self.w(0x8B00_0000 | (10 << 16) | (WORK << 5) | 9);
                }
                self.window = Some(base);
            }
            self.w(opc_imm | ((((off - base) / 8) as u32) << 10) | (9 << 5) | dt);
        } else {
            self.mov_imm(10, off as u64);
            self.w(opc_reg | (10 << 16) | (xn << 5) | dt);
        }
    }
    fn fbin(&mut self, opc: u32, dd: u32, dn: u32, dm: u32) {
        self.w(opc | (dm << 16) | (dn << 5) | dd);
    }
    fn fcmp(&mut self, dn: u32, dm: u32) {
        self.w(0x1E60_2000 | (dm << 16) | (dn << 5));
    }
    fn fcsel(&mut self, dd: u32, dn: u32, dm: u32, cond: u32) {
        self.w(0x1E60_0C00 | (dm << 16) | (cond << 12) | (dn << 5) | dd);
    }
    /// `dd = dn dm + da`, rounded once.
    fn fmadd(&mut self, dd: u32, dn: u32, dm: u32, da: u32) {
        self.w(0x1F40_0000 | (dm << 16) | (da << 10) | (dn << 5) | dd);
    }
    fn fop(&mut self, op: Arith, dd: u32, dn: u32, dm: u32) {
        self.arith(op, dd as u8, dn as u8, dm as u8);
    }
    /// `ldp Dt1, Dt2, [Xn, #off]`.
    fn ldp(&mut self, t1: u32, t2: u32, xn: u32, off: usize) {
        self.w(0x6D40_0000 | (((off / 8) as u32) << 15) | (t2 << 10) | (xn << 5) | t1);
    }
    /// `ldr Dt, [Xn, #off]`.
    fn ldr(&mut self, t: u32, xn: u32, off: usize) {
        self.w(0xFD40_0000 | (((off / 8) as u32) << 10) | (xn << 5) | t);
    }
    /// `fmov Xd, Dn`.
    fn bits_of(&mut self, xd: u32, dn: u32) {
        self.w(0x9E66_0000 | (dn << 5) | xd);
    }
    /// `fmov Dd, Xn`.
    fn float_of(&mut self, dd: u32, xn: u32) {
        self.w(0x9E67_0000 | (xn << 5) | dd);
    }
    /// `add Xd, Xn, Xm, lsl #s`.
    fn add_lsl(&mut self, xd: u32, xn: u32, xm: u32, s: u32) {
        self.w(0x8B00_0000 | (xm << 16) | (s << 10) | (xn << 5) | xd);
    }
    /// `sub Xd, Xn, Xm`.
    fn sub_x(&mut self, xd: u32, xn: u32, xm: u32) {
        self.w(0xCB00_0000 | (xm << 16) | (xn << 5) | xd);
    }
    /// `ubfm Xd, Xn, #immr, #imms` (`ubfx`, `lsl`), or `sbfm` (`asr`).
    fn bfm(&mut self, signed: bool, xd: u32, xn: u32, immr: u32, imms: u32) {
        let op = if signed { 0x9340_0000 } else { 0xD340_0000 };
        self.w(op | (immr << 16) | (imms << 10) | (xn << 5) | xd);
    }
    /// `ubfx Xd, Xn, #lsb, #width`.
    fn ubfx(&mut self, xd: u32, xn: u32, lsb: u32, width: u32) {
        self.bfm(false, xd, xn, lsb, lsb + width - 1);
    }
    /// `lsl Xd, Xn, #s`.
    fn lsl(&mut self, xd: u32, xn: u32, s: u32) {
        self.bfm(
            false,
            xd,
            xn,
            (u64::BITS - s) % u64::BITS,
            u64::BITS - 1 - s,
        );
    }

    /// [`rsdag::semantics::exp_guarded`] step for step.
    fn exp_inline(&mut self, d: u32, x: u32, [a, b, c, e, f]: [u32; 5]) {
        use Arith::*;
        let k = |i: usize| i * F64_BYTES;
        self.mov_imm(K, EXP_K.as_ptr() as u64);
        self.ldp(0, 1, K, k(EXP_K_INV));
        self.fmadd(a, x, 0, 1); // t
        self.fop(Sub, b, a, 1); // kd
        self.ldp(0, 1, K, k(EXP_K_HI));
        self.fmadd(c, b, 0, x);
        self.fmadd(c, b, 1, c); // r
                                // the row of k mod N, and k / N into the exponent
        self.bits_of(G0, a);
        self.ubfx(G1, G0, 0, EXP_TABLE_BITS);
        self.mov_imm(G2, EXP_TABLE.as_ptr() as u64);
        self.add_lsl(G1, G2, G1, EXP_ROW_SHIFT);
        self.ldp(e, f, G1, 0); // tail, scale bits
        self.lsl(G0, G0, EXP_SHIFT);
        self.bits_of(G2, f);
        self.add_lsl(G2, G2, G0, 0);
        self.float_of(f, G2); // scale
        self.ldp(0, 1, K, k(EXP_K_POLY));
        self.ldp(2, 3, K, k(EXP_K_POLY + 2));
        self.fop(Mul, a, c, c); // r^2
        self.fmadd(b, c, 1, 0);
        self.fmadd(2, c, 3, 2);
        self.fop(Add, e, e, c);
        self.fmadd(e, a, b, e);
        self.fop(Mul, a, a, a);
        self.fmadd(e, a, 2, e); // p
        self.fmadd(d, f, e, f);
        self.ldr(0, K, k(EXP_K_UNBIAS));
        self.fop(Mul, d, d, 0);
        // underflow to zero, a NaN kept, the linear tail above the limit
        self.ldp(0, 1, K, k(EXP_K_GUARDS));
        self.ldp(2, 3, K, k(EXP_K_GUARDS + 2)); // EXP_AT_LIMIT, 1
        self.w(0x9E67_0000 | (31 << 5) | 4); // fmov d4, xzr
        self.fcmp(x, 0);
        self.fcsel(d, 4, d, COND_MI);
        self.fcmp(x, x);
        self.fcsel(d, x, d, COND_VS);
        self.fop(Sub, 4, x, 1);
        self.fop(Add, 4, 4, 3);
        self.fop(Mul, 4, 4, 2);
        self.fcmp(x, 1);
        self.fcsel(d, 4, d, COND_GT);
    }

    /// [`rsdag::semantics::ln_guarded`] step for step.
    fn ln_inline(&mut self, d: u32, x: u32, [a, b, c, e, f]: [u32; 5]) {
        use Arith::*;
        let k = |i: usize| i * F64_BYTES;
        // x = 2^k z
        self.bits_of(G0, x);
        self.mov_imm(G1, LN_OFF);
        self.sub_x(G1, G0, G1); // tmp
        self.bfm(true, G2, G1, F64_FRAC_BITS, u64::BITS - 1); // k
        self.w(0x9E62_0000 | (G2 << 5) | b); // scvtf: kd
        self.lsl(G2, G2, F64_FRAC_BITS);
        self.sub_x(G2, G0, G2);
        self.float_of(e, G2); // z
        self.ubfx(G2, G1, LN_SHIFT, LN_TABLE_BITS);
        self.mov_imm(G3, LN_TABLE.as_ptr() as u64);
        self.add_lsl(G2, G3, G2, LN_ROW_SHIFT);
        self.ldp(a, c, G2, 0); // invc, c
        self.ldp(f, d, G2, 2 * F64_BYTES); // logc
        self.fop(Sub, e, e, c);
        self.fop(Mul, e, e, a); // r
        self.mov_imm(K, LN_K.as_ptr() as u64);
        self.ldp(0, 1, K, k(LN_K_LN2));
        self.fmadd(a, b, 0, f); // w
        self.fop(Add, c, a, e); // hi
        self.fop(Sub, a, a, c);
        self.fop(Add, a, a, e);
        self.fmadd(f, b, 1, d);
        self.fop(Add, a, a, f); // lo
        for (i, r) in [0, 2, 4, 6].into_iter().enumerate() {
            self.ldp(r, r + 1, K, k(LN_K_POLY + 2 * i)); // a3 .. a9, -1/2
        }
        self.fop(Mul, f, e, e); // r^2
        self.fmadd(b, e, 1, 0);
        self.fmadd(d, e, 3, 2);
        self.fmadd(b, f, d, b);
        self.fmadd(d, e, 5, 4);
        self.fmadd(d, f, 6, d);
        self.fop(Mul, 0, f, f);
        self.fmadd(b, 0, d, b);
        self.fmadd(a, f, 7, a);
        self.fop(Mul, f, f, e);
        self.fmadd(a, f, b, a);
        self.fop(Add, d, a, c);
        // infinity and NaN kept, the floor
        self.ldp(0, 1, K, k(LN_K_GUARDS)); // LN_FLOOR, LN_AT_FLOOR
        self.ldr(2, K, k(LN_K_GUARDS + 2)); // infinity
        self.fcmp(x, 2);
        self.fcsel(d, x, d, COND_EQ);
        self.fcmp(x, x);
        self.fcsel(d, x, d, COND_VS);
        self.fcmp(x, 0);
        self.fcsel(d, 1, d, COND_LS);
    }
}

// The general registers of the inline exponential and logarithm: the
// constants' address, and scratch.
const K: u32 = 11;
const G0: u32 = 12;
const G1: u32 = 13;
const G2: u32 = 14;
const G3: u32 = 15;
const F64_BYTES: usize = std::mem::size_of::<f64>();
/// `exp`'s rows hold two values, `ln`'s four: the shift of a row index to
/// its byte offset.
const EXP_ROW_SHIFT: u32 = (2 * F64_BYTES).trailing_zeros();
const LN_ROW_SHIFT: u32 = (4 * F64_BYTES).trailing_zeros();

/// The exponential's constants as the code loads them, in pairs.
static EXP_K: [f64; 14] = [
    EXP_INV_LN2_N,
    MAGIC,
    -EXP_LN2_HI_N,
    -EXP_LN2_LO_N,
    EXP_POLY[0],
    EXP_POLY[1],
    EXP_POLY[2],
    EXP_POLY[3],
    EXP_UNBIAS,
    0.0,
    EXP_UNDERFLOW,
    EXP_LIMIT,
    EXP_AT_LIMIT,
    1.0,
];
const EXP_K_INV: usize = 0;
const EXP_K_HI: usize = 2;
const EXP_K_POLY: usize = 4;
const EXP_K_UNBIAS: usize = 8;
const EXP_K_GUARDS: usize = 10;

/// The logarithm's constants as the code loads them, in pairs.
static LN_K: [f64; 14] = [
    LN_LN2_HI,
    LN_LN2_LO,
    LN_POLY[0],
    LN_POLY[1],
    LN_POLY[2],
    LN_POLY[3],
    LN_POLY[4],
    LN_POLY[5],
    LN_POLY[6],
    -0.5,
    LN_FLOOR,
    LN_AT_FLOOR,
    f64::INFINITY,
    0.0,
];
const LN_K_LN2: usize = 0;
const LN_K_POLY: usize = 2;
const LN_K_GUARDS: usize = 10;

impl Isa for A64 {
    const LANES: usize = 1;
    const CACHE: &'static [u8] = &[
        8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31,
    ];
    const SAVED: usize = 8;
    const RESULT: u8 = 0;

    fn new(hot: &[*const ()]) -> A64 {
        A64 {
            code: Vec::with_capacity(4096),
            window: None,
            hot: hot.iter().copied().take(HOT_N).collect(),
        }
    }
    fn finish(self) -> Vec<u8> {
        self.code.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn prologue(&mut self) {
        self.w(0xA9BF_7BFD); // stp x29, x30, [sp, #-16]!
        self.w(0x9100_03FD); // mov x29, sp
        self.w(0xA9BF_53F3); // stp x19, x20, [sp, #-16]!
        self.w(0xA9BF_5BF5); // stp x21, x22, [sp, #-16]!
        self.w(0xA9BF_63F7); // stp x23, x24, [sp, #-16]!
        self.w(0xA9BF_6BF9); // stp x25, x26, [sp, #-16]!
        self.w(0xA9BF_73FB); // stp x27, x28, [sp, #-16]!
        self.w(0x6DBF_27E8); // stp d8, d9, [sp, #-16]!
        self.w(0x6DBF_2FEA); // stp d10, d11, [sp, #-16]!
        self.w(0x6DBF_37EC); // stp d12, d13, [sp, #-16]!
        self.w(0x6DBF_3FEE); // stp d14, d15, [sp, #-16]!
        self.w(0xAA00_03F3); // mov x19, x0
        self.w(0xAA01_03F4); // mov x20, x1
        self.w(0xAA02_03F5); // mov x21, x2
        for k in 0..self.hot.len() {
            let a = self.hot[k] as usize as u64;
            self.mov_imm(HOT_LO + k as u32, a);
        }
    }
    fn epilogue(&mut self) {
        self.w(0x6CC1_3FEE); // ldp d14, d15, [sp], #16
        self.w(0x6CC1_37EC); // ldp d12, d13, [sp], #16
        self.w(0x6CC1_2FEA); // ldp d10, d11, [sp], #16
        self.w(0x6CC1_27E8); // ldp d8, d9, [sp], #16
        self.w(0xA8C1_73FB); // ldp x27, x28, [sp], #16
        self.w(0xA8C1_6BF9); // ldp x25, x26, [sp], #16
        self.w(0xA8C1_63F7); // ldp x23, x24, [sp], #16
        self.w(0xA8C1_5BF5); // ldp x21, x22, [sp], #16
        self.w(0xA8C1_53F3); // ldp x19, x20, [sp], #16
        self.w(0xA8C1_7BFD); // ldp x29, x30, [sp], #16
        self.w(0xD65F_03C0); // ret
    }

    fn load(&mut self, r: u8, base: Base, off: usize) {
        self.mem(0xFD40_0000, 0xFC60_6800, r as u32, Self::base(base), off);
    }
    fn store(&mut self, r: u8, base: Base, off: usize) {
        self.mem(0xFD00_0000, 0xFC20_6800, r as u32, Self::base(base), off);
    }
    fn load_lane(&mut self, r: u8, base: Base, off: usize) {
        self.load(r, base, off);
    }
    fn store_lane(&mut self, r: u8, base: Base, off: usize) {
        self.store(r, base, off);
    }
    fn fconst(&mut self, r: u8, v: f64) {
        if v.to_bits() == 0 {
            self.w(0x9E67_0000 | (31 << 5) | r as u32); // fmov Dr, xzr
        } else {
            self.mov_imm(10, v.to_bits());
            self.w(0x9E67_0000 | (10 << 5) | r as u32); // fmov Dr, x10
        }
    }
    fn mov(&mut self, d: u8, a: u8) {
        if d != a {
            self.w(0x1E60_4000 | ((a as u32) << 5) | d as u32);
        }
    }

    fn arith(&mut self, op: Arith, d: u8, a: u8, b: u8) {
        let opc = match op {
            Arith::Add => 0x1E60_2800,
            Arith::Sub => 0x1E60_3800,
            Arith::Mul => 0x1E60_0800,
            Arith::Div => 0x1E60_1800,
        };
        self.fbin(opc, d as u32, a as u32, b as u32);
    }
    fn neg(&mut self, d: u8, a: u8) {
        self.w(0x1E61_4000 | ((a as u32) << 5) | d as u32);
    }
    const MINMAX: bool = true;
    fn minmax(&mut self, op: ReduceOp, d: u8, a: u8, b: u8) {
        // fminnm / fmaxnm: IEEE minNum / maxNum, what `f64::min` and
        // `f64::max` compile to on this target.
        let opc = match op {
            ReduceOp::Min => 0x1E60_7800,
            _ => 0x1E60_6800,
        };
        self.fbin(opc, d as u32, a as u32, b as u32);
    }
    fn abs(&mut self, d: u8, a: u8) {
        self.w(0x1E60_C000 | ((a as u32) << 5) | d as u32);
    }
    fn sqrt(&mut self, d: u8, a: u8) {
        self.w(0x1E61_C000 | ((a as u32) << 5) | d as u32);
    }
    fn round_inline() -> bool {
        true
    }
    fn round(&mut self, mode: Round, d: u8, a: u8) {
        let opc = match mode {
            Round::Floor => 0x1E65_4000, // frintm
            Round::Ceil => 0x1E64_C000,  // frintp
            Round::Trunc => 0x1E65_C000, // frintz
        };
        self.w(opc | ((a as u32) << 5) | d as u32);
    }

    fn cmp_select(&mut self, op: CmpOp, a: u8, b: u8, t: u8, e: u8, d: u8) {
        self.fcmp(a as u32, b as u32);
        let cond = match op {
            CmpOp::Gt => COND_GT,
            CmpOp::Ge => COND_GE,
            CmpOp::Lt => COND_MI,
            CmpOp::Le => COND_LS,
            CmpOp::Eq => COND_EQ,
            CmpOp::Ne => COND_NE,
        };
        self.fcsel(d as u32, t as u32, e as u32, cond);
    }
    fn select_nz(&mut self, c: u8, t: u8, e: u8, d: u8) {
        self.w(0x1E60_2008 | ((c as u32) << 5)); // fcmp Dc, #0.0
        self.fcsel(d as u32, t as u32, e as u32, COND_NE);
    }
    fn exp_ln(&mut self, ln: bool, d: u8, x: u8, t: [u8; 5]) {
        let t = t.map(u32::from);
        if ln {
            self.ln_inline(d as u32, x as u32, t);
        } else {
            self.exp_inline(d as u32, x as u32, t);
        }
    }

    fn call(&mut self, addr: *const (), args: &[Arg]) {
        // Floats and integers are counted apart: d0.. and x0.. in order.
        let (mut nf, mut ni) = (0u8, 0u32);
        for arg in args {
            match *arg {
                Arg::F(r) => {
                    self.mov(nf, r);
                    nf += 1;
                }
                Arg::I(iarg) => {
                    let xk = ni;
                    ni += 1;
                    match iarg {
                        IArg::Imm(v) => self.mov_imm(xk, v),
                        IArg::WorkAddr(off) if off < 4096 => {
                            self.w(0x9100_0000 | ((off as u32) << 10) | (WORK << 5) | xk);
                        }
                        IArg::WorkAddr(off) => {
                            self.mov_imm(10, off as u64);
                            self.w(0x8B00_0000 | (10 << 16) | (WORK << 5) | xk);
                        }
                        IArg::InputAddr(off) if off < 4096 => {
                            self.w(0x9100_0000 | ((off as u32) << 10) | (INPUTS << 5) | xk);
                        }
                        IArg::InputAddr(off) => {
                            self.mov_imm(10, off as u64);
                            self.w(0x8B00_0000 | (10 << 16) | (INPUTS << 5) | xk);
                        }
                        IArg::Layout => self.w(0xAA00_03E0 | (LAYOUT << 16) | xk),
                    }
                }
            }
        }
        match self.hot.iter().position(|&h| h == addr) {
            Some(k) => self.w(0xD63F_0000 | ((HOT_LO + k as u32) << 5)), // blr xk
            None => {
                self.mov_imm(16, addr as usize as u64);
                self.w(0xD63F_0000 | (16 << 5)); // blr x16
            }
        }
        self.window = None; // x9 is caller-saved
    }
}
