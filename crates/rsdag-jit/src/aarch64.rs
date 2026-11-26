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

use crate::constants::{PREFETCH_AHEAD_BYTES, PREFETCH_EVERY_BYTES};
use crate::isa::*;
use rsdag::node::{CmpOp, ReduceOp};
use rsdag::semantics::exp_ln::*;

const WORK: u32 = 19;
const INPUTS: u32 = 20;
const LAYOUT: u32 = 21;
const WINDOW_X9: u32 = 9;
const SCRATCH_X10: u32 = 10;
const CALL_TARGET_X16: u32 = 16;
const XZR: u32 = 31;
const FIRST_HOT_REG: u32 = 22;
const HOT_REGS: usize = 6;

const D0: u32 = 0;
const D1: u32 = 1;
const D2: u32 = 2;
const D3: u32 = 3;
const D4: u32 = 4;
const D5: u32 = 5;
const D6: u32 = 6;
const D7: u32 = 7;

const COND_EQ: u32 = 0x0;
const COND_NE_TRUE_ON_NAN: u32 = 0x1;
const COND_MI_LT: u32 = 0x4;
const COND_PL_NOT_LT: u32 = 0x5;
const COND_VS_UNORDERED: u32 = 0x6;
const COND_LS_LE: u32 = 0x9;
const COND_GE: u32 = 0xA;
const COND_GT: u32 = 0xC;

const INSN_BYTES: usize = 4;
const IMM12_REACH: usize = 4096;
const ADD_IMM12_LSL12_REACH: usize = 1 << 24;
const WINDOW_BYTES: usize = IMM12_REACH * F64_BYTES;

const PRFM_LITERAL: u32 = 0xD800_0000;
const PLIL2KEEP: u32 = 0b01010;
const MOVZ_X: u32 = 0xD280_0000;
const MOVK_X: u32 = 0xF280_0000;
const ADD_X_IMM12: u32 = 0x9100_0000;
const ADD_X_IMM12_LSL12: u32 = 0x9140_0000;
const SUB_X_REG: u32 = 0xCB00_0000;
const MOV_X_REG: u32 = 0xAA00_03E0;
const SBFM_X: u32 = 0x9340_0000;
const UBFM_X: u32 = 0xD340_0000;
const SCVTF_D_FROM_X: u32 = 0x9E62_0000;
const FMOV_D: u32 = 0x1E60_4000;
const LDR_D_IMM: u32 = 0xFD40_0000;
const LDP_D: u32 = 0x6D40_0000;
const FADD_D: u32 = 0x1E60_2800;
const FSUB_D: u32 = 0x1E60_3800;
const FMUL_D: u32 = 0x1E60_0800;
const FDIV_D: u32 = 0x1E60_1800;
const FMINNM_D: u32 = 0x1E60_7800;
const FMAXNM_D: u32 = 0x1E60_6800;
const FRINTM_D: u32 = 0x1E65_4000;
const FRINTP_D: u32 = 0x1E64_C000;
const FRINTZ_D: u32 = 0x1E65_C000;
const FCMP_D: u32 = 0x1E60_2000;
const FCMP_D_ZERO: u32 = 0x1E60_2008;
const FCSEL_D: u32 = 0x1E60_0C00;
const BLR: u32 = 0xD63F_0000;

const STP_X29_X30_PRE: u32 = 0xA9BF_7BFD;
const MOV_X29_SP: u32 = 0x9100_03FD;
const STP_X19_X20_PRE: u32 = 0xA9BF_53F3;
const STP_X21_X22_PRE: u32 = 0xA9BF_5BF5;
const STP_X23_X24_PRE: u32 = 0xA9BF_63F7;
const STP_X25_X26_PRE: u32 = 0xA9BF_6BF9;
const STP_X27_X28_PRE: u32 = 0xA9BF_73FB;
const STP_D8_D9_PRE: u32 = 0x6DBF_27E8;
const STP_D10_D11_PRE: u32 = 0x6DBF_2FEA;
const STP_D12_D13_PRE: u32 = 0x6DBF_37EC;
const STP_D14_D15_PRE: u32 = 0x6DBF_3FEE;
const MOV_X19_X0: u32 = 0xAA00_03F3;
const MOV_X20_X1: u32 = 0xAA01_03F4;
const MOV_X21_X2: u32 = 0xAA02_03F5;
const LDP_D14_D15_POST: u32 = 0x6CC1_3FEE;
const LDP_D12_D13_POST: u32 = 0x6CC1_37EC;
const LDP_D10_D11_POST: u32 = 0x6CC1_2FEA;
const LDP_D8_D9_POST: u32 = 0x6CC1_27E8;
const LDP_X27_X28_POST: u32 = 0xA8C1_73FB;
const LDP_X25_X26_POST: u32 = 0xA8C1_6BF9;
const LDP_X23_X24_POST: u32 = 0xA8C1_63F7;
const LDP_X21_X22_POST: u32 = 0xA8C1_5BF5;
const LDP_X19_X20_POST: u32 = 0xA8C1_53F3;
const LDP_X29_X30_POST: u32 = 0xA8C1_7BFD;
const RET: u32 = 0xD65F_03C0;

pub(crate) struct A64 {
    code: Vec<u32>,
    window_base_in_caller_saved_x9: Option<usize>,
    hot: Vec<*const ()>,
}

impl A64 {
    fn w(&mut self, word: u32) {
        self.code.push(word);
        if self
            .code
            .len()
            .is_multiple_of(PREFETCH_EVERY_BYTES / INSN_BYTES)
        {
            let imm19 = (PREFETCH_AHEAD_BYTES / INSN_BYTES) as u32;
            self.code.push(PRFM_LITERAL | (imm19 << 5) | PLIL2KEEP);
        }
    }
    fn mov_imm(&mut self, xd: u32, v: u64) {
        let mut first = true;
        for hw in 0..4u32 {
            let part = ((v >> (16 * hw)) & 0xFFFF) as u32;
            if part == 0 && !(first && hw == 3) {
                continue;
            }
            let op = if first { MOVZ_X } else { MOVK_X };
            self.w(op | (hw << 21) | (part << 5) | xd);
            first = false;
        }
        if first {
            self.w(MOVZ_X | xd);
        }
    }
    fn mem(&mut self, opc_imm: u32, opc_reg: u32, dt: u32, xn: u32, off: usize) {
        if off / F64_BYTES < IMM12_REACH {
            self.w(opc_imm | (((off / F64_BYTES) as u32) << 10) | (xn << 5) | dt);
        } else if xn == WORK {
            let base = off - off % WINDOW_BYTES;
            if self.window_base_in_caller_saved_x9 != Some(base) {
                if base < ADD_IMM12_LSL12_REACH {
                    self.w(ADD_X_IMM12_LSL12
                        | (((base >> 12) as u32) << 10)
                        | (WORK << 5)
                        | WINDOW_X9);
                } else {
                    self.mov_imm(SCRATCH_X10, base as u64);
                    self.add_lsl(WINDOW_X9, WORK, SCRATCH_X10, 0);
                }
                self.window_base_in_caller_saved_x9 = Some(base);
            }
            self.w(opc_imm | ((((off - base) / F64_BYTES) as u32) << 10) | (WINDOW_X9 << 5) | dt);
        } else {
            self.mov_imm(SCRATCH_X10, off as u64);
            self.w(opc_reg | (SCRATCH_X10 << 16) | (xn << 5) | dt);
        }
    }
    fn load_or_store(&mut self, store: bool, r: u8, base: Base, off: usize) {
        let (opc_imm, opc_reg) = if store {
            (0xFD00_0000, 0xFC20_6800)
        } else {
            (LDR_D_IMM, 0xFC60_6800)
        };
        self.mem(opc_imm, opc_reg, r as u32, base.register(WORK, INPUTS), off);
    }
    fn fbin(&mut self, opc: u32, dd: u32, dn: u32, dm: u32) {
        self.w(opc | (dm << 16) | (dn << 5) | dd);
    }
    fn fcmp(&mut self, dn: u32, dm: u32) {
        self.w(FCMP_D | (dm << 16) | (dn << 5));
    }
    fn fcsel(&mut self, dd: u32, dn: u32, dm: u32, cond: u32) {
        self.w(FCSEL_D | (dm << 16) | (cond << 12) | (dn << 5) | dd);
    }
    fn fmadd(&mut self, dd: u32, dn: u32, dm: u32, da: u32) {
        self.w(0x1F40_0000 | (dm << 16) | (da << 10) | (dn << 5) | dd);
    }
    fn fop(&mut self, op: Arith, dd: u32, dn: u32, dm: u32) {
        self.arith(op, dd as u8, dn as u8, dm as u8);
    }
    fn ldp(&mut self, t1: u32, t2: u32, xn: u32, off: usize) {
        self.w(LDP_D | (((off / F64_BYTES) as u32) << 15) | (t2 << 10) | (xn << 5) | t1);
    }
    fn ldr(&mut self, t: u32, xn: u32, off: usize) {
        self.w(LDR_D_IMM | (((off / F64_BYTES) as u32) << 10) | (xn << 5) | t);
    }
    fn bits_of(&mut self, xd: u32, dn: u32) {
        self.w(0x9E66_0000 | (dn << 5) | xd);
    }
    fn float_of(&mut self, dd: u32, xn: u32) {
        self.w(0x9E67_0000 | (xn << 5) | dd);
    }
    fn add_lsl(&mut self, xd: u32, xn: u32, xm: u32, s: u32) {
        self.w(0x8B00_0000 | (xm << 16) | (s << 10) | (xn << 5) | xd);
    }
    fn sub_x(&mut self, xd: u32, xn: u32, xm: u32) {
        self.w(SUB_X_REG | (xm << 16) | (xn << 5) | xd);
    }
    fn bfm(&mut self, signed: bool, xd: u32, xn: u32, immr: u32, imms: u32) {
        let op = if signed { SBFM_X } else { UBFM_X };
        self.w(op | (immr << 16) | (imms << 10) | (xn << 5) | xd);
    }
    fn ubfx(&mut self, xd: u32, xn: u32, lsb: u32, width: u32) {
        self.bfm(false, xd, xn, lsb, lsb + width - 1);
    }
    fn lsl(&mut self, xd: u32, xn: u32, s: u32) {
        self.bfm(
            false,
            xd,
            xn,
            (u64::BITS - s) % u64::BITS,
            u64::BITS - 1 - s,
        );
    }

    fn exp_inline(&mut self, d: u32, x: u32, [a, b, c, e, f]: [u32; 5]) {
        use Arith::*;
        let k = |i: usize| i * F64_BYTES;
        self.mov_imm(CONSTS, EXP_K.as_ptr() as u64);
        let (inv_ln2_n, magic) = (D0, D1);
        self.ldp(inv_ln2_n, magic, CONSTS, k(EXP_K_INV));
        let t = a;
        self.fmadd(t, x, inv_ln2_n, magic);
        let kd = b;
        self.fop(Sub, kd, t, magic);
        let (minus_ln2_hi_n, minus_ln2_lo_n) = (D0, D1);
        self.ldp(minus_ln2_hi_n, minus_ln2_lo_n, CONSTS, k(EXP_K_HI));
        let r = c;
        self.fmadd(r, kd, minus_ln2_hi_n, x);
        self.fmadd(r, kd, minus_ln2_lo_n, r);
        let ki = G0;
        self.bits_of(ki, t);
        let row = G1;
        self.ubfx(row, ki, 0, EXP_TABLE_BITS);
        self.mov_imm(G2, EXP_TABLE.as_ptr() as u64);
        self.add_lsl(row, G2, row, EXP_ROW_SHIFT);
        let (tail, scale) = (e, f);
        self.ldp(tail, scale, row, 0);
        self.lsl(ki, ki, EXP_SHIFT);
        self.bits_of(G2, scale);
        self.add_lsl(G2, G2, ki, 0);
        self.float_of(scale, G2);
        let (c2, c3, c4, c5) = (D0, D1, D2, D3);
        self.ldp(c2, c3, CONSTS, k(EXP_K_POLY));
        self.ldp(c4, c5, CONSTS, k(EXP_K_POLY + 2));
        let r2 = a;
        self.fop(Mul, r2, r, r);
        let p23 = b;
        self.fmadd(p23, r, c3, c2);
        let p45 = D2;
        self.fmadd(p45, r, c5, c4);
        let p = e;
        self.fop(Add, p, tail, r);
        self.fmadd(p, r2, p23, p);
        let r4 = a;
        self.fop(Mul, r4, r2, r2);
        self.fmadd(p, r4, p45, p);
        self.fmadd(d, scale, p, scale);
        let unbias = D0;
        self.ldr(unbias, CONSTS, k(EXP_K_UNBIAS));
        self.fop(Mul, d, d, unbias);
        let (underflow, limit, at_limit, one) = (D0, D1, D2, D3);
        self.ldp(underflow, limit, CONSTS, k(EXP_K_GUARDS));
        self.ldp(at_limit, one, CONSTS, k(EXP_K_GUARDS + 2));
        let zero = D4;
        self.float_of(zero, XZR);
        self.fcmp(x, underflow);
        self.fcsel(d, zero, d, COND_MI_LT);
        self.fcmp(x, x);
        self.fcsel(d, x, d, COND_VS_UNORDERED);
        let linear_tail = D4;
        self.fop(Sub, linear_tail, x, limit);
        self.fop(Add, linear_tail, linear_tail, one);
        self.fop(Mul, linear_tail, linear_tail, at_limit);
        self.fcmp(x, limit);
        self.fcsel(d, linear_tail, d, COND_GT);
    }

    fn ln_inline(&mut self, d: u32, x: u32, [a, b, c, e, f]: [u32; 5]) {
        use Arith::*;
        let k = |i: usize| i * F64_BYTES;
        let ix = G0;
        self.bits_of(ix, x);
        let tmp = G1;
        self.mov_imm(tmp, LN_OFF);
        self.sub_x(tmp, ix, tmp);
        let k_int = G2;
        self.bfm(true, k_int, tmp, F64_FRAC_BITS, u64::BITS - 1);
        let kd = b;
        self.w(SCVTF_D_FROM_X | (k_int << 5) | kd);
        self.lsl(G2, k_int, F64_FRAC_BITS);
        self.sub_x(G2, ix, G2);
        let z = e;
        self.float_of(z, G2);
        let row = G2;
        self.ubfx(row, tmp, LN_SHIFT, LN_TABLE_BITS);
        self.mov_imm(G3, LN_TABLE.as_ptr() as u64);
        self.add_lsl(row, G3, row, LN_ROW_SHIFT);
        let (invc, c_row, logc_hi, logc_lo) = (a, c, f, d);
        self.ldp(invc, c_row, row, 0);
        self.ldp(logc_hi, logc_lo, row, 2 * F64_BYTES);
        let r = e;
        self.fop(Sub, r, z, c_row);
        self.fop(Mul, r, r, invc);
        self.mov_imm(CONSTS, LN_K.as_ptr() as u64);
        let (ln2_hi, ln2_lo) = (D0, D1);
        self.ldp(ln2_hi, ln2_lo, CONSTS, k(LN_K_LN2));
        let w = a;
        self.fmadd(w, kd, ln2_hi, logc_hi);
        let hi = c;
        self.fop(Add, hi, w, r);
        let lo = a;
        self.fop(Sub, lo, w, hi);
        self.fop(Add, lo, lo, r);
        let k_ln2_lo_plus_logc_lo = f;
        self.fmadd(k_ln2_lo_plus_logc_lo, kd, ln2_lo, logc_lo);
        self.fop(Add, lo, lo, k_ln2_lo_plus_logc_lo);
        let poly_and_minus_half = [D0, D2, D4, D6];
        for (i, r) in poly_and_minus_half.into_iter().enumerate() {
            self.ldp(r, r + 1, CONSTS, k(LN_K_POLY + 2 * i));
        }
        let [a0, a1, a2, a3, a4, a5, a6, minus_half] = [D0, D1, D2, D3, D4, D5, D6, D7];
        let r2 = f;
        self.fop(Mul, r2, r, r);
        let q01 = b;
        self.fmadd(q01, r, a1, a0);
        let q23 = d;
        self.fmadd(q23, r, a3, a2);
        let q03 = b;
        self.fmadd(q03, r2, q23, q01);
        let q45 = d;
        self.fmadd(q45, r, a5, a4);
        let q46 = d;
        self.fmadd(q46, r2, a6, q45);
        let r4 = D0;
        self.fop(Mul, r4, r2, r2);
        let poly = b;
        self.fmadd(poly, r4, q46, q03);
        let t = a;
        self.fmadd(t, r2, minus_half, lo);
        let r3 = f;
        self.fop(Mul, r3, r2, r);
        self.fmadd(t, r3, poly, t);
        self.fop(Add, d, t, hi);
        let (floor, at_floor, infinity) = (D0, D1, D2);
        self.ldp(floor, at_floor, CONSTS, k(LN_K_GUARDS));
        self.ldr(infinity, CONSTS, k(LN_K_GUARDS + 2));
        self.fcmp(x, infinity);
        self.fcsel(d, x, d, COND_PL_NOT_LT);

        self.fcmp(x, floor);
        self.fcsel(d, at_floor, d, COND_LS_LE);
    }
}

const CONSTS: u32 = 11;
const G0: u32 = 12;
const G1: u32 = 13;
const G2: u32 = 14;
const G3: u32 = 15;
const F64_BYTES: usize = std::mem::size_of::<f64>();
const EXP_ROW_VALUES: usize = 2;
const LN_ROW_VALUES: usize = 4;
const EXP_ROW_SHIFT: u32 = (EXP_ROW_VALUES * F64_BYTES).trailing_zeros();
const LN_ROW_SHIFT: u32 = (LN_ROW_VALUES * F64_BYTES).trailing_zeros();

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
    const RESULT: u8 = D0 as u8;

    fn new(hot: &[*const ()]) -> A64 {
        A64 {
            code: Vec::with_capacity(4096),
            window_base_in_caller_saved_x9: None,
            hot: hot.iter().copied().take(HOT_REGS).collect(),
        }
    }
    fn finish(self) -> Vec<u8> {
        self.code.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn prologue(&mut self) {
        for word in [
            STP_X29_X30_PRE,
            MOV_X29_SP,
            STP_X19_X20_PRE,
            STP_X21_X22_PRE,
            STP_X23_X24_PRE,
            STP_X25_X26_PRE,
            STP_X27_X28_PRE,
            STP_D8_D9_PRE,
            STP_D10_D11_PRE,
            STP_D12_D13_PRE,
            STP_D14_D15_PRE,
            MOV_X19_X0,
            MOV_X20_X1,
            MOV_X21_X2,
        ] {
            self.w(word);
        }
        for k in 0..self.hot.len() {
            let a = self.hot[k] as usize as u64;
            self.mov_imm(FIRST_HOT_REG + k as u32, a);
        }
    }
    fn epilogue(&mut self) {
        for word in [
            LDP_D14_D15_POST,
            LDP_D12_D13_POST,
            LDP_D10_D11_POST,
            LDP_D8_D9_POST,
            LDP_X27_X28_POST,
            LDP_X25_X26_POST,
            LDP_X23_X24_POST,
            LDP_X21_X22_POST,
            LDP_X19_X20_POST,
            LDP_X29_X30_POST,
            RET,
        ] {
            self.w(word);
        }
    }

    fn load(&mut self, r: u8, base: Base, off: usize) {
        self.load_or_store(false, r, base, off);
    }
    fn store(&mut self, r: u8, base: Base, off: usize) {
        self.load_or_store(true, r, base, off);
    }
    fn load_lane(&mut self, r: u8, base: Base, off: usize) {
        self.load(r, base, off);
    }
    fn store_lane(&mut self, r: u8, base: Base, off: usize) {
        self.store(r, base, off);
    }
    fn fconst(&mut self, r: u8, v: f64) {
        if v.to_bits() == 0 {
            self.float_of(r as u32, XZR);
        } else {
            self.mov_imm(SCRATCH_X10, v.to_bits());
            self.float_of(r as u32, SCRATCH_X10);
        }
    }
    fn mov(&mut self, d: u8, a: u8) {
        if d != a {
            self.w(FMOV_D | ((a as u32) << 5) | d as u32);
        }
    }

    fn arith(&mut self, op: Arith, d: u8, a: u8, b: u8) {
        let opc = match op {
            Arith::Add => FADD_D,
            Arith::Sub => FSUB_D,
            Arith::Mul => FMUL_D,
            Arith::Div => FDIV_D,
        };
        self.fbin(opc, d as u32, a as u32, b as u32);
    }
    fn neg(&mut self, d: u8, a: u8) {
        self.w(0x1E61_4000 | ((a as u32) << 5) | d as u32);
    }
    const MINMAX: bool = true;
    fn minmax(&mut self, op: ReduceOp, d: u8, a: u8, b: u8) {
        let ieee_min_max_num_as_f64_min_max_compiles_here = match op {
            ReduceOp::Min => FMINNM_D,
            _ => FMAXNM_D,
        };
        let opc = ieee_min_max_num_as_f64_min_max_compiles_here;
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
            Round::Floor => FRINTM_D,
            Round::Ceil => FRINTP_D,
            Round::Trunc => FRINTZ_D,
        };
        self.w(opc | ((a as u32) << 5) | d as u32);
    }

    fn cmp_select(&mut self, op: CmpOp, a: u8, b: u8, t: u8, e: u8, d: u8) {
        self.fcmp(a as u32, b as u32);
        let cond = match op {
            CmpOp::Gt => COND_GT,
            CmpOp::Ge => COND_GE,
            CmpOp::Lt => COND_MI_LT,
            CmpOp::Le => COND_LS_LE,
            CmpOp::Eq => COND_EQ,
            CmpOp::Ne => COND_NE_TRUE_ON_NAN,
        };
        self.fcsel(d as u32, t as u32, e as u32, cond);
    }
    fn select_nz(&mut self, c: u8, t: u8, e: u8, d: u8) {
        self.w(FCMP_D_ZERO | ((c as u32) << 5));
        self.fcsel(d as u32, t as u32, e as u32, COND_NE_TRUE_ON_NAN);
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
        let (mut next_float_reg, mut next_int_reg) = (0u8, 0u32);
        for arg in args {
            match *arg {
                Arg::F(r) => {
                    self.mov(next_float_reg, r);
                    next_float_reg += 1;
                }
                Arg::I(iarg) => {
                    let xk = next_int_reg;
                    next_int_reg += 1;
                    match iarg {
                        IArg::Imm(v) => self.mov_imm(xk, v),
                        IArg::WorkAddr(off) if off < IMM12_REACH => {
                            self.w(ADD_X_IMM12 | ((off as u32) << 10) | (WORK << 5) | xk);
                        }
                        IArg::WorkAddr(off) => {
                            self.mov_imm(SCRATCH_X10, off as u64);
                            self.add_lsl(xk, WORK, SCRATCH_X10, 0);
                        }
                        IArg::InputAddr(off) if off < IMM12_REACH => {
                            self.w(ADD_X_IMM12 | ((off as u32) << 10) | (INPUTS << 5) | xk);
                        }
                        IArg::InputAddr(off) => {
                            self.mov_imm(SCRATCH_X10, off as u64);
                            self.add_lsl(xk, INPUTS, SCRATCH_X10, 0);
                        }
                        IArg::Layout => self.w(MOV_X_REG | (LAYOUT << 16) | xk),
                    }
                }
            }
        }
        match self.hot.iter().position(|&h| h == addr) {
            Some(k) => self.w(BLR | ((FIRST_HOT_REG + k as u32) << 5)),
            None => {
                self.mov_imm(CALL_TARGET_X16, addr as usize as u64);
                self.w(BLR | (CALL_TARGET_X16 << 5));
            }
        }
        self.window_base_in_caller_saved_x9 = None;
    }
}
