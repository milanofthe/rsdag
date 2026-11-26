//! x86-64 encodings, System V and Windows x64.
//!
//! `rbx` work, `r13` inputs, `r14` layout; `rax` scratch; `xmm0` and
//! `xmm1` carry call arguments, the result and the masks of a select. The
//! cache is `xmm2` to `xmm15`. Under System V every vector register is
//! caller-saved, so nothing survives a host call; under Windows `xmm6` to
//! `xmm15` are callee-saved, so the chunk preserves them and they come
//! first in the cache. Baseline SSE2, with `roundsd` where there is SSE4.1
//! and three-operand VEX forms and `vblendvpd` where there is AVX (the same
//! IEEE operations, fewer bytes); `RSDAG_SSE2` set in the environment
//! keeps to the baseline (as it keeps rsdag's dense kernels to SSE2), so
//! both paths can be tested on one machine.
//! Constants are read rip-relative from a pool at the end of the chunk.
//!
//! `X64<2>` is the same code over two instances at once: the packed forms
//! of the same instructions (`addpd` for `addsd`, `cmppd` for `cmpsd`),
//! each lane the IEEE operation of the scalar code, so a lane's result is
//! the scalar result bit for bit. `X64<4>` runs four in the `ymm`
//! registers, every instruction in its VEX.256 form (AVX). The upper
//! halves of the `ymm` registers are not preserved across calls, not even
//! under Windows, so no value survives a host call there, and `vzeroupper`
//! precedes every call and the return.
//!
//! The exponential and the logarithm are instructions too, rsdag's own
//! algorithm with its integer steps in the packed integer forms and its
//! table rows loaded lane by lane: no call, so the cache keeps its values
//! across them. Its multiply-adds are `vfmadd` where
//! `rsdag::semantics::fused` says the machine fuses, else a product and a
//! sum. Four lanes need AVX2.

use crate::constants::CODE_ALIGN_BYTES;
use crate::isa::*;
use rsdag::node::{CmpOp, ReduceOp};
use rsdag::semantics::exp_ln::*;

const RAX: u8 = 0;
const RCX: u8 = 1;
const RDX: u8 = 2;
const RBX: u8 = 3;
const RSP: u8 = 4;
const RSI: u8 = 6;
const RDI: u8 = 7;
const R8: u8 = 8;
const R9: u8 = 9;
const R10: u8 = 10;
const R11: u8 = 11;
const R12: u8 = 12;
const R13: u8 = 13;
const R14: u8 = 14;
const R15: u8 = 15;

const XMM0: u8 = 0;
const XMM1: u8 = 1;

const WORK: u8 = RBX;
const INPUTS: u8 = R13;
const LAYOUT: u8 = R14;
const WIN: bool = cfg!(windows);
// System V: https://gitlab.com/x86-psABIs/x86-64-ABI
// Windows x64: https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention
const INT_ARGS: &[u8] = if WIN {
    &[RCX, RDX, R8, R9]
} else {
    &[RDI, RSI, RDX, RCX, R8, R9]
};
const FLOAT_ARG_REGS: usize = 4;
const VOLATILE_ON_BOTH_ABIS: [u8; 7] = [RAX, RCX, RDX, R8, R9, R10, R11];
const CALL_ADDRESS_SCRATCH: u8 = RAX;

const STACK_ALIGN: i32 = 16;
const GPR_BYTES: i32 = 8;
const RETURN_ADDRESS_AND_SIX_PUSHES: i32 = 7 * GPR_BYTES;
const CALL_ALIGN_PAD_BYTES: i32 = 8;
const FIRST_WIN_SAVED_XMM: u8 = 6;
const WIN_SAVED_XMMS: u8 = 10;
const XMM_SAVE_BYTES: i32 = 16;
const WIN_FRAME: i32 = WIN_SAVED_XMMS as i32 * XMM_SAVE_BYTES + CALL_ALIGN_PAD_BYTES;
const _: () = assert!((RETURN_ADDRESS_AND_SIX_PUSHES + WIN_FRAME) % STACK_ALIGN == 0);
const _: () = assert!((RETURN_ADDRESS_AND_SIX_PUSHES + CALL_ALIGN_PAD_BYTES) % STACK_ALIGN == 0);
const WIN_SHADOW_SPACE_BYTES: i32 = 32;

const PUSH_RBP_MOV_RBP_RSP: [u8; 4] = [0x55, 0x48, 0x89, 0xE5];
const PUSH_RBX_R12_R13_R14_R15: [u8; 9] = [0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57];
const POP_R15_R14_R13_R12_RBX_RBP_RET: [u8; 11] = [
    0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5B, 0x5D, 0xC3,
];
const SUB_RSP_IMM32: [u8; 3] = [0x48, 0x81, 0xEC];
const ADD_RSP_IMM32: [u8; 3] = [0x48, 0x81, 0xC4];
const SUB_RSP_ALIGN_PAD: [u8; 4] = [0x48, 0x83, 0xEC, CALL_ALIGN_PAD_BYTES as u8];
const ADD_RSP_ALIGN_PAD: [u8; 4] = [0x48, 0x83, 0xC4, CALL_ALIGN_PAD_BYTES as u8];
const MOV_RBX_RCX: [u8; 3] = [0x48, 0x89, 0xCB];
const MOV_R13_RDX: [u8; 3] = [0x49, 0x89, 0xD5];
const MOV_R14_R8: [u8; 3] = [0x4D, 0x89, 0xC6];
const MOV_RBX_RDI: [u8; 3] = [0x48, 0x89, 0xFB];
const MOV_R13_RSI: [u8; 3] = [0x49, 0x89, 0xF5];
const MOV_R14_RDX: [u8; 3] = [0x49, 0x89, 0xD6];
const VZEROUPPER: [u8; 3] = [0xC5, 0xF8, 0x77];
const INT3: u8 = 0xCC;

const REX: u8 = 0x40;
const REX_W: u8 = 0x48;
const REX_B: u8 = 0x41;
const MOD_DISP8: u8 = 0x40;
const MOD_DISP32: u8 = 0x80;
const RM_SIB: u8 = 4;
const SIB_BASE_RSP: u8 = 0x24;
const RM_RIP_DISP32: u8 = 5;
const DISP32_BYTES: usize = 4;

const PFX_66: u8 = 0x66;
const PFX_F2: u8 = 0xF2;
const PFX_F3: u8 = 0xF3;
const ESCAPE_0F: u8 = 0x0F;
const ESCAPE_0F3A: [u8; 2] = [0x0F, 0x3A];

const VEX2: u8 = 0xC5;
const VEX3: u8 = 0xC4;
const VEX_PP_66: u8 = 1;
const VEX_PP_F2: u8 = 3;
const VEX_MAP_0F: u8 = 1;
const VEX_MAP_0F38: u8 = 2;
const VEX_MAP_0F3A: u8 = 3;
const VEX_128: u8 = 0;
const VEX_256: u8 = 1;
const VEX_W1: bool = true;
const NO_VEX_SOURCE: u8 = 0;
const NO_INDEX: u8 = 0;

const MOV_UNALIGNED_LOAD: u8 = 0x10;
const MOV_UNALIGNED_STORE: u8 = 0x11;
const MOVAPD: u8 = 0x28;
const MOVDQU_LOAD: u8 = 0x6F;
const MOVDQU_STORE: u8 = 0x7F;
const MOVQ_GPR_TO_XMM: u8 = 0x6E;
const MOVQ_XMM_TO_GPR: u8 = 0x7E;
const CVTSI2SD: u8 = 0x2A;
const SQRT: u8 = 0x51;
const ANDPD: u8 = 0x54;
const ANDNPD: u8 = 0x55;
const ORPD: u8 = 0x56;
const XORPD: u8 = 0x57;
const MIN: u8 = 0x5D;
const MAX: u8 = 0x5F;
const UNPCKLPD: u8 = 0x14;
const UNPCKHPD: u8 = 0x15;
const PSHUFD: u8 = 0x70;
const PSHUFD_HIGH_HALF_TWICE: u8 = 0xEE;
const PSHIFTQ_IMM8: u8 = 0x73;
const PSLLQ_EXT: u8 = 6;
const PSRLQ_EXT: u8 = 2;
const CMPSD_CMPPD: u8 = 0xC2;
const ROUNDPD: u8 = 0x09;
const ROUNDSD: u8 = 0x0B;
const VPEXTRQ: u8 = 0x16;
const VINSERTF128: u8 = 0x18;
const VEXTRACTI128: u8 = 0x39;
const VBLENDVPD: u8 = 0x4B;
const IS4_XMM0: u8 = 0x00;
const UPPER_HALF: u8 = 1;
const SECOND_LANE: u8 = 1;

const LEA: u8 = 0x8D;
const ALU_IMM32: u8 = 0x81;
const AND_EXT: u8 = 4;
const SHIFT_IMM8: u8 = 0xC1;
const MOV_R_IMM: u8 = 0xB8;
const CALL_RM: u8 = 0xFF;
const MODRM_CALL_REG: u8 = 0xD0;
const CALL_RAX: [u8; 2] = [CALL_RM, MODRM_CALL_REG | CALL_ADDRESS_SCRATCH];

const SIGN_BIT: u64 = 0x8000_0000_0000_0000;

const LEGACY_MEM_OPERAND_ALIGN: usize = 16;
const WORK_ARRAY_ALIGN: usize = std::mem::align_of::<f64>();
const _: () = assert!(CODE_ALIGN_BYTES.is_multiple_of(LEGACY_MEM_OPERAND_ALIGN));

struct RipFixup {
    disp_at: usize,
    constant: usize,
    insn_end: usize,
}

pub(crate) struct X64<const L: usize> {
    code: Vec<u8>,
    avx: bool,
    fma: bool,
    hot: Vec<*const ()>,
    pool: Vec<u64>,
    fixups: Vec<RipFixup>,
}

const HOT_REGS: [u8; 2] = [R12, R15];

#[derive(Clone, Copy)]
pub(crate) struct Features {
    pub sse41: bool,
    pub avx: bool,
    pub avx2: bool,
}

const LN_ROW_MASK: u32 = {
    let mask = LN_INDEX_MASK << LN_ROW_SHIFT;
    assert!(mask <= u32::MAX as u64);
    mask as u32
};

pub(crate) fn features() -> Features {
    static F: std::sync::OnceLock<Features> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let baseline = rsdag::semantics::sse2_baseline();
        Features {
            sse41: !baseline && is_x86_feature_detected!("sse4.1"),
            avx: !baseline && is_x86_feature_detected!("avx"),
            avx2: !baseline && is_x86_feature_detected!("avx2"),
        }
    })
}

impl<const L: usize> X64<L> {
    const PFX: u8 = if L == 1 { PFX_F2 } else { PFX_66 };
    const PP: u8 = if L == 1 { VEX_PP_F2 } else { VEX_PP_66 };
    const VL: u8 = if L == 4 { VEX_256 } else { VEX_128 };
    const POOL_ENTRY_BYTES: usize = F64_BYTES * if L == 4 { 4 } else { 2 };
    const LEGACY_PACKED: bool = L == 2;
    fn b(&mut self, byte: u8) {
        self.code.push(byte);
    }
    fn bytes(&mut self, bs: &[u8]) {
        self.code.extend_from_slice(bs);
    }
    fn rex(&mut self, w: bool, reg: u8, rm: u8) {
        let v = REX | ((w as u8) << 3) | ((reg >> 3) << 2) | (rm >> 3);
        if v != REX {
            self.b(v);
        }
    }
    fn modrm_reg(&mut self, reg: u8, rm: u8) {
        self.b(0xC0 | ((reg & 7) << 3) | (rm & 7));
    }
    fn modrm_mem(&mut self, reg: u8, base: u8, disp: usize) {
        debug_assert_ne!(base & 7, RSP & 7, "this base needs a SIB byte");
        let disp = i32::try_from(disp).expect("work array offset fits i32");
        if let Ok(d8) = i8::try_from(disp) {
            self.b(MOD_DISP8 | ((reg & 7) << 3) | (base & 7));
            self.b(d8 as u8);
        } else {
            self.b(MOD_DISP32 | ((reg & 7) << 3) | (base & 7));
            self.bytes(&disp.to_le_bytes());
        }
    }
    fn sse(&mut self, prefix: u8, op: u8, reg: u8, rm: u8) {
        self.b(prefix);
        self.rex(false, reg, rm);
        self.bytes(&[ESCAPE_0F, op]);
        self.modrm_reg(reg, rm);
    }
    fn sse_mem(&mut self, prefix: u8, op: u8, reg: u8, base: u8, disp: usize) {
        self.b(prefix);
        self.rex(false, reg, base);
        self.bytes(&[ESCAPE_0F, op]);
        self.modrm_mem(reg, base, disp);
    }
    fn mov_mem(&mut self, prefix: u8, pp: u8, l: u8, op: u8, reg: u8, base: u8, disp: usize) {
        if L == 4 {
            self.vex_l(pp, VEX_MAP_0F, l, reg, NO_VEX_SOURCE, base);
            self.b(op);
            self.modrm_mem(reg, base, disp);
        } else {
            self.sse_mem(prefix, op, reg, base, disp);
        }
    }
    fn first_lane_mov(&mut self, store: bool, r: u8, base: Base, off: usize) {
        let op = if store {
            MOV_UNALIGNED_STORE
        } else {
            MOV_UNALIGNED_LOAD
        };
        let b = base.register(WORK, INPUTS);
        self.mov_mem(PFX_F2, VEX_PP_F2, VEX_128, op, r, b, off);
    }
    fn unary_rr(&mut self, prefix: u8, pp: u8, op: u8, d: u8, a: u8) {
        if L == 4 {
            self.vex_rr(pp, op, d, NO_VEX_SOURCE, a);
        } else {
            self.sse(prefix, op, d, a);
        }
    }
    fn zero(&mut self, r: u8) {
        if L == 4 {
            self.vex_rr(VEX_PP_66, XORPD, r, r, r);
        } else {
            self.sse(PFX_66, XORPD, r, r);
        }
    }
    fn mov_imm(&mut self, rd: u8, v: u64) {
        if let Ok(v32_zero_extends) = u32::try_from(v) {
            self.rex(false, 0, rd);
            self.b(MOV_R_IMM | (rd & 7));
            self.bytes(&v32_zero_extends.to_le_bytes());
        } else {
            self.rex(true, 0, rd);
            self.b(MOV_R_IMM | (rd & 7));
            self.bytes(&v.to_le_bytes());
        }
    }
    fn modrm_rsp(&mut self, reg: u8, disp: i32) {
        if let Ok(d8) = i8::try_from(disp) {
            self.bytes(&[
                MOD_DISP8 | ((reg & 7) << 3) | RM_SIB,
                SIB_BASE_RSP,
                d8 as u8,
            ]);
        } else {
            self.bytes(&[MOD_DISP32 | ((reg & 7) << 3) | RM_SIB, SIB_BASE_RSP]);
            self.bytes(&disp.to_le_bytes());
        }
    }
    fn xmm_rsp(&mut self, store: bool, xmm: u8, disp: i32) {
        self.b(PFX_F3);
        self.rex(false, xmm, 0);
        self.bytes(&[ESCAPE_0F, if store { MOVDQU_STORE } else { MOVDQU_LOAD }]);
        self.modrm_rsp(xmm, disp);
    }
    fn move_rsp(&mut self, n: i32) {
        debug_assert_eq!(n % STACK_ALIGN, 0, "rsp moves in 16-byte steps");
        self.bytes(if n >= 0 {
            &ADD_RSP_IMM32
        } else {
            &SUB_RSP_IMM32
        });
        self.bytes(&n.abs().to_le_bytes());
    }
    fn store_rsp(&mut self, r: u8, disp: i32) {
        self.rex(true, r, 0);
        self.b(GPR_MOV);
        self.modrm_rsp(r, disp);
    }
    fn int_arg(&mut self, rk: u8, arg: IArg) {
        match arg {
            IArg::Imm(v) => self.mov_imm(rk, v),
            IArg::WorkAddr(off) => {
                self.rex(true, rk, WORK);
                self.b(LEA);
                self.modrm_mem(rk, WORK, off);
            }
            IArg::InputAddr(off) => {
                self.rex(true, rk, INPUTS);
                self.b(LEA);
                self.modrm_mem(rk, INPUTS, off);
            }
            IArg::Layout => {
                self.rex(true, LAYOUT, rk);
                self.b(GPR_MOV);
                self.modrm_reg(LAYOUT, rk);
            }
        }
    }
    fn vex_l(&mut self, pp: u8, map: u8, l: u8, reg: u8, v: u8, rm: u8) {
        let r = (!reg >> 3) & 1;
        let b = (!rm >> 3) & 1;
        let v = !v & 0xF;
        let tail = (v << 3) | (l << 2) | pp;
        let two_byte_form_encodes_it = map == VEX_MAP_0F && b == 1;
        if two_byte_form_encodes_it {
            self.bytes(&[VEX2, (r << 7) | tail]);
        } else {
            self.bytes(&[VEX3, (r << 7) | (1 << 6) | (b << 5) | map, tail]);
        }
    }
    fn vex(&mut self, pp: u8, map: u8, reg: u8, v: u8, rm: u8) {
        self.vex_l(pp, map, Self::VL, reg, v, rm);
    }
    fn vex_rr(&mut self, pp: u8, op: u8, d: u8, a: u8, rm: u8) {
        self.vex(pp, VEX_MAP_0F, d, a, rm);
        self.b(op);
        self.modrm_reg(d, rm);
    }
    fn modrm_const(&mut self, reg: u8, bits: u64) {
        self.modrm_const_imm(reg, bits, 0);
    }
    fn modrm_const_imm(&mut self, reg: u8, bits: u64, imm_bytes_after_disp: usize) {
        self.b(((reg & 7) << 3) | RM_RIP_DISP32);
        let k = match self.pool.iter().position(|&c| c == bits) {
            Some(k) => k,
            None => {
                self.pool.push(bits);
                self.pool.len() - 1
            }
        };
        let at = self.code.len();
        self.fixups.push(RipFixup {
            disp_at: at,
            constant: k,
            insn_end: at + DISP32_BYTES + imm_bytes_after_disp,
        });
        self.bytes(&[0; DISP32_BYTES]);
    }
    fn op_const(&mut self, prefix: u8, pp: u8, op: u8, d: u8, a: u8, bits: u64) {
        if self.avx {
            self.vex(pp, VEX_MAP_0F, d, a, 0);
        } else {
            self.mov(d, a);
            self.b(prefix);
            self.rex(false, d, 0);
            self.b(ESCAPE_0F);
        }
        self.b(op);
        self.modrm_const(d, bits);
    }
    fn blend_by_xmm0_mask(&mut self, t: u8, e: u8, d: u8) {
        if self.avx {
            self.vex(VEX_PP_66, VEX_MAP_0F3A, d, e, t);
            self.b(VBLENDVPD);
            self.modrm_reg(d, t);
            self.b(IS4_XMM0);
            return;
        }
        self.mov(d, t);
        self.sse(PFX_66, ANDPD, d, XMM0);
        self.sse(PFX_66, ANDNPD, XMM0, e);
        self.sse(PFX_66, ORPD, d, XMM0);
    }
    fn cmp_mask_into_xmm0(&mut self, x: u8, y: u8, pred: u8) {
        if self.avx {
            self.vex_rr(Self::PP, CMPSD_CMPPD, XMM0, x, y);
        } else {
            self.mov(XMM0, x);
            self.sse(Self::PFX, CMPSD_CMPPD, XMM0, y);
        }
        self.b(pred);
    }
    fn arith_op(op: Arith) -> u8 {
        match op {
            Arith::Add => 0x58,
            Arith::Mul => 0x59,
            Arith::Sub => 0x5C,
            Arith::Div => 0x5E,
        }
    }

    fn f(&mut self, op: Arith, d: u8, a: u8, b: u8) {
        self.arith(op, d, a, b);
    }
    fn fc(&mut self, op: Arith, d: u8, a: u8, c: f64) {
        self.op_const(Self::PFX, Self::PP, Self::arith_op(op), d, a, c.to_bits());
    }
    fn vex3(&mut self, w: bool, pp: u8, map: u8, l: u8, reg: u8, v: u8, sib_index: u8, rm: u8) {
        let r = (!reg >> 3) & 1;
        let xb = (!sib_index >> 3) & 1;
        let b = (!rm >> 3) & 1;
        let v = !v & 0xF;
        self.bytes(&[
            VEX3,
            (r << 7) | (xb << 6) | (b << 5) | map,
            ((w as u8) << 7) | (v << 3) | (l << 2) | pp,
        ]);
    }
    fn fma3(&mut self, form: u8, d: u8, v: u8, m: Operand) {
        let sd_form_is_the_next_opcode = (L == 1) as u8;
        let op = form + sd_form_is_the_next_opcode;
        let rm = match m {
            Operand::R(r) => r,
            Operand::C(_) => 0,
        };
        self.vex3(
            VEX_W1,
            VEX_PP_66,
            VEX_MAP_0F38,
            Self::VL,
            d,
            v,
            NO_INDEX,
            rm,
        );
        self.b(op);
        match m {
            Operand::R(r) => self.modrm_reg(d, r),
            Operand::C(c) => self.modrm_const(d, c.to_bits()),
        }
    }
    fn madd(&mut self, d: u8, a: u8, b: Operand, c: Operand) {
        use Arith::*;
        debug_assert!(
            d == a || matches!(c, Operand::R(c) if c == d) || !matches!(b, Operand::R(b) if b == d)
        );
        debug_assert!(d != XMM0 && a != XMM0);
        if !self.fma {
            let product = match c {
                Operand::R(c) if c == d => XMM0,
                _ => d,
            };
            match b {
                Operand::R(b) => self.f(Mul, product, a, b),
                Operand::C(b) => self.fc(Mul, product, a, b),
            }
            match c {
                Operand::R(_) if product == XMM0 => self.f(Add, d, d, XMM0),
                Operand::R(c) => self.f(Add, d, d, c),
                Operand::C(c) => self.fc(Add, d, d, c),
            }
            return;
        }
        match (b, c) {
            (_, Operand::R(c)) if d == c => match b {
                Operand::R(b) => self.fma3(FMA231, d, a, Operand::R(b)),
                Operand::C(_) => self.fma3(FMA231, d, a, b),
            },
            (Operand::C(_), Operand::R(c)) => {
                self.mov(d, a);
                self.fma3(FMA132, d, c, b);
            }
            (Operand::R(b), Operand::R(c)) => {
                let (d_in, other) = if d == b { (b, a) } else { (a, b) };
                self.mov(d, d_in);
                self.fma3(FMA213, d, other, Operand::R(c));
            }
            (Operand::R(b), Operand::C(_)) => {
                let (d_in, other) = if d == b { (b, a) } else { (a, b) };
                self.mov(d, d_in);
                self.fma3(FMA213, d, other, c);
            }
            (Operand::C(b), Operand::C(_)) => {
                debug_assert_ne!(d, a);
                self.fconst(d, b);
                self.fma3(FMA213, d, a, c);
            }
        }
    }
    fn int(&mut self, op: Int, d: u8, a: u8, b: u8) {
        if self.avx {
            self.vex_rr(VEX_PP_66, op as u8, d, a, b);
        } else if d == b && d != a {
            debug_assert!(!matches!(op, Int::Sub), "a difference into its subtrahend");
            self.sse(PFX_66, op as u8, d, a);
        } else {
            self.mov(d, a);
            self.sse(PFX_66, op as u8, d, b);
        }
    }
    fn int_c(&mut self, op: Int, d: u8, a: u8, c: u64) {
        if self.avx {
            self.vex(VEX_PP_66, VEX_MAP_0F, d, a, 0);
        } else {
            self.mov(d, a);
            self.b(PFX_66);
            self.rex(false, d, 0);
            self.b(ESCAPE_0F);
        }
        self.b(op as u8);
        self.modrm_const(d, c);
    }
    fn shift(&mut self, left: bool, d: u8, a: u8, n: u32) {
        let ext = if left { PSLLQ_EXT } else { PSRLQ_EXT };
        if self.avx {
            self.vex(VEX_PP_66, VEX_MAP_0F, ext, d, a);
        } else {
            self.mov(d, a);
            self.b(PFX_66);
            self.rex(false, 0, d);
            self.b(ESCAPE_0F);
        }
        self.b(PSHIFTQ_IMM8);
        self.modrm_reg(ext, if self.avx { a } else { d });
        self.b(n as u8);
    }
    fn cmp_c(&mut self, x: u8, c: f64, pred: u8) {
        debug_assert!(self.avx || matches!(pred, LT_OQ | LE_OQ | NLT_UQ | GT_OQ));
        if self.avx {
            self.vex(Self::PP, VEX_MAP_0F, XMM0, x, 0);
        } else if pred == GT_OQ {
            let c_lt_x = LT_OQ;
            self.fconst(XMM0, c);
            self.sse(Self::PFX, CMPSD_CMPPD, XMM0, x);
            self.b(c_lt_x & LEGACY_CMP_PRED_BITS);
            return;
        } else {
            self.mov(XMM0, x);
            self.b(Self::PFX);
            self.b(ESCAPE_0F);
        }
        self.b(CMPSD_CMPPD);
        self.modrm_const_imm(XMM0, c.to_bits(), 1);
        self.b(if self.avx {
            pred
        } else {
            pred & LEGACY_CMP_PRED_BITS
        });
    }
    fn keep_clobbering_t(&mut self, d: u8, t: u8) {
        if self.avx {
            self.vex(VEX_PP_66, VEX_MAP_0F3A, d, d, t);
            self.b(VBLENDVPD);
            self.modrm_reg(d, t);
            self.b(IS4_XMM0);
        } else {
            self.sse(PFX_66, ANDPD, t, XMM0);
            self.sse(PFX_66, ANDNPD, XMM0, d);
            self.sse(PFX_66, ORPD, XMM0, t);
            self.mov(d, XMM0);
        }
    }
    fn keep_preserving_x(&mut self, d: u8, x: u8) {
        if self.avx {
            self.keep_clobbering_t(d, x);
        } else {
            self.mov(XMM1, x);
            self.keep_clobbering_t(d, XMM1);
        }
    }
    fn modrm_sib_disp8(&mut self, reg: u8, base: u8, index: u8, disp: u8) {
        self.bytes(&[
            MOD_DISP8 | RM_SIB | ((reg & 7) << 3),
            ((index & 7) << 3) | (base & 7),
            disp,
        ]);
    }
    fn load_at(&mut self, pair: bool, r: u8, index: u8, disp: u8) {
        let pp = if pair { VEX_PP_66 } else { VEX_PP_F2 };
        if self.avx {
            self.vex3(
                false,
                pp,
                VEX_MAP_0F,
                VEX_128,
                r,
                NO_VEX_SOURCE,
                index,
                TABLE,
            );
        } else {
            self.b(if pair { PFX_66 } else { PFX_F2 });
            let rex = REX | ((r >> 3) << 2) | ((index >> 3) << 1) | (TABLE >> 3);
            if rex != REX {
                self.b(rex);
            }
            self.b(ESCAPE_0F);
        }
        self.b(MOV_UNALIGNED_LOAD);
        self.modrm_sib_disp8(r, TABLE, index, disp);
    }
    fn insert_upper_half_at(&mut self, r: u8, index: u8, disp: u8) {
        self.vex3(false, VEX_PP_66, VEX_MAP_0F3A, VEX_256, r, r, index, TABLE);
        self.b(VINSERTF128);
        self.modrm_sib_disp8(r, TABLE, index, disp);
        self.b(UPPER_HALF);
    }
    fn unpack(&mut self, high: bool, d: u8, a: u8, b: u8) {
        let op = if high { UNPCKHPD } else { UNPCKLPD };
        if self.avx {
            self.vex_rr(VEX_PP_66, op, d, a, b);
        } else {
            debug_assert!(d == a || d != b);
            self.mov(d, a);
            self.sse(PFX_66, op, d, b);
        }
    }
    fn lane_to_gpr(&mut self, g: u8, x: u8) {
        if self.avx {
            self.vex3(
                VEX_W1,
                VEX_PP_66,
                VEX_MAP_0F,
                VEX_128,
                x,
                NO_VEX_SOURCE,
                NO_INDEX,
                g,
            );
        } else {
            self.b(PFX_66);
            self.b(REX_W | ((x >> 3) << 2) | (g >> 3));
            self.b(ESCAPE_0F);
        }
        self.b(MOVQ_XMM_TO_GPR);
        self.modrm_reg(x, g);
    }
    fn gpr_to_lane(&mut self, x: u8, g: u8) {
        if self.avx {
            self.vex3(
                VEX_W1,
                VEX_PP_66,
                VEX_MAP_0F,
                VEX_128,
                x,
                NO_VEX_SOURCE,
                NO_INDEX,
                g,
            );
        } else {
            self.b(PFX_66);
            self.b(REX_W | ((x >> 3) << 2) | (g >> 3));
            self.b(ESCAPE_0F);
        }
        self.b(MOVQ_GPR_TO_XMM);
        self.modrm_reg(x, g);
    }
    fn int_to_float(&mut self, x: u8, g: u8) {
        if self.avx {
            self.vex3(VEX_W1, VEX_PP_F2, VEX_MAP_0F, VEX_128, x, x, NO_INDEX, g);
        } else {
            self.b(PFX_F2);
            self.b(REX_W | ((x >> 3) << 2) | (g >> 3));
            self.b(ESCAPE_0F);
        }
        self.b(CVTSI2SD);
        self.modrm_reg(x, g);
    }
    fn gpr_rr(&mut self, op: u8, d: u8, s: u8) {
        self.rex(true, s, d);
        self.b(op);
        self.modrm_reg(s, d);
    }
    fn gpr_shift(&mut self, ext: u8, d: u8, n: u32) {
        self.rex(true, 0, d);
        self.b(SHIFT_IMM8);
        self.modrm_reg(ext, d);
        self.b(n as u8);
    }
    fn gpr_and(&mut self, d: u8, v: u32) {
        self.rex(true, 0, d);
        self.b(ALU_IMM32);
        self.modrm_reg(AND_EXT, d);
        self.bytes(&v.to_le_bytes());
    }
    fn lane1_to_gpr(&mut self, g: u8, x: u8) {
        if self.avx {
            self.vex3(
                VEX_W1,
                VEX_PP_66,
                VEX_MAP_0F3A,
                VEX_128,
                x,
                NO_VEX_SOURCE,
                NO_INDEX,
                g,
            );
            self.b(VPEXTRQ);
            self.modrm_reg(x, g);
            self.b(SECOND_LANE);
        } else {
            self.sse(PFX_66, PSHUFD, XMM0, x);
            self.b(PSHUFD_HIGH_HALF_TWICE);
            self.lane_to_gpr(g, XMM0);
        }
    }
    fn rows(&mut self, table: *const u8, idx: u8, cols: &[u8], tmp: u8) {
        let g = &INDEX_GPRS[..L];
        self.lane_to_gpr(g[0], idx);
        if L > 1 {
            self.lane1_to_gpr(g[1], idx);
        }
        if L == 4 {
            self.vex3(
                false,
                VEX_PP_66,
                VEX_MAP_0F3A,
                VEX_256,
                idx,
                NO_VEX_SOURCE,
                NO_INDEX,
                idx,
            );
            self.b(VEXTRACTI128);
            self.modrm_reg(idx, idx);
            self.b(UPPER_HALF);
            self.lane_to_gpr(g[2], idx);
            self.lane1_to_gpr(g[3], idx);
        }
        self.rows_from(table, cols, tmp);
    }
    fn rows_from(&mut self, table: *const u8, cols: &[u8], tmp: u8) {
        let g = &INDEX_GPRS[..L];
        self.mov_imm(TABLE, table as u64);
        for (p, pair) in cols.chunks(2).enumerate() {
            let at = (p * PAIR_BYTES) as u8;
            let [c0, c1] = [pair[0], pair[1]];
            match L {
                1 => {
                    self.load_at(false, c0, g[0], at);
                    self.load_at(false, c1, g[0], at + F64_BYTES as u8);
                }
                _ => {
                    self.load_at(true, c0, g[0], at);
                    self.load_at(true, tmp, g[1], at);
                    if L == 4 {
                        self.insert_upper_half_at(c0, g[2], at);
                        self.insert_upper_half_at(tmp, g[3], at);
                    }
                    self.unpack(true, c1, c0, tmp);
                    self.unpack(false, c0, c0, tmp);
                }
            }
        }
    }

    fn exp_inline(&mut self, d: u8, x: u8, [a, b, c, e, f]: [u8; 5]) {
        use Arith::*;
        use Operand::{C, R};
        let s = XMM1;
        let t = a;
        self.madd(t, x, C(EXP_INV_LN2_N), C(MAGIC));
        let kd = b;
        self.fc(Sub, kd, t, MAGIC);
        let r = c;
        self.madd(r, kd, C(-EXP_LN2_HI_N), R(x));
        self.madd(r, kd, C(-EXP_LN2_LO_N), R(r));
        let row = e;
        self.int_c(Int::And, row, t, EXP_INDEX_MASK);
        self.shift(true, row, row, EXP_ROW_SHIFT);
        let ki_into_exponent = b;
        self.shift(true, ki_into_exponent, t, EXP_SHIFT);
        let (tail, scale) = (f, d);
        self.rows(EXP_TABLE.as_ptr().cast(), row, &[tail, scale], row);
        self.int(Int::Add, scale, scale, ki_into_exponent);
        let [c2, c3, c4, c5] = EXP_POLY;
        let r2 = a;
        self.f(Mul, r2, r, r);
        let p23 = b;
        self.madd(p23, r, C(c3), C(c2));
        let p45 = e;
        self.madd(p45, r, C(c5), C(c4));
        let p = f;
        self.f(Add, p, tail, r);
        self.madd(p, r2, R(p23), R(p));
        let r4 = a;
        self.f(Mul, r4, r2, r2);
        self.madd(p, r4, R(p45), R(p));
        self.madd(d, scale, R(p), R(scale));
        self.fc(Mul, d, d, EXP_UNBIAS);
        let zero = s;
        self.zero(zero);
        self.cmp_c(x, EXP_UNDERFLOW, LT_OQ);
        self.keep_clobbering_t(d, zero);
        self.cmp_mask_into_xmm0(x, x, UNORD_Q);
        self.keep_preserving_x(d, x);
        let linear_tail = s;
        self.fc(Sub, linear_tail, x, EXP_LIMIT);
        self.fc(Add, linear_tail, linear_tail, 1.0);
        self.fc(Mul, linear_tail, linear_tail, EXP_AT_LIMIT);
        self.cmp_c(x, EXP_LIMIT, GT_OQ);
        self.keep_clobbering_t(d, linear_tail);
    }

    fn ln_inline(&mut self, d: u8, x: u8, [a, b, c, e, f]: [u8; 5]) {
        use Arith::*;
        use Operand::{C, R};
        let s = XMM1;
        let table = LN_TABLE.as_ptr().cast();
        let (kd, z) = (b, e);
        let (invc, c_row, logc_hi, logc_lo) = (a, c, f, d);
        if L == 1 {
            let [tmp, k, ix] = [INDEX_GPRS[0], INDEX_GPRS[1], INDEX_GPRS[2]];
            self.lane_to_gpr(ix, x);
            self.mov_imm(k, LN_OFF);
            self.gpr_rr(GPR_MOV, tmp, ix);
            self.gpr_rr(GPR_SUB, tmp, k);
            self.gpr_rr(GPR_MOV, k, tmp);
            self.gpr_shift(GPR_SAR, k, F64_FRAC_BITS);
            self.zero(kd);
            self.int_to_float(kd, k);
            self.gpr_shift(GPR_SHL, k, F64_FRAC_BITS);
            self.gpr_rr(GPR_SUB, ix, k);
            self.gpr_to_lane(z, ix);
            let row = tmp;
            self.gpr_shift(GPR_SHR, row, LN_SHIFT - LN_ROW_SHIFT);
            self.gpr_and(row, LN_ROW_MASK);
            self.rows_from(table, &[invc, c_row, logc_hi, logc_lo], s);
        } else {
            let tmp = a;
            self.int_c(Int::Sub, tmp, x, LN_OFF);
            self.shift(false, kd, tmp, F64_FRAC_BITS);
            self.int_c(Int::Xor, kd, kd, LN_K_OFFSET);
            self.int_c(Int::Add, kd, kd, MAGIC.to_bits());
            self.fc(Sub, kd, kd, LN_K_MAGIC);
            let exponent_bits = c;
            self.int_c(Int::And, exponent_bits, tmp, LN_EXP_MASK);
            self.int(Int::Sub, z, x, exponent_bits);
            let row = c;
            self.shift(false, row, tmp, LN_SHIFT - LN_ROW_SHIFT);
            self.int_c(Int::And, row, row, LN_INDEX_MASK << LN_ROW_SHIFT);
            self.rows(table, row, &[invc, c_row, logc_hi, logc_lo], s);
        }
        let r = e;
        self.f(Sub, r, z, c_row);
        self.f(Mul, r, r, invc);
        let w = a;
        self.madd(w, kd, C(LN_LN2_HI), R(logc_hi));
        let hi = c;
        self.f(Add, hi, w, r);
        let lo = a;
        self.f(Sub, lo, w, hi);
        self.f(Add, lo, lo, r);
        let k_ln2_lo_plus_logc_lo = f;
        self.madd(k_ln2_lo_plus_logc_lo, kd, C(LN_LN2_LO), R(logc_lo));
        self.f(Add, lo, lo, k_ln2_lo_plus_logc_lo);
        let p = LN_POLY;
        let r2 = s;
        self.f(Mul, r2, r, r);
        let q01 = b;
        self.madd(q01, r, C(p[1]), C(p[0]));
        let q23 = d;
        self.madd(q23, r, C(p[3]), C(p[2]));
        let q03 = b;
        self.madd(q03, r2, R(q23), R(q01));
        let q45 = d;
        self.madd(q45, r, C(p[5]), C(p[4]));
        let q46 = d;
        self.madd(q46, r2, C(p[6]), R(q45));
        let r4 = f;
        self.f(Mul, r4, r2, r2);
        let poly = b;
        self.madd(poly, r4, R(q46), R(q03));
        let t = a;
        self.madd(t, r2, C(-0.5), R(lo));
        let r3 = s;
        self.f(Mul, r3, r2, r);
        self.madd(t, r3, R(poly), R(t));
        self.f(Add, d, t, hi);
        self.cmp_c(x, f64::INFINITY, NLT_UQ);
        self.keep_preserving_x(d, x);
        let at_floor = s;
        self.fconst(at_floor, LN_AT_FLOOR);
        self.cmp_c(x, LN_FLOOR, LE_OQ);
        self.keep_clobbering_t(d, at_floor);
    }
}

#[derive(Clone, Copy)]
enum Operand {
    R(u8),
    C(f64),
}

const GPR_MOV: u8 = 0x89;
const GPR_SUB: u8 = 0x29;
const GPR_SHL: u8 = 4;
const GPR_SHR: u8 = 5;
const GPR_SAR: u8 = 7;

const FMA132: u8 = 0x98;
const FMA213: u8 = 0xA8;
const FMA231: u8 = 0xB8;

const TABLE: u8 = R11;
const INDEX_GPRS: [u8; 4] = [RAX, RCX, RDX, R10];
const fn volatile_on_both_abis(g: u8) -> bool {
    let mut j = 0;
    while j < VOLATILE_ON_BOTH_ABIS.len() {
        if VOLATILE_ON_BOTH_ABIS[j] == g {
            return true;
        }
        j += 1;
    }
    false
}
const _: () = {
    assert!(volatile_on_both_abis(TABLE));
    let mut k = 0;
    while k < INDEX_GPRS.len() {
        assert!(volatile_on_both_abis(INDEX_GPRS[k]));
        k += 1;
    }
};
const F64_BYTES: usize = std::mem::size_of::<f64>();
const PAIR_BYTES: usize = 2 * F64_BYTES;
const EXP_ROW_VALUES: usize = 2;
const LN_ROW_VALUES: usize = 4;
const EXP_ROW_SHIFT: u32 = (EXP_ROW_VALUES * F64_BYTES).trailing_zeros();
const LN_ROW_SHIFT: u32 = (LN_ROW_VALUES * F64_BYTES).trailing_zeros();
const LN_K_OFFSET: u64 = 1 << (u64::BITS - 1 - F64_FRAC_BITS);
const LN_K_MAGIC: f64 = MAGIC + LN_K_OFFSET as f64;

#[derive(Clone, Copy)]
enum Int {
    Add = 0xD4,
    Sub = 0xFB,
    And = 0xDB,
    Xor = 0xEF,
}

const NLT_UQ: u8 = 0x15;
const LT_OQ: u8 = 0x11;
const LE_OQ: u8 = 0x12;
const UNORD_Q: u8 = 0x03;
const GT_OQ: u8 = 0x1E;
const LEGACY_CMP_PRED_BITS: u8 = 7;
const CMP_EQ_OQ: u8 = 0;
const CMP_LT_OS: u8 = 1;
const CMP_LE_OS: u8 = 2;
const CMP_NEQ_UQ: u8 = 4;

impl<const L: usize> Isa for X64<L> {
    const LANES: usize = L;
    const CACHE: &'static [u8] = if WIN {
        &[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5]
    } else {
        &[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    };
    const SAVED: usize = if WIN && L < 4 {
        WIN_SAVED_XMMS as usize
    } else {
        0
    };
    const RESULT: u8 = XMM0;

    fn new(hot: &[*const ()]) -> X64<L> {
        assert!(matches!(L, 1 | 2 | 4), "one, two or four lanes");
        let Features { avx, avx2, .. } = features();
        assert!(L < 4 || avx2, "four lanes need AVX2");
        X64 {
            code: Vec::with_capacity(8192),
            avx,
            fma: avx && rsdag::semantics::fused(),
            hot: hot.iter().copied().take(HOT_REGS.len()).collect(),
            pool: Vec::new(),
            fixups: Vec::new(),
        }
    }
    fn finish(mut self) -> Result<Vec<u8>, crate::JitError> {
        const { assert!(Self::POOL_ENTRY_BYTES.is_multiple_of(LEGACY_MEM_OPERAND_ALIGN)) };
        if !self.pool.is_empty() {
            self.code.resize(
                self.code.len().next_multiple_of(LEGACY_MEM_OPERAND_ALIGN),
                INT3,
            );
            let base = self.code.len();
            for c in &self.pool {
                for _ in 0..Self::POOL_ENTRY_BYTES / F64_BYTES {
                    self.code.extend_from_slice(&c.to_le_bytes());
                }
            }
            for fix in &self.fixups {
                let disp =
                    (base + Self::POOL_ENTRY_BYTES * fix.constant) as i64 - fix.insn_end as i64;
                let disp = i32::try_from(disp).map_err(|_| {
                    crate::JitError::Codegen(
                        "a constant of a chunk lies beyond a 32-bit displacement".into(),
                    )
                })?;
                self.code[fix.disp_at..fix.disp_at + DISP32_BYTES]
                    .copy_from_slice(&disp.to_le_bytes());
            }
        }
        Ok(self.code)
    }

    fn prologue(&mut self) {
        self.bytes(&PUSH_RBP_MOV_RBP_RSP);
        self.bytes(&PUSH_RBX_R12_R13_R14_R15);
        if WIN {
            self.bytes(&SUB_RSP_IMM32);
            self.bytes(&WIN_FRAME.to_le_bytes());
            for k in 0..WIN_SAVED_XMMS {
                let x = FIRST_WIN_SAVED_XMM + k;
                self.xmm_rsp(true, x, XMM_SAVE_BYTES * k as i32);
            }
            self.bytes(&MOV_RBX_RCX);
            self.bytes(&MOV_R13_RDX);
            self.bytes(&MOV_R14_R8);
        } else {
            self.bytes(&SUB_RSP_ALIGN_PAD);
            self.bytes(&MOV_RBX_RDI);
            self.bytes(&MOV_R13_RSI);
            self.bytes(&MOV_R14_RDX);
        }
        for k in 0..self.hot.len() {
            let a = self.hot[k] as usize as u64;
            self.mov_imm(HOT_REGS[k], a);
        }
    }
    fn epilogue(&mut self) {
        if L == 4 {
            self.bytes(&VZEROUPPER);
        }
        if WIN {
            for k in 0..WIN_SAVED_XMMS {
                let x = FIRST_WIN_SAVED_XMM + k;
                self.xmm_rsp(false, x, XMM_SAVE_BYTES * k as i32);
            }
            self.bytes(&ADD_RSP_IMM32);
            self.bytes(&WIN_FRAME.to_le_bytes());
        } else {
            self.bytes(&ADD_RSP_ALIGN_PAD);
        }
        self.bytes(&POP_R15_R14_R13_R12_RBX_RBP_RET);
    }

    fn load(&mut self, r: u8, base: Base, off: usize) {
        let b = base.register(WORK, INPUTS);
        self.mov_mem(Self::PFX, Self::PP, Self::VL, MOV_UNALIGNED_LOAD, r, b, off);
    }
    fn store(&mut self, r: u8, base: Base, off: usize) {
        let b = base.register(WORK, INPUTS);
        self.mov_mem(
            Self::PFX,
            Self::PP,
            Self::VL,
            MOV_UNALIGNED_STORE,
            r,
            b,
            off,
        );
    }
    fn load_lane(&mut self, r: u8, base: Base, off: usize) {
        self.first_lane_mov(false, r, base, off);
    }
    fn store_lane(&mut self, r: u8, base: Base, off: usize) {
        self.first_lane_mov(true, r, base, off);
    }
    fn fconst(&mut self, r: u8, v: f64) {
        if v.to_bits() == 0 {
            self.zero(r);
        } else if L == 4 {
            self.vex(VEX_PP_66, VEX_MAP_0F, r, NO_VEX_SOURCE, 0);
            self.b(MOV_UNALIGNED_LOAD);
            self.modrm_const(r, v.to_bits());
        } else {
            self.b(if L == 1 { PFX_F2 } else { PFX_66 });
            self.rex(false, r, 0);
            self.bytes(&[ESCAPE_0F, if L == 1 { MOV_UNALIGNED_LOAD } else { MOVAPD }]);
            self.modrm_const(r, v.to_bits());
        }
    }
    fn mov(&mut self, d: u8, a: u8) {
        if d != a {
            self.unary_rr(PFX_66, VEX_PP_66, MOVAPD, d, a);
        }
    }

    fn arith(&mut self, op: Arith, d: u8, a: u8, b: u8) {
        let opc = Self::arith_op(op);
        if self.avx {
            self.vex_rr(Self::PP, opc, d, a, b);
        } else if d == b && d != a {
            if matches!(op, Arith::Add | Arith::Mul) {
                self.sse(Self::PFX, opc, d, a);
            } else {
                self.mov(XMM0, a);
                self.sse(Self::PFX, opc, XMM0, b);
                self.mov(d, XMM0);
            }
        } else {
            self.mov(d, a);
            self.sse(Self::PFX, opc, d, b);
        }
    }
    const MEM_OPERANDS: bool = !Self::LEGACY_PACKED || WORK_ARRAY_ALIGN >= LEGACY_MEM_OPERAND_ALIGN;
    fn arith_mem(&mut self, op: Arith, d: u8, a: u8, base: Base, off: usize) {
        let base = base.register(WORK, INPUTS);
        if self.avx {
            self.vex(Self::PP, VEX_MAP_0F, d, a, base);
        } else {
            self.mov(d, a);
            self.b(Self::PFX);
            self.rex(false, d, base);
            self.b(ESCAPE_0F);
        }
        self.b(Self::arith_op(op));
        self.modrm_mem(d, base, off);
    }
    const MINMAX: bool = true;
    fn minmax(&mut self, op: ReduceOp, d: u8, a: u8, b: u8) {
        let opc = match op {
            ReduceOp::Min => MIN,
            _ => MAX,
        };
        if self.avx {
            self.vex_rr(Self::PP, opc, XMM1, a, b);
        } else {
            self.mov(XMM1, a);
            self.sse(Self::PFX, opc, XMM1, b);
        }
        let b_is_not_nan = CMP_EQ_OQ;
        self.cmp_mask_into_xmm0(b, b, b_is_not_nan);
        self.blend_by_xmm0_mask(XMM1, a, d);
    }
    fn neg(&mut self, d: u8, a: u8) {
        self.op_const(PFX_66, VEX_PP_66, XORPD, d, a, SIGN_BIT);
    }
    fn abs(&mut self, d: u8, a: u8) {
        self.op_const(PFX_66, VEX_PP_66, ANDPD, d, a, !SIGN_BIT);
    }
    fn sqrt(&mut self, d: u8, a: u8) {
        self.unary_rr(Self::PFX, Self::PP, SQRT, d, a);
    }
    fn round_inline() -> bool {
        features().sse41
    }
    fn round(&mut self, mode: Round, d: u8, a: u8) {
        let imm = match mode {
            Round::Floor => 0x9,
            Round::Ceil => 0xA,
            Round::Trunc => 0xB,
        };
        let op = if L == 1 { ROUNDSD } else { ROUNDPD };
        if L == 4 {
            self.vex(VEX_PP_66, VEX_MAP_0F3A, d, NO_VEX_SOURCE, a);
            self.b(op);
        } else {
            self.b(PFX_66);
            self.rex(false, d, a);
            self.bytes(&ESCAPE_0F3A);
            self.b(op);
        }
        self.modrm_reg(d, a);
        self.b(imm);
    }
    fn cmp_select(&mut self, op: CmpOp, a: u8, b: u8, t: u8, e: u8, d: u8) {
        let (x, y, pred) = match op {
            CmpOp::Gt => (b, a, CMP_LT_OS),
            CmpOp::Ge => (b, a, CMP_LE_OS),
            CmpOp::Lt => (a, b, CMP_LT_OS),
            CmpOp::Le => (a, b, CMP_LE_OS),
            CmpOp::Eq => (a, b, CMP_EQ_OQ),
            CmpOp::Ne => (a, b, CMP_NEQ_UQ),
        };
        self.cmp_mask_into_xmm0(x, y, pred);
        self.blend_by_xmm0_mask(t, e, d);
    }
    fn select_nz(&mut self, c: u8, t: u8, e: u8, d: u8) {
        self.zero(XMM1);
        self.cmp_mask_into_xmm0(c, XMM1, CMP_NEQ_UQ);
        self.blend_by_xmm0_mask(t, e, d);
    }
    fn exp_ln(&mut self, ln: bool, d: u8, x: u8, t: [u8; 5]) {
        if ln {
            self.ln_inline(d, x, t);
        } else {
            self.exp_inline(d, x, t);
        }
    }

    fn call(&mut self, addr: *const (), args: &[Arg]) {
        let counts_floats_and_ints_together = WIN;
        let stacked = if counts_floats_and_ints_together {
            args.len()
        } else {
            args.iter().filter(|a| matches!(a, Arg::I(_))).count()
        }
        .saturating_sub(INT_ARGS.len());
        let shadow = if WIN { WIN_SHADOW_SPACE_BYTES } else { 0 };
        let frame = (shadow + GPR_BYTES * stacked as i32 + STACK_ALIGN - 1) & !(STACK_ALIGN - 1);
        if frame > 0 {
            self.move_rsp(-frame);
        }
        let (mut nf, mut ni) = (0usize, 0usize);
        for (pos, arg) in args.iter().enumerate() {
            match *arg {
                Arg::F(r) => {
                    let k = if counts_floats_and_ints_together {
                        pos
                    } else {
                        nf
                    };
                    nf += 1;
                    assert!(k < FLOAT_ARG_REGS, "float arguments fit the registers");
                    self.mov(k as u8, r);
                }
                Arg::I(iarg) => {
                    let k = if counts_floats_and_ints_together {
                        pos
                    } else {
                        ni
                    };
                    ni += 1;
                    match k.checked_sub(INT_ARGS.len()) {
                        Some(slot) => {
                            self.int_arg(CALL_ADDRESS_SCRATCH, iarg);
                            self.store_rsp(CALL_ADDRESS_SCRATCH, shadow + GPR_BYTES * slot as i32);
                        }
                        None => self.int_arg(INT_ARGS[k], iarg),
                    }
                }
            }
        }
        if L == 4 {
            self.bytes(&VZEROUPPER);
        }
        match self.hot.iter().position(|&h| h == addr) {
            Some(k) => self.bytes(&[REX_B, CALL_RM, MODRM_CALL_REG | (HOT_REGS[k] & 7)]),
            None => {
                self.mov_imm(CALL_ADDRESS_SCRATCH, addr as usize as u64);
                self.bytes(&CALL_RAX);
            }
        }
        if frame > 0 {
            self.move_rsp(frame);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_exponent_through_a_logical_shift_and_the_rounding_constant_is_the_arithmetic_shift() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let random = std::iter::repeat_with(move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        });
        let edges = [0, 1 << 63, u64::MAX, LN_OFF, 0x7FF0u64 << 48];
        for tmp in edges.into_iter().chain(random.take(100_000)) {
            let biased = ((tmp >> F64_FRAC_BITS) ^ LN_K_OFFSET).wrapping_add(MAGIC.to_bits());
            let kd = f64::from_bits(biased) - LN_K_MAGIC;
            assert_eq!(kd, ((tmp as i64) >> F64_FRAC_BITS) as f64, "tmp {tmp:#x}");
        }
    }
}
