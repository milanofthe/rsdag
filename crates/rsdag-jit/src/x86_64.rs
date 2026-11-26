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

use crate::isa::*;
use rsdag::node::{CmpOp, ReduceOp};
use rsdag::semantics::exp_ln::*;

const WORK: u8 = 3; // rbx
const INPUTS: u8 = 13; // r13
const LAYOUT: u8 = 14; // r14
const WIN: bool = cfg!(windows);
/// Integer argument registers: rdi rsi rdx rcx r8 r9, or rcx rdx r8 r9.
const INT_ARGS: &[u8] = if WIN {
    &[1, 2, 8, 9]
} else {
    &[7, 6, 2, 1, 8, 9]
};
/// The Windows frame below the six pushes: `xmm6` to `xmm15` and 8 bytes of
/// alignment. Shadow space and stack arguments are not in it -- a call
/// reserves what it needs (see [`X64::call`]), so no frame has to guess how
/// wide the widest host routine is.
const WIN_FRAME: i32 = 168;
/// Windows wants 32 bytes of shadow space below every call's arguments.
const WIN_SHADOW: i32 = 32;

pub(crate) struct X64<const L: usize> {
    code: Vec<u8>,
    /// Three-operand VEX forms and `vblendvpd`.
    avx: bool,
    /// Fused multiply-adds (`rsdag::semantics::fused`).
    fma: bool,
    /// Host routines held in `r12` and `r15`.
    hot: Vec<*const ()>,
    /// The chunk's constants, and where their displacements go: the byte
    /// offset of each `disp32`, the constant it addresses and where its
    /// instruction ends (what the displacement is relative to).
    pool: Vec<u64>,
    fixups: Vec<(usize, usize, usize)>,
}

const HOT_REGS: [u8; 2] = [12, 15];

/// The instruction sets the encodings may use.
#[derive(Clone, Copy)]
pub(crate) struct Features {
    pub sse41: bool,
    /// Three-operand VEX forms, `vblendvpd` and the 256-bit forms.
    pub avx: bool,
    /// 256-bit integer forms, which four lanes need.
    pub avx2: bool,
}

const LN_ROW_MASK: u32 = {
    let mask = LN_INDEX_MASK << LN_ROW_SHIFT;
    assert!(mask <= u32::MAX as u64);
    mask as u32
};

/// What this CPU has, or the SSE2 baseline when `RSDAG_SSE2` is set.
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
    /// The legacy prefix of the arithmetic forms: `F2` (`sd`) or `66` (`pd`).
    const PFX: u8 = if L == 1 { 0xF2 } else { 0x66 };
    /// The same as a VEX `pp` field.
    const PP: u8 = if L == 1 { 3 } else { 1 };
    /// The VEX `L` bit: 256-bit forms for four lanes.
    const VL: u8 = (L == 4) as u8;
    /// Bytes per constant in the pool: every lane holds it.
    const CONST_BYTES: usize = 8 * if L == 4 { 4 } else { 2 };
    fn b(&mut self, byte: u8) {
        self.code.push(byte);
    }
    fn bytes(&mut self, bs: &[u8]) {
        self.code.extend_from_slice(bs);
    }
    /// A REX prefix when any of its bits is set.
    fn rex(&mut self, w: bool, reg: u8, rm: u8) {
        let v = 0x40 | ((w as u8) << 3) | ((reg >> 3) << 2) | (rm >> 3);
        if v != 0x40 {
            self.b(v);
        }
    }
    fn modrm_reg(&mut self, reg: u8, rm: u8) {
        self.b(0xC0 | ((reg & 7) << 3) | (rm & 7));
    }
    /// `[base + disp]` for `rbx` and `r13` (neither needs a SIB byte).
    fn modrm_mem(&mut self, reg: u8, base: u8, disp: usize) {
        let disp = i32::try_from(disp).expect("work array offset fits i32");
        if let Ok(d8) = i8::try_from(disp) {
            self.b(0x40 | ((reg & 7) << 3) | (base & 7));
            self.b(d8 as u8);
        } else {
            self.b(0x80 | ((reg & 7) << 3) | (base & 7));
            self.bytes(&disp.to_le_bytes());
        }
    }
    /// A legacy-SSE register-register instruction `prefix 0F op /r`.
    fn sse(&mut self, prefix: u8, op: u8, reg: u8, rm: u8) {
        self.b(prefix);
        self.rex(false, reg, rm);
        self.bytes(&[0x0F, op]);
        self.modrm_reg(reg, rm);
    }
    fn sse_mem(&mut self, prefix: u8, op: u8, reg: u8, base: u8, disp: usize) {
        self.b(prefix);
        self.rex(false, reg, base);
        self.bytes(&[0x0F, op]);
        self.modrm_mem(reg, base, disp);
    }
    /// A move between a register and memory: the legacy form, or for four
    /// lanes the VEX form of vector length `l` (the scalar moves `0`).
    fn mov_mem(&mut self, prefix: u8, pp: u8, l: u8, op: u8, reg: u8, base: u8, disp: usize) {
        if L == 4 {
            self.vex_l(pp, 1, l, reg, 0, base);
            self.b(op);
            self.modrm_mem(reg, base, disp);
        } else {
            self.sse_mem(prefix, op, reg, base, disp);
        }
    }
    /// `op d, a`: a register instruction of map `0F` with one source,
    /// legacy, or VEX for four lanes.
    fn unary_rr(&mut self, prefix: u8, pp: u8, op: u8, d: u8, a: u8) {
        if L == 4 {
            self.vex_rr(pp, op, d, 0, a);
        } else {
            self.sse(prefix, op, d, a);
        }
    }
    /// `xorpd r, r`: zero.
    fn zero(&mut self, r: u8) {
        if L == 4 {
            self.vex_rr(1, 0x57, r, r, r);
        } else {
            self.sse(0x66, 0x57, r, r);
        }
    }
    /// `mov r64, imm64` (or the shorter zero-extending `mov r32, imm32`).
    fn mov_imm(&mut self, rd: u8, v: u64) {
        if let Ok(v32) = u32::try_from(v) {
            self.rex(false, 0, rd);
            self.b(0xB8 | (rd & 7));
            self.bytes(&v32.to_le_bytes());
        } else {
            self.rex(true, 0, rd);
            self.b(0xB8 | (rd & 7));
            self.bytes(&v.to_le_bytes());
        }
    }
    /// `[rsp + disp]`, which needs a SIB byte.
    fn modrm_rsp(&mut self, reg: u8, disp: i32) {
        if let Ok(d8) = i8::try_from(disp) {
            self.bytes(&[0x40 | ((reg & 7) << 3) | 4, 0x24, d8 as u8]);
        } else {
            self.bytes(&[0x80 | ((reg & 7) << 3) | 4, 0x24]);
            self.bytes(&disp.to_le_bytes());
        }
    }
    /// `movdqu [rsp + disp], xmm` (`store`) or the load.
    fn xmm_rsp(&mut self, store: bool, xmm: u8, disp: i32) {
        self.b(0xF3);
        self.rex(false, xmm, 0);
        self.bytes(&[0x0F, if store { 0x7F } else { 0x6F }]);
        self.modrm_rsp(xmm, disp);
    }
    /// `add rsp, n`, or `sub rsp, -n`. `n` stays a multiple of 16 so that
    /// `rsp` keeps the alignment every call below relies on.
    fn move_rsp(&mut self, n: i32) {
        debug_assert_eq!(n % 16, 0, "rsp moves in 16-byte steps");
        self.bytes(if n >= 0 {
            &[0x48, 0x81, 0xC4]
        } else {
            &[0x48, 0x81, 0xEC]
        });
        self.bytes(&n.abs().to_le_bytes());
    }
    /// `mov [rsp + disp], r64`.
    fn store_rsp(&mut self, r: u8, disp: i32) {
        self.rex(true, r, 0);
        self.b(0x89);
        self.modrm_rsp(r, disp);
    }
    /// An integer argument into `r64`.
    fn int_arg(&mut self, rk: u8, arg: IArg) {
        match arg {
            IArg::Imm(v) => self.mov_imm(rk, v),
            IArg::WorkAddr(off) => {
                self.rex(true, rk, WORK);
                self.b(0x8D); // lea rk, [rbx + off]
                self.modrm_mem(rk, WORK, off);
            }
            IArg::InputAddr(off) => {
                self.rex(true, rk, INPUTS);
                self.b(0x8D); // lea rk, [r13 + off]
                self.modrm_mem(rk, INPUTS, off);
            }
            IArg::Layout => {
                self.rex(true, LAYOUT, rk);
                self.b(0x89); // mov rk, r14
                self.modrm_reg(LAYOUT, rk);
            }
        }
    }
    /// A VEX prefix for map `map` (1 `0F`, 3 `0F3A`), the legacy prefix
    /// `pp` (0 none, 1 `66`, 2 `F3`, 3 `F2`) and the vector length `l` (0
    /// 128 bits, 1 256): `reg` the ModRM reg field, `v` the extra source,
    /// `rm` the ModRM rm field. The two-byte form where it can encode them.
    fn vex_l(&mut self, pp: u8, map: u8, l: u8, reg: u8, v: u8, rm: u8) {
        let r = (!reg >> 3) & 1;
        let b = (!rm >> 3) & 1;
        let v = !v & 0xF;
        let tail = (v << 3) | (l << 2) | pp;
        if map == 1 && b == 1 {
            self.bytes(&[0xC5, (r << 7) | tail]);
        } else {
            self.bytes(&[0xC4, (r << 7) | (1 << 6) | (b << 5) | map, tail]);
        }
    }
    /// [`vex_l`](Self::vex_l) at the lanes' vector length.
    fn vex(&mut self, pp: u8, map: u8, reg: u8, v: u8, rm: u8) {
        self.vex_l(pp, map, Self::VL, reg, v, rm);
    }
    /// `op d, a, rm`: a VEX register instruction of map `0F`.
    fn vex_rr(&mut self, pp: u8, op: u8, d: u8, a: u8, rm: u8) {
        self.vex(pp, 1, d, a, rm);
        self.b(op);
        self.modrm_reg(d, rm);
    }
    /// ModRM for `[rip + disp32]` addressing the constant `bits` of the
    /// chunk's pool; `finish` writes the displacement.
    fn modrm_const(&mut self, reg: u8, bits: u64) {
        self.modrm_const_imm(reg, bits, 0);
    }
    /// The same for an instruction that ends in `imm` immediate bytes after
    /// the displacement (rip is the next instruction's address).
    fn modrm_const_imm(&mut self, reg: u8, bits: u64, imm: usize) {
        self.b(((reg & 7) << 3) | 5);
        let k = match self.pool.iter().position(|&c| c == bits) {
            Some(k) => k,
            None => {
                self.pool.push(bits);
                self.pool.len() - 1
            }
        };
        let at = self.code.len();
        self.fixups.push((at, k, at + 4 + imm));
        self.bytes(&[0; 4]);
    }
    /// `d = a op [constant]`: VEX where there is AVX, else a copy and the
    /// legacy two-operand form (`prefix 0F op`).
    fn op_const(&mut self, prefix: u8, pp: u8, op: u8, d: u8, a: u8, bits: u64) {
        if self.avx {
            self.vex(pp, 1, d, a, 0);
        } else {
            self.mov(d, a);
            self.b(prefix);
            self.rex(false, d, 0);
            self.b(0x0F);
        }
        self.b(op);
        self.modrm_const(d, bits);
    }
    /// `d = mask ? t : e` with the mask in `xmm0`; clobbers `xmm0`.
    fn blend(&mut self, t: u8, e: u8, d: u8) {
        if self.avx {
            // vblendvpd d, e, t, xmm0
            self.vex(1, 3, d, e, t);
            self.b(0x4B);
            self.modrm_reg(d, t);
            self.b(0x00);
            return;
        }
        self.mov(d, t);
        self.sse(0x66, 0x54, d, 0); // andpd d, xmm0
        self.sse(0x66, 0x55, 0, e); // andnpd xmm0, e
        self.sse(0x66, 0x56, d, 0); // orpd d, xmm0
    }
    /// `xmm0 = (x pred y) ? all ones : 0`, `cmpsd`.
    fn cmp_mask(&mut self, x: u8, y: u8, pred: u8) {
        if self.avx {
            self.vex_rr(Self::PP, 0xC2, 0, x, y);
        } else {
            self.mov(0, x);
            self.sse(Self::PFX, 0xC2, 0, y);
        }
        self.b(pred);
    }
    /// The scalar opcode (`F2 0F op`) of an arithmetic op.
    fn arith_op(op: Arith) -> u8 {
        match op {
            Arith::Add => 0x58,
            Arith::Mul => 0x59,
            Arith::Sub => 0x5C,
            Arith::Div => 0x5E,
        }
    }

    // The inline exponential and logarithm: VEX where there is AVX (and
    // AVX2 for the integer forms on four lanes), else the legacy SSE2
    // forms; a fused multiply-add where `rsdag::semantics::fused` says.

    /// `d = a op b`.
    fn f(&mut self, op: Arith, d: u8, a: u8, b: u8) {
        self.arith(op, d, a, b);
    }
    /// `d = a op c`, `c` from the pool.
    fn fc(&mut self, op: Arith, d: u8, a: u8, c: f64) {
        self.op_const(Self::PFX, Self::PP, Self::arith_op(op), d, a, c.to_bits());
    }
    /// The three-operand VEX prefix with every field: `w` its W bit, `map`
    /// (1 `0F`, 2 `0F38`, 3 `0F3A`), `l` the vector length, `reg` the ModRM
    /// reg field, `v` the extra source, `x` a SIB index and `rm` the ModRM
    /// rm field (or SIB base).
    fn vex3(&mut self, w: bool, pp: u8, map: u8, l: u8, reg: u8, v: u8, x: u8, rm: u8) {
        let r = (!reg >> 3) & 1;
        let xb = (!x >> 3) & 1;
        let b = (!rm >> 3) & 1;
        let v = !v & 0xF;
        self.bytes(&[
            0xC4,
            (r << 7) | (xb << 6) | (b << 5) | map,
            ((w as u8) << 7) | (v << 3) | (l << 2) | pp,
        ]);
    }
    /// `d = a b + c` as `vfmadd{form}` (`0x98` 132, `0xA8` 213, `0xB8`
    /// 231) on `d`, the register `v` and `m`, a register or a constant.
    fn fma3(&mut self, form: u8, d: u8, v: u8, m: Operand) {
        let op = form + (L == 1) as u8; // the `sd` form is the next opcode
        let rm = match m {
            Operand::R(r) => r,
            Operand::C(_) => 0,
        };
        self.vex3(true, 1, 2, Self::VL, d, v, 0, rm);
        self.b(op);
        match m {
            Operand::R(r) => self.modrm_reg(d, r),
            Operand::C(c) => self.modrm_const(d, c.to_bits()),
        }
    }
    /// `d = a b + c`, rounded once where the machine fuses, else the product
    /// and the sum each rounded (`rsdag::semantics::MulAdd`). `a` is a
    /// register; `d` is `a` or `c` or none of the operands; `xmm0` is the
    /// product's scratch.
    fn madd(&mut self, d: u8, a: u8, b: Operand, c: Operand) {
        use Arith::*;
        if !self.fma {
            // the product into `d`, or into `xmm0` where `d` holds `c`
            let p = match c {
                Operand::R(c) if c == d => 0,
                _ => d,
            };
            match b {
                Operand::R(b) => self.f(Mul, p, a, b),
                Operand::C(b) => self.fc(Mul, p, a, b),
            }
            match c {
                Operand::R(_) if p == 0 => self.f(Add, d, d, 0),
                Operand::R(c) => self.f(Add, d, d, c),
                Operand::C(c) => self.fc(Add, d, d, c),
            }
            return;
        }
        match (b, c) {
            // d = d b + c, b a register or the constant
            (_, Operand::R(c)) if d == c => match b {
                Operand::R(b) => self.fma3(FMA231, d, a, Operand::R(b)),
                Operand::C(_) => self.fma3(FMA231, d, a, b),
            },
            (Operand::C(_), Operand::R(c)) => {
                // d = a [b] + c: d = d [b] + c with d = a
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
    /// `d = a op b` over 64-bit integer lanes (`paddq` and the like).
    fn int(&mut self, op: Int, d: u8, a: u8, b: u8) {
        if self.avx {
            self.vex_rr(1, op as u8, d, a, b);
        } else if d == b && d != a {
            debug_assert!(!matches!(op, Int::Sub), "a difference into its subtrahend");
            self.sse(0x66, op as u8, d, a);
        } else {
            self.mov(d, a);
            self.sse(0x66, op as u8, d, b);
        }
    }
    /// `d = a op c` over 64-bit integer lanes, `c` from the pool.
    fn int_c(&mut self, op: Int, d: u8, a: u8, c: u64) {
        if self.avx {
            self.vex(1, 1, d, a, 0);
        } else {
            self.mov(d, a);
            self.b(0x66);
            self.rex(false, d, 0);
            self.b(0x0F);
        }
        self.b(op as u8);
        self.modrm_const(d, c);
    }
    /// `d = a << n`, or logical `a >> n`, over 64-bit lanes: `psllq`
    /// (`66 0F 73 /6 ib`), `psrlq` (`/2`).
    fn shift(&mut self, left: bool, d: u8, a: u8, n: u32) {
        let ext = if left { 6 } else { 2 };
        if self.avx {
            self.vex(1, 1, ext, d, a);
        } else {
            self.mov(d, a);
            self.b(0x66);
            self.rex(false, 0, d);
            self.b(0x0F);
        }
        self.b(0x73);
        self.modrm_reg(ext, if self.avx { a } else { d });
        self.b(n as u8);
    }
    /// `xmm0 = (x pred c) ? all ones : 0`, `c` from the pool, `pred` one
    /// of VEX's 32 (of them `LT_OQ`, `LE_OQ`, `NLT_UQ`, `GT_OQ` without AVX).
    fn cmp_c(&mut self, x: u8, c: f64, pred: u8) {
        if self.avx {
            self.vex(Self::PP, 1, 0, x, 0);
        } else if pred == GT_OQ {
            // c < x
            self.fconst(0, c);
            self.sse(Self::PFX, 0xC2, 0, x);
            self.b(LT_OQ & 7);
            return;
        } else {
            self.mov(0, x);
            self.b(Self::PFX);
            self.b(0x0F);
        }
        self.b(0xC2);
        self.modrm_const_imm(0, c.to_bits(), 1);
        self.b(if self.avx { pred } else { pred & 7 });
    }
    /// `d = xmm0 ? t : d`; clobbers `xmm0`, and `t` without AVX.
    fn keep(&mut self, d: u8, t: u8) {
        if self.avx {
            self.vex(1, 3, d, d, t); // vblendvpd d, d, t, xmm0
            self.b(0x4B);
            self.modrm_reg(d, t);
            self.b(0x00);
        } else {
            self.sse(0x66, 0x54, t, 0); // andpd t, xmm0
            self.sse(0x66, 0x55, 0, d); // andnpd xmm0, d
            self.sse(0x66, 0x56, 0, t); // orpd xmm0, t
            self.mov(d, 0);
        }
    }
    /// `d = xmm0 ? x : d`, `x` kept (through `xmm1` without AVX).
    fn keep_x(&mut self, d: u8, x: u8) {
        if self.avx {
            self.keep(d, x);
        } else {
            self.mov(1, x);
            self.keep(d, 1);
        }
    }
    /// A ModRM and SIB byte for `[base + index + disp8]`.
    fn modrm_sib(&mut self, reg: u8, base: u8, index: u8, disp: u8) {
        self.bytes(&[
            0x44 | ((reg & 7) << 3),
            ((index & 7) << 3) | (base & 7),
            disp,
        ]);
    }
    /// `r` from `[TABLE + index + disp]`: `movsd`, or `movupd` (two values)
    /// with `pair`; VEX.128 where there is AVX.
    fn load_at(&mut self, pair: bool, r: u8, index: u8, disp: u8) {
        let pp = if pair { 1 } else { 3 };
        if self.avx {
            self.vex3(false, pp, 1, 0, r, 0, index, TABLE);
        } else {
            self.b(if pair { 0x66 } else { 0xF2 });
            let rex = 0x40 | ((r >> 3) << 2) | ((index >> 3) << 1) | (TABLE >> 3);
            if rex != 0x40 {
                self.b(rex);
            }
            self.b(0x0F);
        }
        self.b(0x10);
        self.modrm_sib(r, TABLE, index, disp);
    }
    /// The upper half of `ymm r` from `[TABLE + index + disp]`
    /// (`vinsertf128 r, r, m128, 1`).
    fn insert_at(&mut self, r: u8, index: u8, disp: u8) {
        self.vex3(false, 1, 3, 1, r, r, index, TABLE);
        self.b(0x18);
        self.modrm_sib(r, TABLE, index, disp);
        self.b(1);
    }
    /// `d = unpcklpd(a, b)` (the lanes' first values), or `unpckhpd` with
    /// `high`.
    fn unpack(&mut self, high: bool, d: u8, a: u8, b: u8) {
        let op = if high { 0x15 } else { 0x14 };
        if self.avx {
            self.vex_rr(1, op, d, a, b);
        } else {
            debug_assert!(d == a || d != b);
            self.mov(d, a);
            self.sse(0x66, op, d, b);
        }
    }
    /// `movq g, x`: a register's first lane into the general register `g`.
    fn lane_to_gpr(&mut self, g: u8, x: u8) {
        if self.avx {
            self.vex3(true, 1, 1, 0, x, 0, 0, g);
        } else {
            self.b(0x66);
            self.b(0x48 | ((x >> 3) << 2) | (g >> 3));
            self.b(0x0F);
        }
        self.b(0x7E);
        self.modrm_reg(x, g);
    }
    /// `movq x, g`: the general register `g` into a register's first lane.
    fn gpr_to_lane(&mut self, x: u8, g: u8) {
        if self.avx {
            self.vex3(true, 1, 1, 0, x, 0, 0, g);
        } else {
            self.b(0x66);
            self.b(0x48 | ((x >> 3) << 2) | (g >> 3));
            self.b(0x0F);
        }
        self.b(0x6E);
        self.modrm_reg(x, g);
    }
    /// `cvtsi2sd x, g`: the signed integer `g` as a double into `x`'s first
    /// lane (exact for the exponents here).
    fn int_to_float(&mut self, x: u8, g: u8) {
        if self.avx {
            self.vex3(true, 3, 1, 0, x, x, 0, g);
        } else {
            self.b(0xF2);
            self.b(0x48 | ((x >> 3) << 2) | (g >> 3));
            self.b(0x0F);
        }
        self.b(0x2A);
        self.modrm_reg(x, g);
    }
    /// `op d, s` on 64-bit general registers (`op r/m64, r64`).
    fn gpr_rr(&mut self, op: u8, d: u8, s: u8) {
        self.rex(true, s, d);
        self.b(op);
        self.modrm_reg(s, d);
    }
    /// A shift of a 64-bit general register by `n` (`C1 /ext ib`).
    fn gpr_shift(&mut self, ext: u8, d: u8, n: u32) {
        self.rex(true, 0, d);
        self.b(0xC1);
        self.modrm_reg(ext, d);
        self.b(n as u8);
    }
    /// `and d, imm32` on a 64-bit general register.
    fn gpr_and(&mut self, d: u8, v: u32) {
        self.rex(true, 0, d);
        self.b(0x81);
        self.modrm_reg(4, d);
        self.bytes(&v.to_le_bytes());
    }
    /// `g` = the second lane of `x` (`vpextrq g, x, 1`; without AVX through
    /// `xmm0`).
    fn lane1_to_gpr(&mut self, g: u8, x: u8) {
        if self.avx {
            self.vex3(true, 1, 3, 0, x, 0, 0, g);
            self.b(0x16);
            self.modrm_reg(x, g);
            self.b(1);
        } else {
            self.sse(0x66, 0x70, 0, x); // pshufd xmm0, x, the high half twice
            self.b(0xEE);
            self.lane_to_gpr(g, 0);
        }
    }
    /// The rows of a table at `table`, one per lane at the byte offsets in
    /// `idx`'s lanes, into `cols` by column (`tmp` the scratch of the
    /// transposition; `idx` is free afterwards and may be `tmp`).
    fn rows(&mut self, table: *const u8, idx: u8, cols: &[u8], tmp: u8) {
        let g = &INDEX_GPRS[..L];
        self.lane_to_gpr(g[0], idx);
        if L > 1 {
            self.lane1_to_gpr(g[1], idx);
        }
        if L == 4 {
            // vextracti128 idx, idx, 1
            self.vex3(false, 1, 3, 1, idx, 0, 0, idx);
            self.b(0x39);
            self.modrm_reg(idx, idx);
            self.b(1);
            self.lane_to_gpr(g[2], idx);
            self.lane1_to_gpr(g[3], idx);
        }
        self.rows_from(table, cols, tmp);
    }
    /// [`rows`](Self::rows) with the offsets already in [`INDEX_GPRS`].
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
                        self.insert_at(c0, g[2], at);
                        self.insert_at(tmp, g[3], at);
                    }
                    self.unpack(true, c1, c0, tmp);
                    self.unpack(false, c0, c0, tmp);
                }
            }
        }
    }

    /// [`rsdag::semantics::exp_guarded`] step for step.
    fn exp_inline(&mut self, d: u8, x: u8, [a, b, c, e, f]: [u8; 5]) {
        use Arith::*;
        use Operand::{C, R};
        let s = 1;
        // t = x N / ln 2 + MAGIC, k = round(x N / ln 2) in its low bits
        self.madd(a, x, C(EXP_INV_LN2_N), C(MAGIC));
        self.fc(Sub, b, a, MAGIC); // kd
        self.madd(c, b, C(-EXP_LN2_HI_N), R(x));
        self.madd(c, b, C(-EXP_LN2_LO_N), R(c)); // r
                                                 // the row of k mod N, and k / N into the exponent
        self.int_c(Int::And, e, a, EXP_INDEX_MASK);
        self.shift(true, e, e, EXP_ROW_SHIFT);
        self.shift(true, b, a, EXP_SHIFT);
        self.rows(EXP_TABLE.as_ptr().cast(), e, &[f, d], e); // tail, scale bits
        self.int(Int::Add, d, d, b); // scale
        let [c2, c3, c4, c5] = EXP_POLY;
        self.f(Mul, a, c, c); // r^2
        self.madd(b, c, C(c3), C(c2));
        self.madd(e, c, C(c5), C(c4));
        self.f(Add, f, f, c);
        self.madd(f, a, R(b), R(f));
        self.f(Mul, a, a, a);
        self.madd(f, a, R(e), R(f)); // p
        self.madd(d, d, R(f), R(d));
        self.fc(Mul, d, d, EXP_UNBIAS);
        // underflow to zero, a NaN kept, the linear tail above the limit
        self.zero(s);
        self.cmp_c(x, EXP_UNDERFLOW, LT_OQ);
        self.keep(d, s);
        self.cmp_mask(x, x, UNORD_Q);
        self.keep_x(d, x);
        self.fc(Sub, s, x, EXP_LIMIT);
        self.fc(Add, s, s, 1.0);
        self.fc(Mul, s, s, EXP_AT_LIMIT);
        self.cmp_c(x, EXP_LIMIT, GT_OQ);
        self.keep(d, s);
    }

    /// [`rsdag::semantics::ln_guarded`] step for step.
    fn ln_inline(&mut self, d: u8, x: u8, [a, b, c, e, f]: [u8; 5]) {
        use Arith::*;
        use Operand::{C, R};
        let s = 1;
        // x = 2^k z, and the row's offset
        let table = LN_TABLE.as_ptr().cast();
        if L == 1 {
            // one lane: the integer steps in general registers
            let [t, k, ix] = [INDEX_GPRS[0], INDEX_GPRS[1], INDEX_GPRS[2]];
            self.lane_to_gpr(ix, x);
            self.mov_imm(k, LN_OFF);
            self.gpr_rr(GPR_MOV, t, ix);
            self.gpr_rr(GPR_SUB, t, k); // tmp
            self.gpr_rr(GPR_MOV, k, t);
            self.gpr_shift(GPR_SAR, k, F64_FRAC_BITS); // k
            self.zero(b);
            self.int_to_float(b, k); // kd
            self.gpr_shift(GPR_SHL, k, F64_FRAC_BITS);
            self.gpr_rr(GPR_SUB, ix, k);
            self.gpr_to_lane(e, ix); // z
            self.gpr_shift(GPR_SHR, t, LN_SHIFT - LN_ROW_SHIFT);
            self.gpr_and(t, LN_ROW_MASK);
            self.rows_from(table, &[a, c, f, d], s); // invc, c, logc
        } else {
            self.int_c(Int::Sub, a, x, LN_OFF); // tmp
            self.shift(false, b, a, F64_FRAC_BITS);
            self.int_c(Int::Xor, b, b, LN_K_OFFSET);
            self.int_c(Int::Add, b, b, MAGIC.to_bits());
            self.fc(Sub, b, b, LN_K_MAGIC); // kd
            self.int_c(Int::And, c, a, LN_EXP_MASK);
            self.int(Int::Sub, e, x, c); // z
            self.shift(false, c, a, LN_SHIFT - LN_ROW_SHIFT);
            self.int_c(Int::And, c, c, LN_INDEX_MASK << LN_ROW_SHIFT);
            self.rows(table, c, &[a, c, f, d], s); // invc, c, logc
        }
        self.f(Sub, e, e, c);
        self.f(Mul, e, e, a); // r
        self.madd(a, b, C(LN_LN2_HI), R(f)); // w
        self.f(Add, c, a, e); // hi
        self.f(Sub, a, a, c);
        self.f(Add, a, a, e);
        self.madd(f, b, C(LN_LN2_LO), R(d));
        self.f(Add, a, a, f); // lo
        let p = LN_POLY;
        self.f(Mul, s, e, e); // r^2
        self.madd(b, e, C(p[1]), C(p[0]));
        self.madd(d, e, C(p[3]), C(p[2]));
        self.madd(b, s, R(d), R(b));
        self.madd(d, e, C(p[5]), C(p[4]));
        self.madd(d, s, C(p[6]), R(d));
        self.f(Mul, f, s, s);
        self.madd(b, f, R(d), R(b));
        self.madd(a, s, C(-0.5), R(a));
        self.f(Mul, s, s, e);
        self.madd(a, s, R(b), R(a));
        self.f(Add, d, a, c);
        // infinity and NaN kept (not below infinity), the floor
        self.cmp_c(x, f64::INFINITY, NLT_UQ);
        self.keep_x(d, x);
        self.fconst(s, LN_AT_FLOOR);
        self.cmp_c(x, LN_FLOOR, LE_OQ);
        self.keep(d, s);
    }
}

/// A multiply-add's operand: a register or a constant from the pool.
#[derive(Clone, Copy)]
enum Operand {
    R(u8),
    C(f64),
}

// General-register opcodes: `mov` and `sub` (`op r/m64, r64`), and the
// shifts' ModRM extensions under `C1`.
const GPR_MOV: u8 = 0x89;
const GPR_SUB: u8 = 0x29;
const GPR_SHL: u8 = 4;
const GPR_SHR: u8 = 5;
const GPR_SAR: u8 = 7;

const FMA132: u8 = 0x98;
const FMA213: u8 = 0xA8;
const FMA231: u8 = 0xB8;

/// The general registers the inline exponential and logarithm use: the
/// table's address, and each lane's row offset; all volatile under both
/// conventions and holding nothing between ops.
const TABLE: u8 = 11; // r11
const INDEX_GPRS: [u8; 4] = [0, 1, 2, 10]; // rax rcx rdx r10
const F64_BYTES: usize = std::mem::size_of::<f64>();
const PAIR_BYTES: usize = 2 * F64_BYTES;
/// `exp`'s rows hold two values, `ln`'s four.
const EXP_ROW_SHIFT: u32 = PAIR_BYTES.trailing_zeros();
const LN_ROW_SHIFT: u32 = (2 * PAIR_BYTES).trailing_zeros();
/// `k`, the top 12 bits of `ln`'s `tmp` as a signed number, is `(tmp >>
/// 52) ^ LN_K_OFFSET` less `LN_K_OFFSET`, and as a float that through the
/// rounding constant: no 64-bit arithmetic shift nor integer conversion
/// below AVX-512.
const LN_K_OFFSET: u64 = 1 << (u64::BITS - 1 - F64_FRAC_BITS);
const LN_K_MAGIC: f64 = MAGIC + LN_K_OFFSET as f64;

/// The packed 64-bit integer ops, as their `66 0F` opcodes.
#[derive(Clone, Copy)]
enum Int {
    Add = 0xD4,
    Sub = 0xFB,
    And = 0xDB,
    Xor = 0xEF,
}

// VEX comparison predicates, ordered (false on a NaN) but `UNORD_Q` and
// `NLT_UQ`; the legacy `cmpsd` has their low three bits, but for `GT_OQ`.
const NLT_UQ: u8 = 0x15;
const LT_OQ: u8 = 0x11;
const LE_OQ: u8 = 0x12;
const UNORD_Q: u8 = 0x03;
const GT_OQ: u8 = 0x1E;

impl<const L: usize> Isa for X64<L> {
    const LANES: usize = L;
    const CACHE: &'static [u8] = if WIN {
        &[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5]
    } else {
        &[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    };
    const SAVED: usize = if WIN && L < 4 { 10 } else { 0 };
    const RESULT: u8 = 0;

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
    fn finish(mut self) -> Vec<u8> {
        // The constant pool after the code, each use rip-relative. An entry
        // is 16 bytes (32 for four lanes), 16-byte aligned (a chunk is placed
        // so): a legacy `andpd` or `xorpd` reads 16 aligned bytes from
        // memory. Every lane holds the constant, so packed code finds it in
        // each.
        if !self.pool.is_empty() {
            self.code.resize(self.code.len().next_multiple_of(16), 0xCC);
            let base = self.code.len();
            for c in &self.pool {
                for _ in 0..Self::CONST_BYTES / 8 {
                    self.code.extend_from_slice(&c.to_le_bytes());
                }
            }
            for &(at, k, end) in &self.fixups {
                let disp = (base + Self::CONST_BYTES * k) as i64 - end as i64;
                let disp = i32::try_from(disp).expect("a chunk is smaller than 2 GB");
                self.code[at..at + 4].copy_from_slice(&disp.to_le_bytes());
            }
        }
        self.code
    }

    fn prologue(&mut self) {
        // Six pushes and the frame: 16-byte aligned at every call below.
        self.bytes(&[0x55, 0x48, 0x89, 0xE5]); // push rbp; mov rbp, rsp
        self.bytes(&[0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57]); // push rbx r12 r13 r14 r15
        if WIN {
            self.bytes(&[0x48, 0x81, 0xEC]); // sub rsp, WIN_FRAME
            self.bytes(&WIN_FRAME.to_le_bytes());
            for x in 6..16u8 {
                self.xmm_rsp(true, x, 16 * (x as i32 - 6));
            }
            self.bytes(&[0x48, 0x89, 0xCB]); // mov rbx, rcx
            self.bytes(&[0x49, 0x89, 0xD5]); // mov r13, rdx
            self.bytes(&[0x4D, 0x89, 0xC6]); // mov r14, r8
        } else {
            self.bytes(&[0x48, 0x83, 0xEC, 0x08]); // sub rsp, 8
            self.bytes(&[0x48, 0x89, 0xFB]); // mov rbx, rdi
            self.bytes(&[0x49, 0x89, 0xF5]); // mov r13, rsi
            self.bytes(&[0x49, 0x89, 0xD6]); // mov r14, rdx
        }
        for k in 0..self.hot.len() {
            let a = self.hot[k] as usize as u64;
            self.mov_imm(HOT_REGS[k], a);
        }
    }
    fn epilogue(&mut self) {
        if L == 4 {
            self.bytes(&[0xC5, 0xF8, 0x77]); // vzeroupper
        }
        if WIN {
            for x in 6..16u8 {
                self.xmm_rsp(false, x, 16 * (x as i32 - 6));
            }
            self.bytes(&[0x48, 0x81, 0xC4]); // add rsp, WIN_FRAME
            self.bytes(&WIN_FRAME.to_le_bytes());
        } else {
            self.bytes(&[0x48, 0x83, 0xC4, 0x08]); // add rsp, 8
        }
        self.bytes(&[
            0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5B, 0x5D, 0xC3,
        ]);
    }

    fn load(&mut self, r: u8, base: Base, off: usize) {
        // movsd, or movupd (the work array is 8-byte aligned)
        let b = base.register(WORK, INPUTS);
        self.mov_mem(Self::PFX, Self::PP, Self::VL, 0x10, r, b, off);
    }
    fn store(&mut self, r: u8, base: Base, off: usize) {
        let b = base.register(WORK, INPUTS);
        self.mov_mem(Self::PFX, Self::PP, Self::VL, 0x11, r, b, off);
    }
    fn load_lane(&mut self, r: u8, base: Base, off: usize) {
        self.mov_mem(0xF2, 3, 0, 0x10, r, base.register(WORK, INPUTS), off); // movsd
    }
    fn store_lane(&mut self, r: u8, base: Base, off: usize) {
        self.mov_mem(0xF2, 3, 0, 0x11, r, base.register(WORK, INPUTS), off);
    }
    fn fconst(&mut self, r: u8, v: f64) {
        if v.to_bits() == 0 {
            self.zero(r);
        } else if L == 4 {
            // vmovupd r, [rip + constant] (the pool entry holds it four times)
            self.vex(1, 1, r, 0, 0);
            self.b(0x10);
            self.modrm_const(r, v.to_bits());
        } else {
            // movsd r, [rip + constant], or movapd for both lanes (the pool
            // entry is 16 aligned bytes holding the constant twice)
            self.b(if L == 1 { 0xF2 } else { 0x66 });
            self.rex(false, r, 0);
            self.bytes(&[0x0F, if L == 1 { 0x10 } else { 0x28 }]);
            self.modrm_const(r, v.to_bits());
        }
    }
    fn mov(&mut self, d: u8, a: u8) {
        if d != a {
            self.unary_rr(0x66, 1, 0x28, d, a); // movapd
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
                self.mov(0, a);
                self.sse(Self::PFX, opc, 0, b);
                self.mov(d, 0);
            }
        } else {
            self.mov(d, a);
            self.sse(Self::PFX, opc, d, b);
        }
    }
    // A legacy packed memory operand must be 16-byte aligned; the work
    // array is 8-byte aligned, so two lanes load into a register. Four
    // lanes are VEX, whose memory operands need no alignment.
    const MEM_OPERANDS: bool = L != 2;
    fn arith_mem(&mut self, op: Arith, d: u8, a: u8, base: Base, off: usize) {
        let base = base.register(WORK, INPUTS);
        if self.avx {
            self.vex(Self::PP, 1, d, a, base);
        } else {
            self.mov(d, a);
            self.b(Self::PFX);
            self.rex(false, d, base);
            self.b(0x0F);
        }
        self.b(Self::arith_op(op));
        self.modrm_mem(d, base, off);
    }
    const MINMAX: bool = true;
    fn minmax(&mut self, op: ReduceOp, d: u8, a: u8, b: u8) {
        // `minsd a, b` (`maxsd`) is `a` where it is the smaller (larger),
        // else `b`: on a tie and wherever either is a NaN. A NaN `b` then
        // takes `a`: the reference's rule (`f64::min`, `f64::max` as LLVM
        // lowers them), a tie the second operand's.
        let opc = match op {
            ReduceOp::Min => 0x5D,
            _ => 0x5F,
        };
        if self.avx {
            self.vex_rr(Self::PP, opc, 1, a, b);
        } else {
            self.mov(1, a);
            self.sse(Self::PFX, opc, 1, b);
        }
        self.cmp_mask(b, b, 0); // ordered: false where b is a NaN
        self.blend(1, a, d);
    }
    fn neg(&mut self, d: u8, a: u8) {
        self.op_const(0x66, 1, 0x57, d, a, 0x8000_0000_0000_0000); // xorpd
    }
    fn abs(&mut self, d: u8, a: u8) {
        self.op_const(0x66, 1, 0x54, d, a, 0x7FFF_FFFF_FFFF_FFFF); // andpd
    }
    fn sqrt(&mut self, d: u8, a: u8) {
        self.unary_rr(Self::PFX, Self::PP, 0x51, d, a);
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
        let op = if L == 1 { 0x0B } else { 0x09 }; // roundsd, or roundpd
        if L == 4 {
            self.vex(1, 3, d, 0, a);
            self.b(op);
        } else {
            self.b(0x66);
            self.rex(false, d, a);
            self.bytes(&[0x0F, 0x3A, op]);
        }
        self.modrm_reg(d, a);
        self.b(imm);
    }
    fn cmp_select(&mut self, op: CmpOp, a: u8, b: u8, t: u8, e: u8, d: u8) {
        // cmpsd predicates: 0 eq, 1 lt, 2 le (ordered, so false on NaN),
        // 4 neq (true on NaN). Gt and Ge are Lt and Le with swapped operands.
        let (x, y, pred) = match op {
            CmpOp::Gt => (b, a, 1),
            CmpOp::Ge => (b, a, 2),
            CmpOp::Lt => (a, b, 1),
            CmpOp::Le => (a, b, 2),
            CmpOp::Eq => (a, b, 0),
            CmpOp::Ne => (a, b, 4),
        };
        self.cmp_mask(x, y, pred);
        self.blend(t, e, d);
    }
    fn select_nz(&mut self, c: u8, t: u8, e: u8, d: u8) {
        self.zero(1);
        self.cmp_mask(c, 1, 4); // cmpneqsd: true when c != 0 or c is NaN
        self.blend(t, e, d);
    }
    fn exp_ln(&mut self, ln: bool, d: u8, x: u8, t: [u8; 5]) {
        if ln {
            self.ln_inline(d, x, t);
        } else {
            self.exp_inline(d, x, t);
        }
    }

    fn call(&mut self, addr: *const (), args: &[Arg]) {
        // System V counts floats and integers apart and passes the first six
        // integers in registers; Windows counts them together, passes four,
        // and wants shadow space on top. What is left over goes on the stack,
        // reserved right here and released after the call -- a kernel that
        // grows an argument then costs one more slot, not a corrupted frame.
        let stacked = if WIN {
            args.len()
        } else {
            args.iter().filter(|a| matches!(a, Arg::I(_))).count()
        }
        .saturating_sub(INT_ARGS.len());
        let shadow = if WIN { WIN_SHADOW } else { 0 };
        let frame = (shadow + 8 * stacked as i32 + 15) & !15;
        if frame > 0 {
            self.move_rsp(-frame);
        }
        let (mut nf, mut ni) = (0usize, 0usize);
        for (pos, arg) in args.iter().enumerate() {
            match *arg {
                Arg::F(r) => {
                    let k = if WIN { pos } else { nf };
                    nf += 1;
                    assert!(k < 4, "float arguments fit the registers");
                    self.mov(k as u8, r);
                }
                Arg::I(iarg) => {
                    let k = if WIN { pos } else { ni };
                    ni += 1;
                    // `rax` is the scratch the address goes through below, so
                    // a stacked argument can borrow it here.
                    match k.checked_sub(INT_ARGS.len()) {
                        Some(slot) => {
                            self.int_arg(0, iarg);
                            self.store_rsp(0, shadow + 8 * slot as i32);
                        }
                        None => self.int_arg(INT_ARGS[k], iarg),
                    }
                }
            }
        }
        if L == 4 {
            self.bytes(&[0xC5, 0xF8, 0x77]); // vzeroupper
        }
        match self.hot.iter().position(|&h| h == addr) {
            Some(k) => self.bytes(&[0x41, 0xFF, 0xD0 | (HOT_REGS[k] & 7)]), // call r12 / r15
            None => {
                self.mov_imm(0, addr as usize as u64);
                self.bytes(&[0xFF, 0xD0]); // call rax
            }
        }
        if frame > 0 {
            self.move_rsp(frame);
        }
    }
}
