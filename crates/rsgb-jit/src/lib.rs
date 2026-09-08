//! Chunked Cranelift JIT backend for SANE's evaluation [`Tape`].
//!
//! The tape is SANE's single evaluation IR; [`Tape::eval`] is the interpreting
//! backend and this crate is the native one. The previous JIT lowered a tape to
//! *one* Cranelift function and died on BSIM4-amplifier tapes (~500k ops):
//! register allocation on a single huge straight-line function is superlinear,
//! Cranelift ground for tens of seconds and aborted. The fix is structural and
//! cheap because tape slots are **memory-based SSA**: values live in a
//! caller-owned work array, so the op stream can be cut into chunks of
//! [`CHUNK_OPS`] instructions, each lowered to its own small function
//! `fn(work, inputs, ctx)`. A value crossing a chunk boundary simply stays in
//! the work array -- the interpreter's storage model *is* the spill model --
//! and the chunks compile independently (in parallel, on the rayon pool).
//!
//! Bit-exactness is a hard invariant: the same IEEE operation sequence as the
//! interpreter (no fast-math, no FMA contraction -- `MulAdd` lowers to `fmul`
//! + `fadd`), and every transcendental routes through the *same*
//! [`unary_f64`] host trampolines, so the domain guards (limexp, ln-floor,
//! sqrt-clamp) hold identically. `tests/parity.rs` fuzzes arena == tape ==
//! chunked-JIT to the bit.

use std::sync::Arc;

use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{types, AbiParam, FuncRef, InstBuilder, MemFlags, Value};
use cranelift_codegen::ir::{StackSlotData, StackSlotKind};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use rustc_hash::FxHashMap as HashMap;

use rsgb::extern_fn::ExternBundle;
use rsgb::node::{
    binary_f64, dot_slice, reduce_slice, unary_f64, BinOp, CmpOp, ReduceOp, UnaryOp,
    REDUCE_SIMD_MIN,
};
use rsgb::{Tape, TapeVisitor};

/// Instructions per compiled function for the solver's outer tapes (residual
/// and Jacobian over the whole circuit). Bounds Cranelift's superlinear
/// register allocation (the failure mode of the unchunked predecessor).
/// Measured on the pressure-scheduled 364k-op step tape of an 18-transistor
/// PSP103 ring: 1024 compiles ~5x faster than 16384 at equal or better native
/// eval speed -- the scheduler keeps the live set small, so a chunk boundary
/// costs almost nothing there; 256 loses eval speed to boundary traffic and
/// per-chunk fixed cost.
pub const CHUNK_OPS: usize = 1024;

/// Instructions per compiled function for device-bundle *bodies*: small (5k
/// to 30k ops), evaluated thousands of times per Newton step through the lane
/// backends, and compiled off the critical path. There the per-chunk fixed
/// cost (call, callee-saved vector registers, boundary stripes) is what
/// counts: on the same ring, 1024-op body chunks cost 9% of the whole
/// transient against 16384.
pub const CHUNK_OPS_BODY: usize = 16384;

/// Reasons a tape cannot be compiled (callers fall back to the interpreter).
#[derive(Debug)]
pub enum JitError {
    /// Cranelift module/codegen error.
    Codegen(String),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::Codegen(s) => write!(f, "cranelift codegen error: {s}"),
        }
    }
}
impl std::error::Error for JitError {}

// --- host trampolines: bit-identical to the interpreter's elementary ops ----
//
// Every unary op routes through `unary_f64`, the one source of truth the arena
// and tape interpreters use. Calling the bare libm op here would silently drop
// its domain guards, so compiled code would diverge on out-of-domain iterates.

extern "C" fn h_exp(x: f64) -> f64 {
    unary_f64(UnaryOp::Exp, x)
}
extern "C" fn h_ln(x: f64) -> f64 {
    unary_f64(UnaryOp::Ln, x)
}
extern "C" fn h_sqrt(x: f64) -> f64 {
    unary_f64(UnaryOp::Sqrt, x)
}
extern "C" fn h_sin(x: f64) -> f64 {
    unary_f64(UnaryOp::Sin, x)
}
extern "C" fn h_cos(x: f64) -> f64 {
    unary_f64(UnaryOp::Cos, x)
}
extern "C" fn h_sinh(x: f64) -> f64 {
    unary_f64(UnaryOp::Sinh, x)
}
extern "C" fn h_cosh(x: f64) -> f64 {
    unary_f64(UnaryOp::Cosh, x)
}
extern "C" fn h_tanh(x: f64) -> f64 {
    unary_f64(UnaryOp::Tanh, x)
}
extern "C" fn h_atan(x: f64) -> f64 {
    unary_f64(UnaryOp::Atan, x)
}
extern "C" fn h_floor(x: f64) -> f64 {
    unary_f64(UnaryOp::Floor, x)
}
/// The floating-point extension of the unary set: one trampoline, the op as
/// a constant code (see `unary_code`), the same `unary_f64` as everywhere.
extern "C" fn h_unary_ext(op: u32, x: f64) -> f64 {
    unary_f64(unary_from_code(op), x)
}
extern "C" fn h_binary(op: u32, x: f64, y: f64) -> f64 {
    binary_f64(binary_from_code(op), x, y)
}
extern "C" fn h_powi(x: f64, n: i64) -> f64 {
    x.powi(n as i32)
}

extern "C" fn h_reduce(op: u32, ptr: *const f64, len: usize) -> f64 {
    let xs = unsafe { std::slice::from_raw_parts(ptr, len) };
    let op = match op {
        0 => ReduceOp::Sum,
        1 => ReduceOp::Product,
        2 => ReduceOp::Min,
        _ => ReduceOp::Max,
    };
    reduce_slice(op, xs)
}

extern "C" fn h_dot(a: *const f64, b: *const f64, len: usize) -> f64 {
    let va = unsafe { std::slice::from_raw_parts(a, len) };
    let vb = unsafe { std::slice::from_raw_parts(b, len) };
    dot_slice(va, vb)
}

/// Per-eval host context handed to every chunk. `#[repr(C)]` with `bscratch`
/// first: `BundlePick` loads the pointer straight off offset 0.
#[repr(C)]
struct HostCtx {
    bscratch: *mut f64,
    bundles: *const Vec<Arc<dyn ExternBundle>>,
}

extern "C" fn h_bundle(ctx: *const HostCtx, idx: usize, args: *const f64, len: usize, base: usize) {
    let (bundles, xs, ctxr) = unsafe {
        (
            &*(*ctx).bundles,
            std::slice::from_raw_parts(args, len),
            &*ctx,
        )
    };
    let b = &bundles[idx];
    let out = unsafe { std::slice::from_raw_parts_mut(ctxr.bscratch.add(base), b.n_outputs()) };
    b.call(xs, out);
}

extern "C" fn h_bundle_batch(
    ctx: *const HostCtx,
    idx: usize,
    args: *const f64,
    n_groups: usize,
    n_args: usize,
    base: usize,
) {
    let (bundles, xs, ctxr) = unsafe {
        (
            &*(*ctx).bundles,
            std::slice::from_raw_parts(args, n_groups * n_args),
            &*ctx,
        )
    };
    let b = &bundles[idx];
    let out = unsafe {
        std::slice::from_raw_parts_mut(ctxr.bscratch.add(base), n_groups * b.n_outputs())
    };
    b.call_batch(xs, n_groups, n_args, out);
}

/// The dedicated trampoline of a unary op, `None` for the extension ops
/// (which go through `h_unary_ext` with their code).
fn unary_sym(op: UnaryOp) -> Option<&'static str> {
    Some(match op {
        UnaryOp::Exp => "h_exp",
        UnaryOp::Ln => "h_ln",
        UnaryOp::Sqrt => "h_sqrt",
        UnaryOp::Sin => "h_sin",
        UnaryOp::Cos => "h_cos",
        UnaryOp::Sinh => "h_sinh",
        UnaryOp::Cosh => "h_cosh",
        UnaryOp::Tanh => "h_tanh",
        UnaryOp::Atan => "h_atan",
        UnaryOp::Floor => "h_floor",
        _ => return None,
    })
}

/// The op codes the trampolines take are the vocabulary's own codes
/// ([`UnaryOp::code`]), so the JIT carries no second numbering to keep in
/// step with the enum.
fn unary_code(op: UnaryOp) -> u32 {
    op.code()
}
fn unary_from_code(code: u32) -> UnaryOp {
    UnaryOp::from_code(code)
}
fn binary_code(op: BinOp) -> u32 {
    op.code()
}
fn binary_from_code(code: u32) -> BinOp {
    BinOp::from_code(code)
}

fn reduce_code(op: ReduceOp) -> i64 {
    match op {
        ReduceOp::Sum => 0,
        ReduceOp::Product => 1,
        ReduceOp::Min => 2,
        ReduceOp::Max => 3,
    }
}

const HOST_NAMES: [&str; 17] = [
    "h_unary_ext",
    "h_binary",
    "h_exp",
    "h_ln",
    "h_sqrt",
    "h_sin",
    "h_cos",
    "h_sinh",
    "h_cosh",
    "h_tanh",
    "h_atan",
    "h_floor",
    "h_powi",
    "h_reduce",
    "h_dot",
    "h_bundle",
    "h_bundle_batch",
];

fn host_addr(name: &str) -> *const u8 {
    match name {
        "h_exp" => h_exp as *const u8,
        "h_ln" => h_ln as *const u8,
        "h_sqrt" => h_sqrt as *const u8,
        "h_sin" => h_sin as *const u8,
        "h_cos" => h_cos as *const u8,
        "h_sinh" => h_sinh as *const u8,
        "h_cosh" => h_cosh as *const u8,
        "h_tanh" => h_tanh as *const u8,
        "h_atan" => h_atan as *const u8,
        "h_floor" => h_floor as *const u8,
        "h_powi" => h_powi as *const u8,
        "h_unary_ext" => h_unary_ext as *const u8,
        "h_binary" => h_binary as *const u8,
        "h_reduce" => h_reduce as *const u8,
        "h_dot" => h_dot as *const u8,
        "h_bundle" => h_bundle as *const u8,
        "h_bundle_batch" => h_bundle_batch as *const u8,
        _ => unreachable!(),
    }
}

// --- inline reductions: `reduce_slice` / `dot_slice` as IR -------------------
//
// The 4-accumulator fold of the interpreter emitted in the identical
// association order, so the value is bit-exact -- without the stack spill and
// host call a long KCL sum or matrix row otherwise pays.

/// The 4-lane sum / product fold of `reduce_slice` over scalar values
/// (`xs.len() >= REDUCE_SIMD_MIN`).
fn emit_reduce_4lane(b: &mut FunctionBuilder, op: ReduceOp, xs: &[Value]) -> Value {
    let (init, comb): (f64, fn(&mut FunctionBuilder, Value, Value) -> Value) = match op {
        ReduceOp::Sum => (0.0, |b, x, y| b.ins().fadd(x, y)),
        ReduceOp::Product => (1.0, |b, x, y| b.ins().fmul(x, y)),
        _ => unreachable!("min/max keep the host call"),
    };
    let mut a: [Value; 4] = std::array::from_fn(|_| b.ins().f64const(init));
    let ch = xs.len() / 4;
    for c in 0..ch {
        for i in 0..4 {
            a[i] = comb(b, a[i], xs[4 * c + i]);
        }
    }
    let l = comb(b, a[0], a[1]);
    let r = comb(b, a[2], a[3]);
    let mut acc = comb(b, l, r);
    for &x in &xs[ch * 4..] {
        acc = comb(b, acc, x);
    }
    acc
}

/// `dot_slice` as IR (bit-exact association order for both its branches).
fn emit_dot(b: &mut FunctionBuilder, xs: &[Value], ys: &[Value]) -> Value {
    let n = xs.len();
    if n >= REDUCE_SIMD_MIN {
        let mut acc: [Value; 4] = std::array::from_fn(|_| b.ins().f64const(0.0));
        let ch = n / 4;
        for c in 0..ch {
            for i in 0..4 {
                let p = b.ins().fmul(xs[4 * c + i], ys[4 * c + i]);
                acc[i] = b.ins().fadd(acc[i], p);
            }
        }
        let l = b.ins().fadd(acc[0], acc[1]);
        let r = b.ins().fadd(acc[2], acc[3]);
        let mut s = b.ins().fadd(l, r);
        for k in ch * 4..n {
            let p = b.ins().fmul(xs[k], ys[k]);
            s = b.ins().fadd(s, p);
        }
        s
    } else {
        let mut s = b.ins().f64const(0.0);
        for k in 0..n {
            let p = b.ins().fmul(xs[k], ys[k]);
            s = b.ins().fadd(s, p);
        }
        s
    }
}

// --- pass 1: record the tape's op stream ------------------------------------
//
// `Tape::lower` drives a visitor; recording into a plain vector decouples the
// chunk partitioning and the (parallel) per-chunk codegen from the visitor
/// Exponent bound for unrolling `powi` into inline multiplies (4 fmuls at 16).
const POWI_UNROLL_MAX: u32 = 16;

// callback structure. Bundle bodies are interned by `Arc` identity
// into tables the compiled code indexes through the host context.

enum ROp {
    Const(u32, f64),
    Input(u32, u32),
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
    Bundle(u32, Vec<u32>, u32),
    /// (bundle idx, group-major arg slots, n_groups, n_args, base0)
    BundleBatch(u32, Vec<u32>, u32, u32, u32),
    Pick(u32, u32),
}

impl ROp {
    /// Destination work slot, if the op writes one.
    fn dst(&self) -> Option<u32> {
        match self {
            ROp::Const(d, _)
            | ROp::Input(d, _)
            | ROp::Add(d, _, _)
            | ROp::Mul(d, _, _)
            | ROp::MulAdd(d, _, _, _)
            | ROp::Sub(d, _, _)
            | ROp::Neg(d, _)
            | ROp::Powi(d, _, _)
            | ROp::Unary(d, _, _)
            | ROp::Binary(d, _, _, _)
            | ROp::Cmp(d, _, _, _)
            | ROp::Select(d, _, _, _)
            | ROp::Reduce(d, _, _)
            | ROp::Dot(d, _, _)
            | ROp::Pick(d, _) => Some(*d),
            ROp::Bundle(..) | ROp::BundleBatch(..) => None,
        }
    }
    /// Visit every work slot the op reads.
    fn for_each_read(&self, mut f: impl FnMut(u32)) {
        match self {
            ROp::Const(..) | ROp::Input(..) | ROp::Pick(..) => {}
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
            ROp::Dot(_, a, b) => a.iter().chain(b.iter()).copied().for_each(f),
            ROp::Bundle(_, args, _) | ROp::BundleBatch(_, args, _, _, _) => {
                args.iter().copied().for_each(f)
            }
        }
    }
}

/// Which work slots a chunked compile must materialize in the work array:
/// the tape outputs, plus every slot some chunk reads without having written
/// it first (a cross-chunk value -- including the re-entry reads of a repeated
/// `eval_main` over a shared prolog buffer). Everything else lives purely in
/// registers inside its chunk; dropping the store-through on those is the
/// difference between memory-bound and register-resident model evaluation.
fn store_mask(jobs: &[&[ROp]], outputs: &[u32], n_work: usize) -> Vec<bool> {
    let mut mask = vec![false; n_work];
    for &o in outputs {
        mask[o as usize] = true;
    }
    let mut written = vec![false; n_work];
    for ops in jobs {
        written.iter_mut().for_each(|w| *w = false);
        for op in *ops {
            op.for_each_read(|r| {
                if !written[r as usize] {
                    mask[r as usize] = true;
                }
            });
            if let Some(d) = op.dst() {
                written[d as usize] = true;
            }
        }
    }
    mask
}

#[derive(Default)]
struct Recorder {
    ops: Vec<ROp>,
    bundles: Vec<Arc<dyn ExternBundle>>,
    bundle_idx: HashMap<usize, u32>,
}

impl Recorder {
    fn intern_bundle(&mut self, b: &Arc<dyn ExternBundle>) -> u32 {
        let key = Arc::as_ptr(b) as *const () as usize;
        *self.bundle_idx.entry(key).or_insert_with(|| {
            self.bundles.push(b.clone());
            (self.bundles.len() - 1) as u32
        })
    }
}

impl TapeVisitor for Recorder {
    fn constant(&mut self, dst: u32, v: f64) {
        self.ops.push(ROp::Const(dst, v));
    }
    fn input(&mut self, dst: u32, k: u32) {
        self.ops.push(ROp::Input(dst, k));
    }
    fn add(&mut self, dst: u32, a: u32, b: u32) {
        self.ops.push(ROp::Add(dst, a, b));
    }
    fn mul(&mut self, dst: u32, a: u32, b: u32) {
        self.ops.push(ROp::Mul(dst, a, b));
    }
    fn mul_add(&mut self, dst: u32, a: u32, b: u32, c: u32) {
        self.ops.push(ROp::MulAdd(dst, a, b, c));
    }
    fn sub(&mut self, dst: u32, a: u32, b: u32) {
        self.ops.push(ROp::Sub(dst, a, b));
    }
    fn neg(&mut self, dst: u32, a: u32) {
        self.ops.push(ROp::Neg(dst, a));
    }
    fn powi(&mut self, dst: u32, a: u32, n: i32) {
        self.ops.push(ROp::Powi(dst, a, n));
    }
    fn unary(&mut self, dst: u32, op: UnaryOp, a: u32) {
        self.ops.push(ROp::Unary(dst, op, a));
    }
    fn binary(&mut self, dst: u32, op: BinOp, a: u32, b: u32) {
        self.ops.push(ROp::Binary(dst, op, a, b));
    }
    fn cmp(&mut self, dst: u32, op: CmpOp, a: u32, b: u32) {
        self.ops.push(ROp::Cmp(dst, op, a, b));
    }
    fn select(&mut self, dst: u32, c: u32, t: u32, e: u32) {
        self.ops.push(ROp::Select(dst, c, t, e));
    }
    fn reduce(&mut self, dst: u32, op: ReduceOp, args: &[u32]) {
        self.ops.push(ROp::Reduce(dst, op, args.to_vec()));
    }
    fn dot(&mut self, dst: u32, a: &[u32], b: &[u32]) {
        self.ops.push(ROp::Dot(dst, a.to_vec(), b.to_vec()));
    }
    fn bundle_call(&mut self, b: &Arc<dyn ExternBundle>, args: &[u32], scratch_base: u32) {
        let idx = self.intern_bundle(b);
        self.ops.push(ROp::Bundle(idx, args.to_vec(), scratch_base));
    }
    fn bundle_batch(
        &mut self,
        b: &Arc<dyn ExternBundle>,
        args: &[u32],
        n_groups: u32,
        n_args: u32,
        base0: u32,
    ) {
        let idx = self.intern_bundle(b);
        self.ops.push(ROp::BundleBatch(
            idx,
            args.to_vec(),
            n_groups,
            n_args,
            base0,
        ));
    }
    fn bundle_pick(&mut self, dst: u32, idx: u32) {
        self.ops.push(ROp::Pick(dst, idx));
    }
}

// --- pass 2: per-chunk codegen ----------------------------------------------

type ChunkFn = extern "C" fn(*mut f64, *const f64, *const HostCtx);

/// One compiled chunk. The module is kept alive so the code stays mapped; it is
/// immutable after `finalize_definitions`, so sending it across threads (rayon
/// compiles chunks in parallel) and calling the code from any thread is sound.
struct NativeChunk {
    _module: JITModule,
    func: ChunkFn,
}
unsafe impl Send for NativeChunk {}
unsafe impl Sync for NativeChunk {}

/// Chunk-local lowering state. `cur` caches the SSA value of each work slot
/// written or read within this chunk; a write stores to the work array only
/// when the liveness mask says a later chunk (or the host, for outputs) reads
/// it -- everything else stays register-resident. Reads of slots produced by
/// earlier chunks load from the work array once.
struct ChunkJit<'a> {
    b: FunctionBuilder<'a>,
    ptr_ty: types::Type,
    work_ptr: Value,
    in_ptr: Value,
    ctx_ptr: Value,
    href: &'a HashMap<&'static str, FuncRef>,
    cur: Vec<Option<Value>>,
    /// Slots that must be materialized in the work array (see [`store_mask`]).
    mask: &'a [bool],
}

impl ChunkJit<'_> {
    fn get(&mut self, s: u32) -> Value {
        if let Some(v) = self.cur[s as usize] {
            return v;
        }
        let v = self.b.ins().load(
            types::F64,
            MemFlags::trusted(),
            self.work_ptr,
            (s * 8) as i32,
        );
        self.cur[s as usize] = Some(v);
        v
    }
    fn set(&mut self, dst: u32, v: Value) {
        // The work array is the cross-chunk (and output) truth; chunk-local
        // values skip it entirely.
        if self.mask[dst as usize] {
            self.b
                .ins()
                .store(MemFlags::trusted(), v, self.work_ptr, (dst * 8) as i32);
        }
        self.cur[dst as usize] = Some(v);
    }
    /// f64 immediate via integer bits (a const pool would exceed PC-relative
    /// displacement on op-dense functions; immediates have no such limit).
    fn fconst(&mut self, v: f64) -> Value {
        let bits = self.b.ins().iconst(types::I64, v.to_bits() as i64);
        self.b.ins().bitcast(types::F64, MemFlags::new(), bits)
    }
    fn call(&mut self, name: &'static str, args: &[Value]) -> Value {
        let f = self.href[name];
        let c = self.b.ins().call(f, args);
        self.b.inst_results(c)[0]
    }
    /// Spill `slots`' values into a fresh stack array; returns its base pointer.
    fn spill(&mut self, slots: &[u32]) -> Value {
        let slot = self.b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (slots.len() * 8) as u32,
            3,
        ));
        for (k, &s) in slots.iter().enumerate() {
            let v = self.get(s);
            self.b.ins().stack_store(v, slot, (k * 8) as i32);
        }
        self.b.ins().stack_addr(self.ptr_ty, slot, 0)
    }

    fn lower(&mut self, op: &ROp) {
        match op {
            ROp::Const(dst, v) => {
                let val = self.fconst(*v);
                self.set(*dst, val);
            }
            ROp::Input(dst, k) => {
                // An unmapped symbol evaluates to NaN, exactly like the
                // interpreter; short caller input arrays are padded by `eval`.
                let v = if *k == u32::MAX {
                    self.fconst(f64::NAN)
                } else {
                    self.b.ins().load(
                        types::F64,
                        MemFlags::trusted(),
                        self.in_ptr,
                        (*k * 8) as i32,
                    )
                };
                self.set(*dst, v);
            }
            ROp::Add(dst, a, b) => {
                let (x, y) = (self.get(*a), self.get(*b));
                let v = self.b.ins().fadd(x, y);
                self.set(*dst, v);
            }
            ROp::Mul(dst, a, b) => {
                let (x, y) = (self.get(*a), self.get(*b));
                let v = self.b.ins().fmul(x, y);
                self.set(*dst, v);
            }
            // fmul + fadd on purpose, never fma: the superinstruction fuses the
            // dispatch, the rounding stays the interpreter's two IEEE ops.
            ROp::MulAdd(dst, a, b, c) => {
                let (x, y) = (self.get(*a), self.get(*b));
                let m = self.b.ins().fmul(x, y);
                let z = self.get(*c);
                let v = self.b.ins().fadd(m, z);
                self.set(*dst, v);
            }
            ROp::Sub(dst, a, b) => {
                let (x, y) = (self.get(*a), self.get(*b));
                let v = self.b.ins().fsub(x, y);
                self.set(*dst, v);
            }
            ROp::Neg(dst, a) => {
                let x = self.get(*a);
                let v = self.b.ins().fneg(x);
                self.set(*dst, v);
            }
            ROp::Powi(dst, a, n) => {
                let av = self.get(*a);
                // Small constant exponents unroll to the exact square-and-
                // multiply sequence of the host `powi` (compiler-builtins
                // `__powidf2`), so results stay bit-identical while the hot
                // path loses the libcall. Large exponents keep the call.
                let v = if n.unsigned_abs() <= POWI_UNROLL_MAX {
                    let mut pow = n.unsigned_abs();
                    let mut base = av;
                    let mut mul = self.fconst(1.0);
                    loop {
                        if pow & 1 != 0 {
                            mul = self.b.ins().fmul(mul, base);
                        }
                        pow >>= 1;
                        if pow == 0 {
                            break;
                        }
                        base = self.b.ins().fmul(base, base);
                    }
                    if *n < 0 {
                        let one = self.fconst(1.0);
                        self.b.ins().fdiv(one, mul)
                    } else {
                        mul
                    }
                } else {
                    let nv = self.b.ins().iconst(types::I64, *n as i64);
                    self.call("h_powi", &[av, nv])
                };
                self.set(*dst, v);
            }
            ROp::Unary(dst, op, a) => {
                let av = self.get(*a);
                // Floor is an IEEE-exact hardware op. SANE's Sqrt carries the
                // Newton guard `x > 0 ? sqrt(x) : 0` (see `unary_f64`); the
                // hardware sqrt is IEEE-correct on the taken side and the
                // negative side selects the exact 0.0, so the pair stays
                // bit-identical to the host call -- minus the call.
                let v = match op {
                    UnaryOp::Sqrt => {
                        let zero = self.fconst(0.0);
                        let pos = self.b.ins().fcmp(FloatCC::GreaterThan, av, zero);
                        let sq = self.b.ins().sqrt(av);
                        self.b.ins().select(pos, sq, zero)
                    }
                    UnaryOp::Floor => self.b.ins().floor(av),
                    _ => match unary_sym(*op) {
                        Some(sym) => self.call(sym, &[av]),
                        None => {
                            let code = self.b.ins().iconst(types::I32, unary_code(*op) as i64);
                            self.call("h_unary_ext", &[code, av])
                        }
                    },
                };
                self.set(*dst, v);
            }
            ROp::Binary(dst, op, a, b) => {
                let (x, y) = (self.get(*a), self.get(*b));
                let code = self.b.ins().iconst(types::I32, binary_code(*op) as i64);
                let v = self.call("h_binary", &[code, x, y]);
                self.set(*dst, v);
            }
            ROp::Cmp(dst, op, a, b) => {
                let (x, y) = (self.get(*a), self.get(*b));
                let cc = match op {
                    CmpOp::Gt => FloatCC::GreaterThan,
                    CmpOp::Ge => FloatCC::GreaterThanOrEqual,
                    CmpOp::Lt => FloatCC::LessThan,
                    CmpOp::Le => FloatCC::LessThanOrEqual,
                    CmpOp::Eq => FloatCC::Equal,
                    CmpOp::Ne => FloatCC::NotEqual,
                };
                let c = self.b.ins().fcmp(cc, x, y);
                let one = self.fconst(1.0);
                let zero = self.fconst(0.0);
                let v = self.b.ins().select(c, one, zero);
                self.set(*dst, v);
            }
            ROp::Select(dst, c, t, e) => {
                let (cv, tv, ev) = (self.get(*c), self.get(*t), self.get(*e));
                let zero = self.fconst(0.0);
                // FloatCC::NotEqual is unordered-or-unequal: NaN != 0 is true,
                // matching the interpreter's `c != 0.0`.
                let cond = self.b.ins().fcmp(FloatCC::NotEqual, cv, zero);
                let v = self.b.ins().select(cond, tv, ev);
                self.set(*dst, v);
            }
            ROp::Reduce(dst, op, args) => {
                // Sum/Product inline as the interpreter's own folds
                // (`reduce_slice`: sequential below the 4-lane threshold, the
                // 4-accumulator order above it), so bits match without the
                // spill + libcall. Min/Max keep the call (Cranelift fmin/fmax
                // NaN semantics differ from `f64::min`).
                let inline_fold = matches!(op, ReduceOp::Sum | ReduceOp::Product)
                    && args.len() < REDUCE_SIMD_MIN
                    && !args.is_empty();
                let inline_4lane = matches!(op, ReduceOp::Sum | ReduceOp::Product)
                    && args.len() >= REDUCE_SIMD_MIN;
                let v = if inline_4lane {
                    let xs: Vec<Value> = args.iter().map(|&a| self.get(a)).collect();
                    emit_reduce_4lane(&mut self.b, *op, &xs)
                } else if inline_fold {
                    // Fold from the identity, exactly like the interpreter
                    // (`0.0 + -0.0` is `+0.0`, so starting at x0 would differ).
                    let mut acc = match op {
                        ReduceOp::Sum => self.fconst(0.0),
                        _ => self.fconst(1.0),
                    };
                    for &a in args {
                        let x = self.get(a);
                        acc = match op {
                            ReduceOp::Sum => self.b.ins().fadd(acc, x),
                            _ => self.b.ins().fmul(acc, x),
                        };
                    }
                    acc
                } else {
                    let ptr = self.spill(args);
                    let opc = self.b.ins().iconst(types::I32, reduce_code(*op));
                    let lenv = self.b.ins().iconst(self.ptr_ty, args.len() as i64);
                    self.call("h_reduce", &[opc, ptr, lenv])
                };
                self.set(*dst, v);
            }
            ROp::Dot(dst, a, bb) => {
                let xs: Vec<Value> = a.iter().map(|&s| self.get(s)).collect();
                let ys: Vec<Value> = bb.iter().map(|&s| self.get(s)).collect();
                let v = emit_dot(&mut self.b, &xs, &ys);
                self.set(*dst, v);
            }
            ROp::Bundle(idx, args, base) => {
                let ptr = self.spill(args);
                let iv = self.b.ins().iconst(self.ptr_ty, *idx as i64);
                let lenv = self.b.ins().iconst(self.ptr_ty, args.len() as i64);
                let basev = self.b.ins().iconst(self.ptr_ty, *base as i64);
                let f = self.href["h_bundle"];
                self.b.ins().call(f, &[self.ctx_ptr, iv, ptr, lenv, basev]);
            }
            ROp::BundleBatch(idx, args, n_groups, n_args, base) => {
                let ptr = self.spill(args);
                let iv = self.b.ins().iconst(self.ptr_ty, *idx as i64);
                let ngv = self.b.ins().iconst(self.ptr_ty, *n_groups as i64);
                let nav = self.b.ins().iconst(self.ptr_ty, *n_args as i64);
                let basev = self.b.ins().iconst(self.ptr_ty, *base as i64);
                let f = self.href["h_bundle_batch"];
                self.b
                    .ins()
                    .call(f, &[self.ctx_ptr, iv, ptr, ngv, nav, basev]);
            }
            ROp::Pick(dst, idx) => {
                // ctx->bscratch (offset 0), then bscratch[idx].
                let bs = self
                    .b
                    .ins()
                    .load(self.ptr_ty, MemFlags::trusted(), self.ctx_ptr, 0);
                let v = self
                    .b
                    .ins()
                    .load(types::F64, MemFlags::trusted(), bs, (*idx * 8) as i32);
                self.set(*dst, v);
            }
        }
    }
}

fn compile_chunk(ops: &[ROp], n_work: usize, mask: &[bool]) -> Result<NativeChunk, JitError> {
    let mut flags = settings::builder();
    flags.set("use_colocated_libcalls", "false").unwrap();
    flags.set("is_pic", "false").unwrap();
    flags.set("opt_level", "speed").unwrap();
    let isa = cranelift_native::builder()
        .map_err(|e| JitError::Codegen(e.to_string()))?
        .finish(settings::Flags::new(flags))
        .map_err(|e| JitError::Codegen(e.to_string()))?;

    let mut jb = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    for name in HOST_NAMES {
        jb.symbol(name, host_addr(name));
    }
    let mut module = JITModule::new(jb);
    let ptr_ty = module.target_config().pointer_type();
    let call_conv = module.target_config().default_call_conv;

    let mut host_ids: HashMap<&'static str, cranelift_module::FuncId> = HashMap::default();
    {
        let mut decl = |name: &'static str, params: &[types::Type], ret: Option<types::Type>| {
            let mut sig = module.make_signature();
            sig.call_conv = call_conv;
            for &p in params {
                sig.params.push(AbiParam::new(p));
            }
            if let Some(r) = ret {
                sig.returns.push(AbiParam::new(r));
            }
            let id = module
                .declare_function(name, Linkage::Import, &sig)
                .expect("declare host import");
            host_ids.insert(name, id);
        };
        for op in [
            UnaryOp::Exp,
            UnaryOp::Ln,
            UnaryOp::Sqrt,
            UnaryOp::Sin,
            UnaryOp::Cos,
            UnaryOp::Sinh,
            UnaryOp::Cosh,
            UnaryOp::Tanh,
            UnaryOp::Atan,
            UnaryOp::Floor,
        ] {
            decl(
                unary_sym(op).expect("a dedicated trampoline"),
                &[types::F64],
                Some(types::F64),
            );
        }
        decl("h_unary_ext", &[types::I32, types::F64], Some(types::F64));
        decl(
            "h_binary",
            &[types::I32, types::F64, types::F64],
            Some(types::F64),
        );
        decl("h_powi", &[types::F64, types::I64], Some(types::F64));
        decl("h_reduce", &[types::I32, ptr_ty, ptr_ty], Some(types::F64));
        decl("h_dot", &[ptr_ty, ptr_ty, ptr_ty], Some(types::F64));
        decl("h_bundle", &[ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty], None);
        decl(
            "h_bundle_batch",
            &[ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty],
            None,
        );
    }

    let mut cctx = module.make_context();
    cctx.func.signature.call_conv = call_conv;
    cctx.func.signature.params.push(AbiParam::new(ptr_ty)); // work
    cctx.func.signature.params.push(AbiParam::new(ptr_ty)); // inputs
    cctx.func.signature.params.push(AbiParam::new(ptr_ty)); // host ctx

    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut cctx.func, &mut fbctx);
        let blk = b.create_block();
        b.append_block_params_for_function_params(blk);
        b.switch_to_block(blk);
        b.seal_block(blk);
        let work_ptr = b.block_params(blk)[0];
        let in_ptr = b.block_params(blk)[1];
        let ctx_ptr = b.block_params(blk)[2];

        let mut href: HashMap<&'static str, FuncRef> = HashMap::default();
        for (&name, &id) in &host_ids {
            href.insert(name, module.declare_func_in_func(id, b.func));
        }

        let mut jit = ChunkJit {
            b,
            ptr_ty,
            work_ptr,
            in_ptr,
            ctx_ptr,
            href: &href,
            cur: vec![None; n_work],
            mask,
        };
        for op in ops {
            jit.lower(op);
        }
        jit.b.ins().return_(&[]);
        jit.b.finalize();
    }

    let func_id = module
        .declare_function("chunk", Linkage::Export, &cctx.func.signature)
        .map_err(|e| JitError::Codegen(e.to_string()))?;
    module
        .define_function(func_id, &mut cctx)
        .map_err(|e| JitError::Codegen(e.to_string()))?;
    module.clear_context(&mut cctx);
    module
        .finalize_definitions()
        .map_err(|e| JitError::Codegen(e.to_string()))?;

    let code = module.get_finalized_function(func_id);
    let func = unsafe { std::mem::transmute::<*const u8, ChunkFn>(code) };
    Ok(NativeChunk {
        _module: module,
        func,
    })
}

/// A tape compiled to native code as a sequence of chunk functions sharing the
/// caller's work array. Same eval contract as [`Tape::eval`], bit-exact.
pub struct ChunkedTape {
    chunks: Vec<NativeChunk>,
    /// Chunks `..prolog_chunks` are the source tape's parameter-pure prolog
    /// (chunk boundaries never straddle the split); 0 for unsplit tapes.
    prolog_chunks: usize,
    bundles: Vec<Arc<dyn ExternBundle>>,
    outputs: Vec<u32>,
    n_work: usize,
    bundle_scratch_len: usize,
    /// 1 + highest mapped input index; `eval` pads shorter caller arrays with
    /// NaN so out-of-range reads match the interpreter's `get(k) -> NaN`.
    n_inputs: usize,
}

impl ChunkedTape {
    /// Compile with the default [`CHUNK_OPS`] chunk size.
    /// Compile with extra work slots forced live (stored to the work array)
    /// beyond the outputs/cross-chunk set -- a specialized tape's prolog guard
    /// slots, which the solver reads straight from the buffer after
    /// `eval_prolog`.
    pub fn compile_live(tape: &Tape, extra_live: &[u32]) -> Result<ChunkedTape, JitError> {
        Self::compile_full(tape, CHUNK_OPS, extra_live)
    }

    pub fn compile(tape: &Tape) -> Result<ChunkedTape, JitError> {
        Self::compile_with(tape, CHUNK_OPS)
    }

    /// Compile with an explicit chunk size (tests use tiny chunks to force many
    /// boundary crossings). Chunks compile in parallel on the rayon pool.
    pub fn compile_with(tape: &Tape, chunk_ops: usize) -> Result<ChunkedTape, JitError> {
        Self::compile_full(tape, chunk_ops, &[])
    }

    fn compile_full(
        tape: &Tape,
        chunk_ops: usize,
        extra_live: &[u32],
    ) -> Result<ChunkedTape, JitError> {
        let mut rec = Recorder::default();
        tape.lower(&mut rec);
        let n_inputs = rec
            .ops
            .iter()
            .filter_map(|op| match op {
                ROp::Input(_, k) if *k != u32::MAX => Some(*k as usize + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let n_work = tape.n_work();
        let chunk_ops = chunk_ops.max(1);
        // Chunk the prolog and main phases separately so no chunk straddles
        // the split boundary; the recorded stream is 1:1 with the tape's ops.
        let split = tape.prolog_len().min(rec.ops.len());
        use rayon::prelude::*;
        let (pro, main) = rec.ops.split_at(split);
        let jobs: Vec<&[ROp]> = pro
            .chunks(chunk_ops)
            .chain(main.chunks(chunk_ops))
            .collect();
        let prolog_chunks = pro.chunks(chunk_ops).count();
        let mut mask = store_mask(&jobs, tape.outputs(), n_work);
        for &s in extra_live {
            mask[s as usize] = true;
        }
        let chunks: Result<Vec<NativeChunk>, JitError> = jobs
            .par_iter()
            .map(|ops| compile_chunk(ops, n_work, &mask))
            .collect();
        Ok(ChunkedTape {
            chunks: chunks?,
            prolog_chunks,
            bundles: rec.bundles,
            outputs: tape.outputs().to_vec(),
            n_work,
            bundle_scratch_len: tape.bundle_scratch_len(),
            n_inputs,
        })
    }

    /// Number of chunk functions (diagnostics).
    pub fn n_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Evaluate the parameter-pure prolog chunks into `work` (sized/cleared
    /// here); mirrors [`Tape::eval_prolog`]. The buffer layout is the native
    /// one (`n_work + bundle_scratch`), so pair only with
    /// [`eval_main`](Self::eval_main).
    pub fn eval_prolog(&self, inputs: &[f64], work: &mut Vec<f64>) {
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
        work.resize(self.n_work + self.bundle_scratch_len, 0.0);
        let (w, bs) = work.split_at_mut(self.n_work);
        let ctx = HostCtx {
            bscratch: bs.as_mut_ptr(),
            bundles: &self.bundles,
        };
        let (wp, ip, cp) = (w.as_mut_ptr(), ins.as_ptr(), &ctx as *const HostCtx);
        for c in &self.chunks[..self.prolog_chunks] {
            (c.func)(wp, ip, cp);
        }
    }

    /// Evaluate the main-phase chunks over a buffer prepared by
    /// [`eval_prolog`](Self::eval_prolog); mirrors [`Tape::eval_main`].
    pub fn eval_main(&self, inputs: &[f64], work: &mut [f64], out: &mut Vec<f64>) {
        assert_eq!(
            work.len(),
            self.n_work + self.bundle_scratch_len,
            "eval_main requires a work buffer prepared by eval_prolog"
        );
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
        let (w, bs) = work.split_at_mut(self.n_work);
        let ctx = HostCtx {
            bscratch: bs.as_mut_ptr(),
            bundles: &self.bundles,
        };
        let (wp, ip, cp) = (w.as_mut_ptr(), ins.as_ptr(), &ctx as *const HostCtx);
        for c in &self.chunks[self.prolog_chunks..] {
            (c.func)(wp, ip, cp);
        }
        out.clear();
        out.extend(self.outputs.iter().map(|&o| w[o as usize]));
    }

    /// Evaluate; same contract and bit-identical results as [`Tape::eval`].
    pub fn eval(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        // Pad short input arrays with NaN (interpreter semantics for reads past
        // the end); the common case passes the full vector and skips this.
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
        work.resize(self.n_work + self.bundle_scratch_len, 0.0);
        let (w, bs) = work.split_at_mut(self.n_work);
        let ctx = HostCtx {
            bscratch: bs.as_mut_ptr(),
            bundles: &self.bundles,
        };
        let (wp, ip, cp) = (w.as_mut_ptr(), ins.as_ptr(), &ctx as *const HostCtx);
        for c in &self.chunks {
            (c.func)(wp, ip, cp);
        }
        out.clear();
        out.extend(self.outputs.iter().map(|&o| w[o as usize]));
    }
}

// ============================================================================
// Lane-batched (SIMD) backend: evaluate a tape on TWO independent input sets
// at once, one f64x2 vector per work slot. Built for device-bundle bodies
// (instance batching): the N instances of a compact-model template become
// lane pairs through `ExternBundle::call_batch`.
//
// Bit-exactness holds per lane: vector fadd/fmul/fsub are IEEE per-lane
// identical to their scalar forms, `MulAdd` stays two rounded vector ops, and
// every transcendental extracts the lanes and routes through the SAME host
// trampolines as the scalar backend. Tapes containing bundle calls cannot be
// lane-compiled (nested bundle scratch has no lane semantics) and return an
// error; callers fall back to per-group scalar evaluation.
// ============================================================================

/// Number of SIMD lanes of the lane backend (f64x2: two instances per pass).
pub const LANES: usize = 2;

/// `S` f64x2 stripes per work slot (lane width = `2*S`). Slot `s` occupies
/// `S*16` bytes at byte offset `s*S*16`; inputs/outputs are lane-interleaved
/// the same way.
struct LaneChunkJit<'a, const S: usize> {
    b: FunctionBuilder<'a>,
    ptr_ty: types::Type,
    work_ptr: Value,
    in_ptr: Value,
    href: &'a HashMap<&'static str, FuncRef>,
    cur: Vec<Option<[Value; S]>>,
    /// Slots that must be materialized in the work array (see [`store_mask`]).
    mask: &'a [bool],
}

impl<const S: usize> LaneChunkJit<'_, S> {
    fn get(&mut self, s: u32) -> [Value; S] {
        if let Some(v) = self.cur[s as usize] {
            return v;
        }
        let v = std::array::from_fn(|j| {
            self.b.ins().load(
                types::F64X2,
                MemFlags::trusted(),
                self.work_ptr,
                (s as i32) * (S as i32) * 16 + (j as i32) * 16,
            )
        });
        self.cur[s as usize] = Some(v);
        v
    }
    fn set(&mut self, dst: u32, v: [Value; S]) {
        if self.mask[dst as usize] {
            for (j, &x) in v.iter().enumerate() {
                self.b.ins().store(
                    MemFlags::trusted(),
                    x,
                    self.work_ptr,
                    (dst as i32) * (S as i32) * 16 + (j as i32) * 16,
                );
            }
        }
        self.cur[dst as usize] = Some(v);
    }
    fn fsplat(&mut self, v: f64) -> [Value; S] {
        let bits = self.b.ins().iconst(types::I64, v.to_bits() as i64);
        let f = self.b.ins().bitcast(types::F64, MemFlags::new(), bits);
        let x = self.b.ins().splat(types::F64X2, f);
        [x; S]
    }
    fn map2(
        &mut self,
        a: [Value; S],
        b: [Value; S],
        f: impl Fn(&mut Self, Value, Value) -> Value,
    ) -> [Value; S] {
        std::array::from_fn(|j| f(self, a[j], b[j]))
    }
    /// Apply a scalar host function lane-wise (same trampolines, same guards).
    /// Apply a coded host function (`h_unary_ext`, `h_binary`) lane-wise.
    fn per_lane_coded(
        &mut self,
        name: &'static str,
        code: Value,
        v: [Value; S],
        w: Option<[Value; S]>,
    ) -> [Value; S] {
        let f = self.href[name];
        std::array::from_fn(|j| {
            let mut rs = [code, code];
            for (l, r) in rs.iter_mut().enumerate() {
                let x = self.b.ins().extractlane(v[j], l as u8);
                let c = match w {
                    Some(w) => {
                        let y = self.b.ins().extractlane(w[j], l as u8);
                        self.b.ins().call(f, &[code, x, y])
                    }
                    None => self.b.ins().call(f, &[code, x]),
                };
                *r = self.b.inst_results(c)[0];
            }
            let z = self.b.ins().splat(types::F64X2, rs[0]);
            self.b.ins().insertlane(z, rs[1], 1)
        })
    }
    fn per_lane(&mut self, name: &'static str, v: [Value; S]) -> [Value; S] {
        let f = self.href[name];
        std::array::from_fn(|j| {
            let x0 = self.b.ins().extractlane(v[j], 0);
            let c0 = self.b.ins().call(f, &[x0]);
            let r0 = self.b.inst_results(c0)[0];
            let x1 = self.b.ins().extractlane(v[j], 1);
            let c1 = self.b.ins().call(f, &[x1]);
            let r1 = self.b.inst_results(c1)[0];
            let z = self.b.ins().splat(types::F64X2, r0);
            self.b.ins().insertlane(z, r1, 1)
        })
    }
    /// Spill lane `l` (0..2S) of `slots` into a fresh stack array.
    fn spill_lane(&mut self, slots: &[u32], l: usize) -> Value {
        let slot = self.b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (slots.len() * 8) as u32,
            3,
        ));
        for (k, &s) in slots.iter().enumerate() {
            let v = self.get(s);
            let x = self.b.ins().extractlane(v[l / 2], (l % 2) as u8);
            self.b.ins().stack_store(x, slot, (k * 8) as i32);
        }
        self.b.ins().stack_addr(self.ptr_ty, slot, 0)
    }
    /// Combine per-lane scalar results (length 2S) into stripe vectors.
    fn join(&mut self, rs: &[Value]) -> [Value; S] {
        std::array::from_fn(|j| {
            let z = self.b.ins().splat(types::F64X2, rs[2 * j]);
            self.b.ins().insertlane(z, rs[2 * j + 1], 1)
        })
    }
    /// `a + b` or `a * b` on every stripe.
    fn vec_combine(&mut self, op: ReduceOp, a: [Value; S], b: [Value; S]) -> [Value; S] {
        std::array::from_fn(|j| match op {
            ReduceOp::Sum => self.b.ins().fadd(a[j], b[j]),
            _ => self.b.ins().fmul(a[j], b[j]),
        })
    }

    /// A `Sum` or `Product` reduction as vector arithmetic, in the exact
    /// order of [`rsgb::node::reduce_slice`]: four accumulators from
    /// `REDUCE_SIMD_MIN` operands on, a left fold from the identity below it
    /// (starting at the identity matters: `0.0 + -0.0` is `+0.0`, folding
    /// from the first operand would not be).
    fn fold_lanes(&mut self, op: ReduceOp, args: &[u32]) -> [Value; S] {
        let ident = self.fsplat(match op {
            ReduceOp::Sum => 0.0,
            _ => 1.0,
        });
        if args.len() >= REDUCE_SIMD_MIN {
            let mut acc = [ident; 4];
            let ch = args.len() / 4;
            for c in 0..ch {
                for (k, a) in acc.iter_mut().enumerate() {
                    let x = self.get(args[4 * c + k]);
                    *a = self.vec_combine(op, *a, x);
                }
            }
            let l = self.vec_combine(op, acc[0], acc[1]);
            let r = self.vec_combine(op, acc[2], acc[3]);
            let mut s = self.vec_combine(op, l, r);
            for &a in &args[ch * 4..] {
                let x = self.get(a);
                s = self.vec_combine(op, s, x);
            }
            s
        } else {
            let mut acc = ident;
            for &a in args {
                let x = self.get(a);
                acc = self.vec_combine(op, acc, x);
            }
            acc
        }
    }

    /// An inner product as vector arithmetic, in the order of
    /// [`rsgb::node::dot_slice`] (two roundings per term, no contraction).
    fn dot_lanes(&mut self, a: &[u32], b: &[u32]) -> [Value; S] {
        let zero = self.fsplat(0.0);
        let term = |s: &mut Self, k: usize| -> [Value; S] {
            let (x, y) = (s.get(a[k]), s.get(b[k]));
            std::array::from_fn(|j| s.b.ins().fmul(x[j], y[j]))
        };
        if a.len() >= REDUCE_SIMD_MIN {
            let mut acc = [zero; 4];
            let ch = a.len() / 4;
            for c in 0..ch {
                for k in 0..4 {
                    let p = term(self, 4 * c + k);
                    acc[k] = self.vec_combine(ReduceOp::Sum, acc[k], p);
                }
            }
            let l = self.vec_combine(ReduceOp::Sum, acc[0], acc[1]);
            let r = self.vec_combine(ReduceOp::Sum, acc[2], acc[3]);
            let mut s = self.vec_combine(ReduceOp::Sum, l, r);
            for k in ch * 4..a.len() {
                let p = term(self, k);
                s = self.vec_combine(ReduceOp::Sum, s, p);
            }
            s
        } else {
            let mut s = zero;
            for k in 0..a.len() {
                let p = term(self, k);
                s = self.vec_combine(ReduceOp::Sum, s, p);
            }
            s
        }
    }

    /// Vector compare -> per-lane 1.0 / 0.0 via mask bit-select.
    fn cmp_mask(&mut self, cc: FloatCC, x: [Value; S], y: [Value; S]) -> [Value; S] {
        let one = self.fsplat(1.0);
        let zero = self.fsplat(0.0);
        std::array::from_fn(|j| {
            let mask = self.b.ins().fcmp(cc, x[j], y[j]);
            let m = self.b.ins().bitcast(types::F64X2, MemFlags::new(), mask);
            self.b.ins().bitselect(m, one[j], zero[j])
        })
    }

    fn lower(&mut self, op: &ROp) -> Result<(), JitError> {
        match op {
            ROp::Const(dst, v) => {
                let val = self.fsplat(*v);
                self.set(*dst, val);
            }
            ROp::Input(dst, k) => {
                let v = if *k == u32::MAX {
                    self.fsplat(f64::NAN)
                } else {
                    std::array::from_fn(|j| {
                        self.b.ins().load(
                            types::F64X2,
                            MemFlags::trusted(),
                            self.in_ptr,
                            (*k as i32) * (S as i32) * 16 + (j as i32) * 16,
                        )
                    })
                };
                self.set(*dst, v);
            }
            ROp::Add(dst, a, b) => {
                let (x, y) = (self.get(*a), self.get(*b));
                let v = self.map2(x, y, |t, p, q| t.b.ins().fadd(p, q));
                self.set(*dst, v);
            }
            ROp::Mul(dst, a, b) => {
                let (x, y) = (self.get(*a), self.get(*b));
                let v = self.map2(x, y, |t, p, q| t.b.ins().fmul(p, q));
                self.set(*dst, v);
            }
            ROp::MulAdd(dst, a, b, c) => {
                let (x, y) = (self.get(*a), self.get(*b));
                let m = self.map2(x, y, |t, p, q| t.b.ins().fmul(p, q));
                let z = self.get(*c);
                let v = self.map2(m, z, |t, p, q| t.b.ins().fadd(p, q));
                self.set(*dst, v);
            }
            ROp::Sub(dst, a, b) => {
                let (x, y) = (self.get(*a), self.get(*b));
                let v = self.map2(x, y, |t, p, q| t.b.ins().fsub(p, q));
                self.set(*dst, v);
            }
            ROp::Neg(dst, a) => {
                let x = self.get(*a);
                let v = std::array::from_fn(|j| self.b.ins().fneg(x[j]));
                self.set(*dst, v);
            }
            ROp::Powi(dst, a, n) => {
                let av = self.get(*a);
                // Same unrolled `__powidf2` sequence as the scalar chunk, but
                // as whole-vector multiplies -- one fmul per SIMD register
                // instead of a libcall per lane.
                let v = if n.unsigned_abs() <= POWI_UNROLL_MAX {
                    let mut pow = n.unsigned_abs();
                    let mut base = av;
                    let mut mul = self.fsplat(1.0);
                    loop {
                        if pow & 1 != 0 {
                            for k in 0..S {
                                mul[k] = self.b.ins().fmul(mul[k], base[k]);
                            }
                        }
                        pow >>= 1;
                        if pow == 0 {
                            break;
                        }
                        for k in 0..S {
                            base[k] = self.b.ins().fmul(base[k], base[k]);
                        }
                    }
                    if *n < 0 {
                        let one = self.fsplat(1.0);
                        for k in 0..S {
                            mul[k] = self.b.ins().fdiv(one[k], mul[k]);
                        }
                    }
                    mul
                } else {
                    let nv = self.b.ins().iconst(types::I64, *n as i64);
                    let f = self.href["h_powi"];
                    let rs: Vec<Value> = (0..2 * S)
                        .map(|l| {
                            let x = self.b.ins().extractlane(av[l / 2], (l % 2) as u8);
                            let c = self.b.ins().call(f, &[x, nv]);
                            self.b.inst_results(c)[0]
                        })
                        .collect();
                    self.join(&rs)
                };
                self.set(*dst, v);
            }
            ROp::Unary(dst, op, a) => {
                let av = self.get(*a);
                // Same guarded-sqrt / exact-floor vectorization as the scalar
                // chunk (see there); everything else stays a per-lane libm call.
                let v = match op {
                    UnaryOp::Sqrt => {
                        let zero = self.fsplat(0.0);
                        std::array::from_fn(|j| {
                            let mask = self.b.ins().fcmp(FloatCC::GreaterThan, av[j], zero[j]);
                            let m = self.b.ins().bitcast(types::F64X2, MemFlags::new(), mask);
                            let sq = self.b.ins().sqrt(av[j]);
                            self.b.ins().bitselect(m, sq, zero[j])
                        })
                    }
                    UnaryOp::Floor => std::array::from_fn(|j| self.b.ins().floor(av[j])),
                    _ => match unary_sym(*op) {
                        Some(sym) => self.per_lane(sym, av),
                        None => {
                            let code = self.b.ins().iconst(types::I32, unary_code(*op) as i64);
                            self.per_lane_coded("h_unary_ext", code, av, None)
                        }
                    },
                };
                self.set(*dst, v);
            }
            ROp::Binary(dst, op, a, b) => {
                let (x, y) = (self.get(*a), self.get(*b));
                let code = self.b.ins().iconst(types::I32, binary_code(*op) as i64);
                let v = self.per_lane_coded("h_binary", code, x, Some(y));
                self.set(*dst, v);
            }
            ROp::Cmp(dst, op, a, b) => {
                let (x, y) = (self.get(*a), self.get(*b));
                let cc = match op {
                    CmpOp::Gt => FloatCC::GreaterThan,
                    CmpOp::Ge => FloatCC::GreaterThanOrEqual,
                    CmpOp::Lt => FloatCC::LessThan,
                    CmpOp::Le => FloatCC::LessThanOrEqual,
                    CmpOp::Eq => FloatCC::Equal,
                    CmpOp::Ne => FloatCC::NotEqual,
                };
                let v = self.cmp_mask(cc, x, y);
                self.set(*dst, v);
            }
            ROp::Select(dst, c, t, e) => {
                let (cv, tv, ev) = (self.get(*c), self.get(*t), self.get(*e));
                let zero = self.fsplat(0.0);
                let v = std::array::from_fn(|j| {
                    let mask = self.b.ins().fcmp(FloatCC::NotEqual, cv[j], zero[j]);
                    let m = self.b.ins().bitcast(types::F64X2, MemFlags::new(), mask);
                    self.b.ins().bitselect(m, tv[j], ev[j])
                });
                self.set(*dst, v);
            }
            ROp::Reduce(dst, op, args) => {
                // Sum and Product fold as vector arithmetic in the reference
                // order, like the scalar backend does. Min and Max keep the
                // per-lane host call: Cranelift's `fmin`/`fmax` disagree with
                // `f64::min` on NaN, and the reduction is the *one* place
                // arrays and matrices reach the backend, so it has to be
                // bit-exact rather than merely close.
                let v = if matches!(op, ReduceOp::Sum | ReduceOp::Product) && !args.is_empty() {
                    self.fold_lanes(*op, args)
                } else {
                    let opc = self.b.ins().iconst(types::I32, reduce_code(*op));
                    let lenv = self.b.ins().iconst(self.ptr_ty, args.len() as i64);
                    let f = self.href["h_reduce"];
                    let rs: Vec<Value> = (0..2 * S)
                        .map(|l| {
                            let p = self.spill_lane(args, l);
                            let c = self.b.ins().call(f, &[opc, p, lenv]);
                            self.b.inst_results(c)[0]
                        })
                        .collect();
                    self.join(&rs)
                };
                self.set(*dst, v);
            }
            ROp::Dot(dst, a, bb) => {
                let v = self.dot_lanes(a, bb);
                self.set(*dst, v);
            }
            ROp::Bundle(..) | ROp::BundleBatch(..) | ROp::Pick(..) => {
                return Err(JitError::Codegen(
                    "bundle calls have no lane semantics (body tapes are call-free)".into(),
                ));
            }
        }
        Ok(())
    }
}

fn compile_lane_chunk<const S: usize>(
    ops: &[ROp],
    n_work: usize,
    mask: &[bool],
) -> Result<NativeChunk, JitError> {
    let mut flags = settings::builder();
    flags.set("use_colocated_libcalls", "false").unwrap();
    flags.set("is_pic", "false").unwrap();
    flags.set("opt_level", "speed").unwrap();
    let isa = cranelift_native::builder()
        .map_err(|e| JitError::Codegen(e.to_string()))?
        .finish(settings::Flags::new(flags))
        .map_err(|e| JitError::Codegen(e.to_string()))?;

    let mut jb = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    for name in HOST_NAMES {
        jb.symbol(name, host_addr(name));
    }
    let mut module = JITModule::new(jb);
    let ptr_ty = module.target_config().pointer_type();
    let call_conv = module.target_config().default_call_conv;

    let mut host_ids: HashMap<&'static str, cranelift_module::FuncId> = HashMap::default();
    {
        let mut decl = |name: &'static str, params: &[types::Type], ret: Option<types::Type>| {
            let mut sig = module.make_signature();
            sig.call_conv = call_conv;
            for &p in params {
                sig.params.push(AbiParam::new(p));
            }
            if let Some(r) = ret {
                sig.returns.push(AbiParam::new(r));
            }
            let id = module
                .declare_function(name, Linkage::Import, &sig)
                .expect("declare host import");
            host_ids.insert(name, id);
        };
        for op in [
            UnaryOp::Exp,
            UnaryOp::Ln,
            UnaryOp::Sqrt,
            UnaryOp::Sin,
            UnaryOp::Cos,
            UnaryOp::Sinh,
            UnaryOp::Cosh,
            UnaryOp::Tanh,
            UnaryOp::Atan,
            UnaryOp::Floor,
        ] {
            decl(
                unary_sym(op).expect("a dedicated trampoline"),
                &[types::F64],
                Some(types::F64),
            );
        }
        decl("h_unary_ext", &[types::I32, types::F64], Some(types::F64));
        decl(
            "h_binary",
            &[types::I32, types::F64, types::F64],
            Some(types::F64),
        );
        decl("h_powi", &[types::F64, types::I64], Some(types::F64));
        decl("h_reduce", &[types::I32, ptr_ty, ptr_ty], Some(types::F64));
        decl("h_dot", &[ptr_ty, ptr_ty, ptr_ty], Some(types::F64));
    }

    let mut cctx = module.make_context();
    cctx.func.signature.call_conv = call_conv;
    cctx.func.signature.params.push(AbiParam::new(ptr_ty)); // work (lane pairs)
    cctx.func.signature.params.push(AbiParam::new(ptr_ty)); // inputs (lane pairs)
    cctx.func.signature.params.push(AbiParam::new(ptr_ty)); // host ctx

    let mut fbctx = FunctionBuilderContext::new();
    let res: Result<(), JitError> = {
        let mut b = FunctionBuilder::new(&mut cctx.func, &mut fbctx);
        let blk = b.create_block();
        b.append_block_params_for_function_params(blk);
        b.switch_to_block(blk);
        b.seal_block(blk);
        let work_ptr = b.block_params(blk)[0];
        let in_ptr = b.block_params(blk)[1];

        let mut href: HashMap<&'static str, FuncRef> = HashMap::default();
        for (&name, &id) in &host_ids {
            href.insert(name, module.declare_func_in_func(id, b.func));
        }

        let mut jit = LaneChunkJit::<S> {
            b,
            ptr_ty,
            work_ptr,
            in_ptr,
            href: &href,
            cur: vec![None; n_work],
            mask,
        };
        let mut r = Ok(());
        for op in ops {
            r = jit.lower(op);
            if r.is_err() {
                break;
            }
        }
        if r.is_ok() {
            jit.b.ins().return_(&[]);
            jit.b.finalize();
        }
        r
    };
    res?;

    let func_id = module
        .declare_function("lane_chunk", Linkage::Export, &cctx.func.signature)
        .map_err(|e| JitError::Codegen(e.to_string()))?;
    module
        .define_function(func_id, &mut cctx)
        .map_err(|e| JitError::Codegen(e.to_string()))?;
    module.clear_context(&mut cctx);
    module
        .finalize_definitions()
        .map_err(|e| JitError::Codegen(e.to_string()))?;

    let code = module.get_finalized_function(func_id);
    let func = unsafe { std::mem::transmute::<*const u8, ChunkFn>(code) };
    Ok(NativeChunk {
        _module: module,
        func,
    })
}

/// A tape compiled to native SIMD code evaluating TWO independent input sets
/// per pass ([`LANES`] f64 lanes per slot). Same numeric contract per lane as
/// [`Tape::eval`], bit-exact. All buffers are lane-interleaved: input `k`'s
/// lanes at `inputs[k*2] / inputs[k*2+1]`, output `j` likewise.
pub struct LaneTape {
    chunks: Vec<NativeChunk>,
    prolog_chunks: usize,
    bundles: Vec<Arc<dyn ExternBundle>>,
    outputs: Vec<u32>,
    n_work: usize,
    n_inputs: usize,
    /// Lane count (2 = one f64x2 stripe per slot, 4 = two stripes).
    width: usize,
}

impl LaneTape {
    /// Compile the pair (2-lane) backend; fails on tapes containing bundle
    /// calls (see module docs).
    pub fn compile(tape: &Tape) -> Result<LaneTape, JitError> {
        Self::compile_with(tape, CHUNK_OPS)
    }

    /// Compile the wide (4-lane, two-stripe) backend.
    pub fn compile_wide(tape: &Tape) -> Result<LaneTape, JitError> {
        Self::compile_stripes::<2>(tape, CHUNK_OPS)
    }

    /// [`compile_wide`](Self::compile_wide) with an explicit chunk size.
    pub fn compile_wide_with(tape: &Tape, chunk_ops: usize) -> Result<LaneTape, JitError> {
        Self::compile_stripes::<2>(tape, chunk_ops)
    }

    pub fn compile_with(tape: &Tape, chunk_ops: usize) -> Result<LaneTape, JitError> {
        Self::compile_stripes::<1>(tape, chunk_ops)
    }

    /// Lane count of this backend (interleave stride of all buffers).
    pub fn width(&self) -> usize {
        self.width
    }

    fn compile_stripes<const S: usize>(
        tape: &Tape,
        chunk_ops: usize,
    ) -> Result<LaneTape, JitError> {
        let mut rec = Recorder::default();
        tape.lower(&mut rec);
        let n_inputs = rec
            .ops
            .iter()
            .filter_map(|op| match op {
                ROp::Input(_, k) if *k != u32::MAX => Some(*k as usize + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let n_work = tape.n_work();
        let chunk_ops = chunk_ops.max(1);
        let split = tape.prolog_len().min(rec.ops.len());
        use rayon::prelude::*;
        let (pro, main) = rec.ops.split_at(split);
        let jobs: Vec<&[ROp]> = pro
            .chunks(chunk_ops)
            .chain(main.chunks(chunk_ops))
            .collect();
        let prolog_chunks = pro.chunks(chunk_ops).count();
        let mask = store_mask(&jobs, tape.outputs(), n_work);
        let chunks: Result<Vec<NativeChunk>, JitError> = jobs
            .par_iter()
            .map(|ops| compile_lane_chunk::<S>(ops, n_work, &mask))
            .collect();
        Ok(LaneTape {
            chunks: chunks?,
            prolog_chunks,
            bundles: rec.bundles,
            outputs: tape.outputs().to_vec(),
            n_work,
            n_inputs,
            width: 2 * S,
        })
    }

    fn ctx(&self) -> HostCtx {
        HostCtx {
            bscratch: std::ptr::null_mut(),
            bundles: &self.bundles,
        }
    }

    fn pad<'a>(&self, inputs: &'a [f64], padded: &'a mut Vec<f64>) -> &'a [f64] {
        if inputs.len() < self.n_inputs * self.width {
            padded.extend_from_slice(inputs);
            padded.resize(self.n_inputs * self.width, f64::NAN);
            padded
        } else {
            inputs
        }
    }

    /// Evaluate the prolog chunks into the lane-interleaved `work` buffer.
    pub fn eval_prolog(&self, inputs: &[f64], work: &mut Vec<f64>) {
        let mut padded = Vec::new();
        let ins = self.pad(inputs, &mut padded);
        work.clear();
        work.resize(self.n_work * self.width, 0.0);
        let ctx = self.ctx();
        let (wp, ip, cp) = (work.as_mut_ptr(), ins.as_ptr(), &ctx as *const HostCtx);
        for c in &self.chunks[..self.prolog_chunks] {
            (c.func)(wp, ip, cp);
        }
    }

    /// Evaluate the main chunks over a prolog-prepared buffer; `out` receives
    /// the lane-interleaved outputs (`n_outputs * LANES`).
    pub fn eval_main(&self, inputs: &[f64], work: &mut [f64], out: &mut Vec<f64>) {
        assert_eq!(
            work.len(),
            self.n_work * self.width,
            "lane work buffer mismatch"
        );
        let mut padded = Vec::new();
        let ins = self.pad(inputs, &mut padded);
        let ctx = self.ctx();
        let (wp, ip, cp) = (work.as_mut_ptr(), ins.as_ptr(), &ctx as *const HostCtx);
        for c in &self.chunks[self.prolog_chunks..] {
            (c.func)(wp, ip, cp);
        }
        out.clear();
        for &o in &self.outputs {
            for l in 0..self.width {
                out.push(work[o as usize * self.width + l]);
            }
        }
    }

    /// Evaluate both phases (fresh buffer).
    pub fn eval(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        self.eval_prolog(inputs, work);
        let mut w = std::mem::take(work);
        self.eval_main(inputs, &mut w, out);
        *work = w;
    }
}
