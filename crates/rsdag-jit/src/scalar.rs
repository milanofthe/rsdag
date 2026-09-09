//! The scalar backend: the tape recorded into an op stream, cut into
//! chunks, each compiled to a native function; `ChunkedTape` runs them.

use crate::host::*;
use crate::*;
use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{types, AbiParam, FuncRef, InstBuilder, MemFlags, Value};
use cranelift_codegen::ir::{StackSlotData, StackSlotKind};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use rsdag::extern_fn::ExternBundle;
use rsdag::node::{BinOp, CmpOp, ReduceOp, UnaryOp, REDUCE_SIMD_MIN};
use rsdag::{Tape, TapeVisitor};
use rustc_hash::FxHashMap as HashMap;
use std::sync::Arc;

// --- inline reductions: `reduce_slice` / `dot_slice` as IR -------------------
//
// The 4-accumulator fold of the interpreter emitted in the identical
// association order, so the value is bit-exact -- without the stack spill and
// host call a long KCL sum or matrix row otherwise pays.

/// The 4-lane sum / product fold of `reduce_slice` over scalar values
/// (`xs.len() >= REDUCE_SIMD_MIN`).
pub(crate) fn emit_reduce_4lane(b: &mut FunctionBuilder, op: ReduceOp, xs: &[Value]) -> Value {
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
pub(crate) fn emit_dot(b: &mut FunctionBuilder, xs: &[Value], ys: &[Value]) -> Value {
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
pub(crate) const POWI_UNROLL_MAX: u32 = 16;

// callback structure. Bundle bodies are interned by `Arc` identity
// into tables the compiled code indexes through the host context.

pub(crate) enum ROp {
    Const(u32, f64),
    Input(u32, u32),
    Add(u32, u32, u32),
    Mul(u32, u32, u32),
    MulAdd(u32, u32, u32, u32),
    Fma(u32, u32, u32, u32),
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
            | ROp::Fma(d, _, _, _)
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
            ROp::MulAdd(_, a, b, c) | ROp::Fma(_, a, b, c) | ROp::Select(_, a, b, c) => {
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
pub(crate) fn store_mask(jobs: &[&[ROp]], outputs: &[u32], n_work: usize) -> Vec<bool> {
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
pub(crate) struct Recorder {
    pub(crate) ops: Vec<ROp>,
    pub(crate) bundles: Vec<Arc<dyn ExternBundle>>,
    pub(crate) bundle_idx: HashMap<usize, u32>,
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
    fn fma(&mut self, dst: u32, a: u32, b: u32, c: u32) {
        self.ops.push(ROp::Fma(dst, a, b, c));
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

pub(crate) type ChunkFn = extern "C" fn(*mut f64, *const f64, *const HostCtx);

/// One compiled chunk. The module is kept alive so the code stays mapped; it is
/// immutable after `finalize_definitions`, so sending it across threads (rayon
/// compiles chunks in parallel) and calling the code from any thread is sound.
pub(crate) struct NativeChunk {
    pub(crate) _module: JITModule,
    pub(crate) func: ChunkFn,
}
unsafe impl Send for NativeChunk {}
unsafe impl Sync for NativeChunk {}

/// Chunk-local lowering state. `cur` caches the SSA value of each work slot
/// written or read within this chunk; a write stores to the work array only
/// when the liveness mask says a later chunk (or the host, for outputs) reads
/// it -- everything else stays register-resident. Reads of slots produced by
/// earlier chunks load from the work array once.
pub(crate) struct ChunkJit<'a> {
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
            ROp::Fma(dst, a, b, c) => {
                let (x, y, z) = (self.get(*a), self.get(*b), self.get(*c));
                let v = self.b.ins().fma(x, y, z);
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
                // The ops a machine does in one instruction are emitted, not
                // called: rounding to integral is IEEE-exact, `fabs` clears a
                // bit, and the guarded `sqrt` (`x > 0 ? sqrt(x) : 0`, see
                // `unary_f64`) is a compare, a hardware sqrt and a select.
                // All of them stay bit-identical to the host routine, minus
                // the call. `round` is not among them: the reference rounds
                // half away from zero, Cranelift's `nearest` rounds to even.
                let v = match op {
                    UnaryOp::Sqrt => {
                        let zero = self.fconst(0.0);
                        let pos = self.b.ins().fcmp(FloatCC::GreaterThan, av, zero);
                        let sq = self.b.ins().sqrt(av);
                        self.b.ins().select(pos, sq, zero)
                    }
                    UnaryOp::Floor => self.b.ins().floor(av),
                    UnaryOp::Ceil => self.b.ins().ceil(av),
                    UnaryOp::Trunc => self.b.ins().trunc(av),
                    UnaryOp::Abs => self.b.ins().fabs(av),
                    // `sign` is the reference's own cascade: `+-0.0` and NaN
                    // fall through unchanged because neither comparison
                    // holds for them.
                    UnaryOp::Sign => {
                        let zero = self.fconst(0.0);
                        let one = self.fconst(1.0);
                        let minus = self.fconst(-1.0);
                        let pos = self.b.ins().fcmp(FloatCC::GreaterThan, av, zero);
                        let neg = self.b.ins().fcmp(FloatCC::LessThan, av, zero);
                        let n = self.b.ins().select(neg, minus, av);
                        self.b.ins().select(pos, one, n)
                    }
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

pub(crate) fn compile_chunk(
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
