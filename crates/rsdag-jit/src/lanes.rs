//! The lane backend: the same op stream over `f64x2` stripes, several
//! parameter sets per pass; `LaneTape` runs it, `suggest_lanes` picks a
//! width.

use crate::host::*;
use crate::scalar::*;
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

/// Number of SIMD lanes of the lane backend's default width (f64x2: two
/// instances per pass).
pub const LANES: usize = 2;

/// The lane counts [`LaneTape::compile_lanes`] accepts. Each is a whole
/// number of f64x2 stripes per value, so lane `k` of a wider tape computes
/// exactly what lane `k` of a narrower one does.
pub const LANE_WIDTHS: [usize; 4] = [2, 4, 8, 16];

/// `S` f64x2 stripes per work slot (lane width = `2*S`). Slot `s` occupies
/// `S*16` bytes at byte offset `s*S*16`; inputs/outputs are lane-interleaved
/// the same way.
pub(crate) struct LaneChunkJit<'a, const S: usize> {
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
    /// order of [`rsdag::node::reduce_slice`]: four accumulators from
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
    /// [`rsdag::node::dot_slice`] (two roundings per term, no contraction).
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
            ROp::Fma(dst, a, b, c) => {
                let (x, y, z) = (self.get(*a), self.get(*b), self.get(*c));
                let v = std::array::from_fn(|j| self.b.ins().fma(x[j], y[j], z[j]));
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
                    UnaryOp::Ceil => std::array::from_fn(|j| self.b.ins().ceil(av[j])),
                    UnaryOp::Trunc => std::array::from_fn(|j| self.b.ins().trunc(av[j])),
                    UnaryOp::Abs => std::array::from_fn(|j| self.b.ins().fabs(av[j])),
                    UnaryOp::Sign => {
                        let zero = self.fsplat(0.0);
                        let one = self.fsplat(1.0);
                        let minus = self.fsplat(-1.0);
                        std::array::from_fn(|j| {
                            let pos = self.b.ins().fcmp(FloatCC::GreaterThan, av[j], zero[j]);
                            let pm = self.b.ins().bitcast(types::F64X2, MemFlags::new(), pos);
                            let neg = self.b.ins().fcmp(FloatCC::LessThan, av[j], zero[j]);
                            let nm = self.b.ins().bitcast(types::F64X2, MemFlags::new(), neg);
                            let n = self.b.ins().bitselect(nm, minus[j], av[j]);
                            self.b.ins().bitselect(pm, one[j], n)
                        })
                    }
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

pub(crate) fn compile_lane_chunk<const S: usize>(
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

/// Counts the ops that the lane backend cannot vectorise, so they cost one
/// host call *per lane* and get more expensive as the width grows: the
/// elementary functions, the binary functions beyond the ring, and the
/// ordered reductions (see the `Reduce` arm of the lane lowering).
#[derive(Default)]
pub(crate) struct HostCallCount {
    host: usize,
    total: usize,
}

impl TapeVisitor for HostCallCount {
    fn constant(&mut self, _: u32, _: f64) {
        self.total += 1;
    }
    fn input(&mut self, _: u32, _: u32) {
        self.total += 1;
    }
    fn add(&mut self, _: u32, _: u32, _: u32) {
        self.total += 1;
    }
    fn mul(&mut self, _: u32, _: u32, _: u32) {
        self.total += 1;
    }
    fn mul_add(&mut self, _: u32, _: u32, _: u32, _: u32) {
        self.total += 1;
    }
    fn sub(&mut self, _: u32, _: u32, _: u32) {
        self.total += 1;
    }
    fn neg(&mut self, _: u32, _: u32) {
        self.total += 1;
    }
    fn powi(&mut self, _: u32, _: u32, _: i32) {
        self.total += 1;
    }
    fn unary(&mut self, _: u32, _: UnaryOp, _: u32) {
        self.total += 1;
        self.host += 1;
    }
    fn binary(&mut self, _: u32, _: BinOp, _: u32, _: u32) {
        self.total += 1;
        self.host += 1;
    }
    fn cmp(&mut self, _: u32, _: CmpOp, _: u32, _: u32) {
        self.total += 1;
    }
    fn select(&mut self, _: u32, _: u32, _: u32, _: u32) {
        self.total += 1;
    }
    fn reduce(&mut self, _: u32, op: ReduceOp, _: &[u32]) {
        self.total += 1;
        if matches!(op, ReduceOp::Min | ReduceOp::Max) {
            self.host += 1;
        }
    }
    fn dot(&mut self, _: u32, _: &[u32], _: &[u32]) {
        self.total += 1;
    }
    fn bundle_call(&mut self, _: &Arc<dyn ExternBundle>, _: &[u32], _: u32) {
        self.total += 1;
        self.host += 1;
    }
    fn bundle_batch(&mut self, _: &Arc<dyn ExternBundle>, _: &[u32], _: u32, _: u32, _: u32) {
        self.total += 1;
        self.host += 1;
    }
    fn bundle_pick(&mut self, _: u32, _: u32) {
        self.total += 1;
    }
}

/// The lane width to compile this tape at, or `None` when no width beats
/// running the scalar program once per parameter set.
///
/// Two properties of the program decide it, and both were measured rather
/// than reasoned (the `bench` example prints the table):
///
/// - How many values are live at once. A chain of dependent ops leaves the
///   register file idle and gains the most from lanes -- 15x per set at
///   eight lanes on a pure mul/add chain, because the lanes fill a pipeline
///   that was stalling. A program with hundreds of live values already uses
///   the registers, and wide lanes spill them: the same chain with 512 live
///   values gives 1.9x at four lanes and 0.8x at sixteen.
/// - How much of the program the lane backend cannot vectorise. Every
///   elementary function and every ordered reduction costs one host call per
///   lane, so their cost grows with the width instead of shrinking: a
///   transcendental-heavy program is already at 0.8x with two lanes and 0.4x
///   with sixteen.
pub fn suggest_lanes(tape: &Tape) -> Option<usize> {
    let mut count = HostCallCount::default();
    tape.lower(&mut count);
    if count.total == 0 {
        return None;
    }
    let host = count.host as f64 / count.total as f64;
    if host > 0.25 {
        // Dominated by per-lane host calls: the scalar backend wins.
        return None;
    }
    if host > 0.05 {
        // Some per-lane calls: two lanes still pay, wider ones do not.
        return Some(2);
    }
    match tape.n_slots() {
        0..=255 => Some(8),
        256..=1023 => Some(4),
        _ => Some(2),
    }
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

    /// Compile for an explicit lane count: 2, 4, 8 or 16 parameter sets per
    /// pass, held as `lanes / 2` vector registers per live value.
    ///
    /// Which width pays depends on the program, not on the machine alone. A
    /// program with few live values at a time (a chain) has registers to
    /// spare and gains from more lanes; a wide program already keeps the
    /// register file busy, and more lanes spill it. [`LANE_WIDTHS`] lists
    /// what is available, and the `bench` example prices them.
    pub fn compile_lanes(
        tape: &Tape,
        lanes: usize,
        chunk_ops: usize,
    ) -> Result<LaneTape, JitError> {
        match lanes {
            2 => Self::compile_stripes::<1>(tape, chunk_ops),
            4 => Self::compile_stripes::<2>(tape, chunk_ops),
            8 => Self::compile_stripes::<4>(tape, chunk_ops),
            16 => Self::compile_stripes::<8>(tape, chunk_ops),
            _ => Err(JitError::Codegen(format!(
                "lane count {lanes} is not one of {LANE_WIDTHS:?}"
            ))),
        }
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
