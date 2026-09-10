//! The op stream: the tape lowered through [`TapeVisitor`] into a flat
//! vector the emitter walks. Recording decouples the chunk partitioning and
//! the parallel per-chunk codegen from the visitor callback structure.
//! Bundle bodies are interned by `Arc` identity into a table the compiled
//! code indexes through the pointer every chunk receives.

use rsdag::extern_fn::ExternBundle;
use rsdag::node::{BinOp, CmpOp, ReduceOp, UnaryOp};
use rsdag::TapeVisitor;
use rustc_hash::FxHashMap;
use std::sync::Arc;

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
    /// `(bundle, args, scratch_base)`.
    Bundle(u32, Vec<u32>, u32),
    /// `(bundle, args, n_groups, n_args, base0)`.
    BundleBatch(u32, Vec<u32>, u32, u32, u32),
    /// `(dst, scratch index)`.
    Pick(u32, u32),
}

impl ROp {
    /// How many slots the op hands to a host routine through the gather
    /// area of the work array.
    pub(crate) fn gather_len(&self) -> usize {
        match self {
            ROp::Reduce(_, ReduceOp::Min | ReduceOp::Max, args) => args.len(),
            ROp::Bundle(_, args, _) | ROp::BundleBatch(_, args, ..) => args.len(),
            _ => 0,
        }
    }
}

#[derive(Default)]
pub(crate) struct Recorder {
    pub(crate) ops: Vec<ROp>,
    pub(crate) bundles: Vec<Arc<dyn ExternBundle>>,
    bundle_idx: FxHashMap<usize, u32>,
}

impl Recorder {
    fn intern(&mut self, b: &Arc<dyn ExternBundle>) -> u32 {
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
        let idx = self.intern(b);
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
        let idx = self.intern(b);
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
