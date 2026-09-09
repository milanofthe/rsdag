//! Host trampolines: the routines native code calls for what it does not
//! emit itself, bit-identical to the interpreter's, and the table that
//! resolves them by name.

use rsdag::extern_fn::ExternBundle;
use rsdag::node::{binary_f64, dot_slice, reduce_slice, unary_f64, BinOp, ReduceOp, UnaryOp};
use std::sync::Arc;

// --- host trampolines: bit-identical to the interpreter's elementary ops ----
//
// Every unary op routes through `unary_f64`, the one source of truth the arena
// and tape interpreters use. Calling the bare libm op here would silently drop
// its domain guards, so compiled code would diverge on out-of-domain iterates.

pub(crate) extern "C" fn h_exp(x: f64) -> f64 {
    unary_f64(UnaryOp::Exp, x)
}
pub(crate) extern "C" fn h_ln(x: f64) -> f64 {
    unary_f64(UnaryOp::Ln, x)
}
pub(crate) extern "C" fn h_sqrt(x: f64) -> f64 {
    unary_f64(UnaryOp::Sqrt, x)
}
pub(crate) extern "C" fn h_sin(x: f64) -> f64 {
    unary_f64(UnaryOp::Sin, x)
}
pub(crate) extern "C" fn h_cos(x: f64) -> f64 {
    unary_f64(UnaryOp::Cos, x)
}
pub(crate) extern "C" fn h_sinh(x: f64) -> f64 {
    unary_f64(UnaryOp::Sinh, x)
}
pub(crate) extern "C" fn h_cosh(x: f64) -> f64 {
    unary_f64(UnaryOp::Cosh, x)
}
pub(crate) extern "C" fn h_tanh(x: f64) -> f64 {
    unary_f64(UnaryOp::Tanh, x)
}
pub(crate) extern "C" fn h_atan(x: f64) -> f64 {
    unary_f64(UnaryOp::Atan, x)
}
pub(crate) extern "C" fn h_floor(x: f64) -> f64 {
    unary_f64(UnaryOp::Floor, x)
}
/// The floating-point extension of the unary set: one trampoline, the op as
/// a constant code (see `unary_code`), the same `unary_f64` as everywhere.
pub(crate) extern "C" fn h_unary_ext(op: u32, x: f64) -> f64 {
    unary_f64(unary_from_code(op), x)
}
pub(crate) extern "C" fn h_binary(op: u32, x: f64, y: f64) -> f64 {
    binary_f64(binary_from_code(op), x, y)
}
pub(crate) extern "C" fn h_powi(x: f64, n: i64) -> f64 {
    x.powi(n as i32)
}

pub(crate) extern "C" fn h_reduce(op: u32, ptr: *const f64, len: usize) -> f64 {
    let xs = unsafe { std::slice::from_raw_parts(ptr, len) };
    let op = match op {
        0 => ReduceOp::Sum,
        1 => ReduceOp::Product,
        2 => ReduceOp::Min,
        _ => ReduceOp::Max,
    };
    reduce_slice(op, xs)
}

pub(crate) extern "C" fn h_dot(a: *const f64, b: *const f64, len: usize) -> f64 {
    let va = unsafe { std::slice::from_raw_parts(a, len) };
    let vb = unsafe { std::slice::from_raw_parts(b, len) };
    dot_slice(va, vb)
}

/// Per-eval host context handed to every chunk. `#[repr(C)]` with `bscratch`
/// first: `BundlePick` loads the pointer straight off offset 0.
#[repr(C)]
pub(crate) struct HostCtx {
    pub(crate) bscratch: *mut f64,
    pub(crate) bundles: *const Vec<Arc<dyn ExternBundle>>,
}

pub(crate) extern "C" fn h_bundle(
    ctx: *const HostCtx,
    idx: usize,
    args: *const f64,
    len: usize,
    base: usize,
) {
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

pub(crate) extern "C" fn h_bundle_batch(
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
pub(crate) fn unary_sym(op: UnaryOp) -> Option<&'static str> {
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
pub(crate) fn unary_code(op: UnaryOp) -> u32 {
    op.code()
}
pub(crate) fn unary_from_code(code: u32) -> UnaryOp {
    UnaryOp::from_code(code)
}
pub(crate) fn binary_code(op: BinOp) -> u32 {
    op.code()
}
pub(crate) fn binary_from_code(code: u32) -> BinOp {
    BinOp::from_code(code)
}

pub(crate) fn reduce_code(op: ReduceOp) -> i64 {
    match op {
        ReduceOp::Sum => 0,
        ReduceOp::Product => 1,
        ReduceOp::Min => 2,
        ReduceOp::Max => 3,
    }
}

pub(crate) const HOST_NAMES: [&str; 17] = [
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

pub(crate) fn host_addr(name: &str) -> *const u8 {
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
