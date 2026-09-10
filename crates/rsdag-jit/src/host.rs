//! Host routines: what the emitted code calls for everything that is not
//! an instruction. Every transcendental goes through the same
//! [`unary_f64`](rsdag::node::unary_f64) as the interpreter, so the domain
//! guards hold identically and the backends agree to the bit.

use rsdag::extern_fn::ExternBundle;
use rsdag::node::{binary_f64, reduce_slice, unary_f64, BinOp, ReduceOp, UnaryOp};
use std::sync::Arc;

pub(crate) type Bundles = Vec<Arc<dyn ExternBundle>>;

macro_rules! unary_trampolines {
    ($($name:ident = $op:ident),* $(,)?) => {
        $(pub(crate) extern "C" fn $name(x: f64) -> f64 {
            unary_f64(UnaryOp::$op, x)
        })*
        /// The dedicated routine of a unary op, or the coded one.
        pub(crate) fn unary_addr(op: UnaryOp) -> (*const (), Option<u32>) {
            match op {
                $(UnaryOp::$op => ($name as *const (), None),)*
                _ => (h_unary_ext as *const (), Some(op.code())),
            }
        }
    };
}
unary_trampolines! {
    h_exp = Exp, h_ln = Ln, h_sqrt = Sqrt, h_sin = Sin, h_cos = Cos,
    h_sinh = Sinh, h_cosh = Cosh, h_tanh = Tanh, h_atan = Atan, h_floor = Floor,
}

pub(crate) extern "C" fn h_unary_ext(op: u32, x: f64) -> f64 {
    unary_f64(UnaryOp::from_code(op), x)
}
pub(crate) extern "C" fn h_binary(op: u32, x: f64, y: f64) -> f64 {
    binary_f64(BinOp::from_code(op), x, y)
}
pub(crate) extern "C" fn h_powi(x: f64, n: i64) -> f64 {
    x.powi(n as i32)
}

pub(crate) fn reduce_code(op: ReduceOp) -> u64 {
    match op {
        ReduceOp::Sum => 0,
        ReduceOp::Product => 1,
        ReduceOp::Min => 2,
        ReduceOp::Max => 3,
    }
}
pub(crate) extern "C" fn h_reduce(op: u64, ptr: *const f64, len: usize) -> f64 {
    let xs = unsafe { std::slice::from_raw_parts(ptr, len) };
    let op = match op {
        0 => ReduceOp::Sum,
        1 => ReduceOp::Product,
        2 => ReduceOp::Min,
        _ => ReduceOp::Max,
    };
    reduce_slice(op, xs)
}

pub(crate) extern "C" fn h_bundle(
    bundles: *const Bundles,
    idx: usize,
    args: *const f64,
    len: usize,
    out: *mut f64,
) {
    let b = unsafe { &(&*bundles)[idx] };
    let xs = unsafe { std::slice::from_raw_parts(args, len) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, b.n_outputs()) };
    b.call(xs, out);
}

pub(crate) extern "C" fn h_bundle_batch(
    bundles: *const Bundles,
    idx: usize,
    args: *const f64,
    n_groups: usize,
    n_args: usize,
    out: *mut f64,
) {
    let b = unsafe { &(&*bundles)[idx] };
    let xs = unsafe { std::slice::from_raw_parts(args, n_groups * n_args) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n_groups * b.n_outputs()) };
    b.call_batch(xs, n_groups, n_args, out);
}
