//! Host routines: what the emitted code calls for everything that is not
//! an instruction. Every transcendental goes through the same
//! [`unary_f64`](rsdag::semantics::unary_f64) as the interpreter, so the domain
//! guards hold identically and the backends agree to the bit.

use rsdag::extern_fn::ExternBundle;
use rsdag::node::{BinOp, ReduceOp, UnaryOp};
use rsdag::semantics::{binary_f64, reduce_slice, unary_f64};
use rsdag::tape::calls::{run_call, run_stage, Call, Shared};
use std::sync::Arc;

use crate::native::Layout;

pub(crate) type Bundles = Vec<Arc<dyn ExternBundle>>;

macro_rules! unary_trampolines {
    ($($name:ident = $op:ident),* $(,)?) => {
        $(pub(crate) extern "C" fn $name(x: f64) -> f64 {
            unary_f64(UnaryOp::$op, x)
        })*
        pub(crate) fn unary_addr(op: UnaryOp) -> (*const (), Option<u32>) {
            match op {
                $(UnaryOp::$op => ($name as *const (), None),)*
                _ => (h_unary_ext as *const (), Some(op.code())),
            }
        }
    };
}
unary_trampolines! {
    h_sqrt = Sqrt, h_sin = Sin, h_cos = Cos,
    h_sinh = Sinh, h_cosh = Cosh, h_tanh = Tanh, h_atan = Atan, h_floor = Floor,
}

pub(crate) extern "C" fn h_unary_ext(op: u32, x: f64) -> f64 {
    unary_f64(UnaryOp::from_code(op), x)
}
pub(crate) extern "C" fn h_binary(op: u32, x: f64, y: f64) -> f64 {
    binary_f64(BinOp::from_code(op), x, y)
}
pub(crate) extern "C" fn h_powi(x: f64, n: i64) -> f64 {
    rsdag::semantics::powi_f64(x, n as i32)
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

// Unwinding out of `extern "C"` aborts: https://doc.rust-lang.org/nomicon/ffi.html#ffi-and-unwinding
std::thread_local! {
    static FIRST_PANIC_HELD_PAST_EMITTED_FRAMES: std::cell::Cell<Option<Box<dyn std::any::Any + Send>>> =
        const { std::cell::Cell::new(None) };
}

fn hold_panic_past_emitted_frames(f: impl FnOnce()) {
    if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        FIRST_PANIC_HELD_PAST_EMITTED_FRAMES.with(|c| {
            let first = c.take().unwrap_or(p);
            c.set(Some(first));
        });
    }
}

pub(crate) fn resume_panic() {
    if let Some(p) = FIRST_PANIC_HELD_PAST_EMITTED_FRAMES.with(|c| c.take()) {
        std::panic::resume_unwind(p);
    }
}

unsafe fn layout_regions<'a>(
    layout: &Layout,
    work: *mut f64,
    inputs: *const f64,
) -> (Shared, Shared, &'a [f64]) {
    let (g, s) = (layout.gather_start, layout.scratch_start);
    unsafe {
        (
            Shared::from_raw(work, g),
            Shared::from_raw(work.add(g), s - g),
            std::slice::from_raw_parts(inputs, layout.n_inputs),
        )
    }
}

pub(crate) extern "C" fn h_call(
    layout: *const Layout,
    c: *const Call<'static>,
    work: *mut f64,
    inputs: *const f64,
) {
    let (l, c) = unsafe { (&*layout, &*c) };
    // SAFETY: `work` and `inputs` are the program's buffers, laid out as
    // `layout`, and the call's regions are disjoint by the layout.
    hold_panic_past_emitted_frames(|| unsafe {
        let (slots, gather, inputs) = layout_regions(l, work, inputs);
        let scratch = std::slice::from_raw_parts_mut(work.add(l.scratch_start), l.scratch_len);
        run_call(c, slots, gather, inputs, 0..c.n_groups, scratch)
    });
}

pub(crate) extern "C" fn h_stage(
    layout: *const Layout,
    c: *const Call<'static>,
    n: u64,
    work: *mut f64,
    inputs: *const f64,
) {
    let (l, calls) = unsafe { (&*layout, std::slice::from_raw_parts(c, n as usize)) };
    // SAFETY: `work` and `inputs` are the program's buffers, laid out as
    // `layout`; the calls of a stage read nothing another one writes
    // (`rsdag`'s stage planning), their arguments apart.
    hold_panic_past_emitted_frames(|| unsafe {
        let (slots, gather, inputs) = layout_regions(l, work, inputs);
        run_stage(calls.len(), |k| calls[k], slots, gather, inputs)
    });
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Place {
    pub(crate) base: u64,
    pub(crate) byte_offset: u64,
}

impl Place {
    pub(crate) const WORK: u64 = 0;
    pub(crate) const INPUTS: u64 = 1;
    const UNREAD_ACCUMULATOR: u64 = 2;
    pub(crate) const NONE: Place = Place {
        base: Self::UNREAD_ACCUMULATOR,
        byte_offset: 0,
    };
}

#[repr(C)]
pub(crate) struct KernelDesc {
    pub(crate) kind: u64,
    pub(crate) m: u64,
    pub(crate) k: u64,
    pub(crate) n: u64,
    pub(crate) operands_a_xb_accumulator: [Place; KernelDesc::OPERANDS],
    pub(crate) fold_codes_addr_or_zero: u64,
    pub(crate) out_byte_offset: u64,
}

impl KernelDesc {
    pub(crate) const OPERANDS: usize = 3;
    pub(crate) const GEMV: u64 = 0;
    pub(crate) const GEMM: u64 = 1;
    pub(crate) const SOLVE_M_SYSTEMS_K_RHS: u64 = 2;
}

pub(crate) extern "C" fn h_kernel(
    layout: *const Layout,
    d: *const KernelDesc,
    work: *mut f64,
    inputs: *const f64,
) {
    use rsdag::semantics as s;
    let (l, d) = unsafe { (&*layout, &*d) };
    let (m, k, n) = (d.m as usize, d.k as usize, d.n as usize);
    // SAFETY: the code placed every operand, in the work array or the
    // inputs, and the output block apart from all of them (a kernel's
    // outputs are fresh slots); the lengths are the kernel's.
    let operand = |i: usize, len: usize| -> Option<&[f64]> {
        let p = d.operands_a_xb_accumulator[i];
        let base = match p.base {
            Place::WORK => work as *const f64,
            Place::INPUTS => inputs,
            _ => return None,
        };
        Some(unsafe { std::slice::from_raw_parts(base.byte_add(p.byte_offset as usize), len) })
    };
    let len_out = match d.kind {
        KernelDesc::GEMV => m,
        KernelDesc::GEMM => m * n,
        _ => m * n * k,
    };
    let out = unsafe {
        std::slice::from_raw_parts_mut(work.byte_add(d.out_byte_offset as usize), len_out)
    };
    let codes = (d.fold_codes_addr_or_zero != 0).then(|| unsafe {
        std::slice::from_raw_parts(d.fold_codes_addr_or_zero as *const u32, len_out)
    });
    let (a, b) = match d.kind {
        KernelDesc::GEMV => (operand(0, m * n), operand(1, n)),
        KernelDesc::GEMM => (operand(0, m * k), operand(1, n * k)),
        _ => (operand(0, m * n * n), operand(1, m * n * k)),
    };
    let (a, b) = (a.expect("a kernel's first operand"), b.expect("its second"));
    match (d.kind, codes) {
        (KernelDesc::GEMV, None) => s::gemv(a, b, m, n, out),
        (KernelDesc::GEMV, Some(codes)) => s::gemv_fold(a, b, m, n, operand(2, m), codes, out),
        (KernelDesc::GEMM, None) => s::gemm(a, b, m, k, n, out),
        (KernelDesc::GEMM, Some(codes)) => {
            s::gemm_fold(a, b, m, k, n, operand(2, m * n), codes, out)
        }
        _ => {
            let scratch =
                unsafe { std::slice::from_raw_parts_mut(work.add(l.scratch_start), l.scratch_len) };
            s::solve_batch_into(a, b, n, k, m, out, scratch)
        }
    }
}
