//! Host routines: what the emitted code calls for everything that is not
//! an instruction. Every transcendental goes through the same
//! [`unary_f64`](rsdag::semantics::unary_f64) as the interpreter, so the domain
//! guards hold identically and the backends agree to the bit.

use rsdag::extern_fn::ExternBundle;
use rsdag::node::{BinOp, ReduceOp, UnaryOp};
use rsdag::semantics::{binary_f64, reduce_slice, unary_f64};
use rsdag::tape::calls::{run_call, run_stage, Call, Phase, Shared};
use std::sync::Arc;

#[cfg(target_arch = "x86_64")]
use crate::constants::MAX_LANES;

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

/// The exponential (logarithm) of the `n` lanes of a vector at `p`, in
/// place, each as `unary_f64` computes it: lane code makes one call per op,
/// not one per lane, four lanes in one vector where AVX2 is.
pub(crate) extern "C" fn h_exp_lanes(p: *mut f64, n: u64) {
    // SAFETY: the emitted code passes its gather area, `n` lanes long.
    let xs = unsafe { std::slice::from_raw_parts_mut(p, n as usize) };
    #[cfg(target_arch = "x86_64")]
    if xs.len() == MAX_LANES && crate::x86_64::features().avx2 {
        // SAFETY: AVX2 is there, and four lanes at `p`.
        return unsafe { lanes4::exp(p) };
    }
    for x in xs {
        *x = rsdag::semantics::exp_guarded(*x);
    }
}
pub(crate) extern "C" fn h_ln_lanes(p: *mut f64, n: u64) {
    // SAFETY: as for `h_exp_lanes`.
    let xs = unsafe { std::slice::from_raw_parts_mut(p, n as usize) };
    #[cfg(target_arch = "x86_64")]
    if xs.len() == MAX_LANES && crate::x86_64::features().avx2 {
        // SAFETY: AVX2 is there, and four lanes at `p`.
        return unsafe { lanes4::ln(p) };
    }
    for x in xs {
        *x = rsdag::semantics::ln_guarded(*x);
    }
}

/// `rsdag::semantics::exp_guarded` and `ln_guarded` on four lanes at once:
/// the same IEEE operations, step for step (the tests below hold
/// them to the bit). AVX2 has no 64-bit arithmetic shift, so `exp`'s `k >>
/// 1` is `floor(k / 2)` through the rounding constant, the same integer.
#[cfg(target_arch = "x86_64")]
pub(crate) mod lanes4 {
    use rsdag::semantics::exp_ln::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn c(v: f64) -> __m256d {
        _mm256_set1_pd(v)
    }

    #[inline(always)]
    unsafe fn ci(v: i64) -> __m256i {
        _mm256_set1_epi64x(v)
    }

    /// `2^k` for each lane's `-1022 <= k <= 1023`.
    #[inline(always)]
    unsafe fn pow2(k: __m256i) -> __m256d {
        unsafe {
            _mm256_castsi256_pd(_mm256_slli_epi64::<{ F64_FRAC_BITS as i32 }>(
                _mm256_add_epi64(k, ci(F64_EXP_BIAS)),
            ))
        }
    }

    /// # Safety
    /// AVX2, and four lanes at `p`.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn exp(p: *mut f64) {
        unsafe {
            let x = _mm256_loadu_pd(p);
            let t = _mm256_add_pd(_mm256_mul_pd(x, c(INV_LN2)), c(MAGIC));
            let kf = _mm256_sub_pd(t, c(MAGIC));
            let magic = ci(MAGIC.to_bits() as i64);
            let k = _mm256_sub_epi64(_mm256_castpd_si256(t), magic);
            let hi = _mm256_sub_pd(x, _mm256_mul_pd(kf, c(LN2_HI)));
            let lo = _mm256_mul_pd(kf, c(LN2_LO));
            let r = _mm256_sub_pd(hi, lo);
            let t2 = _mm256_mul_pd(r, r);
            let mut q = _mm256_add_pd(c(EP4), _mm256_mul_pd(t2, c(EP5)));
            q = _mm256_add_pd(c(EP3), _mm256_mul_pd(t2, q));
            q = _mm256_add_pd(c(EP2), _mm256_mul_pd(t2, q));
            q = _mm256_add_pd(c(EP1), _mm256_mul_pd(t2, q));
            let cc = _mm256_sub_pd(r, _mm256_mul_pd(t2, q));
            let frac = _mm256_div_pd(_mm256_mul_pd(r, cc), _mm256_sub_pd(c(2.0), cc));
            let y = _mm256_sub_pd(c(1.0), _mm256_sub_pd(_mm256_sub_pd(lo, frac), hi));
            let k1f = _mm256_floor_pd(_mm256_mul_pd(kf, c(0.5)));
            let k1 = _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(k1f, c(MAGIC))), magic);
            let k2 = _mm256_sub_epi64(k, k1);
            let mut v = _mm256_mul_pd(_mm256_mul_pd(y, pow2(k1)), pow2(k2));
            let under = _mm256_cmp_pd::<_CMP_LT_OQ>(x, c(EXP_UNDERFLOW));
            v = _mm256_blendv_pd(v, c(0.0), under);
            let nan = _mm256_cmp_pd::<_CMP_UNORD_Q>(x, x);
            v = _mm256_blendv_pd(v, x, nan);
            let tail = _mm256_mul_pd(
                c(EXP_AT_LIMIT),
                _mm256_add_pd(c(1.0), _mm256_sub_pd(x, c(EXP_LIMIT))),
            );
            let above = _mm256_cmp_pd::<_CMP_GT_OQ>(x, c(EXP_LIMIT));
            _mm256_storeu_pd(p, _mm256_blendv_pd(v, tail, above));
        }
    }

    /// # Safety
    /// AVX2, and four lanes at `p`.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn ln(p: *mut f64) {
        unsafe {
            let x = _mm256_loadu_pd(p);
            let bits = _mm256_castpd_si256(x);
            const HIGH: i32 = F64_HIGH_WORD as i32;
            const HIGH_FRAC: i32 = F64_HIGH_FRAC_BITS as i32;
            let high = _mm256_srli_epi64::<HIGH>(bits);
            let hm = _mm256_and_si256(high, ci(LN_HIGH_FRAC_MASK));
            let i = _mm256_and_si256(_mm256_add_epi64(hm, ci(LN_SQRT2_CARRY)), ci(LN_HIGH_UNIT));
            let e = _mm256_sub_epi64(_mm256_srli_epi64::<HIGH_FRAC>(high), ci(F64_EXP_BIAS));
            let k = _mm256_add_epi64(e, _mm256_srli_epi64::<HIGH_FRAC>(i));
            let m_hi = _mm256_or_si256(hm, _mm256_xor_si256(i, ci(LN_HIGH_ONE)));
            let low = _mm256_and_si256(bits, ci(F64_LOW_WORD_MASK as i64));
            let m = _mm256_castsi256_pd(_mm256_or_si256(_mm256_slli_epi64::<HIGH>(m_hi), low));
            let f = _mm256_sub_pd(m, c(1.0));
            let s = _mm256_div_pd(f, _mm256_add_pd(c(2.0), f));
            let magic = ci(MAGIC.to_bits() as i64);
            let dk = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(magic, k)), c(MAGIC));
            let z = _mm256_mul_pd(s, s);
            let w = _mm256_mul_pd(z, z);
            let mut t1 = _mm256_add_pd(c(LG4), _mm256_mul_pd(w, c(LG6)));
            t1 = _mm256_mul_pd(w, _mm256_add_pd(c(LG2), _mm256_mul_pd(w, t1)));
            let mut t2 = _mm256_add_pd(c(LG5), _mm256_mul_pd(w, c(LG7)));
            t2 = _mm256_add_pd(c(LG3), _mm256_mul_pd(w, t2));
            t2 = _mm256_mul_pd(z, _mm256_add_pd(c(LG1), _mm256_mul_pd(w, t2)));
            let r = _mm256_add_pd(t2, t1);
            let hfsq = _mm256_mul_pd(_mm256_mul_pd(c(0.5), f), f);
            let inner = _mm256_add_pd(
                _mm256_mul_pd(s, _mm256_add_pd(hfsq, r)),
                _mm256_mul_pd(dk, c(LN2_LO)),
            );
            let v = _mm256_sub_pd(
                _mm256_mul_pd(dk, c(LN2_HI)),
                _mm256_sub_pd(_mm256_sub_pd(hfsq, inner), f),
            );
            let keep = _mm256_or_pd(
                _mm256_cmp_pd::<_CMP_UNORD_Q>(x, x),
                _mm256_cmp_pd::<_CMP_EQ_OQ>(x, c(f64::INFINITY)),
            );
            let v = _mm256_blendv_pd(v, x, keep);
            let floor = _mm256_cmp_pd::<_CMP_LE_OQ>(x, c(LN_FLOOR));
            _mm256_storeu_pd(p, _mm256_blendv_pd(v, c(LN_AT_FLOOR), floor));
        }
    }
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

// A panic out of a host call, held until the emitted code has returned:
// unwinding cannot cross the emitted frames (they carry no unwind
// tables, and an `extern "C"` boundary aborts), so a trampoline catches
// it, the chunk runs to its end on whatever the failed call left in its
// outputs, and `resume_panic` raises it again on the caller's side. The
// first panic of an evaluation is the one kept.
std::thread_local! {
    static PANIC: std::cell::Cell<Option<Box<dyn std::any::Any + Send>>> =
        const { std::cell::Cell::new(None) };
}

/// Run `f`, holding a panic out of it for [`resume_panic`].
fn guarded(f: impl FnOnce()) {
    if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        PANIC.with(|c| {
            let first = c.take().unwrap_or(p);
            c.set(Some(first));
        });
    }
}

/// Raise a panic a host call held while the emitted code ran.
pub(crate) fn resume_panic() {
    if let Some(p) = PANIC.with(|c| c.take()) {
        std::panic::resume_unwind(p);
    }
}

/// A call site as the emitted code hands it to [`h_call`]: every size
/// fixed when the tape was compiled, every place an index into the work
/// array's regions: its slots `0..gather`, the gather area
/// `gather..scratch` and the bundles' scratch from `scratch`. The
/// descriptors live as long as the code that points at them.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct CallDesc {
    pub(crate) bundle: u64,
    /// 0 whole, 1 the main phase, 2 the prolog.
    pub(crate) kind: u64,
    pub(crate) n_groups: u64,
    pub(crate) n_args: u64,
    pub(crate) n_in: u64,
    pub(crate) n_out: u64,
    /// The arguments in the gather area; the outputs (the states for a
    /// prolog) and the states (a main phase) among the slots.
    pub(crate) args: u64,
    pub(crate) out: u64,
    pub(crate) state: u64,
    pub(crate) gather: u64,
    pub(crate) scratch: u64,
    pub(crate) scratch_len: u64,
    /// Ops the call carries (its instances times its body's), for
    /// [`h_stage`]'s choice of running in parallel.
    pub(crate) ops: u64,
    /// The operands the arguments are gathered from (a slot, or an input
    /// tagged as in the tape), for a call too wide to gather in its code;
    /// `0` when the code gathered them.
    pub(crate) table: u64,
    /// The argument each of an instance's `n_in` operands is, for a call
    /// that gathers only some; `0` for all, in order.
    pub(crate) places: u64,
    /// The inputs the operands read reach this far.
    pub(crate) n_inputs: u64,
}

/// `d` as the tape's call runner takes it.
///
/// # Safety
/// `d` describes a call of the code `bundles` belong to.
unsafe fn call_of<'a>(bundles: &'a Bundles, d: &'a CallDesc) -> Call<'a> {
    let (ng, ni) = (d.n_groups as usize, d.n_in as usize);
    let table = |p: u64, n: usize| unsafe { std::slice::from_raw_parts(p as *const u32, n) };
    Call {
        bundle: &*bundles[d.bundle as usize],
        phase: match d.kind {
            0 => Phase::Whole,
            1 => Phase::Main,
            _ => Phase::Prolog,
        },
        n_groups: ng,
        n_args: d.n_args as usize,
        n_out: d.n_out as usize,
        operands: (d.table != 0).then(|| table(d.table, ng * ni)),
        places: (d.places != 0).then(|| table(d.places, ni)),
        args: d.args as usize,
        out: d.out as usize,
        state: d.state as usize,
        ops: d.ops,
    }
}

/// The regions of the work array `d` names, and the inputs its operands
/// read.
///
/// # Safety
/// `work` and `inputs` are the program's buffers.
unsafe fn regions<'a>(
    d: &CallDesc,
    work: *mut f64,
    inputs: *const f64,
) -> (Shared<f64>, Shared<f64>, &'a [f64]) {
    let (g, s) = (d.gather as usize, d.scratch as usize);
    unsafe {
        (
            Shared::from_raw(work, g),
            Shared::from_raw(work.add(g), s - g),
            std::slice::from_raw_parts(inputs, d.n_inputs as usize),
        )
    }
}

/// Every bundle call: whole, main phase over its states, or prolog.
pub(crate) extern "C" fn h_call(
    bundles: *const Bundles,
    d: *const CallDesc,
    work: *mut f64,
    inputs: *const f64,
) {
    let d = unsafe { &*d };
    let b = unsafe { &*bundles };
    // SAFETY: the call's regions are disjoint by the layout.
    guarded(|| unsafe {
        let c = call_of(b, d);
        let (slots, gather, inputs) = regions(d, work, inputs);
        let scratch =
            std::slice::from_raw_parts_mut(work.add(d.scratch as usize), d.scratch_len as usize);
        run_call(&c, slots, gather, inputs, 0..c.n_groups, scratch)
    });
}

/// The calls of a stage ([`rsdag::Stage`]): `n` descriptors from `d` on,
/// run by the tape's [`run_stage`].
pub(crate) extern "C" fn h_stage(
    bundles: *const Bundles,
    d: *const CallDesc,
    n: u64,
    work: *mut f64,
    inputs: *const f64,
) {
    let descs = unsafe { std::slice::from_raw_parts(d, n as usize) };
    let b = unsafe { &*bundles };
    guarded(|| unsafe {
        let n_inputs = descs.iter().map(|d| d.n_inputs).max().unwrap_or(0);
        let wide = CallDesc {
            n_inputs,
            ..descs[0]
        };
        let (slots, gather, inputs) = regions(&wide, work, inputs);
        // SAFETY: the calls of a stage read nothing another one writes
        // (`rsdag`'s stage planning), their arguments apart.
        let call = |k: usize| call_of(b, &descs[k]);
        run_stage(descs.len(), call, slots, gather, inputs)
    });
}

/// Where a kernel operand is: a byte offset into the work array or into
/// the inputs, or nowhere (an accumulator no fold reads).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Place {
    pub(crate) base: u64,
    pub(crate) off: u64,
}

impl Place {
    pub(crate) const WORK: u64 = 0;
    pub(crate) const INPUTS: u64 = 1;
    pub(crate) const NONE: Place = Place { base: 2, off: 0 };
}

/// A dense kernel as the code hands it over: its kind (0 `Gemv`, 1 `Gemm`,
/// 2 the solves of `k` right-hand sides, `m` systems of them), its dimensions, where its
/// operands are (`a`, then `x` or `b`, then the accumulator), the address
/// of its fold codes (0 without), the byte offset of its outputs, and
/// the scratch a solve works in (the layout's, apart from slots and gather).
#[repr(C)]
pub(crate) struct KernelDesc {
    pub(crate) kind: u64,
    pub(crate) m: u64,
    pub(crate) k: u64,
    pub(crate) n: u64,
    pub(crate) operands: [Place; 3],
    pub(crate) codes: u64,
    pub(crate) out: u64,
    /// The byte offset and the length of the scratch a solve works in.
    pub(crate) scratch: u64,
    pub(crate) scratch_len: u64,
}

/// Run a dense kernel through the reference kernels of `rsdag::semantics`.
pub(crate) extern "C" fn h_kernel(work: *mut f64, inputs: *const f64, d: *const KernelDesc) {
    use rsdag::semantics as s;
    let d = unsafe { &*d };
    let (m, k, n) = (d.m as usize, d.k as usize, d.n as usize);
    // SAFETY: the code placed every operand, in the work array or the
    // inputs, and the output block apart from all of them (a kernel's
    // outputs are fresh slots); the lengths are the kernel's.
    let operand = |i: usize, len: usize| -> Option<&[f64]> {
        let p = d.operands[i];
        let base = match p.base {
            Place::WORK => work as *const f64,
            Place::INPUTS => inputs,
            _ => return None,
        };
        Some(unsafe { std::slice::from_raw_parts(base.byte_add(p.off as usize), len) })
    };
    let len_out = match d.kind {
        0 => m,
        1 => m * n,
        _ => m * n * k,
    };
    let out = unsafe { std::slice::from_raw_parts_mut(work.byte_add(d.out as usize), len_out) };
    let codes = (d.codes != 0)
        .then(|| unsafe { std::slice::from_raw_parts(d.codes as *const u32, len_out) });
    let (a, b) = match d.kind {
        0 => (operand(0, m * n), operand(1, n)),
        1 => (operand(0, m * k), operand(1, n * k)),
        _ => (operand(0, m * n * n), operand(1, m * n * k)),
    };
    let (a, b) = (a.expect("a kernel's first operand"), b.expect("its second"));
    match (d.kind, codes) {
        (0, None) => s::gemv(a, b, m, n, out),
        (0, Some(codes)) => s::gemv_fold(a, b, m, n, operand(2, m), codes, out),
        (1, None) => s::gemm(a, b, m, k, n, out),
        (1, Some(codes)) => s::gemm_fold(a, b, m, k, n, operand(2, m * n), codes, out),
        _ => {
            let scratch = unsafe {
                std::slice::from_raw_parts_mut(
                    work.byte_add(d.scratch as usize),
                    d.scratch_len as usize,
                )
            };
            s::solve_batch_into(a, b, n, k, m, out, scratch)
        }
    }
}

#[cfg(test)]
mod tests {
    use rsdag::semantics::{exp_guarded, ln_guarded};

    /// Values over every range the functions treat apart: the specials, the
    /// guards' edges, subnormal and huge magnitudes, and a log-uniform stream.
    fn values() -> Vec<f64> {
        let mut v = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            5e-324,
            80.0,
            80.0000001,
            79.9999999,
            -745.2,
            -745.1,
            -708.4,
            709.0,
            1e-30,
            1e-30 * (1.0 + f64::EPSILON),
            1e300,
            f64::MAX,
        ];
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..400_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            let sign = if s & 1 == 0 { 1.0 } else { -1.0 };
            v.push(sign * 10f64.powf(-320.0 + 640.0 * u));
            v.push(-800.0 + 900.0 * u);
        }
        v
    }

    #[test]
    fn four_lanes_are_the_scalar_bit_for_bit() {
        if !crate::x86_64::features().avx2 {
            return;
        }
        let v = values();
        for (name, lanes, one) in [
            (
                "exp",
                super::h_exp_lanes as extern "C" fn(*mut f64, u64),
                exp_guarded as fn(f64) -> f64,
            ),
            ("ln", super::h_ln_lanes, ln_guarded),
        ] {
            let mut w = v.clone();
            w.truncate(w.len() / 4 * 4);
            for c in w.as_chunks_mut::<4>().0 {
                lanes(c.as_mut_ptr(), 4);
            }
            for (x, y) in v.iter().zip(&w) {
                let r = one(*x);
                assert!(
                    r.to_bits() == y.to_bits() || (r.is_nan() && y.is_nan()),
                    "{name}({x:e}): lanes {y:e}, scalar {r:e}"
                );
            }
        }
    }
}
