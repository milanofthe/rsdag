//! The `f64` twins of the dot-fold kernels: the same fold as the generic
//! reference in [`crate::semantics`] (four accumulators, merged as
//! `(0 + 1) + (2 + 3)`, then the tail in order), as two-lane vectors where
//! the target has them. Bit-identical to the reference by construction:
//! lane `l` of the vector pair is accumulator `l`, every product and every
//! sum rounds once, nothing is fused. The generic reference stays the
//! definition; the tests hold the twins to it.

/// Two `f64` lanes: NEON on AArch64, SSE2 on x86-64 (part of the base
/// architecture, so no dispatch), two scalars elsewhere.
#[derive(Clone, Copy)]
struct V2(Inner);

#[cfg(target_arch = "aarch64")]
type Inner = std::arch::aarch64::float64x2_t;
#[cfg(target_arch = "x86_64")]
type Inner = std::arch::x86_64::__m128d;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
type Inner = [f64; 2];

#[cfg(target_arch = "aarch64")]
impl V2 {
    #[inline(always)]
    fn zero() -> V2 {
        unsafe { V2(std::arch::aarch64::vdupq_n_f64(0.0)) }
    }
    /// # Safety
    /// `p` points at two readable `f64`.
    #[inline(always)]
    unsafe fn load(p: *const f64) -> V2 {
        V2(std::arch::aarch64::vld1q_f64(p))
    }
    #[inline(always)]
    fn mul(self, o: V2) -> V2 {
        unsafe { V2(std::arch::aarch64::vmulq_f64(self.0, o.0)) }
    }
    #[inline(always)]
    fn add(self, o: V2) -> V2 {
        unsafe { V2(std::arch::aarch64::vaddq_f64(self.0, o.0)) }
    }
    #[inline(always)]
    fn lanes(self) -> [f64; 2] {
        unsafe {
            [
                std::arch::aarch64::vgetq_lane_f64(self.0, 0),
                std::arch::aarch64::vgetq_lane_f64(self.0, 1),
            ]
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl V2 {
    #[inline(always)]
    fn zero() -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_setzero_pd()) }
    }
    /// # Safety
    /// `p` points at two readable `f64`.
    #[inline(always)]
    unsafe fn load(p: *const f64) -> V2 {
        V2(std::arch::x86_64::_mm_loadu_pd(p))
    }
    #[inline(always)]
    fn mul(self, o: V2) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_mul_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn add(self, o: V2) -> V2 {
        unsafe { V2(std::arch::x86_64::_mm_add_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn lanes(self) -> [f64; 2] {
        let mut out = [0.0; 2];
        unsafe { std::arch::x86_64::_mm_storeu_pd(out.as_mut_ptr(), self.0) };
        out
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
impl V2 {
    #[inline(always)]
    fn zero() -> V2 {
        V2([0.0; 2])
    }
    /// # Safety
    /// `p` points at two readable `f64`.
    #[inline(always)]
    unsafe fn load(p: *const f64) -> V2 {
        V2([*p, *p.add(1)])
    }
    #[inline(always)]
    fn mul(self, o: V2) -> V2 {
        V2([self.0[0] * o.0[0], self.0[1] * o.0[1]])
    }
    #[inline(always)]
    fn add(self, o: V2) -> V2 {
        V2([self.0[0] + o.0[0], self.0[1] + o.0[1]])
    }
    #[inline(always)]
    fn lanes(self) -> [f64; 2] {
        self.0
    }
}

/// The merge of the four accumulators and the tail past the last chunk
/// of four, in the reference order.
#[inline(always)]
fn finish(c0: V2, c1: V2, a: &[f64], b: &[f64], from: usize) -> f64 {
    let [l0, l1] = c0.lanes();
    let [l2, l3] = c1.lanes();
    let mut s = (l0 + l1) + (l2 + l3);
    for l in from..a.len() {
        s += a[l] * b[l];
    }
    s
}

/// [`crate::semantics::dot_slice_t`] in `f64`.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    assert!(b.len() >= n, "dot operands of one length");
    let ch = n / 4;
    let (mut c0, mut c1) = (V2::zero(), V2::zero());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    for c in 0..ch {
        let o = 4 * c;
        // SAFETY: `o + 3 < 4 * ch <= n <= a.len(), b.len()`.
        unsafe {
            c0 = c0.add(V2::load(pa.add(o)).mul(V2::load(pb.add(o))));
            c1 = c1.add(V2::load(pa.add(o + 2)).mul(V2::load(pb.add(o + 2))));
        }
    }
    finish(c0, c1, a, b, ch * 4)
}

/// [`crate::semantics::gemv_t`] in `f64`: four rows at a time, each row
/// its own accumulator pair, the vector loaded once per chunk for all
/// four.
pub(crate) fn gemv(a: &[f64], x: &[f64], m: usize, n: usize, out: &mut [f64]) {
    assert!(a.len() >= m * n && x.len() >= n && out.len() >= m);
    let ch = n / 4;
    let px = x.as_ptr();
    let mut i = 0;
    while i + 4 <= m {
        let p = [
            a[i * n..].as_ptr(),
            a[(i + 1) * n..].as_ptr(),
            a[(i + 2) * n..].as_ptr(),
            a[(i + 3) * n..].as_ptr(),
        ];
        let mut acc = [[V2::zero(); 2]; 4];
        for c in 0..ch {
            let o = 4 * c;
            // SAFETY: row `r` starts at `(i + r) * n` with `n` elements,
            // `o + 3 < n`; `x` has `n`.
            unsafe {
                let x0 = V2::load(px.add(o));
                let x1 = V2::load(px.add(o + 2));
                for (r, ar) in acc.iter_mut().enumerate() {
                    ar[0] = ar[0].add(V2::load(p[r].add(o)).mul(x0));
                    ar[1] = ar[1].add(V2::load(p[r].add(o + 2)).mul(x1));
                }
            }
        }
        for (r, ar) in acc.iter().enumerate() {
            out[i + r] = finish(ar[0], ar[1], &a[(i + r) * n..][..n], x, ch * 4);
        }
        i += 4;
    }
    for r in i..m {
        out[r] = dot(&a[r * n..][..n], x);
    }
}

/// [`crate::semantics::gemm_t`] in `f64`: four rows of `a` against two
/// rows of `b` at a time, sixteen accumulator pairs.
pub(crate) fn gemm(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, out: &mut [f64]) {
    assert!(a.len() >= m * k && b.len() >= n * k && out.len() >= m * n);
    let ch = k / 4;
    let mut i = 0;
    while i + 4 <= m {
        let pa = [
            a[i * k..].as_ptr(),
            a[(i + 1) * k..].as_ptr(),
            a[(i + 2) * k..].as_ptr(),
            a[(i + 3) * k..].as_ptr(),
        ];
        let mut j = 0;
        while j + 2 <= n {
            let pb0 = b[j * k..].as_ptr();
            let pb1 = b[(j + 1) * k..].as_ptr();
            let mut acc = [[[V2::zero(); 2]; 2]; 4];
            for c in 0..ch {
                let o = 4 * c;
                // SAFETY: every row has `k` elements and `o + 3 < k`.
                unsafe {
                    let b00 = V2::load(pb0.add(o));
                    let b01 = V2::load(pb0.add(o + 2));
                    let b10 = V2::load(pb1.add(o));
                    let b11 = V2::load(pb1.add(o + 2));
                    for (r, ar) in acc.iter_mut().enumerate() {
                        let a0 = V2::load(pa[r].add(o));
                        let a1 = V2::load(pa[r].add(o + 2));
                        ar[0][0] = ar[0][0].add(a0.mul(b00));
                        ar[0][1] = ar[0][1].add(a1.mul(b01));
                        ar[1][0] = ar[1][0].add(a0.mul(b10));
                        ar[1][1] = ar[1][1].add(a1.mul(b11));
                    }
                }
            }
            for (r, ar) in acc.iter().enumerate() {
                for (q, aq) in ar.iter().enumerate() {
                    out[(i + r) * n + j + q] = finish(
                        aq[0],
                        aq[1],
                        &a[(i + r) * k..][..k],
                        &b[(j + q) * k..][..k],
                        ch * 4,
                    );
                }
            }
            j += 2;
        }
        for r in 0..4 {
            for jj in j..n {
                out[(i + r) * n + jj] = dot(&a[(i + r) * k..][..k], &b[jj * k..][..k]);
            }
        }
        i += 4;
    }
    for r in i..m {
        for j in 0..n {
            out[r * n + j] = dot(&a[r * k..][..k], &b[j * k..][..k]);
        }
    }
}
