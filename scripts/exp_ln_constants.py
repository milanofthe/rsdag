"""The constants of rsdag's exponential and logarithm.

Writes crates/rsdag/src/constants/exp_ln.rs: the tables, the reduction
constants and the polynomial coefficients that `rsdag::semantics::exp_core`
and `ln_core` and every backend's encoding of them read. Computed with
mpmath at 256 bits; every double is the round-to-nearest of its exact value
unless the comment in the output says otherwise.

    python3 scripts/exp_ln_constants.py > crates/rsdag/src/constants/exp_ln.rs
"""

import struct
import sys

import mpmath as mp

mp.mp.prec = 256

# exp: x = k ln2 / N + r, exp(x) = 2^(k / N) exp(r); 2^(j / N) from a table.
EXP_BITS = 7
EXP_N = 1 << EXP_BITS
# The scale is built 2^EXP_BIAS too large and the result multiplied back
# once, so the scale stays normal down to the subnormal results.
EXP_BIAS = 64
EXP_DEGREE = 5  # exp(r) - 1 - r = r^2 (c2 + c3 r + c4 r^2 + c5 r^3)

# ln: x = 2^k z, z in [LN_OFF, 2 LN_OFF), log z = -log(invc) + log1p(r) with
# r = (z - c) invc, c and invc = 1/c from a table of LN_N subintervals.
LN_BITS = 7
LN_N = 1 << LN_BITS
LN_OFF = 0x3FE6_0000_0000_0000  # 0.6875
# Subintervals at 1 take c = 1 (r = z - 1, exact) from 1 - 2^-6 to 1 + 2^-6:
# the result there is about r, and r's rounding elsewhere is small against it.
LN_CENTER = mp.mpf(2) ** -6
LN_DEGREE = 9  # log1p(r) = r - r^2/2 + r^3 (a3 + a4 r + ... + a9 r^6)
# |c invc - 1| below this, so that r = (z - c) invc misses no term.
LN_CINV_EPS = mp.mpf(2) ** -72


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def dbl(x):
    """Round to nearest double."""
    return float(mp.mpf(x))


def round_to(x, lsb_exp):
    """x rounded to a multiple of 2^lsb_exp."""
    s = mp.mpf(2) ** lsb_exp
    return float(mp.nint(x / s) * s)


def ulp(x):
    m, e = mp.frexp(x)
    return mp.mpf(2) ** (e - 53)


def chebfit(f, R, deg):
    """Coefficients of the degree `deg` polynomial interpolating f at the
    Chebyshev nodes of [-R, R], low first, and its max error on a fine grid."""
    n = deg + 1
    nodes = [R * mp.cos(mp.pi * (2 * k + 1) / (2 * n)) for k in range(n)]
    A = mp.matrix([[t**j for j in range(n)] for t in nodes])
    c = mp.lu_solve(A, mp.matrix([f(t) for t in nodes]))
    return [c[j] for j in range(n)]


def max_err(f, p, R, pts=2001):
    worst = mp.mpf(0)
    for k in range(pts):
        t = -R + 2 * R * k / (pts - 1)
        if t == 0:
            continue
        v = sum(pj * t**j for j, pj in enumerate(p))
        worst = max(worst, abs(v - f(t)))
    return worst


def exp_constants():
    ln2 = mp.log(2)
    out = {}
    out["EXP_INV_LN2_N"] = dbl(EXP_N / ln2)
    # k ln2/N exact for |k| < 2^18 (|x| < 745.2 gives |k| < 137700): 35 bits.
    step = ln2 / EXP_N
    e = int(mp.floor(mp.log(step, 2)))
    hi = round_to(step, e - 34)
    out["EXP_LN2_HI_N"] = hi
    out["EXP_LN2_LO_N"] = dbl(step - hi)
    R = step / 2 * (1 + mp.mpf(2) ** -10)
    g = lambda t: (mp.exp(t) - 1 - t) / t**2 if abs(t) > 1e-30 else mp.mpf(1) / 2 + t / 6
    c = chebfit(g, R, EXP_DEGREE - 2)
    cd = [dbl(v) for v in c]
    rel = max(
        abs(1 + t + t**2 * sum(cj * t**j for j, cj in enumerate(cd)) - mp.exp(t)) / mp.exp(t)
        for t in [-R + 2 * R * k / 2000 for k in range(2001)]
    )
    table = []
    for j in range(EXP_N):
        v = mp.mpf(2) ** (mp.mpf(j) / EXP_N)
        s = dbl(v)
        tail = dbl((v - s) / s)
        sb = bits(s * 2.0**EXP_BIAS) - (j << (52 - EXP_BITS))
        table += [bits(tail), sb]
    return out, cd, rel, R, table


def ln_constants():
    ln2 = mp.log(2)
    out = {}
    # k ln2 exact for |k| <= 1075, and a multiple of 2^-43 as logc_hi.
    hi = round_to(ln2, -42)
    out["LN_LN2_HI"] = hi
    out["LN_LN2_LO"] = dbl(ln2 - hi)
    rows = []
    rmax = mp.mpf(0)
    for i in range(LN_N):
        lo_b = LN_OFF + (i << (52 - LN_BITS))
        hi_b = LN_OFF + ((i + 1) << (52 - LN_BITS))
        zlo = mp.mpf(struct.unpack("<d", struct.pack("<Q", lo_b))[0])
        zhi = mp.mpf(struct.unpack("<d", struct.pack("<Q", hi_b))[0])
        if zlo >= 1 - LN_CENTER and zhi <= 1 + LN_CENTER:
            rows.append((1.0, 1.0, 0.0, 0.0))
            rmax = max(rmax, abs(zlo - 1), abs(zhi - 1))
            continue
        m = (zlo + zhi) / 2
        # c near the middle with invc = RN(1/c) and c invc = 1 within
        # LN_CINV_EPS, checked exactly: c = cm 2^(e - 52), invc = im 2^(-e - 53)
        # with im = RN(2^105 / cm), c invc - 1 = (cm im - 2^105) / 2^105.
        c0 = bits(dbl(m))
        e = ((c0 >> 52) & 0x7FF) - 1023
        found = None
        for d in range(1 << 24):
            for s in (1, -1):
                cb = c0 + s * d
                cm = (cb & ((1 << 52) - 1)) | (1 << 52)
                im = ((1 << 106) // cm + 1) // 2
                if abs(cm * im - (1 << 105)) < (1 << 105) * LN_CINV_EPS:
                    c = struct.unpack("<d", struct.pack("<Q", cb))[0]
                    inv = float(mp.mpf(im) * mp.mpf(2) ** (-e - 53))
                    found = (c, inv)
                    break
            if found:
                break
        c, inv = found
        logc = -mp.log(mp.mpf(inv))
        lhi = round_to(logc, -43)
        llo = dbl(logc - lhi)
        rows.append((inv, c, lhi, llo))
        rmax = max(rmax, abs((zlo - c) * inv), abs((zhi - c) * inv))
    R = rmax * (1 + mp.mpf(2) ** -10)
    g = lambda t: (mp.log1p(t) - t + t**2 / 2) / t**3 if abs(t) > 1e-30 else mp.mpf(1) / 3 - t / 4
    a = chebfit(g, R, LN_DEGREE - 3)
    ad = [dbl(v) for v in a]
    rel = max(
        abs(t - t**2 / 2 + t**3 * sum(aj * t**j for j, aj in enumerate(ad)) - mp.log1p(t))
        / abs(mp.log1p(t))
        for t in [-R + 2 * R * k / 2000 for k in range(2001)]
        if t != 0
    )
    return out, ad, rel, R, rows


def lit(x):
    return f"f64::from_bits(0x{bits(x):016x})"


def main():
    eo, ec, erel, eR, etab = exp_constants()
    lo, la, lrel, lR, ltab = ln_constants()
    w = sys.stdout.write
    log2 = lambda v: mp.nstr(mp.log(v, 2), 4)
    w("""//! The exponential's and the logarithm's constants, generated by
//! `scripts/exp_ln_constants.py` (mpmath at 256 bits; every double the
//! nearest to its exact value unless said otherwise). Part of the contract:
//! [`exp_core`](crate::semantics::exp_core), [`ln_core`](crate::semantics::ln_core)
//! and every backend's encoding of them read these.
//!
//! `exp`: `x = k ln 2 / N + r` with `|r| <= ln 2 / 2N`, `exp(x) = 2^(k / N)
//! exp(r)`, `2^(j / N)` from a table (the method of Tang, ACM TOMS 15, 1989,
//! as ARM's optimized-routines and glibc take it). Error: below 0.51 ulp,
//! 0.76 ulp for subnormal results (rounded twice).
//!
//! `ln`: `x = 2^k z` with `z` in `[0.6875, 1.375)`, `log x = k ln 2 - log(invc)
//! + log1p(r)` with `r = (z - c) invc` for the `c` of `z`'s subinterval and
//! `invc = 1/c`. Near 1, `c = 1` and `r = z - 1` is exact; elsewhere `|log
//! x| > 2^-6` and the rounding of `r` costs at most 1/8 ulp. Error: below
//! 0.63 ulp.
//!
//! Without a fused multiply-add the same steps round twice; the errors hold
//! for both.

""")
    w(f"/// `x + MAGIC - MAGIC` rounds `x` (`|x| < 2^51`) to an integer, the low\n")
    w(f"/// bits of `x + MAGIC` that integer: `1.5 2^52`.\n")
    w(f"pub const MAGIC: f64 = {lit(1.5 * 2.0**52)};\n\n")
    w(f"/// `exp`'s table: `N = 2^{EXP_BITS}` subintervals.\n")
    w(f"pub const EXP_TABLE_BITS: u32 = {EXP_BITS};\n")
    w(f"/// `exp`'s scale is built `2^{EXP_BIAS}` too large, so it stays normal down\n")
    w("/// to the subnormal results, and the result multiplied back once.\n")
    w(f"pub const EXP_SCALE_BIAS: i64 = {EXP_BIAS};\n")
    w("/// `N / ln 2`.\n")
    w(f"pub const EXP_INV_LN2_N: f64 = {lit(eo['EXP_INV_LN2_N'])};\n")
    w("/// `ln 2 / N` in two parts, the high one of 35 significant bits, so `k\n")
    w("/// EXP_LN2_HI_N` is exact for every `|k| < 2^18` (`|x| < 745.2`).\n")
    w(f"pub const EXP_LN2_HI_N: f64 = {lit(eo['EXP_LN2_HI_N'])};\n")
    w("/// See [`EXP_LN2_HI_N`].\n")
    w(f"pub const EXP_LN2_LO_N: f64 = {lit(eo['EXP_LN2_LO_N'])};\n")
    w("/// `exp(r) = 1 + r + r^2 (c2 + c3 r + c4 r^2 + c5 r^3)`: interpolation at\n")
    w(f"/// the Chebyshev nodes of `[-{mp.nstr(eR, 6)}, {mp.nstr(eR, 6)}]`, relative error 2^{log2(erel)}\n")
    w("/// with the coefficients as doubles.\n")
    w("pub const EXP_POLY: [f64; 4] = [\n")
    for c in ec:
        w(f"    {lit(c)},\n")
    w("];\n")
    w("/// Per `j`: the bits of `tail_j` and of `2^(EXP_SCALE_BIAS) s_j` minus `j\n")
    w(f"/// << {52 - EXP_BITS}` (which the integer `k` shifted into the exponent adds\n")
    w("/// back), with `2^(j / N) = s_j (1 + tail_j)` and `s_j` the double nearest.\n")
    w("#[rustfmt::skip]\n")
    w(f"pub static EXP_TABLE: [u64; {2 * EXP_N}] = [\n")
    for j in range(EXP_N):
        w(f"    0x{etab[2 * j]:016x}, 0x{etab[2 * j + 1]:016x},\n")
    w("];\n")
    w("/// `exp(EXP_LIMIT)`, the slope of the linear tail.\n")
    w(f"pub const EXP_AT_LIMIT: f64 = {lit(dbl(mp.exp(80)))};\n\n")
    w(f"/// `ln`'s table: `N = 2^{LN_BITS}` subintervals of `z`.\n")
    w(f"pub const LN_TABLE_BITS: u32 = {LN_BITS};\n")
    w("/// The bits of `0.6875`, where `z`'s range starts.\n")
    w(f"pub const LN_OFF: u64 = 0x{LN_OFF:016x};\n")
    w("/// `ln 2` in two parts, the high one a multiple of `2^-42`, so `k LN_LN2_HI\n")
    w("/// + logc_hi` is exact for every `|k| <= 1075`.\n")
    w(f"pub const LN_LN2_HI: f64 = {lit(lo['LN_LN2_HI'])};\n")
    w("/// See [`LN_LN2_HI`].\n")
    w(f"pub const LN_LN2_LO: f64 = {lit(lo['LN_LN2_LO'])};\n")
    w("/// `log1p(r) = r - r^2/2 + r^3 (a3 + a4 r + ... + a9 r^6)`: interpolation\n")
    w(f"/// at the Chebyshev nodes of `[-{mp.nstr(lR, 6)}, {mp.nstr(lR, 6)}]`, relative error 2^{log2(lrel)}\n")
    w("/// with the coefficients as doubles.\n")
    w(f"pub const LN_POLY: [f64; {len(la)}] = [\n")
    for a in la:
        w(f"    {lit(a)},\n")
    w("];\n")
    w("/// Per subinterval: `invc`, `c`, `logc_hi`, `logc_lo`. `c` is a double near\n")
    w("/// the subinterval's middle for which `invc`, the double nearest `1/c`, has\n")
    w("/// `|c invc - 1| < 2^-72` (found by search), so `(z - c) invc` is `z invc -\n")
    w("/// 1` but for its rounding; `logc_hi + logc_lo = -log(invc)`, `logc_hi` a\n")
    w("/// multiple of `2^-43`. From `1 - 2^-6` to `1 + 2^-6`, `c = 1`.\n")
    w("#[rustfmt::skip]\n")
    w(f"pub static LN_TABLE: [[f64; 4]; {LN_N}] = [\n")
    for row in ltab:
        w("    [" + ", ".join(lit(v) for v in row) + "],\n")
    w("];\n")
    ln_floor = mp.mpf(1e-30)
    w("/// `ln(LN_FLOOR)`.\n")
    w(f"pub const LN_AT_FLOOR: f64 = {lit(dbl(mp.log(ln_floor)))};\n")


if __name__ == "__main__":
    main()
