"""Write crates/rsdag/src/math/table.rs: 2^(i/128) with its tail, and the
reduction constants of rsdag::math::exp, computed with mpmath at 200 bits:

    python3 scripts/exp_table.py > crates/rsdag/src/math/table.rs.body

the file body after its doc comment."""

import mpmath, struct
mpmath.mp.prec = 200
N = 128
def bits(x): return struct.unpack('<Q', struct.pack('<d', x))[0]
def dbl(m): return float(m)  # round to nearest
rows = []
for i in range(N):
    e = mpmath.power(2, mpmath.mpf(i) / N)
    s = dbl(e)
    tail = dbl((e - mpmath.mpf(s)) / mpmath.mpf(s))
    rows.append((bits(tail), (bits(s) - (i << 45)) & (2**64 - 1)))
ln2 = mpmath.log(2)
ln2n = ln2 / N
# hi with 36 significant bits so k*hi is exact for |k| < 2^17
m, ex = mpmath.frexp(ln2n)
hi = mpmath.ldexp(mpmath.floor(mpmath.ldexp(m, 36)), ex - 36)
lo = dbl(ln2n - hi)
hi = dbl(hi)
inv = dbl(N / ln2)
print(f"pub(super) const INV_LN2_N: f64 = {inv!r};")
print(f"pub(super) const NEG_LN2_HI_N: f64 = {-hi!r};")
print(f"pub(super) const NEG_LN2_LO_N: f64 = {-lo!r};")
print("pub(super) static EXP: [u64; 256] = [")
for t, sb in rows:
    print(f"    0x{t:016x},")
    print(f"    0x{sb:016x},")
print("];")
