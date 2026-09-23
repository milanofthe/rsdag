"""Write crates/rsdag/tests/data/math_ref.txt: arguments of rsdag's own
elementary functions and their exact values as double-double (mpmath at 160
bits), one line per point: `name x hi lo` as hex bit patterns."""

import random
import struct

import mpmath

mpmath.mp.prec = 160
random.seed(7)
N = 600


def bits(d):
    return struct.unpack("<Q", struct.pack("<d", d))[0]


def logu(lo, hi):
    return 10 ** random.uniform(lo, hi)


def sign():
    return random.choice([-1, 1])


gens = {
    "exp": lambda: random.choice([
        random.uniform(-700, 700), random.uniform(-1, 1), random.uniform(-0.01, 0.01),
        sign() * logu(-20, 0), random.uniform(-745.13, -700), random.uniform(700, 709.78),
    ]),
    "ln": lambda: random.choice([
        logu(-307, 308), random.uniform(0.5, 2), 1 + sign() * logu(-15, -1), random.uniform(1e-320, 1e-308),
    ]),
    "tanh": lambda: sign() * random.choice([logu(-10, 1.4), random.uniform(0.2, 0.7), random.uniform(0, 3)]),
    "sinh": lambda: sign() * random.choice([logu(-10, 1.4), random.uniform(0.3, 1.2), random.uniform(20, 24), random.uniform(700, 710.4)]),
    "cosh": lambda: sign() * random.choice([logu(-10, 1.4), random.uniform(0.3, 1.2), random.uniform(20, 24), random.uniform(700, 710.4)]),
}
fns = {"exp": mpmath.exp, "ln": mpmath.log, "tanh": mpmath.tanh, "sinh": mpmath.sinh, "cosh": mpmath.cosh}

with open("crates/rsdag/tests/data/math_ref.txt", "w") as f:
    for name, gen in gens.items():
        for _ in range(N):
            x = float(gen())
            v = fns[name](mpmath.mpf(x))
            hi = float(v)
            lo = float(v - mpmath.mpf(hi))
            f.write(f"{name} {bits(x):x} {bits(hi):x} {bits(lo):x}\n")
