"""rsdag against CasADi and JAX on one model, from Python, on one core.

    python docs/bench/compare.py > docs/bench/data/compare.csv

The model is the 1D Brusselator on a periodic grid of `n` points (2n states,
a sparse Jacobian of 8 nonzeros per row pair), written in each tool's own
idiom: numpy over rsdag tracers, CasADi SX vectors, jax.numpy. Per tool and
size: the setup to a first result of the right-hand side and of its Jacobian
(tracing, differentiation, compilation), then the time of one call of each.
rsdag and CasADi build the sparse Jacobian; JAX builds the dense one
(`jacfwd`) and stops at the largest size it is measured at. Needs rsdag,
casadi and jax.
"""
import os
import sys
import time

# One core for every tool, as rsdag's own benchmarks.
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["OMP_NUM_THREADS"] = "1"

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
import rsdag

jax.config.update("jax_enable_x64", True)

A, B = 1.0, 3.0
SIZES = [10, 30, 100, 300, 1000, 3000, 10000]
JAX_DENSE_MAX = 1000


def alpha(n):
    return 0.02 * (n + 1) ** 2


def brusselator_np(n):
    """numpy formulation, traced by rsdag."""
    a = alpha(n)

    def f(x):
        u, v = x[:n], x[n:]
        lap_u = np.roll(u, 1) - 2 * u + np.roll(u, -1)
        lap_v = np.roll(v, 1) - 2 * v + np.roll(v, -1)
        uuv = u * u * v
        return np.concatenate([A + uuv - (B + 1) * u + a * lap_u, B * u - uuv + a * lap_v])

    return f


def brusselator_jax(n):
    a = alpha(n)

    def f(x):
        u, v = x[:n], x[n:]
        lap_u = jnp.roll(u, 1) - 2 * u + jnp.roll(u, -1)
        lap_v = jnp.roll(v, 1) - 2 * v + jnp.roll(v, -1)
        uuv = u * u * v
        return jnp.concatenate([A + uuv - (B + 1) * u + a * lap_u, B * u - uuv + a * lap_v])

    return f


def brusselator_casadi(n):
    a = alpha(n)
    x = ca.SX.sym("x", 2 * n)
    u, v = x[:n], x[n:]

    def roll(w, k):
        """numpy.roll by k = +1 or -1."""
        return ca.vertcat(w[-1], w[:-1]) if k == 1 else ca.vertcat(w[1:], w[0])

    lap_u = roll(u, 1) - 2 * u + roll(u, -1)
    lap_v = roll(v, 1) - 2 * v + roll(v, -1)
    uuv = u * u * v
    return x, ca.vertcat(A + uuv - (B + 1) * u + a * lap_u, B * u - uuv + a * lap_v)


def per_call(fn, x):
    """Seconds per call: the best of five batches, a batch at least 50 ms."""
    reps = 1
    while True:
        t = time.perf_counter()
        for _ in range(reps):
            fn(x)
        dt = time.perf_counter() - t
        if dt > 0.05:
            break
        reps *= 4
    best = dt / reps
    for _ in range(4):
        t = time.perf_counter()
        for _ in range(reps):
            fn(x)
        best = min(best, (time.perf_counter() - t) / reps)
    return best


def timed(make):
    t = time.perf_counter()
    r = make()
    return r, time.perf_counter() - t


def x0(n):
    i = np.arange(n)
    return np.concatenate([1 + np.sin(2 * np.pi * i / n), np.full(n, 3.0)])


def first(make, x):
    """A compiled callable and its first result."""
    c = make()
    c(x)
    return c


def rsdag_rows(n, native):
    x = x0(n)
    f = brusselator_np(n)
    F, sf = timed(lambda: first(lambda: rsdag.jit(f, native=native), x))
    J, sj = timed(lambda: first(lambda: rsdag.jacobian(f, sparse=True, native=native), x))
    return sf, sj, per_call(F, x), per_call(J, x), len(J(x))


def casadi_rows(n):
    x = x0(n)

    def build():
        xs, fx = brusselator_casadi(n)
        F = ca.Function("F", [xs], [fx])
        F(x)
        return xs, fx, F

    (xs, fx, F), sf = timed(build)

    def build_j():
        Jx = ca.jacobian(fx, xs)
        J = ca.Function("J", [xs], [Jx])
        J(x)
        return J, Jx.nnz()

    (J, nnz), sj = timed(build_j)
    return sf, sj, per_call(F, x), per_call(J, x), nnz


def jax_rows(n):
    x = jnp.asarray(x0(n))
    f = brusselator_jax(n)
    F = jax.jit(f)
    _, sf = timed(lambda: F(x).block_until_ready())
    fcall = per_call(lambda y: F(y).block_until_ready(), x)
    if n > JAX_DENSE_MAX:
        return sf, None, fcall, None, None
    J = jax.jit(jax.jacfwd(f))
    _, sj = timed(lambda: J(x).block_until_ready())
    return sf, sj, fcall, per_call(lambda y: J(y).block_until_ready(), x), 4 * n * n


def check(n=10):
    """The three tools agree on the right-hand side and the Jacobian."""
    x = x0(n)
    f = brusselator_np(n)
    J = rsdag.jacobian(f, sparse=True)
    rows, cols = J.pattern(x)
    jr = np.zeros((2 * n, 2 * n))
    jr[rows, cols] = J(x)
    xs, fx = brusselator_casadi(n)
    fc = np.array(ca.Function("F", [xs], [fx])(x)).ravel()
    jc = np.array(ca.Function("J", [xs], [ca.jacobian(fx, xs)])(x))
    fj = np.asarray(brusselator_jax(n)(jnp.asarray(x)))
    jj = np.asarray(jax.jacfwd(brusselator_jax(n))(jnp.asarray(x)))
    fr = rsdag.jit(f)(x)
    for a, b in ((fr, fc), (fr, fj), (jr, jc), (jr, jj)):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12), np.abs(a - b).max()


def main():
    check()
    print("tool,states,setup_f_s,setup_j_s,call_f_us,call_j_us,j_entries")
    tools = [
        ("rsdag native", lambda n: rsdag_rows(n, True)),
        ("rsdag interpreter", lambda n: rsdag_rows(n, False)),
        ("CasADi", casadi_rows),
        ("JAX", jax_rows),
    ]
    for n in SIZES:
        for name, run in tools:
            sf, sj, cf, cj, nnz = run(n)
            fmt = lambda v, k=1.0: "" if v is None else f"{v * k:.6g}"
            print(f"{name},{2 * n},{fmt(sf)},{fmt(sj)},{fmt(cf, 1e6)},{fmt(cj, 1e6)},{'' if nnz is None else nnz}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
