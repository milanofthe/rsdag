import numpy as np
import pytest

import rsdag
from rsdag import grad, gt, jacobian, jit, trace, where


def lorenz(x, t):
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    return [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]


def test_scalar_and_numpy_ufuncs():
    f = jit(lambda x: np.sin(x) * np.exp(-x) + x**3 / 2)
    x = 0.7
    assert f(x) == pytest.approx(np.sin(x) * np.exp(-x) + x**3 / 2, rel=1e-15)


def test_array_function_matches_numpy():
    f = jit(lorenz)
    x = np.array([1.0, 2.0, 3.0])
    y = f(x, 0.0)
    assert y.shape == (3,)
    assert np.allclose(y, lorenz(x, 0.0), rtol=1e-15)


def test_jacobian_matches_finite_differences():
    J = jacobian(lorenz)
    x = np.array([1.0, 2.0, 3.0])
    got = J(x, 0.0)
    assert got.shape == (3, 3)
    h = 1e-6
    fd = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        fd[:, j] = (np.asarray(lorenz(x + e, 0.0)) - np.asarray(lorenz(x - e, 0.0))) / (2 * h)
    assert np.allclose(got, fd, atol=1e-6)


def test_gradient_reverse_mode():
    g = grad(lambda x: np.sum(x**2) + np.tanh(x[0] * x[1]))
    x = np.array([0.3, -0.4, 1.2])
    got = g(x)
    expect = 2 * x
    c = 1 - np.tanh(x[0] * x[1]) ** 2
    expect[0] += c * x[1]
    expect[1] += c * x[0]
    assert np.allclose(got, expect, rtol=1e-12)


def test_where_and_comparisons():
    f = jit(lambda x: where(x > 0.0, x, -x * 2.0))
    assert f(2.0) == 2.0
    assert f(-1.5) == 3.0
    v = jit(lambda x: where(gt(x, 1.0), x, 0.0))
    assert np.array_equal(v(np.array([0.5, 2.0])), np.array([0.0, 2.0]))


def test_native_matches_the_interpreter():
    f = jit(lorenz, native=True)
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(f(x, 0.0), lorenz(x, 0.0), rtol=1e-15)


def test_control_flow_is_rejected():
    def bad(x):
        if x > 0:
            return x
        return -x

    with pytest.raises(TypeError):
        trace(bad, 1.0)


def test_retrace_on_new_shape():
    f = jit(lambda x: np.sum(x))
    assert f(np.array([1.0, 2.0])) == 3.0
    assert f(np.array([1.0, 2.0, 3.0])) == 6.0


def test_jacobian_with_respect_to_second_argument():
    f = lambda x, t: [x[0] * t, x[1] * t * t]
    Jt = jacobian(f, wrt=1)
    got = Jt(np.array([2.0, 3.0]), 0.5)
    assert got.shape == (2, 1)
    assert np.allclose(got[:, 0], [2.0, 3.0])


def test_numpy_linear_algebra_patterns_trace():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])

    def f(x):
        y = A @ x
        z = np.dot(x, y)
        return [np.linalg.norm(x), z, np.sum(y * y)]

    x = np.array([0.3, -1.2])
    got = jit(f)(x)
    y = A @ x
    assert np.allclose(got, [np.linalg.norm(x), np.dot(x, y), np.sum(y * y)], rtol=1e-14)
    J = jacobian(f)(x)
    assert J.shape == (3, 2)
    assert np.allclose(J[0], x / np.linalg.norm(x), rtol=1e-12)


def test_program_reports_ops_and_dump():
    f = trace(lambda x: np.exp(x) * x, 1.0)
    p = f.program(1.0)
    # An input is read where it is used, not copied: two ops here.
    assert p.n_inputs == 1 and p.n_outputs == 1 and p.n_ops >= 2
    assert "i0" in p.dump()


def test_tracer_defers_to_arrays_on_the_left():
    """A traced scalar times an array of tracers broadcasts elementwise: the
    binary operators return NotImplemented for an operand they cannot trace,
    so numpy's reflected operator runs."""
    import numpy as np
    from rsdag import Scope

    s = Scope()
    k = s.input()
    arr = np.array([s.input(), s.input()], dtype=object)
    for left, right in [(k * arr, arr * k), (k + arr, arr + k)]:
        assert isinstance(left, np.ndarray) and left.shape == (2,)
        assert [e.expr() for e in left] == [e.expr() for e in right]
    prog = s.compile(list(np.exp(arr / 4.0) * k) + list(k * arr) + list(k - arr))
    out = prog.eval([2.0, 0.5, 1.5])
    assert out == pytest.approx(
        [np.exp(0.125) * 2, np.exp(0.375) * 2, 1.0, 3.0, 1.5, 0.5]
    )

    # a genuinely untraceable operand still raises, from Python's own message
    with pytest.raises(TypeError):
        k * "no"
