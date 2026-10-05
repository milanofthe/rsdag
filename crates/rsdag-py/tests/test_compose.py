"""Programs compose: a traced function called inside another trace is one
instance of its program there, and the outer program evaluates and
differentiates as the unrolled function would."""
import numpy as np
import pytest

import rsdag

cell = rsdag.jit(lambda a, b: [a * b, np.sin(a)])


def unrolled(x):
    return [x[k] * x[k + 1] + np.sin(x[k + 1]) for k in range(len(x) - 1)]


def composed(x):
    return [cell(x[k], x[k + 1])[0] + cell(x[k + 1], x[k])[1] for k in range(len(x) - 1)]


X = np.array([0.3, -0.7, 1.1, 0.4, 2.0])


@pytest.mark.parametrize("native", [False, True])
def test_a_composed_program_evaluates_as_unrolled(native):
    f = rsdag.jit(composed, native=native)
    assert np.allclose(f(X), unrolled(X), rtol=1e-15, atol=0)


def test_a_composed_program_differentiates_as_unrolled():
    J = rsdag.jacobian(composed)(X)
    R = rsdag.jacobian(unrolled)(X)
    assert J.shape == R.shape
    assert np.allclose(J, R, rtol=1e-14, atol=1e-15)
    g = rsdag.grad(lambda x: sum(composed(x)))(X)
    assert np.allclose(g, R.sum(axis=0), rtol=1e-14, atol=1e-15)


def test_programs_nest_and_take_numbers():
    pair = rsdag.jit(lambda u, v: cell(u, v)[0] - cell(v, 2.0)[1])
    outer = rsdag.jit(lambda x: [pair(x[k], x[k + 1]) for k in range(len(x) - 1)])
    want = [X[k] * X[k + 1] - np.sin(X[k + 1]) for k in range(len(X) - 1)]
    assert np.allclose(outer(X), want, rtol=1e-15, atol=0)
    J = rsdag.jacobian(outer)(X)
    for k in range(len(X) - 1):
        assert J[k, k] == pytest.approx(X[k + 1])
        assert J[k, k + 1] == pytest.approx(X[k] - np.cos(X[k + 1]))


def test_a_program_of_arrays_composes_elementwise():
    norm2 = rsdag.jit(lambda v: [v[0] * v[0] + v[1] * v[1]])
    f = rsdag.jit(lambda x: [norm2(x[0:2])[0], norm2(x[2:4])[0]])
    assert np.allclose(f(X[:4]), [X[0] ** 2 + X[1] ** 2, X[2] ** 2 + X[3] ** 2])
