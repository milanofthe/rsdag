"""rsdag: trace Python functions into a symbolic graph, differentiate, compile.

    from rsdag import jit, jacobian, grad
    f = jit(lambda x, t: [x[1], -x[0] * t])
    y = f([1.0, 2.0], 0.5)             # numpy array
    J = jacobian(lambda x, t: ...)([1.0, 2.0], 0.5)   # (n_out, n_in) array

Tracing is operator overloading: the function is called once with tracer
values (scalars and numpy object arrays of tracers), every arithmetic
operation and numpy ufunc records a node. Data-dependent Python control flow
is not traceable; use `rsdag.where(cond, a, b)`.
"""

import builtins
import math

import numpy as np

from ._rsdag import (
    Dispatch, Program, Scope, Tracer,
    select as _select, matmul as _matmul, reduce as _reduce, solve as _solve,
)

__all__ = [
    "Scope", "Tracer", "Program", "trace", "jit", "jacobian", "grad", "where", "clip",
    "gt", "ge", "lt", "le", "eq", "ne",
    "dot", "matmul", "sum", "solve",
]


def _flat(v):
    return np.asarray(v, dtype=object).ravel()


def dot(a, b):
    """Inner product of two vectors; traced, one `Dot` node (rows of one
    vector fuse into a matrix-vector kernel)."""
    if _is_traced(a) or _is_traced(b):
        a, b = _flat(a), _flat(b)
        if a.size != b.size:
            raise ValueError("dot takes two vectors of one length")
        return _matmul(a, b, 1)[0]
    return np.dot(a, b)


def matmul(a, b):
    """`a @ b` over tracers, vectors and matrices: one `Dot` per entry of
    the product, fused into one kernel (`Gemv`, `Gemm`) once compiled."""
    if not (_is_traced(a) or _is_traced(b)):
        return np.matmul(a, b)
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    if not (a.ndim in (1, 2) and b.ndim in (1, 2) and a.shape[-1] == b.shape[0]):
        raise ValueError("matmul over tracers takes vectors and matrices of matching sizes")
    entries = _matmul(a.ravel(), b.ravel(), b.shape[1] if b.ndim == 2 else 1)
    out = np.empty(len(entries), dtype=object)
    out[:] = entries
    out = out.reshape(a.shape[:-1] + b.shape[1:])
    return out if out.ndim else out[()]


def sum(x):
    """The sum of a vector, one `Reduce` node in the reference fold order."""
    if _is_traced(x):
        return _reduce("sum", _flat(x))
    return np.sum(x)


def solve(a, b):
    """The solution of the dense system `a x = b`, one pivoting kernel once
    compiled, differentiable through the inverse."""
    if not (_is_traced(a) or _is_traced(b)):
        return np.linalg.solve(a, b)
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    n = b.shape[0]
    if a.shape != (n, n):
        raise ValueError("solve takes an n by n matrix and n right-hand sides")
    xs = _solve(a.ravel(), b)
    return np.asarray(xs, dtype=object)


def where(cond, a, b):
    """Elementwise select over tracers, numbers and arrays of them."""
    if _is_traced(cond) or _is_traced(a) or _is_traced(b):
        c, x, y = np.broadcast_arrays(*(np.asarray(v, dtype=object) for v in (cond, a, b)))
        out = np.empty(c.shape, dtype=object)
        for idx in np.ndindex(c.shape):
            ci, xi, yi = c[idx], x[idx], y[idx]
            if isinstance(ci, Tracer) or isinstance(xi, Tracer) or isinstance(yi, Tracer):
                out[idx] = _select(ci, xi, yi)
            else:
                out[idx] = xi if ci else yi
        return out if out.shape else out[()]
    return np.where(cond, a, b)


def _compare(ufunc, a, b):
    """Elementwise comparison yielding tracers (0.0/1.0) instead of bools:
    numpy's comparison ufuncs coerce object results to bool, which a traced
    value cannot be."""
    if _is_traced(a) or _is_traced(b):
        out = ufunc(np.asarray(a, dtype=object), np.asarray(b, dtype=object), dtype=object)
        return out if np.ndim(out) else out[()]
    return ufunc(a, b)


def gt(a, b):
    return _compare(np.greater, a, b)


def ge(a, b):
    return _compare(np.greater_equal, a, b)


def lt(a, b):
    return _compare(np.less, a, b)


def le(a, b):
    return _compare(np.less_equal, a, b)


def eq(a, b):
    return _compare(np.equal, a, b)


def ne(a, b):
    return _compare(np.not_equal, a, b)


def clip(x, lo, hi):
    """Elementwise clamp through `maximum` and `minimum`."""
    return np.minimum(np.maximum(x, lo), hi)


def _is_traced(v):
    if isinstance(v, Tracer):
        return True
    if isinstance(v, np.ndarray) and v.dtype == object:
        return any(isinstance(e, Tracer) for e in v.flat)
    if isinstance(v, (list, tuple)):
        return any(_is_traced(e) for e in v)
    return False


def _size(shape):
    return 1 if shape is None else math.prod(shape)


def _inputs(scope, shape):
    """Fresh inputs for an argument of `shape`: a tracer or an array of them."""
    if shape is None:
        return scope.input()
    arr = np.empty(_size(shape), dtype=object)
    for k in range(arr.size):
        arr[k] = scope.input()
    return arr.reshape(shape)


def _flatten_outputs(result):
    """Flatten a scalar, list or array result into tracers and record its
    shape (None for a scalar)."""
    if isinstance(result, Tracer) or np.isscalar(result):
        return [result], None
    arr = np.asarray(result, dtype=object)
    return list(arr.reshape(-1)), arr.shape


class Compiled(Dispatch):
    """A traced function: shape-specialized programs per argument shapes,
    traced on first use for each. Calls and `program(*args)` run in
    `Dispatch`, which asks `_trace` for the program of a new shape."""

    def __init__(self, func, mode="value", native=False, wrt=0):
        self.func = func
        self.mode = mode
        self.native = native
        self.wrt = wrt

    def _trace(self, key):
        """The program for argument shapes `key` (None for a scalar) and
        the shape of its result."""
        scope = Scope()
        outputs, shape = _flatten_outputs(self.func(*(_inputs(scope, s) for s in key)))
        if self.mode == "value":
            program = scope.compile(outputs)
        else:
            start = builtins.sum(map(_size, key[:self.wrt]))
            wrt = list(range(start, start + _size(key[self.wrt])))
            if self.mode == "jacobian":
                program = scope.jacobian(outputs, wrt)
                shape = (len(outputs), len(wrt))
            elif self.mode == "sparse_jacobian":
                program = scope.sparse_jacobian(outputs, wrt)
                shape = (program.n_outputs,)
            elif self.mode == "grad":
                if len(outputs) != 1:
                    raise ValueError("grad needs a scalar-valued function")
                program = scope.gradient(outputs[0], wrt)
                shape = (len(wrt),)
            else:
                raise ValueError(self.mode)
        if self.native:
            program.compile_native()
        return program, shape

    def pattern(self, *args):
        """`(rows, cols)` of a sparse Jacobian's values for arguments of the
        shapes of `args`, as numpy index arrays."""
        rows, cols = self.program(*args).pattern
        return np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp)


def trace(func, *example_args, native=False):
    """Trace `func` on the shapes of `example_args` and return the compiled
    callable (eager form of `jit`)."""
    c = Compiled(func, "value", native)
    c.program(*example_args)
    return c


def jit(func, native=False):
    """Trace lazily on first call; a new argument shape traces again."""
    return Compiled(func, "value", native)


def jacobian(func, wrt=0, native=False, sparse=False):
    """The Jacobian of `func` with respect to its argument `wrt` (the first
    by default) by symbolic differentiation: a dense `(n_out, n_in)` array,
    or with `sparse` the structurally nonzero values, row by row in
    ascending columns, whose indices `pattern(*args)` returns."""
    return Compiled(func, "sparse_jacobian" if sparse else "jacobian", native, wrt)


def grad(func, wrt=0, native=False):
    """The gradient `(n_in,)` of a scalar-valued `func` with respect to its
    argument `wrt` by reverse mode."""
    return Compiled(func, "grad", native, wrt)
