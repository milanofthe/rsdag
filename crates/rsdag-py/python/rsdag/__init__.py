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

import numpy as np

from ._rsdag import Program, Scope, Tracer, select as _select

__all__ = [
    "Scope", "Tracer", "Program", "trace", "jit", "jacobian", "grad", "where", "clip",
    "gt", "ge", "lt", "le", "eq", "ne",
]


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


class _Spec:
    """Shapes of the example arguments: scalars stay scalars, everything
    else is an array of that shape."""

    def __init__(self, args):
        self.shapes = []
        for a in args:
            if np.isscalar(a) or (isinstance(a, np.ndarray) and a.ndim == 0):
                self.shapes.append(None)
            else:
                self.shapes.append(np.shape(np.asarray(a, dtype=float)))

    def key(self):
        return tuple(self.shapes)

    def n_inputs(self):
        return sum(1 if s is None else int(np.prod(s)) for s in self.shapes)

    def input_range(self, arg):
        """Flat input indices of argument `arg`."""
        start = self.n_inputs_before(arg)
        s = self.shapes[arg]
        n = 1 if s is None else int(np.prod(s))
        return list(range(start, start + n))

    def n_inputs_before(self, arg):
        return sum(1 if s is None else int(np.prod(s)) for s in self.shapes[:arg])


def _make_inputs(scope, spec):
    args = []
    for s in spec.shapes:
        if s is None:
            args.append(scope.input())
        else:
            n = int(np.prod(s))
            arr = np.empty(n, dtype=object)
            for k in range(n):
                arr[k] = scope.input()
            args.append(arr.reshape(s))
    return args


def _flatten_outputs(result):
    """Flatten a scalar, list or array result into tracers and record its
    shape (None for a scalar)."""
    if isinstance(result, Tracer) or np.isscalar(result):
        return [result], None
    arr = np.asarray(result, dtype=object)
    return list(arr.reshape(-1)), arr.shape


def _flatten_inputs(args):
    flat = []
    for a in args:
        if np.isscalar(a) or (isinstance(a, np.ndarray) and a.ndim == 0):
            flat.append(float(a))
        else:
            flat.extend(np.asarray(a, dtype=float).reshape(-1).tolist())
    return flat


class Compiled:
    """A traced function: shape-specialized programs per argument shapes,
    traced on first use for each."""

    def __init__(self, func, mode="value", native=False, wrt=0):
        self.func = func
        self.mode = mode
        self.native = native
        self.wrt = wrt
        self._programs = {}

    def _program(self, args):
        spec = _Spec(args)
        key = spec.key()
        hit = self._programs.get(key)
        if hit is not None:
            return hit
        scope = Scope()
        inputs = _make_inputs(scope, spec)
        result = self.func(*inputs)
        outputs, out_shape = _flatten_outputs(result)
        if self.mode == "value":
            program = scope.compile(outputs)
            shape = out_shape
        elif self.mode == "jacobian":
            wrt = spec.input_range(self.wrt)
            program = scope.jacobian(outputs, wrt)
            shape = (len(outputs), len(wrt))
        elif self.mode == "grad":
            if len(outputs) != 1:
                raise ValueError("grad needs a scalar-valued function")
            wrt = spec.input_range(self.wrt)
            program = scope.gradient(outputs[0], wrt)
            shape = (len(wrt),)
        else:
            raise ValueError(self.mode)
        if self.native:
            program.compile_native()
        entry = (program, shape)
        self._programs[key] = entry
        return entry

    def __call__(self, *args):
        program, shape = self._program(args)
        out = program.eval(_flatten_inputs(args))
        if shape is None:
            return out[0]
        return np.asarray(out).reshape(shape)

    def program(self, *args):
        """The program traced for these argument shapes."""
        return self._program(args)[0]



def trace(func, *example_args, native=False):
    """Trace `func` on the shapes of `example_args` and return the compiled
    callable (eager form of `jit`)."""
    c = Compiled(func, "value", native)
    c._program(example_args)
    return c


def jit(func, native=False):
    """Trace lazily on first call; a new argument shape traces again."""
    return Compiled(func, "value", native)


def jacobian(func, wrt=0, native=False):
    """The dense Jacobian `(n_out, n_in)` of `func` with respect to its
    argument `wrt` (the first by default) by symbolic differentiation."""
    return Compiled(func, "jacobian", native, wrt)


def grad(func, wrt=0, native=False):
    """The gradient `(n_in,)` of a scalar-valued `func` with respect to its
    argument `wrt` by reverse mode."""
    return Compiled(func, "grad", native, wrt)
