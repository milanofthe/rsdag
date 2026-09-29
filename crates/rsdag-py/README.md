# rsdag

Python bindings for [rsdag](https://github.com/milanofthe/rsdag), an
expression graph backend written in Rust. A Python function is traced into
a hash-consed graph, differentiated symbolically and evaluated by an
interpreter or as native machine code (x86-64 and AArch64).

```
pip install rsdag
```

Wheels for Linux (x86-64, aarch64), macOS (arm64 and x86-64) and Windows
(x86-64), CPython 3.11 and newer.

## Example

```python
import numpy as np
from rsdag import jit, jacobian, matmul, solve

def lorenz(x, t):
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    return [s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]]

f = jit(lorenz, native=True)              # traces on first call, then native
y = f(np.array([1.0, 2.0, 3.0]), 0.0)
J = jacobian(lorenz)(np.array([1.0, 2.0, 3.0]), 0.0)   # (3, 3), symbolic
Js = jacobian(lorenz, sparse=True)       # the nonzeros; Js.pattern(x, t): rows, cols
g = jit(lambda A, x: solve(A, matmul(A, x)))            # one Gemv, one Solve
```

`jit`, `jacobian` and `grad` trace on the first call and reuse the compiled
program afterwards. Data-dependent Python control flow is not traceable;
`where(cond, a, b)` with `gt`, `lt`, ... expresses elementwise conditions.

## License

GNU Affero General Public License v3.0. The copyright holder licenses rsdag
on other terms as well; for a commercial license contact
info@milanrother.com.
