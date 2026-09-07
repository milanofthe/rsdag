# rsgb

Rust symbolic graph backend: the shared expression-graph substrate of SANE
and fastsim. One hash-consed scalar graph with exact or floating constants,
symbolic differentiation (forward and reverse), a flat tape with an
interpreter for `f64`, `f32` and complex values, a Cranelift JIT, a C source
backend, a symbolic layer (determinants, polynomials and rational forms,
e-graph simplification) and a Python tracer, so that every optimization
lands in every consumer once.

Private for now. Licensed under PolyForm Noncommercial 1.0.0 (see LICENSE).

Development runs through GitHub issues, not documents: the inventory (#1),
the architecture (#2) and the phases (#3 onwards, each depending on the
previous one) live there, with dates, dependencies and history.

## Crates

- `rsgb`: `Graph<K: Field>` (`BigRational` or `F64` constants), functions
  with calls and roles, `differentiate`, `gradient` (reverse mode),
  `jacobian`, `hessian`, `rebuild` (CSE, dead code, identities after
  transforms), `Tape` (the program: interpreter, `eval_typed` for `f32` and
  `Complex<f64>`, choice specialization, lanes), and `symbolic`
  (`determinant`, `collect`, `rational_form`, `simplify_egraph`).
- `rsgb-jit`: Cranelift chunked JIT (`ChunkedTape`) and SIMD lanes
  (`LaneTape`), bit-identical to the interpreter.
- `rsgb-c`: C source emission (`emit`) and a compile-and-compare harness
  (`verify::run_c`).
- `rsgb-py`: the Python package `rsgb` (`trace`, `jit`, `jacobian`, `grad`,
  `where`, comparison helpers), built with maturin.

## Rust

```rust
use rsgb::{Graph, Tape, Node, SymbolId};

let mut g: Graph = Graph::new();          // exact rational constants
let (x, y) = (g.sym("x"), g.sym("y"));
let e = { let s = g.sin(x); g.mul(s, y) };
let dx = rsgb::differentiate(&mut g, e, SymbolId(0));
let tape = Tape::compile(&g, &[e, dx], &[SymbolId(0), SymbolId(1)]);
let (mut work, mut out) = (Vec::new(), Vec::new());
tape.eval(&[0.5, 2.0], &mut work, &mut out);
```

## Python

```python
import numpy as np
from rsgb import jit, jacobian, grad, where, gt

def lorenz(x, t):
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    return [s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]]

f = jit(lorenz, native=True)              # traces on first call, Cranelift
y = f(np.array([1.0, 2.0, 3.0]), 0.0)
J = jacobian(lorenz)(np.array([1.0, 2.0, 3.0]), 0.0)   # (3, 3), symbolic
src = f.c_source(np.zeros(3), 0.0, name="lorenz_rhs")  # C function
```

Data-dependent Python control flow is not traceable; use `where(cond, a, b)`
with `gt`, `lt`, ... for elementwise conditions on arrays.

## Bit-exactness

Every backend computes the same IEEE operation sequence as the interpreter:
no fast-math, no fused multiply-add (the C backend emits
`#pragma STDC FP_CONTRACT OFF` and is verified with `-ffp-contract=off`),
one reference routine per transcendental, a fixed four-accumulator order
for long reductions. The fuzzers in `crates/rsgb/tests`,
`crates/rsgb-jit/tests` and `crates/rsgb-c/tests` pin interpreter, JIT,
lanes, typed evaluation and C against each other on random graphs over the
whole op vocabulary.

## Build

```
cargo test --workspace
maturin build --release -m crates/rsgb-py/Cargo.toml   # the Python wheel
```
