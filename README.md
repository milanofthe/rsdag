# rsdag

A hash-consed expression graph with symbolic differentiation and a compiled
evaluator. One graph over exact rational or floating constants, forward and
reverse derivatives, a flat tape with an interpreter for `f64`, `f32` and
complex values, a native code backend, a symbolic layer
(determinants, polynomials and rational forms, e-graph simplification) and a
Python tracer.

It builds, differentiates and evaluates graphs, and stops there: no
integrators, no solver loops, no numeric linear algebra library. A linear
solve on a system the graph knows the structure of can be emitted as graph
ops (`symbolic::solve`), so a Newton step is one program.

Private for now. Licensed under PolyForm Noncommercial 1.0.0 (see LICENSE).
Design decisions and phases live in the GitHub issues.

## Crates

- `rsdag`: `Graph<K: Field>`, functions with calls and roles, `Scope` for
  building one incrementally, `Module` as the serializable form,
  `differentiate`, `gradient` (reverse mode), `jacobian`, `hessian`,
  `rebuild`, `Tape` (interpreter, `eval_typed`, choice specialization) and
  `symbolic` (`determinant`, `collect`, `rational_form`, `simplify_egraph`,
  `solve` with static LU and `newton_step`).
- `rsdag-jit`: the native backend (`NativeTape`), machine code emitted
  straight from the tape for AArch64 and x86-64, bit-identical to the
  interpreter, with `eval_many` for instances in parallel.
- `rsdag-py`: the Python package `rsdag` (`trace`, `jit`, `jacobian`,
  `grad`, `where`, comparison helpers), built with maturin.

## Rust

```rust
use rsdag::{Graph, Tape, SymbolId};

let mut g: Graph = Graph::new();          // exact rational constants
let (x, y) = (g.sym("x"), g.sym("y"));
let e = { let s = g.sin(x); g.mul(s, y) };
let dx = rsdag::differentiate(&mut g, e, SymbolId(0));
let tape = Tape::compile(&g, &[e, dx], &[SymbolId(0), SymbolId(1)]);
let (mut work, mut out) = (Vec::new(), Vec::new());
tape.eval(&[0.5, 2.0], &mut work, &mut out);
```

## Python

```python
import numpy as np
from rsdag import jit, jacobian, where, gt

def lorenz(x, t):
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    return [s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]]

f = jit(lorenz, native=True)              # traces on first call, then native
y = f(np.array([1.0, 2.0, 3.0]), 0.0)
J = jacobian(lorenz)(np.array([1.0, 2.0, 3.0]), 0.0)   # (3, 3), symbolic
```

Data-dependent Python control flow is not traceable; use `where(cond, a, b)`
with `gt`, `lt`, ... for elementwise conditions on arrays.

## Bit-exactness

Every backend computes the same IEEE operation sequence as the interpreter:
no fast-math, fused multiply-add only when asked for (`CompileOptions`),
one reference routine per transcendental, a fixed four-accumulator order
for long reductions. `rsdag::synth` generates random programs over the
whole op vocabulary, and the suites in `crates/rsdag/tests` and
`crates/rsdag-jit/tests` pin the arena sweep, the tape, the native code and
typed evaluation against each other on them. `TapeVisitor` documents what
a further backend has to reproduce.

## Benchmarks

`cargo run --release --example bench -p rsdag-jit --features rsdag/synth`
prices the interpreter and the native code per tape op, and the compile
time per op, and checks the two against each other bit for bit. Its corpus
is wide, like an assembled residual or Jacobian: those are thousands of
nodes at a depth of seven to ten, one level per row, and a narrow corpus
prices a shape no consumer produces.

On an M3 a ring op costs about 0.5 ns natively and 10 ns interpreted; an
elementary function adds a call into the same routine the interpreter
uses. Emitting costs 30 to 60 ns per op on a large program, so a program
compiles in about the time of a handful of evaluations.

## Build

```
cargo test --workspace
maturin build --release -m crates/rsdag-py/Cargo.toml   # the Python wheel
```
