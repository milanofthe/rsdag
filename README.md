# rsdag

Expression graph compiler for the equation systems of simulators (DAEs,
ODEs, circuits, state-space blocks): a hash-consed expression graph with
forward and reverse differentiation, a flat instruction tape, an
interpreter over any scalar type, a native code backend for AArch64 and
x86-64, and a sparse linear solve compiled into the same tape.

![Pipeline](docs/diagrams/pipeline.svg)

Private for now. Licensed under PolyForm Noncommercial 1.0.0 (see LICENSE).

## Crates

- `rsdag`: `Graph<K: Field>` (exact rationals or `f64`), functions with
  calls and roles, `Scope`, `Module` (serialization), `differentiate`,
  `gradient` (reverse mode), `sparse_jacobian`, `hessian`, `substitute`,
  `Tape` (interpreter over any `Scalar`, choice specialization, instance
  batching, `Gemv`, `Gemm` and dense `Solve` kernels, prolog/main split),
  `semantics` (the reference arithmetic), `symbolic` (`determinant`,
  `collect`, `rational_form`, `simplify_egraph`, the sparse solve,
  `newton_step`).
- `rsdag-jit`: `NativeTape`, machine code for AArch64 and x86-64 on Linux,
  macOS and Windows; function bodies compiled once and batched over
  instances; `eval_many` over many input sets.
- `rsdag-py`: Python package `rsdag` (`trace`, `jit`, `jacobian`, `grad`,
  `where`, `matmul`, `solve`), built with maturin.

`rsdag` compiles for `wasm32-unknown-unknown` (interpreter only).

## Graph

Nodes are hash-consed; ascending ids are a topological order. Constructors
fold constants in `K` and apply the algebraic identities. A function is a
graph over positional parameters with named outputs; `Call` applies it;
derivative outputs are derived from the body on first demand. Parameters
and outputs carry roles (state, input, parameter, time; residual,
derivative) as metadata.

## Tape

`Tape::compile` lowers a set of roots into an instruction IR, schedules it
for register pressure, allocates slots by liveness and emits the
instruction list. Row dots against one vector lower to `Gemv`, against
several vectors to `Gemm`, a dense system to a pivoting `Solve`; calls of
one function on distinct argument lists lower to one batched call.
`Tape::compile_split` marks parameter-pure inputs; the tape then has a
prolog evaluated once per parameter binding and a main part evaluated per
iteration. `Tape::eval` runs over any `Scalar` (`f64`, `f32`, `Complex64`).
`NativeTape::compile` emits the same instruction sequence as machine code
in chunked functions with a write-back register cache.

## Sparse solve

![Sparse solve](docs/diagrams/solve.svg)

`symbolic::solve` lowers the solve of a system with a known sparsity
pattern into graph ops: block triangular form, minimum-degree ordering per
block, a fill and flop predictor over the elimination tree, and a static LU
in Crout form with the right-hand side as the last column. Pivot rows are
fixed at build time. A step with more than one structural candidate emits a
guard, `|pivot| >= 1e-3 max|column|`; on a failed guard `Plan::repivot`
takes the rows of a numeric elimination on the current values and the
program is rebuilt. With the matrix entries as parameter-pure inputs and
the right-hand side as main inputs, the prolog is the factorization and
the main part the substitution.

## Function bodies

![Function bodies](docs/diagrams/bodies.svg)

A multiply-instantiated model is one function and one call per instance.
The body is compiled once; calls with the same shape lower to one kernel op
that runs the body over all instances, on the rayon pool above a size
threshold. `Graph::set_func_body` registers a body compiled by the caller
(for instance with a per-instance prolog cache); programs whose calls it
covers use it.

## Choice specialization

![Choice specialization](docs/diagrams/specialize.svg)

`Tape::eval_with` records the arm taken by each `Select`. `Tape::specialize`
pins the recorded arms and shortens the tape to that region; the pinned
conditions remain as guards. A failed guard means the region changed: the
full tape is retraced and the specialization rebuilt. Guards that depend
only on parameters are in the prolog and checked once per parameter
binding.

## Bit-exactness

All backends compute the same IEEE operation sequence: no fast-math, no
fused multiply-add, one reference routine per elementary function, one
four-accumulator order for reductions and dot products, the same domain
guards. `rsdag::synth` generates random programs over the whole op
vocabulary; the test suites compare the arena sweep, the tape, the native
code and the typed evaluation on them bit for bit.

## Numbers

One core of an Apple M3, release profile. `docs/bench/plot.py` draws the
figures from the CSVs in `docs/bench/data` (sources in
`docs/bench/README.md`).

![Evaluation and compile cost per op](docs/bench/ops.svg)

Native: 0.5 ns per op on ring programs up to some ten thousand ops, 3 ns at
800k ops (instruction fetch bound); 2 to 5 ns per op with elementary
functions. Interpreter: 10 to 12 ns per op. Emission: about 100 ns per op
on programs above ten thousand ops.

![Sparse solve against a sparse LU library](docs/bench/solve.svg)

Newton step of a circuit-like system: 26 ops per unknown, 31 with pivot
guards, linear to a million unknowns. Against a general sparse LU (rslab,
KLU path): 15x to 150x faster on ring and band patterns; slower on 2D grids
above about a thousand unknowns, where fill grows the program. The cost
predictor gives the program size before it is built.

![Dense kernels](docs/bench/dense.svg)

Dense kernels: `Gemv` 1000 by 1000 in 0.09 ms, `Gemm` 1000 by 1000 at
29 GF/s, dense `Solve` of 1000 unknowns in 40 ms.

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
from rsdag import jit, jacobian, matmul, solve

def lorenz(x, t):
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    return [s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]]

f = jit(lorenz, native=True)              # traces on first call, then native
y = f(np.array([1.0, 2.0, 3.0]), 0.0)
J = jacobian(lorenz)(np.array([1.0, 2.0, 3.0]), 0.0)   # (3, 3), symbolic
g = jit(lambda A, x: solve(A, matmul(A, x)))            # one Gemv, one Solve
```

Data-dependent Python control flow is not traceable; `where(cond, a, b)`
with `gt`, `lt`, ... expresses elementwise conditions.

## Build

```
cargo test --workspace
scripts/ci.sh                                            # the CI gate, locally
maturin build --release -m crates/rsdag-py/Cargo.toml   # the Python wheel
```
