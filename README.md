# rsdag

A differentiable graph compiler for the equation systems of simulators:
DAEs and ODEs, circuits, state-space blocks. One hash-consed expression
graph, forward and reverse derivatives, a flat instruction tape, an
interpreter over any scalar type, and a native code backend that emits
AArch64 and x86-64 machine code from the tape, bit-identical to the
interpreter. The linear solve of a Newton step is a program too: the static
LU of the Jacobian's pattern as graph ops, so a Newton step runs as
straight-line code without a library call.

![The stages from a consumer's model to execution](docs/diagrams/pipeline.svg)

Private for now. Licensed under PolyForm Noncommercial 1.0.0 (see LICENSE).

## Crates

- `rsdag`: `Graph<K: Field>` (exact rationals or `f64`), functions with
  calls and roles, `Scope`, `Module` (the serializable form), `differentiate`,
  `gradient` (reverse mode), `sparse_jacobian`, `hessian`, `substitute`,
  `Tape` (one interpreter over any `Scalar`, choice specialization, instance
  batching, the `Gemv`, `Gemm` and dense `Solve` kernels, prolog/main
  split), `semantics` (the reference arithmetic every backend mirrors) and
  `symbolic` (`determinant`, `collect`, `rational_form`, `simplify_egraph`,
  the graph solve, `newton_step`).
- `rsdag-jit`: the native backend (`NativeTape`) for AArch64 and x86-64 on
  Linux, macOS and Windows. Function bodies are compiled once and called per
  instance, a batch of instances in parallel; `eval_many` runs a program
  over many input sets.
- `rsdag-py`: the Python package `rsdag` (`trace`, `jit`, `jacobian`,
  `grad`, `where`, `matmul`, `solve`), built with maturin.

The graph crate compiles for `wasm32-unknown-unknown`, where a consumer runs
the interpreter; the emitter needs a host.

## How it works

**Graph.** Nodes are hash-consed, ascending ids are a topological order,
and the smart constructors fold constants in `K` and apply the algebraic
identities. A function is a graph over positional parameters with named
outputs; a `Call` node applies it, and a derivative output is derived from
the body on first demand. Roles on parameters and outputs (state, input,
parameter, time; residual, derivative) are metadata for the system layer.

**Tape.** The flat instruction list of a set of roots: reachability,
lowering into an instruction IR, a register-pressure list schedule, slot
allocation with liveness, emission. Row dots against one vector become a
`Gemv`, rows against several vectors a `Gemm`, a dense system a pivoting
`Solve`; calls of one function on distinct argument lists become one
batched call. A tape compiled with a parameter-pure prolog splits into the
part a solve evaluates once per parameter binding and the part it
evaluates per iteration. The interpreter runs the tape over any `Scalar`
(`f64`, `f32`, `Complex64`); the native backend emits the same instruction
sequence as machine code in chunked functions, with a write-back register
cache.

![The linear solve as a program](docs/diagrams/solve.svg)

**Solve.** `symbolic::solve` turns the solve of a system with a known
pattern into graph ops: block triangular form, a minimum-degree order per
block, a cost predictor over the elimination tree, and a static LU in Crout
form with the right-hand side as the last column. The pivot rows are fixed
at build time and guarded: a step with more than one structural candidate
checks at run time that its pivot dominates its column, and on a failed
guard `Plan::repivot` chooses the rows a numeric elimination takes on the
current values and the program is rebuilt. Compiled with the matrix
entries as parameter-pure inputs and the right-hand side as main inputs,
the factorization is the prolog pass and the substitution the main pass.

![Functions and instance batching](docs/diagrams/bodies.svg)

**Bodies.** A multiply-instantiated model is one function and one call per
instance. The body is compiled once, and the calls of a program that share
a shape become one kernel op that runs the body over all instances, on the
rayon pool when the batch is large. A consumer may register a body it
compiled itself, for instance one that caches the prolog over its
parameter-pure arguments per instance.

![Choice specialization](docs/diagrams/specialize.svg)

**Specialization.** Evaluating a tape with a trace sink records the arm
each `Select` takes; specializing on the trace pins the arms and shortens
the tape to that region, keeping the conditions as guards. A guard that
flips retraces the full tape and rebuilds the specialization. Guards that
depend only on parameters live in the prolog and are checked once per
parameter binding.

**Bit-exactness.** Every backend computes the same IEEE operation sequence:
no fast-math, no fused multiply-add, one reference routine per elementary
function, one four-accumulator order for every reduction and dot product,
the same domain guards. `rsdag::synth` generates random programs over the
whole op vocabulary, and the test suites pin the arena sweep, the tape, the
native code and the typed evaluation against each other on them.

## Numbers

One core of an Apple M3, release profile. `docs/bench/plot.py` draws the
figures from the CSVs in `docs/bench/data`; `docs/bench/README.md` names
the sources.

![Evaluation and compile cost per op](docs/bench/ops.svg)

A ring op costs half a nanosecond natively up to programs of a few tens of
thousands of ops and three at a million, where instruction fetch sets the
pace; interpreted it costs ten. An elementary function adds a call into the
same routine the interpreter uses. Emitting costs about a hundred
nanoseconds per op on a large program.

![The Newton step against a sparse LU library](docs/bench/solve.svg)

A Newton step over a circuit-like system is 26 ops per unknown, 31 with the
pivot guard, linear to a million unknowns. Against a general sparse LU
(rslab, KLU path) the graph solve measures 15x to 150x on ring and band
patterns and loses on 2D grids past a thousand unknowns, where fill makes
the program large; the cost predictor tells a consumer which side of that
line a pattern is on.

![The dense kernels](docs/bench/dense.svg)

The dense kernels run at the two-lane peak without fused multiply-add: a
1000-state `A x + B u` evaluates in 0.09 ms, a 1000 by 1000 product at
29 GF/s, the dense solve of 1000 unknowns in 40 ms.

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
