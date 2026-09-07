# Design: the rsgb graph contract

rsgb is the backend both SANE and fastsim build on. It is not a product of
its own: its API follows what the two consumers need, and every optimization
lands in it once. This document fixes the decisions that must hold before
code moves. Open decisions are marked as such.

## 1. Layers

```
  consumers      SANE (DAE, MNA, Verilog-A, solvers)   fastsim (blocks, events, FMI)
  system layer   rsgb::system   states, time, memory, regions, events  (functional graph + writes)
  graph          rsgb::graph    hash-consed scalar DAG, constants, functions, AD, analyses
  execution      rsgb::tape     flat tape, specialization, lanes
                 rsgb::jit      Cranelift chunked JIT
                 rsgb::codegen  C emission and verification
  frontends      rsgb-py        Python tracer (JAX-style jit/jacobian), PyO3
```

The graph is purely functional. State, time, memory and events are not graph
nodes; they are inputs and writes of a region in the system layer, exactly
as fastsim's IR does it today. SANE's DAE `F(x, x', t) = 0` and fastsim's
`dX/dt = F(X, t)` are two region conventions over the same graph.

## 2. Graph data model

- `Graph` (SANE's `Context` renamed): `nodes`, `dedup`, constant table,
  argument pool, symbol table, functions, memo. Ids: `ExprId(u32)`,
  `SymbolId(u32)`, `ConstId(u32)`, `ArgList`, `FuncId`, `OutputId`.
- `Node`, 16 bytes, size asserted:
  `Const(ConstId)`, `Symbol(SymbolId)`, `Add`, `Mul`, `Neg`, `Pow(e, i64)`,
  `Binary(BinOp, a, b)`, `Unary(UnaryOp, e)`, `Cmp(CmpOp, a, b)`,
  `Select(c, t, e)`, `Reduce(ReduceOp, ArgList)`, `Dot(ArgList)`,
  `Call(OutputId, ArgList)`, `Opaque(ExternId, ArgList)`.
  `Add` and `Mul` stay dedicated nodes (they carry the canonical ordering
  and the reduction fusion); `Binary` holds the rest of fastsim's set: Sub
  is not a node (it is `add(neg)`), Div is not a node (`mul(pow -1)`), so
  `BinOp = { Powf, Mod, Min, Max, Atan2, Hypot }`. `Fma` is not a graph node
  (see 4).
- `UnaryOp` is the union of both sets (SANE's 10 plus fastsim's 33 minus
  overlaps), with the domain guards of SANE (`EXP_LIMIT`, `LN_FLOOR`,
  sqrt clamp) as the single reference implementation shared by the arena
  evaluator, the tape, the JIT trampolines and the C backend.
- Ascending `ExprId` is a topological order; the tape relies on it.

## 3. Constants

Decision: exact rationals stay the canonical constant model, with the f64
fast path SANE already has (`konst_f64` cache). Reasons: SANE's linear
symbolic analysis and DAE export need exact arithmetic, and fastsim never
observes a difference because every constant that reaches a tape is
converted once at lowering. Rules:

- Folding at construction happens in rationals. An f64 that is not exactly
  representable as a small rational (irrational results of folding, values
  from tracing) is stored as its exact rational of the f64 bit pattern, so
  round-tripping is lossless and hashing is by value.
- Folding of transcendental functions of constants is not done in the graph
  (fastsim's `unary_fold_in_domain` rule moves to the optimizer, where it
  folds in f64 with the same reference implementation the backends use, and
  only inside the function's domain).

Open: whether to keep `BigRational` or a small-integer fast path with
`BigRational` overflow (the current SANE hot path allocates for every
constant).

## 4. Bit-exactness contract

Every backend computes the same IEEE operation sequence as the arena
evaluator: no fast-math, no FMA contraction, transcendentals through one
reference routine. This is SANE's invariant and fastsim's `Fma` node breaks
it (single rounding). Decision: `Fma` is not a graph node. The optimizer may
mark a `MulAdd` dispatch (two roundings, as SANE's tape does today); a real
fused rounding is a tape/JIT/C option (`fast_fma`) that consumers opt into,
and the parity suite runs both modes. Reductions use the fixed 4-lane
merge order above `REDUCE_SIMD_MIN` in every backend.

## 5. Functions and hierarchy

SANE's functions with calls and memoised derivative outputs are the
hierarchy primitive. fastsim blocks and subsystems map onto them: a block
type is a function, an instance is a call, and the IR's `Call { ExternId }`
is the opaque operator bound to an `ExternBundle`. Inlining is a graph
transform, so a consumer chooses between one function with many calls
(SANE's compact models) and a fully inlined graph (fastsim's static
compile).

## 6. Optimizer

One pass pipeline over the graph, run to a fixed point, from fastsim:
dead code elimination, canonicalize, constant folding (with domain checks),
strength reduction, algebraic simplification, chain reassociation. Rules
applied at construction (SANE) stay, because they keep the graph small
during building; the passes are for what construction cannot see. Every
pass is bit-preserving under section 4 or gated behind an explicit
`unsafe_math` level; reassociation of floating sums is gated.

## 7. Differentiation

- Forward symbolic on the graph (SANE's implementation: `differentiate`,
  `gradient`, `jacobian`, `hessian`, `sparsity`, `time_derivative`), plus
  fastsim's slot-oriented conveniences (`jacobian_wrt_input_slot`,
  feedthrough patterns) as thin wrappers.
- Reverse mode (adjoint) is planned as a graph transform producing the
  gradient of one scalar output; added when a consumer needs it (parameter
  estimation in fastsim, optimization in pathsim). It shares the tape.
- Derivative outputs of functions are memoised in the function, as today.

## 8. Tape and execution

- One tape format: SANE's memory-based SSA over a caller-owned work
  buffer, positional inputs, prolog split, ops as listed in the inventory,
  plus fastsim's mutable parameter slots (a parameter is an input bound to
  a persistent buffer, not a node kind).
- Compilation: reachability, list scheduling, superinstruction fusion,
  liveness-driven slot allocation (both have it; SANE's stays).
- Specialization against a recorded Select trace with guards (SANE) is a
  tape feature available to fastsim's region evaluation.
- Lanes (`eval_batch`, `LaneTape`) for instance batching.
- Backends behind one trait: interpreter, Cranelift chunked JIT, C source.
  The C backend takes fastsim's templates for regions and the reference
  implementations of the special functions; the model and solver scaffolds
  stay in fastsim.

## 9. System layer

`rsgb::system` carries fastsim's IR schema, generalized: `Region { ops:
graph outputs, writes: [Output, StateDeriv, StateWrite, MemoryWrite] }`,
`StateVar`, `MemorySlot`, `Lut1d` as an opaque operator, time as a
distinguished input. SANE's DAE becomes a region set with residual writes.
Events: guard functions are region outputs (so they are differentiable and
specializable), effects are regions with writes; scheduling and
zero-crossing detection stay in the consumers' integrators.

## 10. Frontends

- The Python tracer moves into `rsgb-py` as the JAX-style entry (`jit`,
  `jacobian`, `trace_with_signature`, array tracing with numpy ufunc and
  array-function tables). fastsim re-exports it; SANE gets Python device
  models through it.
- Native construction stays `Builder`-style (fastsim) or direct `Graph`
  methods (SANE); the `Builder` trait is implemented by the graph.

## 11. What stays out

SANE's constants, config, logging, profiling and diagnostics; fastsim's
blocks, solvers, simulation loop, FMI, C scaffolds and Python API surface.
rsgb takes a logging sink and a clock as small traits so it has no
dependency on either consumer.

## 12. Bit-exact parity suite

Carried over and extended: arena == tape == JIT == lanes == C for the
union op set, on random graphs (SANE's `fuzz_parity`, jit `parity`,
`lane_parity`) and on fixtures from both consumers. This suite is the
acceptance test for every change in rsgb.
