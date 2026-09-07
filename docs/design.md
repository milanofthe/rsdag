# Design v2: rsgb

rsgb is the backend both SANE and fastsim build on. Its API follows what the
two consumers need, and every optimization lands in it once. Version 2
replaces "SANE core plus parts of fastsim" with one architecture built on
six principles. Where v1 decisions survive they are restated here; the
inventory (`inventory.md`) stays the record of what exists today.

## Principles

1. Functions all the way down. There is no global symbol table and no
   separate region IR: a `Module` holds `Function`s, a `Function` is a
   hash-consed SSA body over positional parameters with named outputs.
2. The system layer is a signature. Inputs and outputs carry roles; the
   graph stays pure.
3. Transformations are `Function -> Function`: jvp, vjp, inline, simplify,
   specialize, map, substitute, collect.
4. Call, Map and lanes are one concept: hierarchy, batching, SIMD.
5. Three stages, three data structures: `Function` for transformations,
   `Program<T>` for execution, backends over `Program`.
6. Piecewise structure once: the `Select` conditions of a function are its
   discontinuity guards, shared by region specialization and event
   location.

## 1. Module and Function

```
Module {
    functions: Vec<Function>,      // by FuncId
    externs:   Vec<ExternDecl>,    // opaque operators with arity and derivative markers
    consts:    ConstTable<K>,      // module-wide, deduplicated
}
Function {
    name, params: Vec<Param>,      // positional; a Param has a role and a name
    nodes: Vec<Node>,              // hash-consed, ascending id is topological
    dedup, arg_pool,               // as in SANE's Context
    outputs: Vec<Output>,          // (role, name, ExprId)
    derived: memo of derivative outputs, inlined variants, specializations
}
```

- `Node`, 16 bytes: `Const(ConstId)`, `Param(u32)`, `Add`, `Mul`, `Neg`,
  `Pow(e, i64)`, `Binary(BinOp, a, b)`, `Unary(UnaryOp, e)`, `Cmp(CmpOp,
  a, b)`, `Select(c, t, e)`, `Reduce(ReduceOp, ArgList)`, `Dot(ArgList)`,
  `Call(FuncId, out, ArgList)`, `Map(FuncId, out, n, ArgList)`,
  `Opaque(ExternId, out, ArgList)`.
- Construction rules of SANE (identities, folding in `K`, canonical
  operand order, fused reductions) stay as the smart constructors; they
  are what keeps a circuit graph small while it is built.
- A `Scope` is an open function under construction: it hands out params
  on demand by name (what SANE's free symbols are today), and `close()`
  turns it into a `Function`. Tracers, netlist lowering and Verilog-A
  lowering all build through a `Scope`.
- Functions are values: cloning, serializing (serde) and hashing a
  function are cheap and deterministic. A `Module` is the interchange
  format for caching compiled artifacts, code generation and FMI export.

## 2. Two scalar axes

- `K: Field` is the constant field of a Module: `Rational` (SANE's exact
  symbolic analysis), `f64` (fastsim), `Complex<f64>`. Smart constructors
  fold in `K`; transcendental folding exists only for floating `K`, through
  the reference math of the backends.
- `T: Scalar` is the execution type of a `Program<T>`: `f64`, `f32`,
  `Complex<f64>`, lanes. Constants are lowered once with `K: Into<T>`.
- `Cmp` and `Select` need real predicates; for complex `T` the comparison
  on real part or magnitude is an explicit op. Ordered ops (`Min`, `Max`,
  `Floor`, `Sign`) exist for `T: Real` only.
- Domain guards belong to `T`; the reference unary implementations are per
  `T` and the parity suite runs per `T`.

## 3. Roles: the system layer

A role on a parameter or output is metadata; nothing in the graph changes.

- Parameter roles: `Input { port, elem }`, `Param`, `State`, `Time`,
  `Memory { slot, offset }`, `Free` (a symbolic unknown).
- Output roles: `Output { port, elem }`, `StateDeriv`, `Residual`,
  `StateWrite`, `MemoryWrite`, `Guard`, `Plain`.
- A SANE DAE is a function with `Free`, `State`, `Time` params and
  `Residual` outputs. A fastsim block is a function with `Input`, `State`,
  `Time`, `Memory`, `Param` params and `Output`, `StateDeriv`,
  `MemoryWrite` outputs. An event is a `Guard` output plus an effect
  function with `StateWrite` outputs. `Lut1d` is an opaque operator.
- Consumers keep their own integrators, schedulers and solvers; rsgb only
  guarantees that a function with roles can be evaluated, differentiated
  with respect to any role subset, specialized and lowered.

## 4. Transformations (Function -> Function)

- `jvp(f, wrt)`: forward symbolic derivative outputs (SANE's `differentiate`,
  `jacobian`, `hessian`, `sparsity`, `time_derivative` generalized to role
  subsets); memoised on the function.
- `vjp(f, out)`: reverse mode as an adjoint transform producing the
  gradient of one output with respect to all parameters. New; needed for
  parameter estimation and optimization.
- `inline(f, calls)`: replace calls by bodies (SANE's `inline_call`,
  fastsim's static compile).
- `simplify(f)`: the pass pipeline (dead code, canonicalize, folding with
  domain checks, strength reduction, algebraic identities, reassociation
  gated), plus the e-graph simplifier from SANE's `simplify` crate as the
  general algebraic simplifier for symbolic work.
- `specialize(f, choices)`: fix Select branches to a recorded choice,
  keep guard outputs (SANE's choice specialization).
- `map(f, n)`: instance batching as a node (`Map`), lowered to lanes.
- `substitute(f, param, expr)` and `compose(f, g)`: symbolic composition,
  what SANE's `substitute_many` and node merging do today.
- `collect(f, var)`: rational canonical form `N(s)/D(s)` in one variable
  (SANE's `symbolic_poly`), for transfer functions and pole-zero work.
- `det(matrix of ExprId)`: symbolic determinant (SANE's `mna::determinant`)
  as a symbolic linear algebra helper.

## 5. Symbolic analysis is a first-class capability

SANE's symbolic mode (exact transfer functions, symbolic sensitivities,
determinants, rational canonical forms, printing and export) is what
requires `K = Rational`, `substitute`, `collect`, `det`, the e-graph
simplifier and a printer. rsgb carries all of them; they are transforms
and analyses over `Function`, not a separate engine. The printer and the
exporters (text, LaTeX-style, Python expression) live in rsgb as well.
The nonlinearity-degree analysis is dropped (unused in SANE).

## 6. Program<T>: the execution form

`lower::<T>(f) -> Program<T>`: reachability, register-pressure list
scheduling, superinstruction fusion (`MulAdd`, `Sub`, fused reductions and
dots), liveness-driven slot allocation, prolog split (parameter-pure
prefix), constants in `T`, positional inputs over a caller-owned work
buffer. Program-level transforms: `specialize` (choice trace with guards),
`batch` (lanes for `Map`), parameter binding (persistent buffers for
`Param` roles).

Backends implement one trait over `Program<T>`:
- `Interp`: the reference evaluator (SANE's tape).
- `Jit`: Cranelift chunked compilation with host trampolines for the
  transcendentals (SANE's jit), per `T`.
- `CSource`: emits C for a Program and its functions (fastsim's region
  and expression templates, reference special functions), with the
  compile-and-compare verification loop.

## 7. Bit-exactness

Every backend computes the same IEEE operation sequence as `Interp`: no
fast-math, no FMA contraction, one reference routine per transcendental
and per `T`, fixed reduction order above `REDUCE_SIMD_MIN`. `Fma` is a
backend option (`fast_fma`), never a graph node. The parity suite runs
`Interp == Jit == lanes == CSource` on random functions over the union op
set and on both consumers' fixtures, per `T`, and is the acceptance test
of every change.

## 8. Piecewise structure and events

The `Select` conditions reachable from a function's outputs are its
guards. `specialize` freezes them (region tracking in SANE's Newton
loops); an integrator can locate their crossing times with the guard
expressions' derivatives (fastsim's zero-crossing events for
Select-induced discontinuities). Explicit `Guard` outputs cover the rest;
scheduled events are the consumer's business.

## 9. Frontends

- `rsgb-py`: the Python tracer (fastsim's `JitTracer`, arrays, numpy
  tables) builds through a `Scope`; JAX-style `jit`, `jacobian`, `grad`.
- Native builders: the `Builder` trait of fastsim implemented by `Scope`
  and by plain `f64`; SANE's lowering crates build through `Scope`.

## 10. Crates

- `rsgb`: module, function, scope, transforms, symbolic analysis, lower,
  `Program`, `Interp`, roles. No dependency on either consumer; logging
  and clock are small traits with no-op defaults.
- `rsgb-jit`: Cranelift backend.
- `rsgb-c`: C backend and verification.
- `rsgb-py`: PyO3 tracer and bindings.

## 11. What stays in the consumers

SANE: netlist and Verilog-A lowering, MNA topology, DAE conventions,
solvers, harmonic balance, constants and configuration, logging and
profiling, diagnostics. fastsim: blocks, simulation loop, solvers, event
scheduling, FMI, C scaffolds and solver templates, the Python API
surface.
