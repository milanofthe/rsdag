# Inventory: what SANE and fastsim have today

Read from the code on 2026-09-07 (SANE main db59ea0 area, fastsim master).
Line numbers are approximate and only meant as pointers.

## 1. SANE core (`sane/crates/core`, 6.8k lines) and jit (1.6k lines)

### Graph
- Hash-consed scalar DAG in `Context` (`context.rs`): `nodes: Vec<Node>`,
  `dedup: HashMap<Node, ExprId>`, exact rational constants
  (`consts: Vec<BigRational>`, deduplicated, plus an `f64_cache`), an
  argument pool for variadic nodes (`arg_pool`, `arg_dedup`), symbol table
  by name, functions and outputs, and a reusable `Memo` for traversals.
- `Node` (`node.rs:263`), 16 bytes, size asserted: `Const(ConstId)`,
  `Symbol(SymbolId)`, `Add`, `Mul` (operands ordered by id), `Neg`,
  `Pow(e, i64)`, `Unary(UnaryOp, e)`, `Cmp(CmpOp, a, b)` yielding 1.0/0.0,
  `Select(c, t, e)`, `Reduce(ReduceOp, ArgList)` (Sum, Product, Min, Max),
  `Dot(ArgList)` (two halves in one list), `Call(OutputId, ArgList)`.
  There is also an opaque operator bound to `ExternBundle` bodies
  (`extern_fn.rs`).
- `UnaryOp`: Exp, Ln, Sqrt, Sin, Cos, Sinh, Cosh, Tanh, Atan, Floor, with
  domain guards (`EXP_LIMIT = 80`, `LN_FLOOR = 1e-30`) applied identically
  in every backend. `CmpOp`: Gt, Ge, Lt, Le, Eq, Ne.
- Ascending `ExprId` is a topological order (a node's id is larger than its
  children), which the tape relies on.

### Construction rules (`context.rs:329..600`)
- `add`: zero identities, constant folding on rationals, operand ordering.
- `mul`: zero and one identities, folding, ordering. `sub` = add(neg),
  `div` = mul(recip), `recip` = pow(-1).
- `neg`: folding, double negation cancels. `pow_i`: 0 and 1 exponents,
  folding (except 0 to a negative power), nested powers multiply.
- `select`: constant condition resolves, equal branches collapse.
- `reduce`: fused associative reductions (KCL sums) instead of binary trees;
  `dot` for inner products (matrix-vector rows). Reductions of at least
  `REDUCE_SIMD_MIN = 16` operands use a fixed 4-lane accumulator order so
  the arena sweep and the tape agree bit for bit.

### Functions and calls (`func.rs`)
- `Func { name, params: Vec<SymbolId>, outputs: Vec<Output>, body:
  FuncBody, compiled: Option<CompiledBody> }`. One compact model is one
  function with many `Call` nodes. Derivative outputs are differentiated
  from the body on first demand and memoised (`derivative_output`,
  `declare_derivative`). Calls can be inlined (`inline_call`,
  `inline_outputs`). `ExternBundle` (object-safe, `Arc<dyn>`) plugs a
  compiled multi-output body into an opaque operator; the tape calls it
  (`BundleCall`, batched as `BundleBatch` when every group's arguments are
  hoistable, `BundlePick` reads an output).

### Differentiation (`autodiff.rs`, 695 lines)
- Symbolic, forward, builds new nodes in the same context: `differentiate`,
  `gradient`, `jacobian`, `hessian`, `sparsity`, `time_derivative`. Handles
  every node kind including Select (branch-wise), Cmp (zero), Reduce, Dot
  and Call (chain rule over derivative outputs). Any order by repetition.
- `dae/sens.rs` builds directional derivatives, forward sensitivity
  augmentation, Hessian blocks and small-signal matrix derivatives purely
  symbolically on top of it.

### Analyses and transforms
- `nonlinearity.rs`: exact polynomial degree of an expression in a chosen
  variable set, flags transcendental, rational, piecewise, opaque parts.
  Consumed by harmonic balance for exact oversampling.
- `transform.rs`: symbol substitution (`substitute`, `substitute_many`).
- `simplify` crate (385 lines): a small rewrite language on top.
- `display.rs`, `diag.rs`: printing and source snippets. `eval.rs`: arena
  sweep reference evaluator over a symbol environment.
- `mathfn.rs`: lowering of named math calls (Verilog-A style) to nodes.

### Tape (`tape/mod.rs`, `tape/compile.rs`, 1.4k lines)
- Flat instruction list over compact slots, memory-based SSA in a
  caller-owned work buffer; inputs positional. Ops: Const, Input, Add, Mul,
  MulAdd (fused dispatch, not fused rounding), Sub, Neg, Powi, Unary, Cmp,
  Select, Reduce, Dot, BundleCall, BundleBatch, BundlePick.
- `compile.rs`: reachability, register-pressure list scheduling,
  superinstruction fusion, liveness-driven slot allocation, and the
  parameter-pure prolog split (`eval_prolog` / `eval_main`).
- `eval_traced` records the Select choices; `specialize.rs` shortens a tape
  against a recorded choice trace and keeps guard outputs that detect a
  region flip (`SpecializedTape`, `eval_checked`, `prolog_guards`).
- `eval_batch<const L>` evaluates lanes; `TapeVisitor` lowers to backends.

### JIT (`jit/src/lib.rs`)
- Cranelift, chunked: the op stream is cut into functions of `CHUNK_OPS =
  1024` (16384 for bundle bodies), compiled in parallel; values crossing a
  chunk stay in the work array. Transcendentals go through host
  trampolines so guards match the interpreter. `ChunkedTape::compile`,
  `compile_live` (keeps extra slots live for specialization guards),
  `eval_prolog`, `eval_main`. `LaneTape` (`LANES = 2`) is the SIMD-lane
  variant. Bit-exactness against the tape is a hard invariant, pinned by
  `jit/tests/parity.rs` and `lane_parity.rs`; `core/tests/fuzz_parity.rs`
  pins arena == tape.
- SANE's solver runs compiles on a dedicated pool (`solve/src/eval.rs`) and
  swaps backends at episode boundaries (`PrologToken`).

### SANE-specific things in core that must not move
- `constants.rs` (physics and solver knobs), `config.rs` (engine config
  from the environment), `log.rs` (SANE logging), `profile.rs` (pipeline
  stage timing), `time.rs` (wasm-safe clock), `mathfn.rs` names bound to
  Verilog-A, `diag.rs`. The backend needs its own minimal logging hook and
  clock, or takes them as traits.

### Consumers of the core (what they call)
- `dae`: `differentiate`, `jacobian`, `substitute_many`, `eval`,
  `eval_real`, `nonlinearity_of`, `sparsity`, `Context` construction,
  `ReduceOp`, `CmpOp`, `lower_math_call`.
- `solve`: `Tape`, `SpecializedTape`, `Context`, `ExternBundle`,
  `CompiledBody`, `FuncId`, `Output`, `Node`, `ReduceOp`, plus the jit
  crate directly.
- `veriloga`, `osdi`, `device`, `netlist`, `analysis`, `export`, `py`:
  `Context`, `ExprId`, `SymbolId`, `Node::Symbol`, `ExternBundle`,
  `Output`, `eval_real`, `gradient`, `hessian`, `nonlinearity`,
  `UnaryOp`, and the SANE-specific infra modules.

## 2. fastsim (`src/ssa` 5.2k, `tracer` 4.1k, `ir` 2.6k, `codegen` 5.1k,
`compile` 2.9k, `events` 1.2k lines)

### Graph (`ssa/graph.rs`, `ssa/op.rs`)
- Hash-consed scalar DAG `Graph { nodes, dedup, outputs, signature:
  InputSignature, n_params, param_defaults, param_names }`.
- `Node`: `Const(u64 bits)`, `Input(flat index)`, `Param(index)`,
  `Binary(BinOp, a, b)`, `Unary(UnaryOp, a)`, `Cmp`, `Select`, `Fma(a, b,
  c)` (single rounding), `Reduce(ReduceOp, Vec)`, `Dot(Vec, Vec)`
  (accumulated with `mul_add`).
- `BinOp`: Add, Sub, Mul, Div, Pow, Mod, Min, Max, Atan2, Hypot.
  `UnaryOp`: Neg, Sin, Cos, Tan, Atan, Sinh, Cosh, Tanh, Exp, Log, Log10,
  Abs, Sqrt, Sign, Floor, Asin, Acos, Asinh, Acosh, Atanh, Ceil, Round,
  Trunc, Log2, Log1p, Expm1, Cbrt, Erf, Erfc, Lgamma, Tgamma, Digamma,
  RandUniform (counter-based hash RNG). Special functions have their own
  implementations (`digamma`, `rand_uniform`) shared with the C backend
  (`unary_c_fn`, `binary_c_fn`).
- Constants are f64 (bit patterns as keys); no rationals.

### Construction (`ssa/build.rs`)
- `Builder` trait: block math written once, instantiated as `F64Builder`
  (plain arithmetic, monomorphised into the native hot path) or
  `GraphBuilder` (records nodes). Methods cover the whole op set (cst, add,
  sub, mul, div, powf, modulo, min, max, hypot, neg, trig, hyperbolic, exp,
  ln, abs, sqrt, sign, floor, rand_uniform, inverse trig, ...). This is the
  seam a shared backend plugs into.

### Optimizer (`ssa/optimize.rs`)
- `optimize` runs to a fixed point: dead code elimination, `canonicalize`
  (operand ordering and re-interning), constant folding with domain checks
  (`unary_fold_in_domain`), strength reduction, algebraic simplification,
  FMA detection, `reassociate_chains`, and `lower_for_tape`.

### Autodiff (`ssa/autodiff.rs`)
- Symbolic forward: `differentiate`, `jacobian`, `jacobian_wrt_slot`,
  `jacobian_wrt_param`, `jacobian_is_constant`, `jacobian_sparse_wrt_slot`,
  `jacobian_wrt_slot_optimized`, `has_feedthrough`, `feedthrough_pattern`.
  Feedthrough structure feeds the block scheduler.

### Tape (`ssa/tape.rs`)
- `TapeOp { opcode: u8, arg0, arg1, arg2: u32 }`, liveness-driven work-slot
  reuse, `InterpretedFn::from_graph`, `call`, `call_into`, mutable params
  (`set_param`). Differential fuzzer pins `Graph::interpret == tape`.

### Tracer (`tracer/`, PyO3-gated frontend)
- Operator-overloading trace: `JitTracer` / `JitTracerArray` values with
  numpy ufunc and array-function tables (`ufunc_table.rs`, `ndshape.rs`),
  `jit_where`, `jit_clip`, random keys, `trace_with_signature`,
  `_trace_ode`, `_trace_function_block`, `_trace_source`. Exposed in Python
  as `fastsim.jit` (`jit(func)`, `jacobian(func)`) and used by block
  constructors. Data-dependent Python control flow is not traceable.

### IR (`ir/schema.rs`, `ir/builder.rs`, `ir/eval.rs`)
- Hierarchical serializable `Module` snapshot of a `Simulation`: blocks,
  subsystems, ports, `StateVar`, `MemorySlot`, `Regions` per block (`alg`:
  outputs, `dyn`: state derivatives, event effects). A region is a list of
  scalar ops plus ordered writes (`Output`, `StateDeriv`, `StateWrite`,
  `MemoryWrite`). Region op set = graph ops plus `Time`, `Input { port,
  elem }`, `Param`, `State`, `Memory`, `Lut1d`, `Call { ExternId }`.
- `builder.rs` derives it from a live simulation (`module_from_sim`,
  `subsystem_to_ir`); `eval.rs` is the IR's own interpreter.

### Codegen (`codegen/`)
- Jinja templates (minijinja) for C: model and block structs, regions,
  solver implementations, scaffold with CMake, A2L; options for numeric
  type, reductions, structure, layout, solver choice. `verify_c` compiles
  the generated C locally and pins it sample by sample against the engine.

### Compile (`compile/`)
- Static fusion of a whole `Module` into one graph `dX/dt = F(X, t)` over a
  global state vector (`flatten`, `splice`), lowered to one tape and
  integrated by the existing solvers (`CompiledSimulation`); trades
  runtime mutability for cross-block CSE.

### Events (`events/`)
- `Event`, `EventDescriptor`, `SimEvent` trait, `ZeroCrossing` with
  direction, `Condition`, `Schedule` / `ScheduleList`, active flags. Event
  functions and effects are Python callbacks or IR regions; zero-crossing
  detection works on sampled guard values, not on graph derivatives.

### Consumers of the graph
- `blocks`: `GraphBuilder`, `Graph`, `InterpretedFn`, `feedthrough_pattern`,
  `ir::builder`.
- `pybindings`: `jacobian_wrt_slot`, `optimize`, `InterpretedFn`,
  `module_from_sim`.
- `compile`: `jacobian_wrt_slot_optimized`, `optimize`, `InterpretedFn`,
  the IR schema.
- `codegen`: `ir::eval`, `ssa::op::{unary_c_fn, binary_c_fn, digamma,
  numpy_sign}`, `autodiff`.
- `fmi`: the IR `Module` and `Direction`.

## 3. Side by side

| capability | SANE core | fastsim ssa |
|---|---|---|
| constants | exact rationals | f64 bits |
| hierarchy | functions with calls, memoised derivatives | blocks in the IR, no calls in the graph |
| ops beyond arithmetic | Pow(int), 10 unary, Reduce, Dot, Call, Opaque | 10 binary, 33 unary, Fma, Reduce, Dot |
| AD | forward symbolic, any order, sparsity, Hessian blocks | forward symbolic, Jacobian sparsity, feedthrough |
| optimizer | rules at construction, fused Reduce/Dot | fixed-point passes on the graph |
| tape | memory SSA, prolog split, specialization, batching | opcode tape, slot reuse, mutable params |
| native | Cranelift chunked JIT, SIMD lanes | C codegen with SIL verification |
| frontend | netlist, Verilog-A, OSDI | Python tracer, block builders |
| state and time | outside the graph (DAE layer) | IR regions: State, Memory, Time, Lut1d |
| events | none | zero crossing, scheduled, conditions |
| parity pins | arena == tape == JIT == lanes | interpret == tape, C == engine |

Neither has reverse-mode AD, and neither has a graph-level batching model
beyond fixed lanes.
