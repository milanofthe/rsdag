# Plan: building rsgb (design v2)

Order chosen so that a consumer is green after every phase. Estimates are
working days for one person with both codebases in hand. The parity suite
is the acceptance test of every phase; timings are measured before and
after on the existing benchmarks (SANE `paper/benchmarks`, fastsim
`benchmarks`) and go into the PR.

## Phase 0: decisions (done with design v2, confirm)

- Functions all the way down, roles as the system layer, transformations
  as `Function -> Function`, `Program<T>` as the execution form, two
  scalar axes, `Fma` as a backend option, symbolic analysis first class,
  nonlinearity analysis dropped.
- Crates: `rsgb`, `rsgb-jit`, `rsgb-c`, `rsgb-py`. One workspace.

## Phase 1: the core, seeded from SANE (5 to 6 days)

- `Module`, `Function`, `Scope`: SANE's `Context` split into the module
  (functions, externs, constants) and the per-function body; free symbols
  become `Param` nodes of an open scope. Node set as in design v2, generic
  over `K: Field` with `Rational` and `f64` instances.
- Transforms carried over: `jvp` (SANE autodiff), `inline`, `substitute`,
  `specialize`; `simplify` as the pass pipeline skeleton; the printer.
- `lower::<T>` and `Program<T>` from SANE's tape compiler; `Interp` from
  the tape evaluator; `Map` lowered to lanes from `eval_batch`.
- Parity fuzzer `Interp` vs the arena reference over the SANE op set.
- Deliverable: `cargo test --workspace` green; SANE's core tests ported.

## Phase 2: SANE consumes rsgb (4 to 5 days)

- `sane-core` becomes rsgb re-exports plus the SANE-only modules. Netlist,
  Verilog-A and OSDI lowering build through a `Scope`; the DAE is a
  function with roles; `substitute_many` becomes `compose`/`substitute`.
- The symbolic analysis moves into rsgb: `collect` (rational canonical
  form), `det`, the e-graph simplifier. SANE's analysis crate calls them.
- `rsgb-jit` from SANE's jit; SANE's solver uses `Program` and `Jit`.
- Vendor rsgb into SANE as a source copy (rslab procedure).
- Deliverable: SANE workspace tests, Python tests, benchmarks and the
  JIT parity suite green, timings unchanged.

## Phase 3: union of the op vocabulary and the optimizer (3 days)

- fastsim's unary and binary ops, reference math (special functions,
  counter RNG), lowering to `Interp`, `Jit` trampolines, differentiation
  rules; parity fuzzer over the union set.
- Optimizer passes from fastsim into `simplify`, each proven
  bit-preserving or gated.
- Deliverable: rsgb evaluates any graph fastsim builds today.

## Phase 4: fastsim consumes rsgb (6 to 8 days)

- `GraphBuilder` targets a `Scope`; `F64Builder` unchanged.
- Blocks become functions with roles; the IR module becomes an rsgb
  `Module` with block and subsystem metadata; `ir::eval` becomes `Interp`.
- `InterpretedFn` and `jacobian_wrt_slot` become thin wrappers over
  `Program` and `jvp` with role subsets; feedthrough patterns from
  `sparsity`.
- Static compile uses `inline` and `Map`.
- Delete `src/ssa` and the IR op set when parity holds: `Interp == C` on
  the fastsim fixtures, block suite, pathsim benchmark numbers.
- Deliverable: fastsim green on rsgb.

## Phase 5: frontends and backends into rsgb (4 days)

- `rsgb-py`: the tracer, `jit`, `jacobian`, `grad`; fastsim re-exports.
- `rsgb-c`: region and expression templates plus the verification loop;
  fastsim keeps model, solver and scaffold templates.
- `Jit` available to fastsim's static compile (opt-in, same parity suite).

## Phase 6: new capabilities (when a consumer needs them)

- `vjp` (reverse mode), guard-based event location, `Complex<f64>`
  programs in `Jit`, `f32` programs, graph-level batching beyond lanes.

## Rules

- A phase ends when the parity suite is green on both consumers, not when
  the code compiles.
- rsgb is vendored into both consumers as a source copy, resynced by
  script. Private repository, PolyForm Noncommercial.
- Short single-line commits, one PR per phase step, no co-author trailers.
