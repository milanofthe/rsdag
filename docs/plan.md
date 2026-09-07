# Plan: extracting rsgb

Order chosen so that a consumer is green after every phase and no phase
depends on a decision that is still open. Estimates are working days for
one person with the two codebases in hand.

## Phase 0: decisions (this document, 1 day)

- Confirm the design decisions marked in `design.md`: rationals as the
  constant model, `Fma` as a backend option rather than a node, the region
  model for the system layer.
- Name the crates: `rsgb` (graph, tape, system), `rsgb-jit`, `rsgb-codegen`,
  `rsgb-py` (PyO3 tracer). One workspace, one version.

## Phase 1: seed from SANE core (3 to 4 days)

- Copy `sane/crates/core` into `crates/rsgb` without `constants.rs`,
  `config.rs`, `log.rs`, `profile.rs`, `diag.rs`, `mathfn.rs` (Verilog-A
  names stay in SANE). Replace the logging and clock uses by two small
  traits with no-op defaults.
- Copy `sane/crates/jit` into `crates/rsgb-jit`.
- Copy the parity tests (`fuzz_parity`, `parity`, `lane_parity`) and the
  bundle tests.
- Rename `Context` to `Graph`; keep every public function name otherwise,
  so SANE's switch is an import change.
- Deliverable: `cargo test --workspace` green in rsgb with SANE's tests.

## Phase 2: SANE consumes rsgb (2 days)

- `sane/crates/core` becomes a thin crate: `pub use rsgb::*` plus the
  SANE-specific modules that stayed. All other crates unchanged except
  imports.
- Vendor rsgb into SANE as a source copy (the rslab procedure:
  `vendor/rsgb`, `VENDOR.md`, resync script).
- Deliverable: SANE workspace tests, benchmarks (`paper/benchmarks`) and
  the JIT parity suite green, timings unchanged.

## Phase 3: union of the op vocabulary (3 days)

- Add fastsim's unary functions and `Binary(BinOp)` to `Node`, the
  reference implementations (`digamma`, `rand_uniform`, ...) to the shared
  math module, and their lowering to tape, JIT trampolines and the arena
  evaluator. Differentiation rules for every new op.
- Extend the parity fuzzer to the union set.
- Optimizer passes from `fastsim/src/ssa/optimize.rs` as `rsgb::optimize`,
  each proven bit-preserving by the fuzzer or gated.
- Deliverable: rsgb evaluates any graph fastsim can build today.

## Phase 4: fastsim consumes rsgb (5 to 7 days)

- `GraphBuilder` targets `rsgb::Graph`; `F64Builder` unchanged.
- `ssa::tape::InterpretedFn` becomes a wrapper over `rsgb::Tape` with
  parameter slots; `ssa::autodiff` wrappers over rsgb's differentiation
  (`jacobian_wrt_slot`, feedthrough patterns).
- The IR schema moves to `rsgb::system` (regions, states, memory, Lut1d as
  opaque operator, time input); `ir::builder` and `ir::eval` stay in
  fastsim but build on it.
- `compile` fuses through rsgb functions and calls (inline transform).
- Delete `src/ssa` when parity holds: `interpret == tape`, C == engine on
  the fastsim fixtures, the block test suite, and the pathsim benchmark
  numbers (no regression allowed).
- Deliverable: fastsim green on rsgb, `src/ssa` gone.

## Phase 5: frontends and backends into rsgb (4 days)

- Tracer: `src/tracer` moves to `crates/rsgb-py`; fastsim re-exports
  `fastsim.jit`. SANE gets a `sane.jit`-style device model entry later.
- C emission: the region and expression templates and `verify_c` move to
  `crates/rsgb-codegen`; fastsim keeps model, solver and scaffold
  templates.
- Cranelift JIT available to fastsim's static compile path (opt-in, same
  parity suite).
- Deliverable: one tracer, one C emitter, one JIT, both consumers green.

## Phase 6: new capabilities (when a consumer needs them)

- Reverse-mode AD as a graph transform (fastsim parameter estimation,
  pathsim optimization).
- Events in the system layer: guard outputs and effect regions, so guards
  are differentiable and specializable (SANE switching circuits, fastsim
  hybrid systems).
- Graph-level batching beyond fixed lanes.

## Rules for the whole extraction

- The parity suite is the acceptance test; a phase ends when it is green
  on both consumers, not when the code compiles.
- Timings are measured before and after every phase on the existing
  benchmarks (SANE `paper/benchmarks`, fastsim `benchmarks`), with the
  MDOF/s and steps/s numbers in the PR.
- rsgb is vendored into both consumers as a source copy, resynced by
  script, like rslab. Private repository, PolyForm Noncommercial.
- Short single-line commits, one PR per phase step, no co-author trailers.
