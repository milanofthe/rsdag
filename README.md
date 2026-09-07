# rsgb

Rust symbolic graph backend: the shared expression-graph substrate of SANE and
fastsim. A hash-consed scalar DAG with symbolic differentiation, graph
optimization, a flat tape evaluator, a native JIT, code generation and a Python
tracer, so that every optimization lands in every consumer once.

Private for now. Licensed under PolyForm Noncommercial 1.0.0 (see LICENSE).

- `docs/inventory.md`: what SANE and fastsim have today and what each consumer
  needs from the backend.
- `docs/design.md`: the architecture (functions all the way down, roles as
  the system layer, transformations, `Program<T>`, symbolic analysis).
- `docs/plan.md`: the extraction plan.
