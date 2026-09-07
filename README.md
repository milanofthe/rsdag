# rsgb

Rust symbolic graph backend: the shared expression-graph substrate of SANE and
fastsim. A hash-consed scalar DAG with symbolic differentiation, graph
optimization, a flat tape evaluator, a native JIT, code generation and a Python
tracer, so that every optimization lands in every consumer once.

Private for now. Licensed under PolyForm Noncommercial 1.0.0 (see LICENSE).

Development runs through GitHub issues, not documents: the inventory (#1),
the architecture (#2) and the phases (#3 onwards, each depending on the
previous one) live there, with dates, dependencies and history.
