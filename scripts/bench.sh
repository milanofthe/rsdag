#!/bin/sh
# The README's benchmark figures, measured afresh on this machine: the op
# sweep, the dense kernels, small matrices and composition (rsdag's
# examples), then docs/bench/plot.py over the CSVs. With RSDAG_MODULES naming a directory
# of circuit modules (SANE's export_module) and casadi, jax and psutil
# installed, also the comparison against CasADi and JAX
# (docs/bench/modules.py). One core at a time; the comparison takes a
# while, JAX up to a minute per circuit. PYTHON names the interpreter.
set -eu
cd "$(dirname "$0")/.."
D=docs/bench/data
mkdir -p "$D"
export CARGO_INCREMENTAL=0
echo "op sweep"
nice cargo run -q --release -p rsdag-jit --example sweep --features rsdag/synth > "$D/ops.csv"
echo "dense kernels"
nice cargo run -q --release -p rsdag --example dense > "$D/dense.csv"
echo "small matrices"
nice cargo run -q --release -p rsdag-jit --example matrices > "$D/matrices.csv"
echo "composition"
nice cargo run -q --release -p rsdag --example hierarchy > "$D/hierarchy.csv"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
if [ -n "${RSDAG_MODULES:-}" ]; then
    M="$RSDAG_MODULES"
    mkdir -p "$M/values" "$M/composed"
    echo "circuits: rsdag"
    nice cargo run -q --release -p rsdag-jit --example modules -- --values "$M/values" --composed "$M/composed" "$M"/*.json > "$D/modules_rsdag.csv"
    # the other tools get the modules as rsdag compiles them (see modules.rs)
    echo "circuits: CasADi, JAX"
    nice "$PY" docs/bench/modules.py "$M/values" "$M"/composed/*.json > "$D/modules_other.csv"
fi
"$PY" docs/bench/plot.py
