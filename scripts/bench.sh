#!/bin/sh
# The README's benchmark figures, measured afresh on this machine: the op
# sweep, the dense kernels and small matrices (rsdag's examples), then
# docs/bench/plot.py over the CSVs. With RSDAG_MODULES naming a directory
# of circuit modules (SANE's export_module) and casadi, jax and psutil
# installed, also the comparison against CasADi and JAX
# (docs/bench/modules.py). One core at a time; the comparison takes a
# while, JAX up to a minute per circuit.
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
PY="$(command -v python3 || command -v python)"
if [ -n "${RSDAG_MODULES:-}" ]; then
    M="$RSDAG_MODULES"
    mkdir -p "$M/values"
    echo "circuits: rsdag"
    nice cargo run -q --release -p rsdag-jit --example modules -- --values "$M/values" "$M"/*.json > "$D/modules_rsdag.csv"
    echo "circuits: CasADi, JAX"
    nice "$PY" docs/bench/modules.py "$M/values" "$M"/*.json > "$D/modules_other.csv"
fi
"$PY" docs/bench/plot.py
