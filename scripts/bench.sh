#!/bin/sh
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
    COMPOSED_AS_RSDAG_COMPILES_THEM="$M/composed"
    mkdir -p "$M/values" "$COMPOSED_AS_RSDAG_COMPILES_THEM"
    echo "circuits: rsdag"
    nice cargo run -q --release -p rsdag-jit --example modules -- --values "$M/values" --composed "$COMPOSED_AS_RSDAG_COMPILES_THEM" "$M"/*.json > "$D/modules_rsdag.csv"
    echo "circuits: CasADi, JAX"
    nice "$PY" docs/bench/modules.py "$M/values" "$COMPOSED_AS_RSDAG_COMPILES_THEM"/*.json > "$D/modules_other.csv"
fi
"$PY" docs/bench/plot.py
