#!/bin/sh
# The README's benchmark figures, measured afresh on this machine: the op
# sweep and the dense kernels (rsdag's examples), then docs/bench/plot.py
# over the CSVs; with casadi and jax installed, also the comparison against
# them (docs/bench/compare.py). One core at a time, a few minutes.
set -eu
cd "$(dirname "$0")/.."
D=docs/bench/data
mkdir -p "$D"
export CARGO_INCREMENTAL=0
echo "op sweep"
nice cargo run -q --release -p rsdag-jit --example sweep --features rsdag/synth > "$D/ops.csv"
echo "dense kernels"
nice cargo run -q --release -p rsdag --example dense > "$D/dense.csv"
PY="$(command -v python3 || command -v python)"
# Against CasADi and JAX, from Python: needs rsdag, casadi and jax there.
if "$PY" -c "import rsdag, casadi, jax" 2>/dev/null; then
    echo "rsdag, CasADi, JAX"
    nice "$PY" docs/bench/compare.py > "$D/compare.csv"
else
    echo "compare skipped: install rsdag, casadi and jax for it"
fi
"$PY" docs/bench/plot.py
