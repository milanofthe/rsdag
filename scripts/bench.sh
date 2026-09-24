#!/bin/sh
# The README's benchmark figures, measured afresh on this machine: the op
# sweep and the dense kernels (rsdag's examples), the sparse solve against
# rslab's KLU (bench/, outside the workspace, so CI never builds it), then
# docs/bench/plot.py over the CSVs. One core at a time, a few minutes.
set -eu
cd "$(dirname "$0")/.."
D=docs/bench/data
mkdir -p "$D"
export CARGO_INCREMENTAL=0
echo "op sweep"
nice cargo run -q --release -p rsdag-jit --example sweep --features rsdag/synth > "$D/ops.csv"
echo "dense kernels"
nice cargo run -q --release -p rsdag --example dense > "$D/dense.csv"
echo "sparse solve against rslab"
nice cargo run -q --release --manifest-path bench/Cargo.toml --target-dir target > "$D/solve.csv"
python3 docs/bench/plot.py
