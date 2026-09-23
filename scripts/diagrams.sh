#!/bin/sh
# The README's diagrams: rsdag draws them (examples/diagrams.rs), Graphviz
# lays them out into docs/diagrams/*.svg.
set -eu
cd "$(dirname "$0")/.."
command -v dot >/dev/null || { echo "Graphviz (dot) is needed"; exit 1; }
OUT=target/diagrams
cargo run -q -p rsdag --example diagrams -- "$OUT"
for f in "$OUT"/*.dot; do
    dot -Tsvg "$f" -o "docs/diagrams/$(basename "$f" .dot).svg"
done
echo "docs/diagrams updated"
