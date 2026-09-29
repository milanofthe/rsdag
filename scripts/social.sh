#!/bin/sh
# The social cards: rsdag draws the diagrams in the cards' theme
# (examples/diagrams.rs --social), Graphviz renders them to PNGs, and
# docs/social/make.py composes them with the benchmark plots into
# docs/social/*.png.
set -eu
cd "$(dirname "$0")/.."
command -v dot >/dev/null || { echo "Graphviz (dot) is needed"; exit 1; }
OUT=target/social
cargo run -q -p rsdag --example diagrams -- "$OUT" --social
for f in "$OUT"/*.dot; do
    dot -Tpng -Gdpi=200 "$f" -o "${f%.dot}.png"
done
"$(command -v python3 || command -v python)" docs/social/make.py "$OUT"
