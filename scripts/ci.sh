#!/bin/sh
# The CI gate, locally and with the toolchain CI uses (rustup stable), so a
# push is green before it happens. Mirrors .github/workflows/ci.yml.
set -eu
cd "$(dirname "$0")/.."
export CARGO_TERM_COLOR=always
FEATURES='rsdag/synth rsdag/serde'
CARGO='rustup run stable cargo'
run() { echo "== $*"; "$@"; }
run $CARGO fmt --all -- --check
run $CARGO clippy --workspace --exclude rsdag-py --all-targets --features "$FEATURES" -- -D warnings
run $CARGO test --workspace --exclude rsdag-py --features "$FEATURES"
run $CARGO clippy -p rsdag --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' run $CARGO doc --no-deps --workspace --exclude rsdag-py --features "$FEATURES"
run $CARGO check -p rsdag-py
echo "== green"
