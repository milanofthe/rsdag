#!/bin/sh
# The CI gate, locally and with the toolchain CI uses (rustup stable), so a
# push is green before it happens. Mirrors .github/workflows/ci.yml.
set -eu
cd "$(dirname "$0")/.."
export CARGO_TERM_COLOR=always
FEATURES='rsdag/synth rsdag/serde'
# The toolchain's own binaries first on the PATH: `rustup run` leaves a
# Homebrew cargo ahead of the toolchain's, and that one lacks the wasm target
# and mixes its rustc, rustdoc and clippy into one target directory.
TC=$(dirname "$(rustup which cargo --toolchain stable)")
export PATH="$TC:$PATH"
CARGO=cargo
run() { echo "== $*"; "$@"; }
run $CARGO fmt --all -- --check
run $CARGO clippy --workspace --exclude rsdag-py --all-targets --features "$FEATURES" -- -D warnings
run $CARGO test --workspace --exclude rsdag-py --features "$FEATURES"
run $CARGO clippy -p rsdag --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' run $CARGO doc --no-deps --workspace --exclude rsdag-py --features "$FEATURES rsdag/exact rsdag/egraph rsdag/complex"
for f in "" exact complex serde egraph exact,serde; do
    run $CARGO clippy -p rsdag --lib --no-default-features --features "$f" -- -D warnings
done
# The graph crate in the browser (a consumer's web build interprets there),
# linked and importing nothing.
run $CARGO build --release -p rsdag --example wasm_probe --target wasm32-unknown-unknown --features exact,complex,serde
run python3 scripts/wasm_imports.py target/wasm32-unknown-unknown/release/examples/wasm_probe.wasm
run $CARGO check -p rsdag-py
# The Python job: the wheel, installed into the interpreter that runs the
# tests. Skipped where maturin or pytest is missing.
if command -v maturin >/dev/null && python3 -m pytest --version >/dev/null 2>&1; then
    DIST="${TMPDIR:-/tmp}/rsdag-dist"
    rm -rf "$DIST"
    run maturin build --release -q -m crates/rsdag-py/Cargo.toml -o "$DIST"
    run python3 -m pip install -q --force-reinstall --no-deps "$DIST"/*.whl
    run python3 -m pytest -q crates/rsdag-py/tests
else
    echo "== python job skipped (maturin or pytest missing)"
fi
echo "== green"
