#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export CARGO_TERM_COLOR=always
FEATURES='rsdag/synth rsdag/serde'
STABLE_BIN=$(dirname "$(rustup which cargo --toolchain stable)")
export PATH="$STABLE_BIN:$PATH"
CARGO=cargo
run() { echo "== $*"; "$@"; }
run $CARGO fmt --all -- --check
for probe in "" rsprobe/on; do
    run $CARGO clippy --workspace --exclude rsdag-py --all-targets --features "$FEATURES $probe" -- -D warnings
    run $CARGO test --workspace --exclude rsdag-py --features "$FEATURES $probe"
done
test_sse2() {
    RSDAG_SSE2=1 run $CARGO test -p rsdag-jit --features rsdag/synth
    RSDAG_SSE2=1 run $CARGO test -p rsdag --features synth --test simd_kernels
}
test_sse2
run $CARGO clippy -p rsdag --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' run $CARGO doc --no-deps --workspace --exclude rsdag-py --features "$FEATURES rsdag/exact"
for f in "" exact serde exact,serde; do
    run $CARGO clippy -p rsdag --lib --no-default-features --features "$f" -- -D warnings
done
check_wasm() {
    for probe in "" ,rsprobe/on; do
        run $CARGO build --release -p rsdag --example wasm_probe --target wasm32-unknown-unknown --features "exact,serde$probe"
        run python3 scripts/wasm_imports.py target/wasm32-unknown-unknown/release/examples/wasm_probe.wasm
    done
}
check_wasm
run $CARGO check -p rsdag-py
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
