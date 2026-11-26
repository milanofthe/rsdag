#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export CARGO_TERM_COLOR=always
FEATURES='rsdag/synth rsdag/serde'
STABLE_TOOLCHAIN_BIN_AHEAD_OF_HOMEBREW_CARGO=$(dirname "$(rustup which cargo --toolchain stable)")
export PATH="$STABLE_TOOLCHAIN_BIN_AHEAD_OF_HOMEBREW_CARGO:$PATH"
CARGO=cargo
run() { echo "== $*"; "$@"; }
run $CARGO fmt --all -- --check
for scopes_compiled_out_or_recording in "" rsprobe/on; do
    run $CARGO clippy --workspace --exclude rsdag-py --all-targets --features "$FEATURES $scopes_compiled_out_or_recording" -- -D warnings
    run $CARGO test --workspace --exclude rsdag-py --features "$FEATURES $scopes_compiled_out_or_recording"
done
test_sse2_baseline_that_cpus_with_sse41_and_avx_never_run() {
    RSDAG_SSE2=1 run $CARGO test -p rsdag-jit --features rsdag/synth
    RSDAG_SSE2=1 run $CARGO test -p rsdag --features synth --test simd_kernels
}
test_sse2_baseline_that_cpus_with_sse41_and_avx_never_run
run $CARGO clippy -p rsdag --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' run $CARGO doc --no-deps --workspace --exclude rsdag-py --features "$FEATURES rsdag/exact"
for f in "" exact serde exact,serde; do
    run $CARGO clippy -p rsdag --lib --no-default-features --features "$f" -- -D warnings
done
check_rsdag_links_for_the_browser_with_no_imports() {
    for probe in "" ,rsprobe/on; do
        run $CARGO build --release -p rsdag --example wasm_probe --target wasm32-unknown-unknown --features "exact,serde$probe"
        run python3 scripts/wasm_imports.py target/wasm32-unknown-unknown/release/examples/wasm_probe.wasm
    done
}
check_rsdag_links_for_the_browser_with_no_imports
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
