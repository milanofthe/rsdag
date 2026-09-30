#!/bin/sh
# The social cards: docs/social/make.py draws them into docs/social/*.png,
# the benchmark card from docs/bench/data.
set -eu
cd "$(dirname "$0")/.."
"$(command -v python3 || command -v python)" docs/social/make.py
