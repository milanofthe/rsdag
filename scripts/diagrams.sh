#!/bin/sh
# The README's diagrams: docs/diagrams/make.py draws them into
# docs/diagrams/*.svg and *.png, both on a transparent background.
set -eu
cd "$(dirname "$0")/.."
"$(command -v python3 || command -v python)" docs/diagrams/make.py
