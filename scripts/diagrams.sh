#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
"$(command -v python3 || command -v python)" docs/diagrams/make.py
