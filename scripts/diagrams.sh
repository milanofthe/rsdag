#!/bin/sh
# The README's diagrams: docs/diagrams/make.py typesets the TikZ sources
# docs/diagrams/*.tex into *.svg and *.png on a transparent background
# (needs pdflatex, dvisvgm and pdftocairo).
set -eu
cd "$(dirname "$0")/.."
"$(command -v python3 || command -v python)" docs/diagrams/make.py
