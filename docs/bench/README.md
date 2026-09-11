# Benchmark figures

`plot.py` draws the README's figures from the CSVs in `data/`:

| file | source |
|---|---|
| `data/ops.csv` | `cargo run --release -p rsdag-jit --example sweep --features rsdag/synth` |
| `data/dense.csv` | `cargo run --release -p rsdag --example dense` |
| `data/solve.csv` | the Newton step of a circuit-like residual as a graph solve against a general sparse LU (rslab, KLU path), measured with the comparison harness outside this repository; columns `family,n,ops_per_unknown,graph_us,rslab_us` |

All numbers on one core of an Apple M3, release profile. Rerun on a change
that moves them and commit the CSVs with the figures.
