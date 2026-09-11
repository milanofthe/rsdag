"""The README's benchmark figures from the CSVs in docs/bench/data.

    python3 docs/bench/plot.py

Writes docs/bench/*.svg. The data files come from the benchmark binaries
named in docs/bench/README.md; rerun them on a change that moves the numbers.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BLUE, ORANGE, GREY, GREEN = "#1f4e79", "#c55a11", "#7f7f7f", "#3a7d44"

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e3e3e3",
    "grid.linewidth": 0.6,
    "svg.fonttype": "none",
})


def rows(name):
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name), format="svg", bbox_inches="tight")
    plt.close(fig)


def ops():
    r = rows("ops.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for vocab, color in (("ring", BLUE), ("elementary", ORANGE), ("full", GREEN)):
        sel = [x for x in r if x["vocab"] == vocab]
        n = [int(x["ops"]) for x in sel]
        a.plot(n, [float(x["interp_ns_per_op"]) for x in sel], "o--", color=color, ms=4, label=f"{vocab}, interpreter")
        a.plot(n, [float(x["native_ns_per_op"]) for x in sel], "o-", color=color, ms=4, label=f"{vocab}, native")
        b.plot(n, [float(x["compile_ns_per_op"]) for x in sel], "o-", color=color, ms=4, label=vocab)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("ops in the program"); a.set_ylabel("ns per op")
    a.set_title("Evaluation")
    a.legend(fontsize=8, frameon=False, loc="center right")
    b.set_xscale("log")
    b.set_xlabel("ops in the program"); b.set_ylabel("ns per op")
    b.set_title("Native compile")
    b.legend(fontsize=8, frameon=False, loc="upper right")
    save(fig, "ops.svg")


def solve():
    r = rows("solve.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for fam, color in (("ring", BLUE), ("band", ORANGE), ("grid", GREEN)):
        sel = [x for x in r if x["family"] == fam]
        n = [int(x["n"]) for x in sel]
        a.plot(n, [float(x["graph_us"]) for x in sel], "o-", color=color, ms=4, label=f"{fam}, graph solve")
        a.plot(n, [float(x["rslab_us"]) for x in sel], "o--", color=color, ms=4, label=f"{fam}, sparse LU library")
        b.plot(n, [float(x["ops_per_unknown"]) for x in sel], "o-", color=color, ms=4, label=fam)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("unknowns"); a.set_ylabel("us per Newton step")
    a.set_title("Newton step: factor and solve")
    a.legend(fontsize=8, frameon=False)
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("unknowns"); b.set_ylabel("ops per unknown")
    b.set_title("Program size")
    b.legend(fontsize=8, frameon=False)
    save(fig, "solve.svg")


def dense():
    r = rows("dense.csv")
    fig, a = plt.subplots(1, 1, figsize=(4.6, 3.2))
    for kernel, color in (("gemv", BLUE), ("gemm", ORANGE), ("solve", GREEN)):
        sel = [x for x in r if x["kernel"] == kernel]
        a.plot([int(x["n"]) for x in sel], [float(x["gflops"]) for x in sel], "o-", color=color, ms=4, label=kernel)
    a.set_xscale("log")
    a.set_xlabel("n (n by n)"); a.set_ylabel("GF/s, one core")
    a.set_title("Dense kernels")
    a.legend(fontsize=8, frameon=False)
    save(fig, "dense.svg")


if __name__ == "__main__":
    ops()
    solve()
    dense()
    print("wrote", ", ".join(f"docs/bench/{n}" for n in ("ops.svg", "solve.svg", "dense.svg")))
