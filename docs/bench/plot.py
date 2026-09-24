"""The README's benchmark figures from the CSVs in docs/bench/data.

    python3 docs/bench/plot.py

Writes docs/bench/*.svg. scripts/bench.sh measures the CSVs afresh and runs
this; the plots alone redraw from the committed ones.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BLUE, ORANGE, GREY, GREEN, PURPLE = "#4f86c6", "#d9822b", "#8b8b8b", "#4f9d5f", "#8e6bb8"

# Transparent, grey axes and text: legible on a light and on a dark page.
plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GREY,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "svg.fonttype": "none",
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.transparent": True,
    "text.color": GREY,
    "axes.labelcolor": GREY,
    "axes.titlecolor": GREY,
    "axes.edgecolor": GREY,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "legend.labelcolor": GREY,
})


def rows(name):
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


def key(fig, colors, styles):
    """One legend under the figure: a colour per series, then the line and
    marker styles."""
    handles = [Line2D([], [], color=c, lw=2, label=l) for l, c in colors]
    handles += [Line2D([], [], color=GREY, ls=ls, marker=m, ms=4, label=l) for l, ls, m in styles]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.06))


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
        b.plot(n, [float(x["tape_compile_ns_per_op"]) for x in sel], "o--", color=color, ms=4, label=f"{vocab}, tape")
        b.plot(n, [float(x["compile_ns_per_op"]) for x in sel], "o-", color=color, ms=4, label=f"{vocab}, native")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("ops in the program"); a.set_ylabel("ns per op")
    a.set_title("Evaluation")
    b.set_xscale("log")
    b.set_xlabel("ops in the program"); b.set_ylabel("ns per op")
    b.set_title("Compile")
    key(fig, (("ring", BLUE), ("elementary", ORANGE), ("full", GREEN)),
        (("native", "-", "o"), ("interpreter, tape", "--", "o")))
    save(fig, "ops.svg")


FAMILIES = (("ring", BLUE), ("band", ORANGE), ("grid", GREEN), ("random", PURPLE))


def solve():
    r = rows("solve.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(10.4, 3.1))
    for fam, color in FAMILIES:
        sel = [x for x in r if x["family"] == fam]
        n = [int(x["n"]) for x in sel]
        a.plot(n, [float(x["rsdag_us"]) for x in sel], "-", color=color, label=f"{fam}, rsdag")
        a.plot(n, [float(x["klu_us"]) for x in sel], "--", color=color, label=f"{fam}, KLU")
        c.plot(n, [float(x["build_ms"]) for x in sel], "-", color=color, label=f"{fam}, rsdag")
        c.plot(n, [float(x["klu_build_ms"]) for x in sel], "--", color=color, label=f"{fam}, KLU")
        b.plot(n, [float(x["ops_per_unknown"]) for x in sel], "-", color=color, label=fam)
        # Squares where the default chose the supernodal program.
        for x in sel:
            m = "s" if x["program"] == "supernodal" else "o"
            a.plot([int(x["n"])], [float(x["rsdag_us"])], m, color=color, ms=4)
            b.plot([int(x["n"])], [float(x["ops_per_unknown"])], m, color=color, ms=4)
            c.plot([int(x["n"])], [float(x["build_ms"])], m, color=color, ms=4)
    for ax in (a, b, c):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("unknowns")
    a.set_ylabel("us per Newton step"); a.set_title("Factor and solve")
    b.set_ylabel("ops per unknown"); b.set_title("Program size")
    c.set_ylabel("ms"); c.set_title("Build (analysis, compile)")
    key(fig, [(f, c) for f, c in FAMILIES],
        (("rsdag", "-", None), ("KLU", "--", None), ("scalar", "", "o"), ("supernodal", "", "s")))
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
