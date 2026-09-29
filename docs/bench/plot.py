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
from matplotlib.patches import Patch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BLUE, ORANGE, GREY, GREEN = "#4f86c6", "#d9822b", "#8b8b8b", "#4f9d5f"

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


def modules():
    """rsdag against CasADi and JAX on SANE's circuits (modules_rsdag.csv,
    modules_other.csv): the residual and its Jacobian per call, with the
    parameters as inputs and as constants; the twelve BSIM4 amplifiers as
    their median and range."""
    ours = rows("modules_rsdag.csv")
    other = rows("modules_other.csv")
    groups = (("uA741, BJT", lambda m: m == "ua741"),
              ("PSP103 ring", lambda m: m == "ring_psp103"),
              ("BSIM4 amplifiers (12)", lambda m: m not in ("ua741", "ring_psp103")))

    def ours_col(col):
        return lambda m: [float(x[col]) for x in ours if x["module"] == m]

    def other_col(tool, col):
        return lambda m: [float(x[col]) for x in other if x["module"] == m and x["tool"] == tool and x[col]]

    bars = (  # label, color, solid (parameters constant), value per module for F and J
        ("rsdag", BLUE, False, ours_col("call_f_native_us"), ours_col("call_j_native_us")),
        ("CasADi", ORANGE, False, other_col("CasADi (parameter inputs)", "call_f_us"),
         other_col("CasADi (parameter inputs)", "call_j_us")),
        ("rsdag", BLUE, True, ours_col("call_f_folded_us"), ours_col("call_j_folded_us")),
        ("CasADi", ORANGE, True, other_col("CasADi", "call_f_us"), other_col("CasADi", "call_j_us")),
        ("JAX", GREEN, True, other_col("JAX", "call_f_us"), other_col("JAX", "call_j_us")),
    )
    modules_ = sorted({x["module"] for x in ours})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    width = 0.15
    for a, (title, k) in zip(axes, (("Residual, per call", 3), ("Jacobian, per call", 4))):
        for gi, (gname, member) in enumerate(groups):
            mods = [m for m in modules_ if member(m)]
            for bi, bar in enumerate(bars):
                vals = [v for m in mods for v in bar[k](m)]
                x = gi + (bi - 2) * width
                if not vals:
                    a.text(x, 1.2, "x", color=bar[1], ha="center", va="bottom", fontsize=9)
                    continue
                med = float(np.median(vals))
                a.bar(x, med, width * 0.9, color=bar[1], alpha=1.0 if bar[2] else 0.35,
                      edgecolor=bar[1], linewidth=0.8)
                if len(vals) > 1:
                    a.errorbar(x, med, yerr=[[med - min(vals)], [max(vals) - med]], color=GREY,
                               lw=0.8, capsize=2)
        a.set_yscale("log")
        a.set_xticks(range(len(groups)))
        a.set_xticklabels([g[0] for g in groups], fontsize=8)
        a.set_title(title)
        a.grid(axis="x", visible=False)
    axes[0].set_ylabel("microseconds")
    key(fig, (("rsdag", BLUE), ("CasADi", ORANGE), ("JAX", GREEN)), ())
    fig.legend(handles=[Patch(fc=GREY, alpha=0.35, label="parameters as inputs"),
                        Patch(fc=GREY, label="parameters as constants"),
                        Line2D([], [], color=GREY, marker="$x$", ls="", label="no compile in 1 min")],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.13))
    save(fig, "modules.svg")


if __name__ == "__main__":
    ops()
    dense()
    names = ["ops.svg", "dense.svg"]
    if os.path.exists(os.path.join(DATA, "modules_other.csv")):
        modules()
        names.append("modules.svg")
    print("wrote", ", ".join(f"docs/bench/{n}" for n in names))
