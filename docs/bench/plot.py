"""The README's benchmark figures from the CSVs in docs/bench/data.

    python3 docs/bench/plot.py

Writes docs/bench/*.svg and *.png, both on a transparent background.
scripts/bench.sh measures the CSVs afresh and runs this; the plots alone
redraw from the committed ones.
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, os.path.join(HERE, ".."))
from style import BLUE, FONT, GREEN, GREY, ORANGE  # noqa: E402

# Transparent, grey axes and text: legible on a light and on a dark page.
plt.rcParams.update({
    "font.family": FONT,
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
    "hatch.linewidth": 1.0,
})


def rows(name):
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


def key(fig, colors, styles, y=-0.06):
    """One legend under the figure (at `y`, in figure coordinates): a colour
    per series, then the line and marker styles."""
    handles = [Line2D([], [], color=c, lw=2, label=l) for l, c in colors]
    handles += [Line2D([], [], color=GREY, ls=ls, marker=m, ms=4, label=l) for l, ls, m in styles]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, y))


def save(fig, name):
    """`name`.svg and `name`.png, both on a transparent background."""
    fig.tight_layout()
    for ext, opts in (("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight", **opts)
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
    save(fig, "ops")


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
    save(fig, "dense")


def matrices():
    """Small matrices through a tape (matrices.csv): `A x`, `A B` and
    `A \\ b` of n by n inputs, one instance and sixteen of one shape (the
    solves batched), per instance, interpreted and native."""
    r = rows("matrices.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    kinds = (("gemv", "A x", BLUE), ("gemm", "A B", ORANGE), ("solve", "A \\ b", GREEN))
    for a, (count, title) in zip(axes, (("1", "One instance"), ("16", "Sixteen instances"))):
        for kind, _, color in kinds:
            sel = [x for x in r if x["kind"] == kind and x["instances"] == count]
            n = [int(x["n"]) for x in sel]
            a.plot(n, [float(x["native_ns"]) for x in sel], "o-", color=color, ms=4)
            a.plot(n, [float(x["interp_ns"]) for x in sel], "o--", color=color, ms=4)
        a.set_xscale("log", base=2); a.set_yscale("log")
        a.set_xticks([2, 4, 8, 16, 32]); a.set_xticklabels(["2", "4", "8", "16", "32"])
        a.set_xlabel("n (n by n)")
        a.set_title(title)
    axes[0].set_ylabel("ns per instance")
    key(fig, [(label, color) for _, label, color in kinds],
        (("native", "-", "o"), ("interpreter", "--", "o")))
    save(fig, "matrices")


def hierarchy():
    """One instance of a wide body (hierarchy.csv): each stage over the
    body's width, every output a call over one argument list."""
    r = rows("hierarchy.csv")
    n = [int(x["n"]) for x in r]
    fig, a = plt.subplots(1, 1, figsize=(4.6, 3.2))
    stages = (("calls", GREY, ("build_ms",)), ("Jacobian", BLUE, ("jacobian_ms",)),
              ("x' = 0, specialized", ORANGE, ("substitute_ms", "specialize_ms")),
              ("program", GREEN, ("program_ms",)))
    for label, color, cols in stages:
        ms = [sum(float(x[c]) for c in cols) for x in r]
        a.plot(n, ms, "o-", color=color, ms=4, label=label)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("circuit nodes in the body (one instance)"); a.set_ylabel("ms")
    a.set_title("A wide body, called once")
    a.legend(fontsize=8, frameon=False)
    save(fig, "hierarchy")


def modules():
    """rsdag against CasADi and JAX on SANE's circuits (modules_rsdag.csv,
    modules_other.csv): the residual and its Jacobian per call, and the
    setup each takes (differentiation and compilation), with the parameters
    as inputs and as constants; the twelve BSIM4 amplifiers as their median
    and range. rsdag's interpreter (hatched) runs the same tape as its
    native code."""
    ours = rows("modules_rsdag.csv")
    other = rows("modules_other.csv")
    groups = (("uA741, BJT", lambda m: m == "ua741"),
              ("PSP103 ring", lambda m: m == "ring_psp103"),
              ("BSIM4 amplifiers (12)", lambda m: m not in ("ua741", "ring_psp103")))

    def ours_col(col, scale=1.0):
        return lambda m: [float(x[col]) * scale for x in ours if x["module"] == m]

    def other_col(tool, col, scale=1.0):
        return lambda m: [float(x[col]) * scale for x in other
                          if x["module"] == m and x["tool"] == tool and x[col]]

    ms = 1e3
    cas_in, cas = "CasADi (parameter inputs)", "CasADi"
    # label, color, solid (parameters constant), hatch; per module the residual
    # and the Jacobian per call (us), then their setup (ms)
    bars = (
        ("rsdag", BLUE, False, "////",
         ours_col("call_f_interp_us"), ours_col("call_j_interp_us"),
         ours_col("setup_f_interp_s", ms), ours_col("setup_j_interp_s", ms)),
        ("rsdag", BLUE, False, None,
         ours_col("call_f_native_us"), ours_col("call_j_native_us"),
         ours_col("setup_f_native_s", ms), ours_col("setup_j_native_s", ms)),
        ("CasADi", ORANGE, False, None,
         other_col(cas_in, "call_f_us"), other_col(cas_in, "call_j_us"),
         other_col(cas_in, "setup_f_s", ms), other_col(cas_in, "setup_j_s", ms)),
        ("rsdag", BLUE, True, None,
         ours_col("call_f_folded_us"), ours_col("call_j_folded_us"),
         ours_col("setup_f_folded_s", ms), ours_col("setup_j_folded_s", ms)),
        ("CasADi", ORANGE, True, None,
         other_col(cas, "call_f_us"), other_col(cas, "call_j_us"),
         other_col(cas, "setup_f_s", ms), other_col(cas, "setup_j_s", ms)),
        ("JAX", GREEN, True, None,
         other_col("JAX", "call_f_us"), other_col("JAX", "call_j_us"),
         other_col("JAX", "setup_f_s", ms), other_col("JAX", "setup_j_s", ms)),
    )
    modules_ = sorted({x["module"] for x in ours})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharey="row")
    width = 0.13
    panels = ((axes[0, 0], "Residual, per call", 4), (axes[0, 1], "Jacobian, per call", 5),
              (axes[1, 0], "Residual, setup", 6), (axes[1, 1], "Jacobian, setup", 7))
    for a, title, k in panels:
        for gi, (gname, member) in enumerate(groups):
            mods = [m for m in modules_ if member(m)]
            for bi, bar in enumerate(bars):
                vals = [v for m in mods for v in bar[k](m)]
                x = gi + (bi - (len(bars) - 1) / 2) * width
                if not vals:
                    a.text(x, 1.2, "x", color=bar[1], ha="center", va="bottom", fontsize=9)
                    continue
                med = float(np.median(vals))
                # Constants filled, inputs outlined, the interpreter hatched:
                # no transparency, legible on a light and a dark page.
                a.bar(x, med, width * 0.9, facecolor=bar[1] if bar[2] else "none",
                      edgecolor=bar[1], linewidth=1.2, hatch=bar[3])
                if len(vals) > 1:
                    a.errorbar(x, med, yerr=[[med - min(vals)], [max(vals) - med]], color=GREY,
                               lw=0.8, capsize=2)
        a.set_yscale("log")
        a.set_xticks(range(len(groups)))
        a.set_xticklabels([g[0] for g in groups], fontsize=8)
        a.set_title(title)
        a.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("microseconds")
    axes[1, 0].set_ylabel("milliseconds")
    key(fig, (("rsdag", BLUE), ("CasADi", ORANGE), ("JAX", GREEN)), (), y=-0.015)
    fig.legend(handles=[Patch(fc="none", ec=GREY, lw=1.2, label="parameters as inputs"),
                        Patch(fc=GREY, ec=GREY, label="parameters as constants"),
                        Patch(fc="none", ec=GREY, lw=1.2, hatch="////", label="rsdag interpreter"),
                        Line2D([], [], color=GREY, marker="$x$", ls="", label="no compile in 1 min")],
               loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    save(fig, "modules")


if __name__ == "__main__":
    ops()
    dense()
    matrices()
    hierarchy()
    names = ["ops", "dense", "matrices", "hierarchy"]
    if os.path.exists(os.path.join(DATA, "modules_other.csv")):
        modules()
        names.append("modules")
    print("wrote", ", ".join(f"docs/bench/{n}.svg/.png" for n in names))
