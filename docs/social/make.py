"""Social cards for rsdag: 1080 x 1350 PNGs, black on white with one accent,
the blue of the benchmark plots.

    python docs/social/make.py

The benchmark card reads docs/bench/data. Writes docs/social/*.png.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "bench", "data")
W, H, DPI = 10.8, 13.5, 100
L, R = 0.9, W - 0.9
INK, GREY, RULE, ACCENT, PAPER = "#000000", "#6b6b6b", "#d4d4d4", "#4f86c6", "#ffffff"
LW = 1.4

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]


def card(title):
    """A page with its title; returns the figure and axes in inches."""
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(L, 12.3, title, color=INK, fontsize=40, va="top")
    return fig, ax


def box(ax, x, y, w, h, label="", filled=False, size=20):
    """A square-cornered box at its lower left corner, outlined or in the
    accent, its label centered."""
    ax.add_patch(Rectangle((x, y), w, h, fc=ACCENT if filled else "none",
                           ec="none" if filled else INK, lw=LW))
    if label:
        ax.text(x + w / 2, y + h / 2, label, color=PAPER if filled else INK,
                fontsize=size, ha="center", va="center")


def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=LW, mutation_scale=20,
                                shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=DPI, facecolor=PAPER)
    plt.close(fig)


def bodies():
    fig, ax = card("Batched device bodies")
    bx, by, bs = L, 4.6, 2.6
    ax.add_patch(Rectangle((bx, by), bs, bs, fc=ACCENT, ec="none"))
    ax.text(bx + 0.25, by + bs - 0.3, "BSIM4", color=PAPER, fontsize=22, va="top")
    ax.text(bx + 0.25, by + 0.3, "body", color=PAPER, fontsize=16, va="bottom")
    cols, rows, s, g = 3, 6, 0.62, 0.34
    gx = R - (cols * s + (cols - 1) * g)
    gy = by + bs / 2 - (rows * s + (rows - 1) * g) / 2
    for r in range(rows):
        for c in range(cols):
            box(ax, gx + c * (s + g), gy + r * (s + g), s, s)
    arrow(ax, (bx + bs + 0.35, by + bs / 2), (gx - 0.35, by + bs / 2))
    save(fig, "bodies.png")


def pipeline():
    fig, ax = card("From a function to native code")
    w, h, gap = 4.2, 1.0, 0.75
    x = (W - w) / 2
    steps = ["Rust or Python", "Graph", "Derivatives", "Tape"]
    y = 9.8
    for label in steps[:-1]:
        box(ax, x, y, w, h, label, filled=label == "Graph")
        arrow(ax, (W / 2, y), (W / 2, y - gap))
        y -= h + gap
    box(ax, x, y, w, h, steps[-1])
    # the tape runs on two backends
    bw = 3.4
    for bx, label, sx in [(W / 2 - 0.3 - bw, "Interpreter", x + 1.0),
                          (W / 2 + 0.3, "Native code", x + w - 1.0)]:
        box(ax, bx, y - h - gap, bw, h, label)
        arrow(ax, (sx, y), (bx + bw / 2, y - gap))
    save(fig, "pipeline.png")


def derivative():
    fig, ax = card("Derivatives share the graph")
    s = 1.0
    nodes = {
        "x": (2.6, 3.4), "y": (7.8, 3.4),
        "xy": (5.2, 5.6),
        "sin": (2.6, 7.8), "cos": (6.2, 7.8),
        "mul": (7.8, 10.0),
    }
    labels = {"x": "x", "y": "y", "xy": "x y", "sin": "sin", "cos": "cos", "mul": "*"}
    for k, (cx, cy) in nodes.items():
        box(ax, cx - s / 2, cy - s / 2, s, s, labels[k], filled=k == "xy", size=22)
    for a, b in [("x", "xy"), ("y", "xy"), ("xy", "sin"), ("xy", "cos"), ("cos", "mul"),
                 ("y", "mul")]:
        (ax_, ay), (bx, by) = nodes[a], nodes[b]
        d = np.array([bx - ax_, by - ay])
        d /= np.linalg.norm(d)
        # from the edge of one box to the edge of the other
        t = s / 2 / max(abs(d[0]), abs(d[1]))
        arrow(ax, (ax_ + d[0] * t, ay + d[1] * t), (bx - d[0] * t, by - d[1] * t))
    for k, name in [("sin", "f"), ("mul", "df/dx")]:
        cx, cy = nodes[k]
        ax.text(cx, cy + s / 2 + 0.3, name, color=INK, fontsize=24, ha="center", va="bottom")
    save(fig, "derivative.png")


def split():
    fig, ax = card("Parameter work runs once")
    y, h = 5.2, 2.6
    pw = 2.6
    ax.add_patch(Rectangle((L, y), pw, h, fc=ACCENT, ec="none"))
    ax.text(L + 0.25, y + h - 0.3, "prolog", color=PAPER, fontsize=22, va="top")
    n, g = 7, 0.22
    mw = (R - L - pw - 0.5 - (n - 1) * g) / n
    for k in range(n):
        box(ax, L + pw + 0.5 + k * (mw + g), y, mw, h)
    ax.text(L + pw + 0.5, y + h + 0.3, "main", color=INK, fontsize=22, va="bottom")
    ax.text(L, y - 0.3, "once per parameter set", color=GREY, fontsize=16, va="top")
    ax.text(L + pw + 0.5, y - 0.3, "every Newton iteration", color=GREY, fontsize=16, va="top")
    save(fig, "split.png")


def circuits():
    """Time per call on the AnalogGym amplifiers (BSIM4), the median over
    the circuits, parameters as constants: the condition all tools ran."""
    ours = list(csv.DictReader(open(os.path.join(DATA, "modules_rsdag.csv"))))
    other = list(csv.DictReader(open(os.path.join(DATA, "modules_other.csv"))))
    amps = [x["module"] for x in ours if x["module"] not in ("ua741", "ring_psp103")]

    def med(rows, col):
        vals = [float(x[col]) for x in rows if x["module"] in amps and x[col]]
        return float(np.median(vals)) if vals else None

    tool = lambda t: [x for x in other if x["tool"] == t]
    blocks = {
        "Residual": [("rsdag", med(ours, "call_f_folded_us")),
                     ("CasADi", med(tool("CasADi"), "call_f_us")),
                     ("JAX", med(tool("JAX"), "call_f_us"))],
        "Jacobian": [("rsdag", med(ours, "call_j_folded_us")),
                     ("CasADi", med(tool("CasADi"), "call_j_us")),
                     ("JAX", med(tool("JAX"), "call_j_us"))],
    }
    fig, ax = card("Time per call, BSIM4 amplifiers")
    scale = max(v for b in blocks.values() for _, v in b if v)
    x0 = L + 1.6
    barw = R - x0 - 1.0
    top = 9.7
    for block, entries in blocks.items():
        ax.text(L, top, block, color=INK, fontsize=24, va="top")
        ax.plot([L, R], [top - 0.62, top - 0.62], color=RULE, lw=1)
        y = top - 1.2
        for name, v in entries:
            ours_ = name == "rsdag"
            ax.text(L, y, name, color=ACCENT if ours_ else INK, fontsize=20, va="center")
            if v is None:
                ax.text(x0, y, "no compile in 1 min", color=GREY, fontsize=18, va="center")
            else:
                w = max(barw * v / scale, 0.04)
                ax.add_patch(Rectangle((x0, y - 0.22), w, 0.44, fc=ACCENT if ours_ else INK,
                                       ec="none"))
                ax.text(x0 + w + 0.2, y, f"{v:.0f} µs" if v >= 10 else f"{v:.1f} µs",
                        color=INK, fontsize=18, va="center")
            y -= 0.8
        top = y - 0.25
    save(fig, "circuits.png")


if __name__ == "__main__":
    for make in (bodies, pipeline, derivative, split, circuits):
        make()
    print("wrote docs/social/*.png")
