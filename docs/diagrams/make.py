"""The README's diagrams: square boxes, round graph nodes, grey ink and one
blue accent on a transparent background (see docs/style.py).

    python docs/diagrams/make.py

Writes docs/diagrams/*.svg and *.png.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from style import BLUE, FONT, GREY, MONO, WHITE  # noqa: E402

plt.rcParams.update({
    "font.family": FONT,
    "svg.fonttype": "none",
    "savefig.transparent": True,
})

LW = 1.2
R = 0.26  # a graph node's radius
BH = 0.42  # a box's height
FS = 10
FADE = 0.3


class Diagram:
    """A figure in inches, its nodes by key for the edges between them."""

    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self.nodes = {}

    def _style(self, accent, faded):
        color = BLUE if accent else GREY
        return color, (FADE if faded else 1.0)

    def circle(self, key, x, y, label, r=R, fill=False, accent=False, dashed=False, faded=False):
        """A graph node: filled in the accent (an input), outlined in the
        accent (a choice), or outlined grey."""
        color, alpha = self._style(accent or fill, faded)
        self.ax.add_patch(Circle((x, y), r, fc=BLUE if fill else "none",
                                 ec="none" if fill else color, lw=LW,
                                 ls="--" if dashed else "-", alpha=alpha))
        self.ax.text(x, y, label, color=WHITE if fill else color, fontsize=FS - 1,
                     ha="center", va="center", alpha=alpha)
        self.nodes[key] = ("circle", x, y, r)

    def box(self, key, x, y, label, w=None, fill=False, accent=False, dashed=False,
            faded=False, fs=FS):
        """A square box centered at `x, y`."""
        w = w or 0.09 * len(label) + 0.4
        color, alpha = self._style(accent or fill, faded)
        self.ax.add_patch(Rectangle((x - w / 2, y - BH / 2), w, BH,
                                    fc=BLUE if fill else "none",
                                    ec="none" if fill else color, lw=LW,
                                    ls="--" if dashed else "-", alpha=alpha))
        self.ax.text(x, y, label, color=WHITE if fill else color, fontsize=fs,
                     ha="center", va="center", alpha=alpha)
        self.nodes[key] = ("rect", x, y, w, BH)

    def region(self, key, x0, y0, x1, y1, label):
        """A group: a square frame, its label at the top left."""
        self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=GREY,
                                    lw=LW * 0.8))
        self.ax.text(x0 + 0.12, y1 - 0.12, label, color=GREY, fontsize=FS - 1,
                     ha="left", va="top")
        self.nodes[key] = ("rect", (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)

    def text(self, x, y, label, accent=False, faded=False, fs=FS, mono=False, **kw):
        color, alpha = self._style(accent, faded)
        self.ax.text(x, y, label, color=color, fontsize=fs, alpha=alpha,
                     family=MONO if mono else FONT, **kw)

    def point(self, key, x, y):
        """An invisible anchor, the end of an edge to a label."""
        self.nodes[key] = ("rect", x, y, 0.0, 0.0)

    def _rim(self, key, toward):
        """Where the line from node `key` toward `toward` leaves it."""
        kind, x, y, *size = self.nodes[key]
        d = np.array([toward[0] - x, toward[1] - y], dtype=float)
        d /= np.linalg.norm(d)
        if kind == "circle":
            t = size[0]
        else:
            hw, hh = size[0] / 2, size[1] / 2
            t = min(hw / abs(d[0]) if d[0] else np.inf, hh / abs(d[1]) if d[1] else np.inf)
        return x + d[0] * (t + 0.03), y + d[1] * (t + 0.03)

    def edge(self, a, b, label=None, accent=False, dashed=False, faded=False, bend=0.0,
             at=0.5, side=0.0, right=False):
        """An arrow from node `a` to node `b`, clipped at both rims, moved
        `side` inches to its left (a pair of opposite edges apart), bent by
        `bend`; its label to its left, or with `right` to its right."""
        ca, cb = self.nodes[a][1:3], self.nodes[b][1:3]
        p, q = np.array(self._rim(a, cb)), np.array(self._rim(b, ca))
        d = (q - p) / np.linalg.norm(q - p)
        left = np.array([-d[1], d[0]])
        p, q = p + left * side, q + left * side
        color, alpha = self._style(accent, faded)
        self.ax.annotate("", xy=q, xytext=p, arrowprops=dict(
            arrowstyle="-|>", color=color, lw=LW, mutation_scale=10, alpha=alpha,
            shrinkA=0, shrinkB=0, linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={bend}"))
        if label:
            # arc3 is a quadratic curve through a control point `bend` times
            # the length to the right of the midpoint: at `at` it is off the
            # chord by 2 at (1 - at) of that.
            off = 2 * at * (1 - at) * bend * np.linalg.norm(q - p)
            if right:
                left = -left
                off = -off
            m = p + at * (q - p) - left * off + left * 0.08
            if abs(left[0]) > abs(left[1]):
                ha, va = ("left" if left[0] > 0 else "right"), "center"
            else:
                ha, va = "center", ("bottom" if left[1] > 0 else "top")
            self.ax.text(m[0], m[1], label, color=color, fontsize=FS - 2, alpha=alpha,
                         ha=ha, va=va)

    def out(self, key, label, dy=0.55, accent=False, dashed=False):
        """An output: an arrow down from `key` to its name."""
        _, x, y, *_ = self.nodes[key]
        self.point(key + "@", x, y - dy)
        self.edge(key, key + "@", accent=accent, dashed=dashed)
        self.text(x, y - dy - 0.05, label, accent=accent, ha="center", va="top")

    def save(self, name):
        for ext, opts in (("svg", {}), ("png", {"dpi": 200})):
            self.fig.savefig(os.path.join(HERE, f"{name}.{ext}"), **opts)
        plt.close(self.fig)


def pipeline():
    d = Diagram(9.6, 2.6)
    for key, y in (("Rust", 2.0), ("Python", 1.3), ("Module", 0.6)):
        d.box(key, 0.75, y, key, w=1.05)
    d.box("graph", 2.35, 1.3, "Graph", w=1.05, fill=True)
    d.box("transforms", 3.85, 1.3, "Transforms", w=1.25)
    d.box("tape", 5.35, 1.3, "Tape", w=1.05)
    d.region("adaptive", 6.3, 0.25, 7.95, 2.45, "Adaptive")
    for key, y in (("Interpreter", 1.85), ("Specialized", 1.25), ("Native code", 0.65)):
        d.box(key, 7.12, y, key, w=1.35)
        d.edge("tape", key)
    d.box("solver", 8.95, 1.3, "Solver", w=0.95)
    for key in ("Rust", "Python", "Module"):
        d.edge(key, "graph")
    d.edge("graph", "transforms")
    d.edge("transforms", "tape")
    d.edge("adaptive", "solver")
    d.save("pipeline")


def solver_path():
    d = Diagram(9.6, 2.3)
    y = 1.1
    d.box("model", 0.65, y, "Model", w=0.95)
    d.box("f", 2.05, y, "F(x, x', p, t)", w=1.3)
    d.box("j", 3.5, y, "Jacobian", w=1.05)
    d.region("program", 4.35, 0.25, 6.9, 2.1, "one program")
    d.box("prolog", 5.0, y, "prolog", w=0.95)
    d.box("main", 6.25, y, "main", w=0.95)
    d.text(5.0, 0.62, "per parameter set", fs=FS - 2, ha="center", va="top")
    d.text(6.25, 0.62, "per iteration", fs=FS - 2, ha="center", va="top")
    d.box("solver", 7.85, y, "Solver loop", w=1.2)
    d.box("events", 9.05, y, "Events", w=0.85, accent=True, dashed=True)
    d.edge("model", "f")
    d.edge("f", "j")
    d.edge("j", "program")
    d.edge("prolog", "main", accent=True, dashed=True)
    d.edge("program", "solver")
    d.edge("solver", "events", accent=True, dashed=True)
    d.save("solver_path")


def derivative():
    """f = sin(xy) and df/dx = y cos(xy): the derivative's own nodes dashed."""
    d = Diagram(4.4, 4.5)
    d.circle("x", 0.9, 3.9, "x", fill=True)
    d.circle("y", 3.3, 3.9, "y", fill=True)
    d.circle("xy", 2.1, 3.0, "x y", r=0.3)
    d.circle("cos", 2.6, 2.1, "cos", dashed=True)
    d.circle("sin", 0.9, 1.2, "sin")
    d.circle("mul", 3.3, 1.2, "*", dashed=True)
    for a, b in (("x", "xy"), ("y", "xy"), ("xy", "sin")):
        d.edge(a, b)
    for a, b in (("xy", "cos"), ("cos", "mul"), ("y", "mul")):
        d.edge(a, b, dashed=True)
    d.out("sin", "f = sin(x y)")
    d.out("mul", "df/dx = y cos(x y)", dashed=True)
    d.save("derivative")


def split():
    """A diode current and its derivative in v, compiled split: 1/(n vt) in
    the prolog, read by the main phase as state (dashed)."""
    d = Diagram(5.0, 5.1)
    d.region("prolog", 0.25, 3.4, 4.75, 4.95, "prolog: once per parameter set")
    d.region("main", 0.25, 0.8, 4.75, 3.2, "main: every iteration")
    d.circle("n", 0.9, 4.28, "n", fill=True)
    d.circle("vt", 0.9, 3.75, "vt", fill=True, r=0.22)
    d.circle("nvt", 1.9, 4.0, "*")
    d.circle("a", 2.9, 4.0, "1/x")
    d.edge("n", "nvt")
    d.edge("vt", "nvt")
    d.edge("nvt", "a")
    d.circle("v", 0.9, 2.6, "v", fill=True)
    d.circle("va", 1.9, 2.6, "*")
    d.circle("exp", 2.9, 2.6, "exp")
    d.circle("ea", 3.9, 2.6, "*")
    d.circle("sub", 2.9, 1.95, "-1")
    d.circle("i", 2.4, 1.3, "*")
    d.circle("is", 3.4, 1.3, "is", fill=True)
    d.circle("di", 4.4, 1.3, "*")
    d.edge("v", "va")
    d.edge("a", "va", accent=True, dashed=True)
    d.edge("a", "ea", accent=True, dashed=True)
    d.edge("va", "exp")
    d.edge("exp", "ea")
    d.edge("exp", "sub")
    d.edge("sub", "i")
    d.edge("is", "i")
    d.edge("is", "di")
    d.edge("ea", "di")
    d.out("i", "i", dy=0.8)
    d.out("di", "di/dv", dy=0.8)
    d.save("split")


def adaptive():
    d = Diagram(5.9, 2.5)
    d.box("tape", 0.75, 1.25, "Tape", w=0.95)
    d.box("interp", 2.65, 1.25, "Interpreter", w=1.3)
    d.box("spec", 5.0, 2.0, "Specialized", w=1.35)
    d.box("native", 5.0, 0.5, "Native code", w=1.35)
    d.edge("tape", "interp")
    d.edge("interp", "spec", label="traced", side=0.1)
    d.edge("spec", "interp", label="guard fails", accent=True, dashed=True, side=0.1)
    d.edge("interp", "native", label="compiled", right=True)
    d.save("adaptive")


def bodies():
    d = Diagram(6.4, 2.4)
    s = 1.5
    d.ax.add_patch(Rectangle((0.3, 0.45), s, s, fc=BLUE, ec="none"))
    d.ax.text(0.45, 0.45 + s - 0.12, "body", color=WHITE, fontsize=FS + 1, va="top")
    d.ax.text(0.45, 0.57, "compiled once", color=WHITE, fontsize=FS - 2, va="bottom")
    d.nodes["body"] = ("rect", 0.3 + s / 2, 0.45 + s / 2, s, s)
    cols, rows, q, g = 5, 2, 0.4, 0.22
    gx, gy = 6.1 - cols * q - (cols - 1) * g, 1.2 - (rows * q + (rows - 1) * g) / 2
    for r in range(rows):
        for c in range(cols):
            d.ax.add_patch(Rectangle((gx + c * (q + g), gy + r * (q + g)), q, q, fc="none",
                                     ec=GREY, lw=LW))
    d.point("grid", gx - 0.15, 1.2)
    d.edge("body", "grid", label="one batched call")
    d.text(gx + (cols * q + (cols - 1) * g) / 2, gy - 0.15, "instances", fs=FS - 1,
           ha="center", va="top")
    d.save("bodies")


def bodies_tape():
    """Three diode instances in a ring, `is` and `n` parameters: the prolog
    runs the body's parameter part for all three, the main phase one
    batched call."""
    d = Diagram(6.6, 3.6)
    d.region("pro", 0.2, 1.05, 3.0, 3.4, "prolog")
    d.region("main", 3.3, 1.05, 6.4, 3.4, "main")
    for k, x in enumerate((0.7, 1.35, 2.0)):
        d.circle(f"is{k}", x, 2.65, f"is{k}", fill=True, r=0.28)
    d.circle("n", 2.6, 2.65, "n", fill=True)
    d.box("prolog", 1.6, 1.6, "prolog \u00d73", w=1.4)
    for key in ("is0", "is1", "is2", "n"):
        d.edge(key, "prolog")
    for k, x in enumerate((3.9, 4.85, 5.8)):
        d.circle(f"v{k}", x, 2.65, f"v{k}", fill=True)
    d.box("call", 4.85, 1.6, "call \u00d73", w=1.3)
    for k in range(3):
        d.edge(f"v{k}", "call")
    d.edge("prolog", "call", label="state", accent=True, dashed=True)
    for k, x in enumerate((3.9, 4.85, 5.8)):
        d.point(f"k{k}", x, 0.55)
        d.edge("call", f"k{k}")
        d.text(x, 0.5, f"kcl{k}", ha="center", va="top")
    d.save("bodies_tape")


def specialize():
    d = Diagram(7.6, 3.0)
    tape = [("v1 = x * p", False), ("v2 = select(c1, v1, x)", True),
            ("v3 = exp(v2)", False), ("v4 = select(c2, v3, p)", True),
            ("v5 = v4 + v1", False), ("v6 = select(c3, v5, v3)", True)]
    spec = [("v1 = x * p", False), ("v3 = exp(v1)", False), ("v5 = p + v1", False)]

    def listing(key, x0, y1, w, title, lines):
        h = 0.34 + 0.27 * len(lines)
        d.region(key, x0, y1 - h, x0 + w, y1, title)
        for k, (line, accent) in enumerate(lines):
            d.text(x0 + 0.15, y1 - 0.42 - 0.27 * k, line, accent=accent, fs=FS - 1,
                   mono=True, va="top")

    listing("tape", 0.2, 2.8, 2.7, "tape", tape)
    listing("spec", 4.6, 2.8, 2.8, "specialized: c1 = 1, c2 = 0, c3 = 1", spec)
    d.text(4.75, 1.45, "guards: c1 == 1, c2 == 0, c3 == 1", accent=True, fs=FS - 1,
           mono=True, va="top")
    d.point("l", 2.95, 2.3)
    d.point("r", 4.55, 2.3)
    d.edge("l", "r", label="specialize")
    d.point("g", 4.55, 1.05)
    d.point("t", 2.95, 1.05)
    d.edge("g", "t", label="fails: retrace", accent=True, dashed=True)
    d.save("specialize")


def specialize_graph():
    """A piecewise model at v = 1: the arms taken and the conditions that
    guard them stay, the faded arms leave the specialized tape."""
    d = Diagram(4.6, 5.3)
    d.circle("vth", 0.7, 4.8, "vth", fill=True, r=0.3)
    d.circle("v", 2.1, 4.8, "v", fill=True)
    d.circle("vsat", 3.9, 4.8, "vsat", fill=True, r=0.3)
    d.circle("c1", 1.4, 3.95, ">", accent=True)
    d.circle("dv", 1.4, 3.1, "-")
    d.circle("c2", 3.9, 3.1, ">", accent=True)
    d.circle("kd", 1.9, 2.2, "k d", r=0.3)
    d.circle("half", 2.9, 2.2, "k d\u00b2/2", r=0.38, faded=True)
    d.box("inner", 2.9, 1.35, "select", w=0.9, accent=True)
    d.box("outer", 0.9, 0.85, "select", w=0.9, accent=True)
    d.text(0.3, 1.75, "0", faded=True, ha="center", va="center")
    d.point("zero", 0.3, 1.75)
    for a in ("vth", "v"):
        d.edge(a, "c1")
        d.edge(a, "dv")
    d.edge("dv", "c2")
    d.edge("vsat", "c2")
    d.edge("dv", "kd")
    d.edge("dv", "half", faded=True)
    d.edge("c2", "inner", label="if", accent=True)
    d.edge("kd", "inner", label="then")
    d.edge("half", "inner", label="else", faded=True, right=True)
    d.edge("c1", "outer", label="if", accent=True, bend=0.25, at=0.4)
    d.edge("inner", "outer", label="then")
    d.edge("zero", "outer", faded=True)
    d.out("outer", "i", dy=0.45)
    d.save("specialize_graph")


if __name__ == "__main__":
    for make in (pipeline, solver_path, derivative, split, adaptive, bodies, bodies_tape,
                 specialize, specialize_graph):
        make()
    print("wrote docs/diagrams/*.svg and *.png")
