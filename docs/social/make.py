"""Social cards for rsdag: 1600 x 900 PNGs of the diagrams and benchmarks.

    scripts/social.sh          # renders the diagrams, then runs this

    python docs/social/make.py <dir of diagram PNGs>

The diagrams are drawn by rsdag (`examples/diagrams.rs --social`, rendered
by Graphviz); the benchmark cards plot docs/bench/data/compare.csv. Writes
docs/social/*.png.
"""
import csv
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "bench", "data")
W, H, DPI = 16, 9, 100

INK, MUTED, RULE = "#1f2937", "#6b7280", "#e5e7eb"
ACCENT = "#3b82f6"
TOOLS = {
    "rsdag native": (ACCENT, "-"),
    "rsdag interpreter": (ACCENT, "--"),
    "CasADi": ("#f59e0b", "-"),
    "JAX": ("#10b981", "-"),
}
FOOTER = "rsdag   |   pip install rsdag   |   github.com/milanofthe/rsdag"

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": RULE,
    "grid.linewidth": 0.8,
})


def card(title, body):
    """A white card with the accent bar, the title, the body text on the
    left column and the footer; returns the figure."""
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="white")
    fig.add_artist(plt.Rectangle((0, 0), 0.008, 1, color=ACCENT, transform=fig.transFigure))
    fig.text(0.05, 0.88, title, fontsize=38, weight="bold", va="top")
    if body:
        fig.text(0.05, 0.74, "\n".join(textwrap.wrap(body, 34)), fontsize=19, color=MUTED,
                 va="top", linespacing=1.55)
    fig.text(0.05, 0.05, FOOTER, fontsize=14, color=MUTED)
    return fig


def image(fig, path, box):
    """The PNG at `path` fitted into `box` (left, bottom, width, height in
    figure fractions), aspect kept, centered."""
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    l, b, bw, bh = box
    scale = min(bw * W / w, bh * H / h)
    iw, ih = w * scale / W, h * scale / H
    ax = fig.add_axes((l + (bw - iw) / 2, b + (bh - ih) / 2, iw, ih))
    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=DPI, facecolor="white")
    plt.close(fig)


def diagram_cards(src):
    d = lambda name: os.path.join(src, f"{name}.png")
    fig = card("From a function to native code", None)
    fig.text(0.05, 0.79, "rsdag traces a model into one hash-consed graph, differentiates it and "
             "compiles it to a tape the interpreter or native code runs.", fontsize=19, color=MUTED, va="top")
    image(fig, d("pipeline"), (0.04, 0.12, 0.92, 0.62))
    save(fig, "pipeline.png")

    fig = card("Derivatives share the graph",
               "f = sin(xy) + xy and df/dx in one graph: the derivative reuses xy, only what f "
               "alone needs is faded. Forward and reverse mode, sparse Jacobians, Hessians.")
    image(fig, d("derivative"), (0.46, 0.1, 0.5, 0.82))
    save(fig, "derivative.png")

    fig = card("Parameter work runs once",
               "A diode current and its derivative with a prolog and a main phase. What depends "
               "on the parameters only runs once per parameter set, a Newton iteration runs the "
               "rest. Dashed: the state the main phase reads.")
    image(fig, d("split"), (0.46, 0.1, 0.5, 0.8))
    save(fig, "split.png")

    fig = card("One body, many instances",
               "Three diodes of one model in a ring: the body is compiled once, its parameter "
               "part runs per instance in the prolog, the main phase is one batched call.")
    image(fig, d("bodies_tape"), (0.36, 0.12, 0.62, 0.66))
    save(fig, "bodies.png")


def rows():
    with open(os.path.join(DATA, "compare.csv"), newline="") as f:
        return list(csv.DictReader(f))


def series(r, tool, col):
    pts = [(int(x["states"]), float(x[col])) for x in r if x["tool"] == tool and x[col]]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_panel(fig, box, r, col, ylabel, title, tools):
    ax = fig.add_axes(box)
    for tool in tools:
        color, ls = TOOLS[tool]
        n, v = series(r, tool, col)
        ax.plot(n, v, ls, color=color, lw=3, marker="o", ms=6, label=tool)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("states", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title, fontsize=18, loc="left", color=INK)
    ax.tick_params(labelsize=13)
    return ax


def bench_cards():
    r = rows()
    tools = ["rsdag native", "CasADi", "JAX"]
    fig = card("Fast calls from Python", None)
    fig.text(0.05, 0.79, "1D Brusselator, one core, time per call from Python. rsdag and CasADi "
             "build the sparse Jacobian, JAX the dense one (jacfwd).", fontsize=19, color=MUTED, va="top")
    plot_panel(fig, (0.07, 0.17, 0.4, 0.52), r, "call_f_us", "microseconds per call", "Right-hand side", tools)
    ax = plot_panel(fig, (0.56, 0.17, 0.4, 0.52), r, "call_j_us", "microseconds per call", "Jacobian", tools)
    ax.legend(fontsize=14, frameon=False, loc="upper left")
    save(fig, "bench_calls.png")

    fig = card("Compiled in milliseconds", None)
    fig.text(0.05, 0.79, "From the Python function to the first result of the right-hand side "
             "and its Jacobian: tracing, differentiation and compilation.", fontsize=19, color=MUTED, va="top")
    r_setup = [dict(x, setup_s=(float(x["setup_f_s"]) + float(x["setup_j_s"])) * 1e3 if x["setup_j_s"] else "")
               for x in r]
    ax = plot_panel(fig, (0.07, 0.17, 0.6, 0.52), r_setup, "setup_s", "milliseconds",
                    "Setup of right-hand side and Jacobian",
                    ["rsdag native", "rsdag interpreter", "CasADi", "JAX"])
    ax.legend(fontsize=14, frameon=False, loc="lower right")
    fig.text(0.71, 0.66, "\n".join(textwrap.wrap(
        "rsdag and CasADi build the graph element by element, so their setup grows with "
        "the model. JAX compiles vector code, near 35 ms for the right-hand side at every "
        "size, and stops where its dense Jacobian stops being practical.", 30)),
        fontsize=15, color=MUTED, va="top", linespacing=1.5)
    save(fig, "bench_setup.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        diagram_cards(sys.argv[1])
    bench_cards()
    print("wrote docs/social/*.png")
