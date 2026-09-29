"""Social cards for rsdag: 1600 x 900 PNGs of the diagrams and benchmarks.

    scripts/social.sh          # renders the diagrams, then runs this

    python docs/social/make.py <dir of diagram PNGs>

The diagrams are drawn by rsdag (`examples/diagrams.rs --social`, rendered
by Graphviz). Writes docs/social/*.png.
"""
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, DPI = 16, 9, 100

INK, MUTED, RULE = "#1f2937", "#6b7280", "#e5e7eb"
ACCENT = "#3b82f6"
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


if __name__ == "__main__":
    diagram_cards(sys.argv[1])
    print("wrote docs/social/*.png")
