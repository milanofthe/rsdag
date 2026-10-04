"""The README's diagrams, drawn in TikZ: docs/diagrams/*.tex over the
shared preamble style.tex, the colors from docs/style.py.

    python docs/diagrams/make.py [name ...]

Needs pdflatex, dvisvgm and pdftocairo (TeX Live or MiKTeX). Writes
docs/diagrams/<name>.svg (glyphs as paths) and <name>.png (200 dpi,
transparent).
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from style import BLUE, GREY  # noqa: E402

COLORS = {"ink": GREY, "accent": BLUE}


def run(*cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{' '.join(cmd)} failed:\n{r.stdout[-3000:]}{r.stderr[-3000:]}")


def build(name, tmp):
    shutil.copy(os.path.join(HERE, f"{name}.tex"), tmp)
    tex = ("-interaction=nonstopmode", "-halt-on-error")
    # The SVG from DVI through TikZ's dvisvgm driver: dvisvgm's PDF import
    # loses the opacity of faded elements.
    run("latex", *tex, f"\\PassOptionsToClass{{dvisvgm}}{{standalone}}\\input{{{name}.tex}}",
        cwd=tmp)
    run("dvisvgm", "--no-fonts", "--optimize", "--zoom=1.25", "-o", f"{name}.svg",
        f"{name}.dvi", cwd=tmp)
    run("pdflatex", *tex, f"{name}.tex", cwd=tmp)
    run("pdftocairo", "-png", "-r", "200", "-transp", "-singlefile", f"{name}.pdf", name,
        cwd=tmp)
    for ext in ("svg", "png"):
        shutil.copy(os.path.join(tmp, f"{name}.{ext}"), HERE)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(
        os.path.basename(p)[:-4] for p in glob.glob(os.path.join(HERE, "*.tex"))
        if not p.endswith("style.tex"))
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(HERE, "style.tex"), tmp)
        with open(os.path.join(tmp, "colors.tex"), "w") as f:
            for name, hex_ in COLORS.items():
                f.write(f"\\definecolor{{{name}}}{{HTML}}{{{hex_[1:].upper()}}}\n")
        for name in names:
            build(name, tmp)
            print(f"wrote docs/diagrams/{name}.svg and .png")
