"""Interactive bonus chapter: lesson, exact XOR network, live training, exercises."""
from __future__ import annotations

from fractions import Fraction

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.fmt import fm, fv
from ..core.matrix import Matrix
from ..topics import BY_ID
from .nn import XOR_X, train_xor


def exact_xor_network():
    """The hand-made ReLU network of the lesson, evaluated EXACTLY with Matrix."""
    W1 = Matrix([[1, 1], [1, 1]])
    b1 = [Fraction(0), Fraction(-1)]
    w2 = Matrix([[1, -2]])
    out = []
    for x in ([0, 0], [1, 0], [0, 1], [1, 1]):
        z = [a + b for a, b in zip(W1.apply(x), b1)]
        h = [max(v, Fraction(0)) for v in z]
        y = w2.apply(h)[0]
        out.append((x, z, h, y))
    return out


def run_chapter(app) -> None:
    t = BY_ID["ai"]
    for k, (head, body) in enumerate(t.lesson):
        app.out(Panel(Text(body), title=f"Bonus — {head}", border_style="blue"))
        if k == 4:
            tab = Table(title="Exact forward pass (Fractions, this trainer's Matrix class)")
            for c in ("x", "W1x + b1", "h = ReLU(...)", "y = (1,-2)·h", "XOR"):
                tab.add_column(c)
            for x, z, h, y in exact_xor_network():
                tab.add_row(fv(x), fv(z), fv(h), str(y), str(x[0] ^ x[1]))
            app.out(tab)
        if app.ask("Enter: continue, q: stop ").strip().lower() in ("q", "quit", "exit", "menu"):
            return
    app.out(Panel(Text("Training a 2-4-1 network (tanh hidden layer, sigmoid output) on XOR with\n"
                       "full-batch gradient descent, learning rate 2.0. Watch the loss ||Y - T||²/N fall:"),
                  title="Live training", border_style="magenta"))
    net, log = train_xor()
    for e, L in log:
        app.out(f"  epoch {e:5d}   loss {L:.5f}   " + "█" * max(1, int(40 * min(L, 0.3) / 0.3)))
    tab = Table(title="Learned predictions")
    for c in ("x", "prediction", "rounded", "target"):
        tab.add_column(c)
    preds = net.predict(XOR_X)
    for j, p in enumerate(preds):
        x = (XOR_X[0][j], XOR_X[1][j])
        tab.add_row(str(x), f"{p:.3f}", str(round(p)), str(x[0] ^ x[1]))
    app.out(tab)
    app.out(Panel(Text("Learned first layer W1 (rows = neurons) — these are 4 hyperplanes in R²:\n" +
                       "\n".join("  (" + ", ".join(f"{w:+.2f}" for w in row) + ")" for row in net.W1)),
                  border_style="cyan"))
    if app.ask("Try the bonus exercises (layers, depth collapse, least squares)? [Y/n] ").strip().lower() not in ("n", "no"):
        app.practice(t)
