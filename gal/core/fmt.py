"""Plain-text rendering of scalars, vectors, matrices, systems (terminal friendly)."""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from .matrix import Matrix

MINUS = "-"


def fs(x) -> str:
    """Scalar to string: 3, -2, 3/2."""
    if isinstance(x, Fraction):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    return str(x)


def fv(v: Sequence) -> str:
    """Vector as (a, b, c) — the same syntax the student types."""
    return "(" + ", ".join(fs(x) for x in v) + ")"


def fvs(vs: Sequence[Sequence]) -> str:
    return "; ".join(fv(v) for v in vs) if vs else "{} (empty basis)"


def fm_inline(M: Matrix) -> str:
    """Matrix in the input syntax: rows separated by ';'."""
    return "; ".join(" ".join(fs(x) for x in r) for r in M.rows)


def fm(M: Matrix | Sequence[Sequence], aug: int | None = None) -> str:
    """Multi-line matrix with brackets. `aug` = number of columns left of the bar."""
    rows = [[fs(x) for x in r] for r in (M.rows if isinstance(M, Matrix) else M)]
    n = len(rows[0])
    widths = [max(len(r[j]) for r in rows) for j in range(n)]
    lines = []
    for r in rows:
        cells = [r[j].rjust(widths[j]) for j in range(n)]
        if aug is not None and 0 < aug < n:
            body = " ".join(cells[:aug]) + " | " + " ".join(cells[aug:])
        else:
            body = " ".join(cells)
        lines.append(body)
    m = len(lines)
    if m == 1:
        return "[ " + lines[0] + " ]"
    out = []
    for i, body in enumerate(lines):
        l, r = ("⎡", "⎤") if i == 0 else ("⎣", "⎦") if i == m - 1 else ("⎢", "⎥")
        out.append(f"{l} {body} {r}")
    return "\n".join(out)


def fcol(v: Sequence) -> str:
    """Column vector, multi-line."""
    return fm([[x] for x in v])


def hcat(*blocks: str, sep: str = "") -> str:
    """Join multi-line blocks horizontally, vertically centred."""
    split = [b.split("\n") for b in blocks]
    h = max(len(s) for s in split)
    cols = []
    for s in split:
        w = max(len(x) for x in s)
        top = (h - len(s)) // 2
        padded = [" " * w] * top + [x.ljust(w) for x in s] + [" " * w] * (h - len(s) - top)
        cols.append(padded)
    return "\n".join(sep.join(c[i] for c in cols).rstrip() for i in range(h))


def named(name: str, M, aug: int | None = None) -> str:
    return hcat(f"{name} = ", fm(M, aug))


def named_many(*pairs, aug=None) -> str:
    """named_many(("A", A), ("B", B)) -> 'A = [..],  B = [..]' on one band."""
    blocks = []
    for k, (name, M) in enumerate(pairs):
        blocks.append(("   " if k else "") + f"{name} = ")
        blocks.append(fm(M, aug))
    return hcat(*blocks)


def flin(coeffs: Sequence, names: Sequence[str], zero: str = "0") -> str:
    """Linear expression: 2x - y + 3z."""
    out = ""
    for c, name in zip(coeffs, names):
        c = Fraction(c)
        if c == 0:
            continue
        a = abs(c)
        coef = "" if a == 1 and name else fs(a)
        if "/" in coef and name:
            coef = f"({coef})"
        term = f"{coef}{name}"
        if not out:
            out = (MINUS if c < 0 else "") + term
        else:
            out += (f" {MINUS} " if c < 0 else " + ") + term
    return out or zero


def var_names(n: int) -> list[str]:
    if n <= 3:
        return ["x", "y", "z"][:n]
    if n == 4:
        return ["x", "y", "z", "w"]
    return [f"x{i + 1}" for i in range(n)]


def fsys(A: Matrix, b: Sequence, names: Sequence[str] | None = None) -> str:
    """A linear system written equation by equation."""
    names = names or var_names(A.n)
    lhs = [flin(A.row(i), names) for i in range(A.m)]
    w = max(len(s) for s in lhs)
    lines = [f"{lhs[i].rjust(w)} = {fs(Fraction(b[i]))}" for i in range(A.m)]
    if len(lines) == 1:
        return "{ " + lines[0]
    out = []
    for i, s in enumerate(lines):
        l = "⎧" if i == 0 else "⎩" if i == len(lines) - 1 else "⎨" if i == len(lines) // 2 and len(lines) % 2 else "⎪"
        out.append(f"{l} {s}")
    return "\n".join(out)


def fparam(point: Sequence | None, basis: Sequence[Sequence], params: str = "tsuv") -> str:
    """Parametric form (p) + t(w1) + s(w2), in the syntax the student types."""
    parts = []
    if point is not None and (any(x != 0 for x in point) or not basis):
        parts.append(fv(point))
    for k, w in enumerate(basis):
        parts.append(f"{params[k]}{fv(w)}")
    return " + ".join(parts)


def indent(text: str, n: int = 4) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.split("\n"))
