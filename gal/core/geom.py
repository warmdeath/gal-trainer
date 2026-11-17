"""Affine geometry and Euclidean helpers (exact)."""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from .fmt import flin, fm, fs, fv, hcat, indent, var_names
from .gauss import in_span, kernel_basis, rank, row_reduce, same_span, span_rank
from .matrix import Matrix, clear_denominators, dot, vadd, vec, vscale, vsub
from .surd import Surd


def _lin_expr(coeffs, const, names) -> str:
    s = flin(coeffs, names, zero="")
    if const:
        if not s:
            return fs(const)
        s += (" - " if const < 0 else " + ") + fs(abs(const))
    return s or "0"


def cartesian_from_param(point: Sequence, dirs: Sequence[Sequence],
                         names: Sequence[str] | None = None):
    """Cartesian equations of P + Span(dirs) with the method of §2.5/§2.7:
    x ∈ S  ⇔  rg(M) = rg(M | x - P), M = (dirs as columns).
    Returns (equations [(coeffs, rhs)], steps)."""
    P = vec(point)
    n = len(P)
    names = list(names or var_names(n))
    dirs = [vec(d) for d in dirs]
    steps = []
    if not dirs:
        eqs = [([Fraction(int(i == j)) for j in range(n)], P[i]) for i in range(n)]
        steps.append("The subspace is the single point " + fv(P) + ": its equations are "
                     + ", ".join(f"{names[i]} = {fs(P[i])}" for i in range(n)) + ".")
        return eqs, steps
    M = Matrix.from_cols(dirs)
    sym = [_lin_expr([Fraction(int(i == j)) for j in range(n)], -P[i], names) for i in range(n)]
    steps.append("x = (" + ", ".join(names) + ") lies in the subspace iff x - P ∈ Span(directions), i.e. iff "
                 "rg(M) = rg(M | x - P), where M has the direction vectors as columns:")
    rows = [[fs(x) for x in M.row(i)] + [sym[i]] for i in range(n)]
    steps.append(indent(_aug_text(rows)))
    # E·M = U (echelon); the same row operations act on the last column
    red = row_reduce(M.hstack(Matrix.identity(n)), ncols=M.n)
    E = red.result.submatrix(range(n), range(M.n, M.n + n))
    U = red.result.submatrix(range(n), range(M.n))
    r = red.rank
    eqs = []
    for i in range(r, n):
        a = E.row(i)
        rhs = dot(a, P)
        full = clear_denominators(a + [rhs])
        eqs.append((full[:-1], full[-1]))
    col = [_lin_expr(E.row(i), -dot(E.row(i), P), names) for i in range(n)]
    rows = [[fs(x) for x in U.row(i)] + [col[i]] for i in range(n)]
    steps.append("Row reduce (the same operations act on the last column):")
    steps.append(indent(_aug_text(rows)))
    steps.append(f"rg(M) = {r}, so the last {n - r} entr{'y' if n - r == 1 else 'ies'} of the last column must vanish:")
    for a, b in eqs:
        steps.append(f"    {flin(a, names)} = {fs(b)}")
    return eqs, steps


def _aug_text(rows: list[list[str]]) -> str:
    w = [max(len(r[j]) for r in rows) for j in range(len(rows[0]))]
    lines = []
    for r in rows:
        left = " ".join(r[j].rjust(w[j]) for j in range(len(r) - 1))
        lines.append(f"[ {left} | {r[-1].ljust(w[-1])} ]")
    return "\n".join(lines)


def param_from_cartesian(A: Matrix, b: Sequence):
    """Sol(A|b) as (point, directions); None if empty."""
    from .gauss import solve
    sol = solve(A, b)
    if sol.kind == "none":
        return None, sol.steps
    return (sol.particular, sol.basis), sol.steps


# ------------------------------------------------------------------ positions in R^3
def line_line_position(P1, v1, P2, v2) -> str:
    """'coincidenti' | 'parallele' | 'incidenti' | 'sghembe' (Osservazione p.49)."""
    d = vsub(P2, P1)
    r12 = span_rank([v1, v2])
    r3 = span_rank([v1, v2, d])
    if r12 == 1:
        return "coincidenti" if r3 == 1 else "parallele"
    return "incidenti" if r3 == 2 else "sghembe"


def line_plane_position(P1, v1, P2, w1, w2) -> str:
    """'contenuta' | 'parallele' | 'incidenti'."""
    if span_rank([v1, w1, w2]) == 3:
        return "incidenti"
    return "contenuta" if in_span(vsub(P2, P1), [w1, w2]) else "parallele"


def plane_plane_position(P1, v1, v2, P2, w1, w2) -> str:
    if span_rank([v1, v2, w1, w2]) == 3:
        return "incidenti"
    return "coincidenti" if in_span(vsub(P2, P1), [v1, v2]) else "paralleli"


# ------------------------------------------------------------------ Euclidean
def gram_inner(S: Matrix | None):
    def ip(v, w):
        if S is None:
            return dot(v, w)
        return dot(vec(v), S.apply(w))
    return ip


def norm(v, S: Matrix | None = None) -> Surd:
    return Surd.sqrt(gram_inner(S)(v, v))


def cross(v, w) -> list[Fraction]:
    v, w = vec(v), vec(w)
    return [v[1] * w[2] - v[2] * w[1], -(v[0] * w[2] - v[2] * w[0]), v[0] * w[1] - v[1] * w[0]]


def ratio(a, b) -> str:
    """'a/b' with parentheses where needed: 3/4, -6/(12/5), 12/(3√3)."""
    sa, sb = (x.pretty() if isinstance(x, Surd) else fs(x) for x in (a, b))
    if "/" in sb or "√" in sb or sb.startswith("-"):
        sb = f"({sb})"
    if "/" in sa and not sa.startswith("("):
        sa = f"({sa})"
    return f"{sa}/{sb}"


def gram_schmidt(basis: Sequence[Sequence], S: Matrix | None = None):
    """Orthogonal basis c_1..c_p (Teorema 5.20) and the steps."""
    ip = gram_inner(S)
    cs: list[list[Fraction]] = []
    steps = []
    for j, b in enumerate(basis):
        b = vec(b)
        c = b
        sym, num = [], []
        for k, ck in enumerate(cs):
            coef = ip(b, ck) / ip(ck, ck)
            sym.append(f"(⟨b{j + 1},c{k + 1}⟩/||c{k + 1}||²)c{k + 1}")
            num.append(f"({ratio(ip(b, ck), ip(ck, ck))})·{fv(ck)}")
            c = vsub(c, vscale(coef, ck))
        if sym:
            steps.append(f"c{j + 1} = b{j + 1} - " + " - ".join(sym))
            steps.append(f"   = {fv(b)} - " + " - ".join(num) + f" = {fv(c)}")
        else:
            steps.append(f"c1 = b1 = {fv(c)}")
        cs.append(c)
    return cs, steps


def projection(v, W_orth_basis: Sequence[Sequence], S: Matrix | None = None) -> list[Fraction]:
    ip = gram_inner(S)
    out = [Fraction(0)] * len(v)
    for b in W_orth_basis:
        out = vadd(out, vscale(ip(v, b) / ip(b, b), b))
    return out


def projection_matrix(W_basis: Sequence[Sequence], S: Matrix | None = None) -> Matrix:
    """M_C^C(π_W) in the coordinates of the given (Gram matrix S) — Teorema 5.24,
    computed with an orthogonal basis: Σ c c^T S / <c,c>."""
    cs, _ = gram_schmidt(W_basis, S)
    ip = gram_inner(S)
    n = len(cs[0])
    Sm = S or Matrix.identity(n)
    P = Matrix.zeros(n, n)
    for c in cs:
        P = P + (Matrix.col_vector(c) * Matrix([c]) * Sm).scale(1 / ip(c, c))
    return P


def orth_complement(W_basis: Sequence[Sequence], n: int, S: Matrix | None = None):
    """Basis of W⊥ = ker of the matrix with rows (S w)^T."""
    if not W_basis:
        return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    Sm = S or Matrix.identity(n)
    rows = [Sm.apply(w) for w in W_basis]
    return kernel_basis(Matrix(rows))
