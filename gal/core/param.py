"""Matrices depending on ONE parameter (λ, k, a, ...), as in the "al variare di λ" exercises.

Entries are polynomials in the parameter. Everything stays exact: the rank of M(λ) can only
drop at the common roots of its r×r minors (r = generic rank), i.e. at the roots of their gcd.
We find those roots in Q; if the gcd has a factor without rational roots, `irrational` is
returned so the caller can refuse (or report it) instead of giving a wrong answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Sequence

from .fmt import fs
from .gauss import laplace, rank
from .matrix import Matrix
from .parse import parse_poly
from .poly import Poly, poly_gcd, real_root_count


class PMatrix:
    def __init__(self, rows: Sequence[Sequence], var: str = "λ"):
        self.var = var
        self.rows = [[_to_poly(x, var) for x in r] for r in rows]
        self.m, self.n = len(self.rows), len(self.rows[0])
        if any(len(r) != self.n for r in self.rows):
            raise ValueError("rows of different length")

    def at(self, value) -> Matrix:
        return Matrix([[p(value) for p in r] for r in self.rows])

    def hstack(self, other: "PMatrix") -> "PMatrix":
        out = PMatrix([[0]], self.var)
        out.rows = [a + b for a, b in zip(self.rows, other.rows)]
        out.m, out.n = self.m, self.n + other.n
        return out

    def det(self) -> Poly:
        if self.m != self.n:
            raise ValueError("det needs a square matrix")
        return laplace(self.rows)

    def minors(self, r: int):
        for I in combinations(range(self.m), r):
            for J in combinations(range(self.n), r):
                yield laplace([[self.rows[i][j] for j in J] for i in I])

    def is_constant(self) -> bool:
        return all(p.degree <= 0 for r in self.rows for p in r)

    def entry_text(self, i, j) -> str:
        return self.rows[i][j].show(self.var)


def _to_poly(x, var) -> Poly:
    if isinstance(x, Poly):
        return x
    if isinstance(x, (int, Fraction)):
        return Poly([x])
    return parse_poly(str(x), var)


@dataclass
class RankProfile:
    generic: int
    drops: list[Fraction]        # rational λ where the rank is lower than generic
    irrational: Poly | None      # factor of the gcd without rational roots (None if absent)

    def rank_at(self, pm: PMatrix, value) -> int:
        return rank(pm.at(value))


def rank_profile(pm: PMatrix) -> RankProfile:
    for r in range(min(pm.m, pm.n), 0, -1):
        g = Poly()
        found = False
        for mnr in pm.minors(r):
            if mnr.c:
                found = True
                g = mnr if not g.c else poly_gcd(g, mnr)
                if g.degree == 0:
                    break
        if found:
            roots, rest = g.rational_roots() if g.degree > 0 else ({}, g)
            # a leftover factor matters only if it has REAL (hence irrational) roots
            irr = rest if rest.degree > 0 and real_root_count(rest) > 0 else None
            return RankProfile(r, sorted(roots), irr)
    return RankProfile(0, [], None)


def generic_value(bad: Sequence[Fraction]) -> Fraction:
    """A rational value avoiding all critical values (used to evaluate the generic case)."""
    v = Fraction(1000003, 7)
    while v in bad:
        v += 1
    return v


def fmt_pmatrix(pm: PMatrix, name: str = "A") -> str:
    from .fmt import hcat
    cells = [[pm.entry_text(i, j) for j in range(pm.n)] for i in range(pm.m)]
    w = [max(len(cells[i][j]) for i in range(pm.m)) for j in range(pm.n)]
    lines = []
    for i in range(pm.m):
        body = "  ".join(cells[i][j].rjust(w[j]) for j in range(pm.n))
        l, r = ("⎡", "⎤") if i == 0 else ("⎣", "⎦") if i == pm.m - 1 else ("⎢", "⎥")
        if pm.m == 1:
            l, r = "[", "]"
        lines.append(f"{l} {body} {r}")
    return hcat(f"{name} = ", "\n".join(lines))


def values_text(vals: Sequence[Fraction]) -> str:
    return "{" + ", ".join(fs(v) for v in vals) + "}" if vals else "∅"
