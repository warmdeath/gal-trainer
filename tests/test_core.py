"""Core engine verified against sympy on many random matrices."""
import random
from fractions import Fraction

import pytest
import sympy as sp

from gal.core.gauss import (column_space_basis, det, in_span, inverse, kernel_basis, rank,
                            row_reduce, rref, same_span, solve)
from gal.core.matrix import Matrix, clear_denominators
from gal.core.poly import Poly, charpoly, factored
from gal.core.surd import Surd

from .helpers import sm, smv, to_frac


def rand_matrix(rng, m, n, lo=-4, hi=4, rank_deficient=False):
    rows = [[rng.randint(lo, hi) for _ in range(n)] for _ in range(m)]
    if rank_deficient and m >= 2:
        a, b = rng.randint(-2, 2), rng.randint(-2, 2)
        rows[-1] = [a * x + b * y for x, y in zip(rows[0], rows[1])]
    return Matrix(rows)


SEEDS = range(300)


@pytest.mark.parametrize("seed", SEEDS)
def test_rank_det_rref_inverse(seed):
    rng = random.Random(seed)
    m, n = rng.randint(1, 5), rng.randint(1, 5)
    A = rand_matrix(rng, m, n, rank_deficient=rng.random() < 0.4)
    S = sm(A)
    assert rank(A) == S.rank()
    assert sm(rref(A)) == S.rref()[0]
    red = row_reduce(A)
    U = red.result
    # echelon form: each pivot strictly right of the previous, zero rows at bottom
    cols = [j for _, j in red.pivots]
    assert cols == sorted(cols) and len(set(cols)) == len(cols)
    for i in range(red.rank, m):
        assert all(x == 0 for x in U.row(i))
    if m == n:
        assert det(A) == to_frac(S.det())
        inv = inverse(A)
        if S.det() == 0:
            assert inv is None
        else:
            assert sm(inv) == S.inv()


@pytest.mark.parametrize("seed", SEEDS)
def test_kernel_and_column_space(seed):
    rng = random.Random(1000 + seed)
    m, n = rng.randint(1, 4), rng.randint(1, 5)
    A = rand_matrix(rng, m, n, rank_deficient=True)
    S = sm(A)
    K = kernel_basis(A)
    assert len(K) == len(S.nullspace())
    for v in K:
        assert all(x.denominator == 1 for x in v)
        assert S * smv(v) == sp.zeros(m, 1)
    if K:
        assert sp.Matrix.hstack(*[smv(v) for v in K]).rank() == len(K)
    C = column_space_basis(A)
    assert len(C) == S.rank()
    if C:
        assert sp.Matrix.hstack(*[smv(v) for v in C]).rank() == S.rank()


@pytest.mark.parametrize("seed", SEEDS)
def test_solve_against_sympy(seed):
    rng = random.Random(2000 + seed)
    m, n = rng.randint(1, 4), rng.randint(1, 4)
    A = rand_matrix(rng, m, n, rank_deficient=rng.random() < 0.5)
    b = [rng.randint(-5, 5) for _ in range(m)]
    sol = solve(A, b)
    S = sm(A)
    Sb = smv(b)
    rA, rAb = S.rank(), S.row_join(Sb).rank()
    assert (sol.rank_A, sol.rank_Ab) == (rA, rAb)
    if rA < rAb:
        assert sol.kind == "none"
        return
    assert S * smv(sol.particular) == Sb
    assert len(sol.basis) == n - rA
    assert sol.kind == ("unique" if rA == n else "inf")
    for w in sol.basis:
        assert S * smv(w) == sp.zeros(m, 1)


@pytest.mark.parametrize("seed", range(150))
def test_charpoly(seed):
    rng = random.Random(3000 + seed)
    n = rng.randint(1, 4)
    A = rand_matrix(rng, n, n, -3, 3)
    t = sp.Symbol("t")
    expected = sp.Poly(sm(A).charpoly(t).as_expr(), t)
    p = charpoly(A)
    assert [to_frac(c) for c in reversed(expected.all_coeffs())] == list(p.c)
    roots, rest = p.rational_roots()
    sroots = {to_frac(r): k for r, k in sp.roots(expected, filter="Q").items()}
    assert roots == sroots
    assert rest.degree == n - sum(roots.values())


def test_poly_basics():
    p = Poly.from_roots([1, 1, 2])
    assert str(p) == "t^3 - 4t^2 + 5t - 2"
    assert factored(p) == "(t - 1)^2(t - 2)"
    q = Poly([1, 0, 1]) * Poly([-2, 1])
    assert factored(q) == "(t - 2)(t^2 + 1)"


def test_surd():
    assert str(Surd.sqrt(14)) == "sqrt(14)"
    assert str(Surd(12) / Surd.sqrt(11)) == "12*sqrt(11)/11"
    assert Surd.sqrt(Fraction(10, 9)) == Surd(Fraction(1, 3), 10)
    assert Surd.sqrt(8) == Surd(2, 2)
    assert (Surd.sqrt(2) * Surd.sqrt(6)) == Surd(2, 3)
    with pytest.raises(ValueError):
        Surd.sqrt(2) + Surd.sqrt(3)
    for q in [Fraction(1, 3), Fraction(50), Fraction(7, 12)]:
        assert Surd.sqrt(q).square() == q


def test_span_helpers():
    assert in_span([2, 4, 6], [[1, 2, 3]])
    assert not in_span([1, 0, 0], [[1, 2, 3]])
    assert same_span([[1, 0, 0], [0, 1, 0]], [[1, 1, 0], [1, -1, 0]])
    assert clear_denominators([Fraction(1, 2), Fraction(-1, 3)]) == [3, -2]


def test_notes_examples():
    # Esempio p.16: M reduced to echelon form, then Gauss-Jordan gives (-28,-12,7)
    M = Matrix([[0, -3, -5], [2, -4, 2], [1, -1, 3]])
    sol = solve(M, [1, 6, 5])
    assert sol.kind == "unique" and sol.particular == [-28, -12, 7]
    # Esempio p.24: det = -33
    assert det(Matrix([[4, 0, 3], [1, 2, 0], [7, 3, 0]])) == -33
    # Esempio p.27: det = -72
    assert det(Matrix([[0, 0, -3, 5], [1, 8, 6, 7], [2, 0, 4, 7], [3, 0, 6, 9]])) == -72
    # Esempio p.23: inverse
    assert inverse(Matrix([[1, 0, 1], [2, 1, 6], [3, 0, 2]])) == Matrix(
        [[-2, 0, 1], [-14, 1, 4], [3, 0, -1]])
    # Esempio p.75: (t-1)^2 (t-2)
    assert factored(charpoly(Matrix([[3, -1, -2], [1, 0, -1], [0, 1, 1]]))) == "(t - 1)^2(t - 2)"
