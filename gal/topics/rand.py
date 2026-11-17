"""Random objects with 'nice numbers' (small integers, exact inverses...)."""
from __future__ import annotations

import random
from fractions import Fraction

from ..core.gauss import det, rank
from ..core.matrix import Matrix


def nz(rng: random.Random, lo=-5, hi=5) -> int:
    while True:
        x = rng.randint(lo, hi)
        if x:
            return x


def ivec(rng, n, lo=-4, hi=4, nonzero=True) -> list[int]:
    while True:
        v = [rng.randint(lo, hi) for _ in range(n)]
        if not nonzero or any(v):
            return v


def imat(rng, m, n, lo=-4, hi=4) -> Matrix:
    return Matrix([[rng.randint(lo, hi) for _ in range(n)] for _ in range(m)])


def unimodular(rng: random.Random, n: int, steps: int | None = None, bound: int = 6) -> Matrix:
    """Integer matrix with det ±1 (integer inverse), small entries."""
    for _ in range(200):
        M = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        for _ in range(steps or 2 * n):
            i, j = rng.sample(range(n), 2) if n > 1 else (0, 0)
            if n == 1:
                break
            k = rng.choice([-2, -1, 1, 2])
            M[i] = [a + k * b for a, b in zip(M[i], M[j])]
        if rng.random() < 0.5 and n > 1:
            i, j = rng.sample(range(n), 2)
            M[i], M[j] = M[j], M[i]
        off = sum(1 for i in range(n) for j in range(n) if i != j and M[i][j] != 0)
        if all(abs(x) <= bound for r in M for x in r) and off >= n - 1 and \
                sum(1 for r in M for x in r if x != 0) > n:
            return Matrix(M)
    return Matrix.identity(n)


def rank_r(rng: random.Random, m: int, n: int, r: int, bound: int = 9) -> Matrix:
    """Integer m×n matrix of rank exactly r."""
    for _ in range(500):
        if r == 0:
            return Matrix.zeros(m, n)
        L = imat(rng, m, r, -2, 2)
        R = imat(rng, r, n, -2, 2)
        A = L * R
        if rank(A) == r and all(abs(x) <= bound for row in A.rows for x in row) \
                and all(any(row) for row in A.rows):
            return A
    raise RuntimeError("could not build matrix of given rank")


def invertible(rng, n, dets=(1, -1, 2, -2), bound=6) -> Matrix:
    """Invertible integer matrix whose determinant is in `dets` (so A^-1 has nice entries)."""
    for _ in range(2000):
        A = imat(rng, n, n, -3, 3)
        if det(A) in dets and all(abs(x) <= bound for r in A.rows for x in r):
            return A
    return unimodular(rng, n)


def nice_frac(rng, den=(1, 1, 1, 2, 3)) -> Fraction:
    return Fraction(nz(rng, -6, 6), rng.choice(den))
