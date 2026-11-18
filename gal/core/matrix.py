"""Exact matrices over Q (fractions.Fraction). No floats anywhere.

Notation follows the course notes: M_K(m, n), rows R_1..R_m, A^T, I_n.
Vectors are plain lists of Fractions (column vectors of K^n).
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Iterable, Sequence


def frac(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        raise TypeError("bool is not a scalar")
    if isinstance(x, (int, str)):
        return Fraction(x)
    raise TypeError(f"not an exact scalar: {x!r}")


def vec(xs: Iterable) -> list[Fraction]:
    return [frac(x) for x in xs]


def vadd(a, b):
    return [x + y for x, y in zip(a, b, strict=True)]


def vsub(a, b):
    return [x - y for x, y in zip(a, b, strict=True)]


def vscale(k, a):
    k = frac(k)
    return [k * x for x in a]


def dot(a, b) -> Fraction:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def is_zero_vec(a) -> bool:
    return all(x == 0 for x in a)


def clear_denominators(a: Sequence[Fraction]) -> list[Fraction]:
    """Positive rational multiple of `a` with coprime integer entries."""
    a = vec(a)
    if is_zero_vec(a):
        return a
    den = 1
    for x in a:
        den = den * x.denominator // gcd(den, x.denominator)
    ints = [int(x * den) for x in a]
    g = 0
    for x in ints:
        g = gcd(g, abs(x))
    return [Fraction(x // g) for x in ints]


class Matrix:
    __slots__ = ("rows", "m", "n")

    def __init__(self, rows: Iterable[Iterable]):
        self.rows = tuple(tuple(frac(x) for x in r) for r in rows)
        self.m = len(self.rows)
        if self.m == 0:
            raise ValueError("matrix needs at least one row")
        self.n = len(self.rows[0])
        if self.n == 0 or any(len(r) != self.n for r in self.rows):
            raise ValueError("matrix rows must be non-empty and of equal length")

    # ---- constructors -------------------------------------------------
    @staticmethod
    def identity(n: int) -> "Matrix":
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])

    @staticmethod
    def zeros(m: int, n: int) -> "Matrix":
        return Matrix([[0] * n for _ in range(m)])

    @staticmethod
    def diag(values: Sequence) -> "Matrix":
        n = len(values)
        return Matrix([[values[i] if i == j else 0 for j in range(n)] for i in range(n)])

    @staticmethod
    def from_cols(cols: Sequence[Sequence]) -> "Matrix":
        return Matrix(list(zip(*cols, strict=True)))

    @staticmethod
    def col_vector(v: Sequence) -> "Matrix":
        return Matrix([[x] for x in v])

    # ---- access -------------------------------------------------------
    def __getitem__(self, ij) -> Fraction:
        i, j = ij
        return self.rows[i][j]

    def row(self, i: int) -> list[Fraction]:
        return list(self.rows[i])

    def col(self, j: int) -> list[Fraction]:
        return [r[j] for r in self.rows]

    def cols(self) -> list[list[Fraction]]:
        return [self.col(j) for j in range(self.n)]

    def tolist(self) -> list[list[Fraction]]:
        return [list(r) for r in self.rows]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.m, self.n)

    @property
    def is_square(self) -> bool:
        return self.m == self.n

    @property
    def T(self) -> "Matrix":
        return Matrix(list(zip(*self.rows)))

    def trace(self) -> Fraction:
        if not self.is_square:
            raise ValueError("trace needs a square matrix")
        return sum((self.rows[i][i] for i in range(self.n)), Fraction(0))

    def minor(self, i: int, j: int) -> "Matrix":
        """The matrix Â_{i,j}: remove row i and column j (0-based)."""
        return Matrix(
            [[x for c, x in enumerate(r) if c != j] for k, r in enumerate(self.rows) if k != i]
        )

    def submatrix(self, rows: Sequence[int], cols: Sequence[int]) -> "Matrix":
        return Matrix([[self.rows[i][j] for j in cols] for i in rows])

    def hstack(self, other: "Matrix") -> "Matrix":
        if self.m != other.m:
            raise ValueError("hstack: different number of rows")
        return Matrix([a + b for a, b in zip(self.rows, other.rows)])

    def vstack(self, other: "Matrix") -> "Matrix":
        if self.n != other.n:
            raise ValueError("vstack: different number of columns")
        return Matrix(self.rows + other.rows)

    # ---- algebra ------------------------------------------------------
    def __eq__(self, other) -> bool:
        return isinstance(other, Matrix) and self.rows == other.rows

    def __hash__(self) -> int:
        return hash(self.rows)

    def __add__(self, other: "Matrix") -> "Matrix":
        self._same_type(other)
        return Matrix([[a + b for a, b in zip(r, s)] for r, s in zip(self.rows, other.rows)])

    def __sub__(self, other: "Matrix") -> "Matrix":
        self._same_type(other)
        return Matrix([[a - b for a, b in zip(r, s)] for r, s in zip(self.rows, other.rows)])

    def __neg__(self) -> "Matrix":
        return Matrix([[-a for a in r] for r in self.rows])

    def scale(self, k) -> "Matrix":
        k = frac(k)
        return Matrix([[k * a for a in r] for r in self.rows])

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.n != other.m:
                raise ValueError(
                    f"product not defined: {self.m}x{self.n} times {other.m}x{other.n}"
                )
            oc = other.cols()
            return Matrix([[dot(r, c) for c in oc] for r in self.rows])
        return self.scale(other)

    def __rmul__(self, k):
        return self.scale(k)

    def __pow__(self, p: int) -> "Matrix":
        if not self.is_square or p < 0:
            raise ValueError("power needs a square matrix and p >= 0")
        out = Matrix.identity(self.n)
        for _ in range(p):
            out = out * self
        return out

    def apply(self, v: Sequence) -> list[Fraction]:
        """A·v for a column vector v given as a list."""
        v = vec(v)
        if len(v) != self.n:
            raise ValueError("apply: wrong vector length")
        return [dot(r, v) for r in self.rows]

    def is_zero(self) -> bool:
        return all(x == 0 for r in self.rows for x in r)

    def is_symmetric(self) -> bool:
        return self.is_square and self == self.T

    def _same_type(self, other: "Matrix") -> None:
        if not isinstance(other, Matrix) or self.shape != other.shape:
            raise ValueError("sum is defined only for matrices of the same type")

    def __repr__(self) -> str:
        return "Matrix(" + "; ".join(" ".join(str(x) for x in r) for r in self.rows) + ")"
