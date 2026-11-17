"""Exact numbers of the form q·√n (q rational, n square-free positive integer).

Enough for norms, distances and cosines: 12/√11, √(10/3), 2√3/3 ...
Closed under product and quotient; sums only between equal radicals.
"""
from __future__ import annotations

from fractions import Fraction

from .fmt import fs
from .matrix import frac


def _squarefree(n: int) -> tuple[int, int]:
    """n = s^2 * r with r square-free. Returns (s, r)."""
    s, r, d = 1, n, 2
    while d * d <= r:
        while r % (d * d) == 0:
            r //= d * d
            s *= d
        d += 1
    return s, r


class Surd:
    __slots__ = ("coef", "rad")

    def __init__(self, coef=0, rad: int = 1):
        coef = frac(coef)
        if rad < 1:
            raise ValueError("radicand must be positive")
        s, r = _squarefree(rad)
        coef *= s
        if coef == 0:
            r = 1
        self.coef, self.rad = coef, r

    @staticmethod
    def sqrt(x) -> "Surd":
        """√x for a rational x >= 0."""
        if isinstance(x, Surd):
            if x.rad != 1:
                raise ValueError("nested square roots are not supported")
            x = x.coef
        x = frac(x)
        if x < 0:
            raise ValueError("square root of a negative number")
        # sqrt(p/q) = sqrt(p*q)/q
        return Surd(Fraction(1, x.denominator), x.numerator * x.denominator)

    @staticmethod
    def lift(x) -> "Surd":
        return x if isinstance(x, Surd) else Surd(x)

    def is_rational(self) -> bool:
        return self.rad == 1

    def __eq__(self, other) -> bool:
        try:
            o = Surd.lift(other)
        except TypeError:
            return False
        return self.coef == o.coef and self.rad == o.rad

    def __hash__(self):
        return hash((self.coef, self.rad))

    def __neg__(self):
        return Surd(-self.coef, self.rad)

    def __mul__(self, other):
        o = Surd.lift(other)
        return Surd(self.coef * o.coef, self.rad * o.rad)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = Surd.lift(other)
        if o.coef == 0:
            raise ZeroDivisionError("division by zero")
        # 1/(c√r) = √r/(c·r)
        return Surd(self.coef / (o.coef * o.rad), self.rad * o.rad)

    def __rtruediv__(self, other):
        return Surd.lift(other) / self

    def __add__(self, other):
        o = Surd.lift(other)
        if self.coef == 0:
            return o
        if o.coef == 0:
            return self
        if self.rad != o.rad:
            raise ValueError("cannot add different square roots: simplify to the form a·sqrt(b)")
        return Surd(self.coef + o.coef, self.rad)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-Surd.lift(other))

    def __rsub__(self, other):
        return Surd.lift(other) - self

    def __pow__(self, k: int):
        if not isinstance(k, int) or k < 0:
            raise ValueError("exponent must be a non-negative integer")
        out = Surd(1)
        for _ in range(k):
            out = out * self
        return out

    def square(self) -> Fraction:
        return self.coef * self.coef * self.rad

    def __float__(self) -> float:
        return float(self.coef) * self.rad ** 0.5

    def __str__(self) -> str:
        """A form that the answer parser reads back: 12*sqrt(11)/11."""
        if self.rad == 1:
            return fs(self.coef)
        num, den = self.coef.numerator, self.coef.denominator
        head = "" if abs(num) == 1 else f"{abs(num)}*"
        s = ("-" if num < 0 else "") + f"{head}sqrt({self.rad})"
        return s + (f"/{den}" if den != 1 else "")

    def pretty(self) -> str:
        return str(self).replace("*sqrt(", "√(").replace("sqrt(", "√(")

    def __repr__(self):
        return f"Surd({self})"
