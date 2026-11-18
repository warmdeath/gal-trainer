"""Polynomials in one variable over Q, and the characteristic polynomial
p_A(t) = det(t·I_n - A) (the convention of the notes: monic)."""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from .fmt import fs
from .gauss import laplace
from .matrix import Matrix, frac


class Poly:
    __slots__ = ("c",)

    def __init__(self, coeffs: Sequence = ()):
        c = [frac(x) for x in coeffs]
        while c and c[-1] == 0:
            c.pop()
        self.c = tuple(c)  # c[k] is the coefficient of t^k

    @staticmethod
    def t() -> "Poly":
        return Poly([0, 1])

    @staticmethod
    def const(x) -> "Poly":
        return Poly([x])

    @staticmethod
    def from_roots(roots: Sequence) -> "Poly":
        p = Poly([1])
        for r in roots:
            p = p * Poly([-frac(r), 1])
        return p

    @property
    def degree(self) -> int:
        return len(self.c) - 1  # -1 for the zero polynomial

    def is_const(self) -> bool:
        return len(self.c) <= 1

    def const_value(self) -> Fraction:
        return self.c[0] if self.c else Fraction(0)

    def __eq__(self, other) -> bool:
        if isinstance(other, (int, Fraction)):
            other = Poly([other])
        return isinstance(other, Poly) and self.c == other.c

    def __hash__(self):
        return hash(self.c)

    @staticmethod
    def _lift(x) -> "Poly":
        return x if isinstance(x, Poly) else Poly([x])

    def __add__(self, other):
        o = self._lift(other)
        n = max(len(self.c), len(o.c))
        a = list(self.c) + [Fraction(0)] * (n - len(self.c))
        b = list(o.c) + [Fraction(0)] * (n - len(o.c))
        return Poly([x + y for x, y in zip(a, b)])

    __radd__ = __add__

    def __neg__(self):
        return Poly([-x for x in self.c])

    def __sub__(self, other):
        return self + (-self._lift(other))

    def __rsub__(self, other):
        return self._lift(other) - self

    def __mul__(self, other):
        o = self._lift(other)
        if not self.c or not o.c:
            return Poly()
        out = [Fraction(0)] * (len(self.c) + len(o.c) - 1)
        for i, a in enumerate(self.c):
            for j, b in enumerate(o.c):
                out[i + j] += a * b
        return Poly(out)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._lift(other)
        if not o.is_const() or not o.c:
            raise ValueError("division only by a non-zero constant")
        return Poly([x / o.c[0] for x in self.c])

    def __pow__(self, k: int):
        if not isinstance(k, int) or k < 0:
            raise ValueError("exponent must be a non-negative integer")
        out = Poly([1])
        for _ in range(k):
            out = out * self
        return out

    def __call__(self, x) -> Fraction:
        x = frac(x)
        acc = Fraction(0)
        for a in reversed(self.c):
            acc = acc * x + a
        return acc

    def deflate(self, r) -> "Poly":
        """Divide by (t - r); r must be a root."""
        r = frac(r)
        out, acc = [], Fraction(0)
        for a in reversed(self.c):
            acc = acc * r + a
            out.append(acc)
        if out[-1] != 0:
            raise ValueError("not a root")
        return Poly(list(reversed(out[:-1])))

    def rational_roots(self) -> tuple[dict[Fraction, int], "Poly"]:
        """All roots in Q with their algebraic multiplicity, and the cofactor
        without rational roots."""
        roots: dict[Fraction, int] = {}
        p = self
        if p.degree < 1:
            return roots, p
        while p.degree >= 1 and p(0) == 0:
            roots[Fraction(0)] = roots.get(Fraction(0), 0) + 1
            p = p.deflate(0)
        if p.degree >= 1:
            den = 1
            for a in p.c:
                den = den * a.denominator // _gcd(den, a.denominator)
            ints = [int(a * den) for a in p.c]
            cands = set()
            for num in _divisors(abs(ints[0])):
                for d in _divisors(abs(ints[-1])):
                    cands.add(Fraction(num, d))
                    cands.add(Fraction(-num, d))
            for r in sorted(cands):
                while p.degree >= 1 and p(r) == 0:
                    roots[r] = roots.get(r, 0) + 1
                    p = p.deflate(r)
        return roots, p

    def __str__(self) -> str:
        return self.show("t")

    def show(self, var: str = "t") -> str:
        if not self.c:
            return "0"
        out = ""
        for k in range(len(self.c) - 1, -1, -1):
            a = self.c[k]
            if a == 0:
                continue
            mag = abs(a)
            mono = "" if k == 0 else var if k == 1 else f"{var}^{k}"
            coef = fs(mag)
            if mono and mag == 1:
                coef = ""
            elif mono and "/" in coef:
                coef = f"({coef})"
            term = coef + mono
            if not out:
                out = ("-" if a < 0 else "") + term
            else:
                out += (" - " if a < 0 else " + ") + term
        return out

    def __repr__(self):
        return f"Poly({self.show()})"


def poly_divmod(a: Poly, b: Poly) -> tuple[Poly, Poly]:
    """Euclidean division a = q·b + r with deg r < deg b."""
    if not b.c:
        raise ZeroDivisionError("division by the zero polynomial")
    q = [Fraction(0)] * max(1, a.degree - b.degree + 1)
    r = list(a.c)
    lead = b.c[-1]
    while len(r) >= len(b.c) and any(r):
        k = len(r) - len(b.c)
        f = r[-1] / lead
        q[k] = f
        for i, c in enumerate(b.c):
            r[i + k] -= f * c
        while r and r[-1] == 0:
            r.pop()
    return Poly(q), Poly(r)


def poly_gcd(a: Poly, b: Poly) -> Poly:
    """Monic gcd over Q (gcd(0, 0) = 0)."""
    while b.c:
        a, b = b, poly_divmod(a, b)[1]
    if not a.c:
        return a
    return a / a.c[-1]


def _gcd(a, b):
    from math import gcd
    return gcd(a, b)


def _divisors(n: int) -> list[int]:
    """Positive divisors of n in O(√n) (pairs d, n/d); [1] for n = 0."""
    if not n:
        return [1]
    from math import isqrt
    small, large = [], []
    for d in range(1, isqrt(n) + 1):
        if n % d == 0:
            small.append(d)
            if d != n // d:
                large.append(n // d)
    return small + large[::-1]


def charpoly(A: Matrix) -> Poly:
    """p_A(t) = det(t·I_n - A)."""
    if not A.is_square:
        raise ValueError("characteristic polynomial needs a square matrix")
    n = A.n
    entries = [[(Poly([-A[i, j], 1]) if i == j else Poly([-A[i, j]])) for j in range(n)]
               for i in range(n)]
    return laplace(entries)


def factored(p: Poly, var: str = "t") -> str:
    """Factorisation over Q as far as rational roots go: (t - 1)^2(t - 2)(t^2 + 1)."""
    roots, rest = p.rational_roots()
    out = ""
    if rest.degree <= 0 and rest != 1:
        out += fs(rest.const_value()) if rest.c else "0"
    for r in sorted(roots):
        lin = var if r == 0 else f"({var} {'-' if r > 0 else '+'} {fs(abs(r))})"
        out += lin + (f"^{roots[r]}" if roots[r] > 1 else "")
    if rest.degree >= 1:
        out += f"({rest.show(var)})"
    return out or "1"


def real_root_count(p: Poly) -> int:
    """Number of DISTINCT real roots of p (Sturm's theorem, exact)."""
    if p.degree <= 0:
        return 0
    seq = [p, Poly([k * p.c[k] for k in range(1, len(p.c))])]
    while seq[-1].degree > 0:
        r = poly_divmod(seq[-2], seq[-1])[1]
        if not r.c:
            break
        seq.append(-r)

    def changes(signs):
        s = [x for x in signs if x != 0]
        return sum(1 for a, b in zip(s, s[1:]) if a * b < 0)

    at_pos = [q.c[-1] for q in seq]
    at_neg = [q.c[-1] * (-1) ** q.degree for q in seq]
    return changes(at_neg) - changes(at_pos)
