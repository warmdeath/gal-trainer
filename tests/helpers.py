"""Conversions between gal objects and sympy, used by every test."""
from fractions import Fraction

import sympy as sp

from gal.core.matrix import Matrix


def to_frac(x) -> Fraction:
    x = sp.nsimplify(x)
    if not x.is_Rational:
        raise ValueError(f"not rational: {x}")
    return Fraction(int(x.p), int(x.q))


def sm(A: Matrix) -> sp.Matrix:
    return sp.Matrix([[sp.Rational(x.numerator, x.denominator) for x in r] for r in A.rows])


def smv(v) -> sp.Matrix:
    return sp.Matrix([sp.Rational(Fraction(x).numerator, Fraction(x).denominator) for x in v])


def sq(x) -> sp.Rational:
    x = Fraction(x)
    return sp.Rational(x.numerator, x.denominator)


def surd_to_sympy(s) -> sp.Expr:
    return sq(s.coef) * sp.sqrt(s.rad)
