"""Helpers to verify exercise answers independently with sympy."""
import random

import sympy as sp

from gal.exercise import (AffineAns, ChoiceAns, EquationsAns, IntAns, MatrixAns, ScalarAns,
                          SpanAns, VectorAns, YesNoAns)
from gal.topics import get

from .helpers import sm, smv, sq, to_frac

N_SEEDS = 60


def run(topic_id, key, verifier, seeds=N_SEEDS):
    g = get(topic_id).gen(key)
    for seed in range(seeds):
        ex = g(random.Random(seed))
        verifier(ex)


def ans(ex, i=0):
    return ex.parts[i].answer


def affine_equal_sympy(A_sym, b_sym, answer: AffineAns):
    """Check an AffineAns against sympy's gauss_jordan_solve."""
    try:
        sol, params = A_sym.gauss_jordan_solve(b_sym)
    except ValueError:
        assert answer.point is None
        return
    assert answer.point is not None
    p = sp.Matrix(answer.point)
    assert A_sym * smv(answer.point) == b_sym
    assert len(answer.dirs) == params.shape[0]
    for d in answer.dirs:
        assert A_sym * smv(d) == sp.zeros(A_sym.rows, 1)
    if answer.dirs:
        assert sp.Matrix.hstack(*[smv(d) for d in answer.dirs]).rank() == len(answer.dirs)


def equations_define(eqs, point, dirs, n):
    """Equations (coeffs, rhs) define exactly point + span(dirs)."""
    A = sp.Matrix([[sq(c) for c in a] for a, _ in eqs])
    b = sp.Matrix([sq(r) for _, r in eqs])
    assert A * smv(point) == b
    for d in dirs:
        assert A * smv(d) == sp.zeros(len(eqs), 1)
    assert A.rank() == n - len(dirs)
