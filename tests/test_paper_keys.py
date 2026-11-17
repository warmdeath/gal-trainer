"""Independent sympy checks of hand-keyed answers in exam papers (values / matexpr with a given
matrix / MCQ keys that are computable). Papers by this year's lecturer are also checked against
the official solutions while transcribing; papers by the previous lecturer have no official
key, so their computable answers are re-derived here."""
import sympy as sp
import pytest

from gal.bank import build, get_bank

BANK = get_bank()


def key(iid):
    """The keyed answer object(s) of an item, by part label."""
    return {p.label: p.answer for p in build(BANK.items[iid]).parts}


def mcq_key(iid):
    a = key(iid)["answer"]
    return chr(97 + a.correct) if hasattr(a, "correct") else None


def M(rows):
    return sp.Matrix([[sp.Rational(str(x)) for x in r] for r in rows])


def sym(mat):
    """gal Matrix → sympy Matrix."""
    return sp.Matrix([[sp.Rational(x.numerator, x.denominator) for x in r] for r in mat.rows])


def val(iid, label):
    return sp.Rational(key(iid)[label].value.numerator, key(iid)[label].value.denominator)


def signature(A):
    ev = A.eigenvals()
    pos = sum(m for e, m in ev.items() if e > 0)
    neg = sum(m for e, m in ev.items() if e < 0)
    return pos, neg, A.shape[0] - pos - neg


def proj_matrix(cols):
    W = sp.Matrix.hstack(*[sp.Matrix(c) for c in cols])
    return W * (W.T * W).inv() * W.T


# ------------------------------------------------------------------ 06/2025
def test_p2506a():
    h, k = sp.symbols("h k")
    A = sp.Matrix([[1, -k, 1], [1, -1, 0], [1, -1, h]])
    assert sp.factor(A.det()) == h * (k - 1)                         # D1 → a
    assert mcq_key("P2506A-D1") == "a"
    ker = sp.Matrix([[1, 0, 1], [0, 1, 0], [-1, 0, -1]]).nullspace()   # D5: ker = Span(x²)
    assert len(ker) == 1 and ker[0][1] == 0 and ker[0][0] == -ker[0][2]
    assert mcq_key("P2506A-D5") == "b"
    assert sp.Matrix([[1, 1, 1, 1, 1, 1], [-1, 0, 1, 1, 0, -1], [2, 1, 0, 0, 1, 2]]).rank() == 2  # D6: 6 − 2
    assert mcq_key("P2506A-D6") == "c"
    assert sp.Matrix([[1, 1, 0], [1, 1, 1], [0, 1, 1]]).det() != 0  # D12 non-degenerate parabola
    A = M([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    E1 = (A - sp.eye(3)).nullspace()
    assert sym(key("P2506A-E2b")["matrix of π_E1"].M) == proj_matrix([list(v) for v in E1])
    assert signature(A) == tuple(val("P2506A-E2d", l) for l in ("n+", "n−", "n0"))
