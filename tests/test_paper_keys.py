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


# ------------------------------------------------------------------ 07/2025
def test_p2507a():
    d1, d2 = sp.Matrix([1, -1, 0]), sp.Matrix([0, 1, 1])
    assert abs(d1.dot(d2)) / (d1.norm() * d2.norm()) == sp.Rational(1, 2) and mcq_key("P2507A-D1") == "b"
    C = M([[2, 0, 0, 0], [1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]]).T
    B = M([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]]).T
    assert C.inv() * B == M([[sp.Rational(1, 2), -sp.Rational(1, 2), sp.Rational(1, 2), 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert mcq_key("P2507A-D10") == "a"
    basis = M([[1, 0, 2, 1], [0, 1, 1, 0], [3, 0, 0, -1], [0, 0, 0, 1]]).T
    assert basis.solve(sp.Matrix([5, 1, 5, 4])) == sp.Matrix([2, 1, 1, 3]) and mcq_key("P2507A-D11") == "d"
    Weq = sp.Matrix([[1, -1, 0, 1], [2, 0, 1, 0], [0, 1, -1, -1]])
    assert sym(key("P2507A-E1d")["matrix of π_W⊥"].M) == proj_matrix([list(r) for r in Weq.T.columnspace()])
    A = sym(key("P2507A-E3a")["A"].M)
    x = sp.symbols("x1:4")
    X = sp.Matrix(x)
    assert sp.expand((X.T * A * X)[0] - (x[0]**2 + x[1]**2 + 4 * x[2]**2 - 2 * x[0] * x[1])) == 0
    assert signature(A) == tuple(val("P2507A-E3b", l) for l in ("n+", "n−", "n0"))


# ------------------------------------------------------------------ 09/2025
def test_p2509a():
    assert sp.Matrix([1, 2, 3]).cross(sp.Matrix([0, 1, 2])) == sp.Matrix([1, -2, 1]) and mcq_key("P2509A-D2") == "a"
    B = M([[1, 0, 0], [1, 2, 0], [1, 2, 3]]).T
    Mb = M([[1, 2, 3], [0, 1, 2], [0, 0, 1]])
    assert B * Mb * B.inv() * sp.Matrix([0, 0, 3]) == sp.Matrix([3, 4, 3]) and mcq_key("P2509A-D4") == "a"
    Bc = M([[1, 1, 1], [1, 1, 0], [0, 1, 1]]).T       # B in canonical coordinates
    Cc = M([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T       # C = {e2, e3, e1}
    assert Cc.inv() * Bc == M([[1, 1, 1], [1, 0, 1], [1, 1, 0]]) and mcq_key("P2509A-D9") == "a"
    assert set(M([[1, -2], [1, 4]]).eigenvals()) == {2, 3} and mcq_key("P2509A-D10") == "c"
    assert signature(M([[1, -1], [-1, -2]])) == (1, 1, 0) and mcq_key("P2509A-D11") == "a"
    C = M([[1, 2], [0, -1]])
    assert C * C == sp.eye(2)                      # C⁻¹ = C
    x, y = sp.symbols("x y")
    F, line = sp.Matrix([sp.Rational(5, 4)] * 2), x + y - sp.Rational(3, 2)
    conic = sp.expand(2 * ((x - F[0])**2 + (y - F[1])**2) - line**2)
    assert conic == x**2 + y**2 - 2*x*y - 2*x - 2*y + 4
    assert conic.subs({x: 1, y: 1}) == 0
