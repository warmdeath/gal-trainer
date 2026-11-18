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
    """(n+, n−, n0) of a real symmetric matrix: n0 exactly from the rank, the signs from the
    (real) roots of the characteristic polynomial computed numerically."""
    n0 = A.shape[0] - A.rank()
    t = sp.symbols("t")
    roots = [complex(r).real for r in sp.Poly(A.charpoly(t).as_expr(), t).nroots(n=30)]
    nz = sorted(roots, key=abs)[n0:]
    return sum(1 for r in nz if r > 0), sum(1 for r in nz if r < 0), n0


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


# ------------------------------------------------------------------ 01/2025 itinere 2
def test_p2501i2():
    k = sp.symbols("k")
    A = sp.Matrix([[1, k, 0], [k, 0, 0], [1, 0, -1], [0, 1, 0]])
    b = sp.Matrix([1, -1, 0, 2 * k])
    assert A.subs(k, 1).rank() == A.subs(k, 1).row_join(b.subs(k, 1)).rank() == 3
    assert all(A.subs(k, v).row_join(b.subs(k, v)).rank() == 4 for v in (-1, 0, 2, 3))
    assert mcq_key("P2501I2-D1") == "a"
    assert signature(M([[2, 0, 1], [0, 4, -1], [1, -1, 1]])) == (3, 0, 0) and mcq_key("P2501I2-D8") == "c"
    assert M([[1, 2, 1], [2, 2, 0], [1, 0, -1]]).det() == 0 and mcq_key("P2501I2-D10") == "a"
    t = sp.symbols("t")
    Lp = lambda p: sp.expand((t + 1) * sp.diff(p, t) + (t**2 + 1) * sp.diff(p, t, 2))
    cols = [sp.Poly(Lp(m), t).all_coeffs()[::-1] for m in (1, t, t**2)]
    A = sp.Matrix([[c[i] if i < len(c) else 0 for c in cols] for i in range(3)])
    assert sym(key("P2501I2-E2a")["A"].M) == A
    assert sym(key("P2501I2-E2d")["coordinates of L¹⁰(3 + 3t + 3t²)"].M) == A**10 * sp.Matrix([3, 3, 3])
    x, y, z, X, Y, Z = sp.symbols("x y z X Y Z")
    s3 = sp.sqrt(3)
    q = 3*x**2 + 2*y**2 + z**2 + 2*s3*x*z - 2*x + 2*s3*z + 2
    sub = {x: (s3*X + Z)/2 + sp.Rational(1, 4), y: Y, z: (X - s3*Z)/2 - s3/4}
    assert sp.simplify(sp.expand(q.subs(sub)) - (4*X**2 + 2*Y**2 - 4*Z)) == 0


# ------------------------------------------------------------------ 01/2024 itinere II
def test_p2401i2():
    assert (sp.Matrix([1, -1, -1]).cross(sp.Matrix([0, -2, -3]))) == sp.Matrix([1, 3, -2]) and mcq_key("P2401I2-D1") == "b"
    x, y = sp.symbols("x y")
    assert sp.factor(x**2 + 4*x*y + 4*y**2 + 2*x + 4*y) == (x + 2*y) * (x + 2*y + 2)
    A = M([[0, 0, 0], [1, -1, 2], [0, 0, 1]])
    assert sym(key("P2401I2-E2e")["A²⁰²⁴v"].M) == A**2024 * sp.Matrix([1, 1, 1]) == sp.Matrix([0, 0, 1])


# ------------------------------------------------------------------ 02/2024
def test_p2402a():
    k = sp.symbols("k")
    A = sp.Matrix([[1, 1, 1], [1, 1, k + 1], [0, 1, 2 * k]])
    b = sp.Matrix([1, k**2 - k + 1, k**2 - 2 * k])
    assert sp.factor(A.det()) == -k and A.subs(k, 0).rank() == A.subs(k, 0).row_join(b.subs(k, 0)).rank() == 2
    H = sp.Matrix([[1, 2, -1, -2], [3, 0, 5, 0]]).T
    K = sp.Matrix([[-2, -1, -2, 1], [1, -1, 3, 1]]).T
    assert H.rank() + K.rank() - H.row_join(K).rank() == 2 and mcq_key("P2402A-D4") == "c"
    T = sym(key("P2402A-E2a")["M^B_B(T)"].M)
    Mm = sp.Matrix([[0, 1], [-1, 0]])
    basis = [sp.Matrix([[1, 0], [0, 0]]), sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, 0], [0, 1]])]
    for j, Bj in enumerate(basis):
        img = Mm.T * Bj * Mm
        assert img == sum((T[i, j] * basis[i] for i in range(3)), sp.zeros(2))
    x1, x2, x3, t = sp.symbols("x1 x2 x3 t")
    q = 2*x1**2 - x2**2 - x3**2 - 2*x2*x3 - 4*x3 - 2
    Aq = sp.Matrix([[2, 0, 0], [0, -1, -1], [0, -1, -1]])
    line = sp.Matrix([0, -1 + t, -t])             # the axis: A x + b ∥ ker A along it
    g = Aq * line + sp.Matrix([0, 0, -2])
    assert sp.Matrix([0, 1, -1]).cross(g) == sp.zeros(3, 1)


# ------------------------------------------------------------------ 07/2024
def test_p2407a():
    C = M([[2, 0, 0, 0], [1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]]).T      # 2, t+1, t²−1, t³−1
    B = M([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]]).T        # 1, t, t², t³+1
    half = sp.Rational(1, 2)
    assert C.inv() * B == sp.Matrix([[half, -half, half, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert mcq_key("P2407A-D5") == "a"
    basis = M([[1, 0, 2, 1], [0, 1, 1, 0], [3, 0, 0, -1], [0, 0, 0, 1]]).T
    assert basis.solve(sp.Matrix([5, 1, 5, 4])) == sp.Matrix([2, 1, 1, 3]) and mcq_key("P2407A-D10") == "d"
    n = sp.Matrix([1, -1, 0]).cross(sp.Matrix([0, 1, 1]))
    assert abs(sp.Matrix([0, 0, 1]).dot(n)) / n.norm() == 1 / sp.sqrt(3) and mcq_key("P2407A-D12") == "a"
    A = M([[5, 0, 3], [0, 8, 0], [3, 0, 5]])
    assert A.det() == 128 and A.eigenvals() == {8: 2, 2: 1}
