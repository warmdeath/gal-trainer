"""Hand-written answer keys (tasks 'values', 'yesno', 'mcq') checked with sympy where the
question is computational."""
import sympy as sp

from gal.bank import build, load_bank

BANK = load_bank()


def ans(item_id):
    return [p.answer for p in build(BANK.items[item_id]).parts]


def test_F03_E2a_det_properties():
    A = sp.diag(2, 1, 1) * sp.Matrix([[1, 2, 3], [0, 1, 4], [0, 0, 1]])  # any matrix with det 2
    assert A.det() == 2
    exp = [A.inv().det(), (A ** 2).det(), (A.T * A).det(), (A * A.T).det(), (2 * A).det(), (3 * A.inv()).det()]
    assert [sp.Rational(x.value) for x in ans("F03-E2a")] == exp


def test_F03_E2b_column_operations():
    A = sp.Matrix([[3, 1, 0, 2], [0, 1, 0, 0], [1, 0, 1, 5], [0, 2, 0, 1]])
    A = A * sp.diag(sp.Rational(3) / A.det(), 1, 1, 1)
    assert A.det() == 3
    c = [A[:, i] for i in range(4)]
    B = sp.Matrix.hstack(c[0], 2 * c[1], -3 * c[2], 4 * c[3])
    C = sp.Matrix.hstack(c[0], c[0] + c[1], c[0] + c[1] + c[2], c[3])
    D = sp.Matrix.hstack(c[3], c[0], c[1], c[2])
    E = sp.Matrix.hstack(c[0] + c[1], c[1] + c[2], c[2] + c[0], c[0] + c[1] + c[2])
    assert [x.value for x in ans("F03-E2b")] == [B.det(), C.det(), D.det(), E.det()]


def test_F03_E11_pairs():
    a, b = sp.symbols("a b")
    found = []
    for av in range(-4, 5):
        for bv in range(-4, 5):
            M = sp.Matrix([[1, 1, av], [av, bv, 1], [1, av, -1]])
            rhs = sp.Matrix([av, 0, av])
            homog = rhs == sp.zeros(3, 1)
            if homog and 3 - M.rank() == 1:
                found.append((av, bv))
    assert found == [(0, 1)]
    assert [x.value for x in ans("F03-E11")] == [0, 1]


def test_F05_E5_triple():
    a, b, c = sp.symbols("a b c")
    A = sp.Matrix([[1, b, a, c], [0, a, c, b], [-1, c, b, a]])
    sols = []
    for av in range(-3, 4):
        for bv in range(-3, 4):
            for cv in range(-3, 4):
                M = A.subs({a: av, b: bv, c: cv})
                in_ker = (M * sp.Matrix([3, -2, 1, 1])).is_zero_matrix
                in_im = M.rank() == M.row_join(sp.Matrix([3, -2, 1])).rank()
                if in_ker and not in_im:
                    sols.append((av, bv, cv))
    assert sols == [(0, 1, -1)]
    assert tuple(x.value for x in ans("F05-E5")) == (0, 1, -1)


def test_F05_E6_existence():
    def exists(pairs):
        # least squares-free check: f exists iff the linear system for its matrix is solvable
        X = sp.Matrix(2, 2, sp.symbols("x0:4"))
        eqs = []
        for v, w in pairs:
            eqs += list(X * sp.Matrix(v) - sp.Matrix(w))
        return bool(sp.linsolve(eqs, list(X)))
    cases = {"F05-E6a": [((1, 2), (3, 0)), ((2, 7), (4, 5)), ((1, 5), (1, 4))],
             "F05-E6b": [((1, 2), (3, 0)), ((2, 4), (5, 0)), ((0, 1), (1, 1))],
             "F05-E6c": [((1, 2), (1, 0)), ((2, 3), (1, 3)), ((3, 4), (-1, 6))],
             "F05-E6d": [((1, 2), (1, 0)), ((2, 4), (2, 0))]}
    for iid, pairs in cases.items():
        assert ans(iid)[0].value == exists(pairs), iid


def test_F05_E2b_equality():
    A = sp.Matrix([[1, 2, 1, -2], [2, 4, 2, 4], [0, 0, 0, -1]])
    W = sp.Matrix([[-3, 1, 1, 0], [0, 1, -2, 0], [-3, -1, 5, 0]]).T
    assert (A * W).is_zero_matrix
    assert (W.rank() == 4 - A.rank()) == ans("F05-E2b")[0].value


def test_F07_E3_parameters():
    k, s, t = sp.symbols("k s t")
    A = sp.Matrix([[1, -1], [1, k]])
    sol_ii = sp.solve(A.charpoly(t).as_expr().subs(t, 3), k)
    v = sp.Matrix([4, -1])
    w = A * v
    sol_iii = sp.solve(w[0] * v[1] - w[1] * v[0], k)
    assert [x.value for x in ans("F07-E3a")] == [sp.Rational(sol_ii[0]), sp.Rational(sol_iii[0])]
    B = sp.Matrix([[s, 1, 0], [1, 3, 1], [0, 1, s]])
    roots = sorted(sp.solve((-3 * sp.eye(3) - B).det(), s))
    assert ans("F07-E3bi")[0].values == roots
    w = sp.Matrix([1, 0, -1])
    assert sp.simplify(B * w - s * w) == sp.zeros(3, 1) and ans("F07-E3bii")[0].value is True


def test_F07_E4_similar_to_D_minus_I():
    T = sp.Matrix([[1, 2, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1], [0, 0, 1, 2]])
    assert T.det() != 0
    A = T * sp.diag(-1, 2, -1, 2) * T.inv() - sp.eye(4)
    a = ans("F07-E4")
    assert a[0].value == A.det()
    assert a[1].values == sorted(A.eigenvals())


def test_F07_E7_similarity():
    tt = sp.Symbol("tt")
    A = sp.Matrix([[2, 0, 2], [0, 2, 0], [0, 0, 0]])
    B = sp.Matrix([[2, 1, 0], [0, 2, 0], [0, 0, 1]])

    def similar(X, Y):
        # both diagonalisable with the same p, or identical Jordan structure (via sympy)
        return X.charpoly().as_expr() == Y.charpoly().as_expr() and X.jordan_form()[1] == Y.jordan_form()[1]
    assert ans("F07-E7a")[0].value == similar(A, B)
    vals_C = [v for v in range(-4, 5) if similar(A, sp.diag(2, v, 2))]
    vals_D = [v for v in range(-4, 5) if similar(A, sp.Matrix([[2, 0, 1], [0, v, 0], [0, 0, 2]]))]
    a = ans("F07-E7bc")
    assert a[0].values == vals_C and a[1].values == vals_D
