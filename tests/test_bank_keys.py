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
