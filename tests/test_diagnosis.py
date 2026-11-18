"""Typical mistakes are recognised and explained."""
from fractions import Fraction

from gal.core.gauss import inverse
from gal.core.matrix import Matrix
from gal.exercise import MatrixAns, ScalarAns, VectorAns
from gal.topics.ch1 import ex_det, ex_inverse, ex_product
from gal.topics.ch4 import ex_charpoly
from gal.topics.ch5 import ex_projection


def test_product_BA_and_entrywise():
    A, B = Matrix([[1, 2], [3, 4]]), Matrix([[0, 1], [1, 0]])
    ex = ex_product(None, (A, B))
    a = ex.parts[0].answer
    assert a.check("2 1; 4 3").ok                     # AB
    assert "commutative" in a.check("3 4; 1 2").msg   # BA
    assert "entry by entry" in a.check("0 2; 3 0").msg


def test_inverse_transposed():
    A = Matrix([[1, 2], [3, 4]])
    ex = ex_inverse(None, A)
    inv = inverse(A)
    assert "transpose" in ex.parts[0].answer.check("; ".join(" ".join(str(x) for x in r) for r in inv.T.rows)).msg


def test_det_missing_signs():
    ex = ex_det(None, Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]]))
    a = ex.parts[0].answer
    assert a.value == 1
    assert "(-1)^(i+j)" in a.check("-79").msg  # 1·(-24) + 2·(-20) + 3·(-5): expansion without signs
    assert "sign" in a.check("-1").msg.lower()


def test_charpoly_sign_convention():
    ex = ex_charpoly(None, Matrix([[2, 0, 0], [0, 3, 0], [0, 0, 4]]))
    assert "det(tI - A)" in ex.parts[0].answer.check("-t^3 + 9t^2 - 26t + 24").msg


def test_projection_non_orthogonal_basis():
    ex = ex_projection(None, ([1, 0, 0], [[1, 1, 0], [1, 0, 1]]))
    a = ex.parts[0].answer
    # naive Fourier with a non-orthogonal basis: (1/2)(1,1,0) + (1/2)(1,0,1) = (1, 1/2, 1/2)
    assert "Gram-Schmidt" in a.check("(1, 1/2, 1/2)").msg


def test_generic_matrix_diagnoses():
    a = MatrixAns(Matrix([[1, 2], [3, 4]]))
    assert "transpose" in a.check("1 3; 2 4").msg
    assert "entry (2,1)" in a.check("1 2; 5 4").msg
    assert "2 times" in a.check("2 4; 6 8").msg
    v = VectorAns([1, 2, 3])
    assert "component 2" in v.check("(1, 5, 3)").msg
    assert "wrong sign" in v.check("(-1, -2, -3)").msg
    s = ScalarAns(Fraction(3, 4))
    assert "reciprocal" in s.check("4/3").msg
