from fractions import Fraction as F

import pytest

from gal.core.parse import (ParseError, parse_equations, parse_matrix, parse_param, parse_poly,
                            parse_real, parse_scalar, parse_vector, parse_vectors, parse_yesno)
from gal.core.poly import Poly
from gal.core.surd import Surd


@pytest.mark.parametrize("text,val", [
    ("3", 3), ("-2", -2), ("3/4", F(3, 4)), ("-3/4", F(-3, 4)), ("0.5", F(1, 2)),
    ("1/2 + 1/3", F(5, 6)), ("2*(1-3)", -4), ("2(1-3)", -4), ("(-1)^3", -1), ("2^-1", None),
    ("1/(2/3)", F(3, 2)), ("- -2", 2),
])
def test_scalar(text, val):
    if val is None:
        with pytest.raises(ParseError):
            parse_scalar(text)
    else:
        assert parse_scalar(text) == val


@pytest.mark.parametrize("text,val", [
    ("sqrt(14)", Surd(1, 14)), ("12/sqrt(11)", Surd(F(12, 11), 11)), ("√2", Surd(1, 2)),
    ("2sqrt(3)/3", Surd(F(2, 3), 3)), ("sqrt(10/3)", Surd(F(1, 3), 30)), ("sqrt(8)", Surd(2, 2)),
    ("sqrt(2)*sqrt(2)", Surd(2)), ("11/sqrt(6)", Surd(F(11, 6), 6)), ("-sqrt(2)/2", Surd(F(-1, 2), 2)),
])
def test_real(text, val):
    assert parse_real(text) == val


def test_bad_inputs():
    for t in ["", "1 +", "sqrt(-1)", "x", "1/0", "sqrt(2)+sqrt(3)", "(1"]:
        with pytest.raises(ParseError):
            parse_real(t)


def test_vectors_and_matrices():
    assert parse_vector("(1, -2, 3/2)") == [1, -2, F(3, 2)]
    assert parse_vector("1 -2 3") == [1, -2, 3]
    assert parse_vector("[1,2]") == [1, 2]
    with pytest.raises(ParseError):
        parse_vector("(1,2)", 3)
    assert parse_vectors("(1,0,1); (0,1,1)") == [[1, 0, 1], [0, 1, 1]]
    assert parse_vectors("(1,0,1), (0,1,1)") == [[1, 0, 1], [0, 1, 1]]
    assert parse_vectors("span((1,0),(0,1))") == [[1, 0], [0, 1]]
    assert parse_vectors("none") == []
    M = parse_matrix("1 2; 3 4")
    assert M.tolist() == [[1, 2], [3, 4]]
    assert parse_matrix("[[1,2],[3,4]]") == M
    assert parse_matrix("1, -1/2; 0, 1").tolist() == [[1, F(-1, 2)], [0, 1]]
    with pytest.raises(ParseError):
        parse_matrix("1 2; 3", (2, 2))


def test_poly():
    assert parse_poly("t^3 - 4t^2 + 5t - 2") == Poly([-2, 5, -4, 1])
    assert parse_poly("(t-1)^2(t-2)") == Poly([-2, 5, -4, 1])
    assert parse_poly("t**2+1") == Poly([1, 0, 1])
    assert parse_poly("5") == Poly([5])


def test_param():
    pt, dirs = parse_param("(1,2,0) + t(1,0,1) + s(0,1,1)", 3)
    assert pt == [1, 2, 0] and dirs == [[1, 0, 1], [0, 1, 1]]
    pt, dirs = parse_param("t(1,1)", 2)
    assert pt == [0, 0] and dirs == [[1, 1]]
    pt, dirs = parse_param("(1,2) - t(1,-1)", 2)
    assert pt == [1, 2] and dirs == [[-1, 1]]
    pt, dirs = parse_param("(3,0,2) + span((0,1,0))", 3)
    assert dirs == [[0, 1, 0]]
    assert parse_param("none", 3) is None
    pt, dirs = parse_param("(-28,-12,7)", 3)
    assert pt == [-28, -12, 7] and dirs == []


def test_equations():
    eqs = parse_equations("x - 2z - 3w = 0; y + 2z + 2w = 0", ["x", "y", "z", "w"])
    assert eqs == [([1, 0, -2, -3], 0), ([0, 1, 2, 2], 0)]
    assert parse_equations("7x - y - 3z = 6", ["x", "y", "z"]) == [([7, -1, -3], 6)]
    assert parse_equations("2x = y + 1", ["x", "y"]) == [([2, -1], 1)]
    with pytest.raises(ParseError):
        parse_equations("x*y = 1", ["x", "y"])


def test_yesno():
    assert parse_yesno("sì") and parse_yesno("Yes") and not parse_yesno("no")
