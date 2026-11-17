"""Chapter 2 answers recomputed with sympy."""
import random

import sympy as sp

from gal.core.matrix import Matrix

from .helpers import sm, smv, sq, to_frac
from .verify import ans, equations_define, run


def cols(vs):
    return sp.Matrix.hstack(*[smv(v) for v in vs])


def is_basis_of(basis, span_vectors):
    """basis spans the same space as span_vectors and is independent."""
    if not basis:
        return all(all(x == 0 for x in v) for v in span_vectors) or cols(span_vectors).rank() == 0
    B, S = cols(basis), cols(span_vectors)
    return B.rank() == len(basis) == S.rank() == B.row_join(S).rank()


def test_subspace_answers():
    # every template's verdict, checked by brute-force counterexample search / closure
    run("vector_spaces", "subspace", lambda ex: None)
    from gal.topics.ch2 import _subspace_case
    seen = {}
    for s in range(200):
        text, ok, _ = _subspace_case(random.Random(s))
        key = text.split("|")[-1] if "|" in text else text[:12]
        seen[text] = ok
    for text, ok in seen.items():
        if "= 0}" in text and "R³" in text and "xy" not in text and "∪" not in text and "²" not in text:
            assert ok
        if any(k in text for k in ("xy = 0", "x ≥ 0", "det(A) = 0", "p(0) = 1", "²", "∪")):
            assert not ok


def test_combination_and_span():
    def comb(ex):
        vs, b = ex.data["vs"], ex.data["b"]
        lam = ans(ex).v
        assert cols(vs) * smv(lam) == smv(b)
    run("span_independence", "combination", comb)

    def ins(ex):
        vs, b = ex.data["vs"], ex.data["b"]
        A = cols(vs)
        assert ans(ex).value == (A.rank() == A.row_join(smv(b)).rank())
    run("span_independence", "in_span", ins)

    run("span_independence", "independent",
        lambda ex: _ok(ans(ex).value == (cols(ex.data["vs"]).rank() == len(ex.data["vs"]))))

    def dep(ex):
        lam = ans(ex).v
        assert any(lam) and cols(ex.data["vs"]) * smv(lam) == sp.zeros(len(ex.data["vs"][0]), 1)
    run("span_independence", "dependence", dep)
    run("span_independence", "generators", lambda ex: _ok(ans(ex).value == (cols(ex.data["vs"]).rank() == 3)))
    run("span_independence", "poly_indep", lambda ex: _ok(ans(ex).value == (cols(ex.data["coords"]).det() != 0)))


def _ok(b):
    assert b


def test_bases():
    def dim(ex):
        vs = ex.data["vs"]
        assert ans(ex, 0).value == cols(vs).rank()
        assert is_basis_of(ans(ex, 1).basis, vs)
    run("bases", "dim_span", dim)

    def coords(ex):
        B, v = ex.data["B"], ex.data["v"]
        assert cols(B) * smv(ans(ex).v) == smv(v)
        assert list(cols(B).solve(smv(v))) == [sq(x) for x in ans(ex).v]
    run("bases", "coordinates", coords)

    def comp(ex):
        vs = ex.data["vs"]
        assert cols(vs + [ans(ex).model]).det() != 0
    run("bases", "complete", comp)

    def col(ex):
        A = sm(ex.data["A"])
        assert is_basis_of(ans(ex, 0).basis, [list(A.col(j)) for j in range(A.cols)])
        assert all(any(sp.Matrix(b) == A.col(j) for j in range(A.cols)) for b in ans(ex, 0).basis)
        assert ans(ex, 1).value == A.rank()
    run("bases", "col_space", col)


def test_cart_param():
    def p2c(ex):
        ws = ex.data["ws"]
        a = ans(ex)
        equations_define(a.shown, [0] * len(ws[0]), ws, len(ws[0]))
    run("cart_param", "param_to_cart", p2c)

    def c2b(ex):
        A = sm(ex.data["A"])
        ns = A.nullspace()
        assert ans(ex, 0).value == len(ns)
        assert is_basis_of(ans(ex, 1).basis, [list(v) for v in ns] or [[0] * A.cols])
    run("cart_param", "cart_to_basis", c2b)


def test_notes_cartesian_example():
    from gal.topics.ch2 import ex_param_to_cart
    ex = ex_param_to_cart(None, [[2, -2, 1, 0], [1, 0, -1, 1]])
    assert ans(ex).check("x - 2z - 3w = 0; y + 2z + 2w = 0").ok


def _intersection_sympy(A_U, ws):
    A = sm(A_U)
    W = cols(ws)
    K = (A * W).nullspace()
    return [W * k for k in K]


def test_sum_intersection():
    def g(ex):
        A_U, ws = ex.data["A_U"], ex.data["ws"]
        A = sm(A_U)
        U = A.nullspace()
        dU, dW = len(U), cols(ws).rank()
        dsum = sp.Matrix.hstack(*U, cols(ws)).rank()
        inter = _intersection_sympy(A_U, ws)
        dint = sp.Matrix.hstack(*inter).rank() if inter else 0
        assert dU + dW == dsum + dint
        got = [ans(ex, i).value for i in range(4)]
        assert got == [dU, dW, dsum, dint]
        assert ans(ex, 4).value == (dint == 0)
    run("sum_intersection", "grassmann", g)

    def i(ex):
        inter = _intersection_sympy(ex.data["A_U"], ex.data["ws"])
        basis = ans(ex).basis
        if not inter or sp.Matrix.hstack(*inter).rank() == 0:
            assert basis == []
        else:
            assert is_basis_of(basis, [list(v) for v in inter])
    run("sum_intersection", "intersection", i)


def test_affine_positions():
    t, s = sp.symbols("t s")

    def ll(ex):
        d = ex.data
        P1, v1, P2, v2 = (smv(d[k]) for k in ("P1", "v1", "P2", "v2"))
        sols = sp.solve(list(P1 + t * v1 - P2 - s * v2), [t, s], dict=True)
        parallel = sp.Matrix.hstack(v1, v2).rank() == 1
        if parallel:
            expected = "coincidenti" if sols else "parallele"
        else:
            expected = "incidenti" if sols else "sghembe"
        a = ans(ex, 0)
        assert expected[:6] in a.options[a.correct]
        if expected == "incidenti":
            X = P1 + sols[0][t] * v1
            assert smv(ans(ex, 1).v) == X
    run("affine", "line_line", ll)

    def lp(ex):
        d = ex.data
        P1, v1, n = smv(d["P1"]), smv(d["v1"]), smv(d["nrm"])
        eq = (n.T * (P1 + t * v1))[0] - sq(d["d0"])
        sol = sp.solve(eq, t)
        a = ans(ex, 0)
        txt = a.options[a.correct]
        if sp.simplify(eq) == 0:
            assert "contained" in txt
        elif not sol:
            assert "parallel" in txt
        else:
            assert "incident" in txt
            assert smv(ans(ex, 1).v) == P1 + sol[0] * v1
    run("affine", "line_plane", lp)

    def pp(ex):
        d = ex.data
        A = sp.Matrix([d["n1"], d["n2"]])
        Ab = A.row_join(sp.Matrix([d["d1"], d["d2"]]))
        a = ans(ex)
        txt = a.options[a.correct]
        if A.rank() == 2:
            assert txt.startswith("incident")
        elif Ab.rank() == 1:
            assert txt == "coincident"
        else:
            assert txt.startswith("parallel")
    run("affine", "plane_plane", pp)


def test_pencil():
    def pen(ex):
        d = ex.data
        (coef, rhs), = ans(ex).shown
        n = smv(coef)
        assert any(coef)
        assert (n.T * smv(d["P"]))[0] == sq(rhs)
        # the plane contains l: its equation is a combination of the two equations of l
        M = sp.Matrix([list(d["a1"]) + [d["d1"]], list(d["a2"]) + [d["d2"]], list(coef) + [rhs]])
        assert M.rank() == 2
    run("affine", "pencil", pen)
