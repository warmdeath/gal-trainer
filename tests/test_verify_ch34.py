"""Chapters 3–4 answers recomputed with sympy."""
import sympy as sp

from gal.core.matrix import Matrix

from .helpers import sm, smv, sq, to_frac
from .verify import ans, run

t = sp.Symbol("t")


def cols(vs):
    return sp.Matrix.hstack(*[smv(v) for v in vs])


def basis_of_nullspace(basis, M):
    ns = M.nullspace()
    if not ns:
        return basis == []
    B = cols(basis)
    return B.rank() == len(basis) == len(ns) and (M * B).is_zero_matrix


def test_ker_im():
    def k(ex):
        A = sm(ex.data["A"])
        assert basis_of_nullspace(ans(ex, 0).basis, A)
        assert ans(ex, 1).value == A.rank()
        assert ans(ex, 2).value == (A.rank() == A.cols)
        assert ans(ex, 3).value == (A.rank() == A.rows)
    run("linear_maps", "ker_im", k)


def test_nullity_rank():
    def nr(ex):
        d = ex.data
        if d["case"] == "im":
            assert ans(ex).value == d["n"] - d["k"]
        elif d["case"] == "ker":
            assert ans(ex).value == d["n"] - d["r"]
        else:
            assert ans(ex).value == (d["n"] <= d["m"])
    run("linear_maps", "nullity_rank", nr)


POLY_MAPS = {
    "f(p) = t·p''(t) - p'(t)": lambda p: sp.expand(t * sp.diff(p, t, 2) - sp.diff(p, t)),
    "f(p) = p'(t) + p(0)": lambda p: sp.expand(sp.diff(p, t) + p.subs(t, 0)),
    "f(p) = p(t + 1) - p(t)": lambda p: sp.expand(p.subs(t, t + 1) - p),
    "f(p) = (t + 1)·p'(t)": lambda p: sp.expand((t + 1) * sp.diff(p, t)),
    "f(p) = p'(t) - p''(t)": lambda p: sp.expand(sp.diff(p, t) - sp.diff(p, t, 2)),
    "f(p) = p(1) + p(-1)·t": lambda p: sp.expand(p.subs(t, 1) + p.subs(t, -1) * t),
    "f(p) = p(1)·t² + p(0)": lambda p: sp.expand(p.subs(t, 1) * t ** 2 + p.subs(t, 0)),
}


def test_poly_map():
    def pm(ex):
        f = POLY_MAPS[ex.data["text"]]
        k = ex.data["k"]
        a0, a1, a2 = sp.symbols("a0 a1 a2")
        img = sp.Poly(f(a0 + a1 * t + a2 * t ** 2), t)
        # matrix in canonical bases via sympy
        M = sp.zeros(k + 1, 3)
        for j, a in enumerate((a0, a1, a2)):
            for i in range(k + 1):
                M[i, j] = sp.Poly(img.as_expr(), t).coeff_monomial(t ** i).coeff(a)
        assert sm(ex.data["A"]) == M
        r = M.rank()
        assert [ans(ex, 0).value, ans(ex, 1).value, ans(ex, 2).value] == [3 - r, r, r == k + 1]
    run("linear_maps", "poly_map", pm)


def test_rep_matrix():
    def canon(ex):
        # parse the formula shown to the student with sympy and rebuild the matrix
        line = ex.prompt.split("\n")[0]
        lhs, rhs = line.split(") = (", 1)
        names = lhs[lhs.index("f(") + 2:].split(", ")
        syms = sp.symbols(names)
        from sympy.parsing.sympy_parser import (implicit_multiplication_application,
                                                standard_transformations, parse_expr)
        tr = standard_transformations + (implicit_multiplication_application,)
        comps = [parse_expr(c, local_dict=dict(zip(names, syms)), transformations=tr)
                 for c in rhs.rstrip(").").split(", ")]
        M = sp.Matrix([[sp.diff(c, s) for s in syms] for c in comps])
        assert sm(ans(ex).M) == M
    run("rep_matrix", "canonical", canon)

    def interp(ex):
        B, Ws, v = ex.data["B"], ex.data["Ws"], ex.data["v"]
        A = cols(Ws) * cols(B).inv()  # the unique linear map with f(b_i) = w_i
        assert A * smv(v) == smv(ans(ex).v)
    run("rep_matrix", "interpolation", interp)

    def rep(ex):
        A, B, C = sm(ex.data["A"]), cols(ex.data["B"]), cols(ex.data["C"])
        M = sm(ans(ex).M)
        # column j = coordinates of f(b_j) w.r.t. C:  C·M = A·B
        assert C * M == A * B
    run("rep_matrix", "rep", rep)

    def cb(ex):
        P = cols(ex.data["Bp"])
        assert sm(ans(ex, 0).M) == P.inv()
        assert P * smv(ans(ex, 1).v) == smv(ex.data["v"])
    run("rep_matrix", "change_basis", cb)

    def endo(ex):
        A, P = sm(ex.data["A"]), sm(ex.data["P"])
        assert sm(ans(ex).M) == P.inv() * A * P
    run("rep_matrix", "endo_change", endo)


def test_eigen():
    def cp(ex):
        A = sm(ex.data["A"])
        expected = sp.Poly(A.charpoly(t).as_expr(), t)
        got = ans(ex).p
        assert [sq(c) for c in reversed(got.c)] == expected.all_coeffs()
    run("eigen", "charpoly", cp)

    def ev(ex):
        A = sm(ex.data["A"])
        real = sorted(to_frac(r) for r in A.eigenvals() if r.is_real)
        assert real == ans(ex).values
    run("eigen", "eigenvalues", ev)

    def es(ex):
        A, lam = sm(ex.data["A"]), sq(ex.data["lam"])
        assert basis_of_nullspace(ans(ex).basis, lam * sp.eye(A.rows) - A)
    run("eigen", "eigenspace", es)

    def ie(ex):
        A, v = sm(ex.data["A"]), smv(ex.data["v"])
        Av = A * v
        is_e = sp.Matrix.hstack(Av, v).rank() == 1
        assert ans(ex, 0).value == is_e
        if is_e:
            assert Av == sq(ans(ex, 1).value) * v
    run("eigen", "is_eigvec", ie)


def test_diagonalization():
    def d(ex):
        A = sm(ex.data["A"])
        real = all(r.is_real for r in A.eigenvals())
        assert ans(ex).value == (real and A.is_diagonalizable())
    run("diagonalization", "diagonalizable", d)

    def m(ex):
        A, lam = sm(ex.data["A"]), sq(ex.data["lam"])
        assert ans(ex, 0).value == A.eigenvals()[lam]
        assert ans(ex, 1).value == len((lam * sp.eye(A.rows) - A).nullspace())
    run("diagonalization", "mult", m)

    def fp(ex):
        A = sm(ex.data["A"])
        P = sm(ans(ex).P)
        D = P.inv() * A * P
        assert P.det() != 0 and D.is_diagonal()
    run("diagonalization", "find_P", fp)

    def param(ex):
        k = sp.Symbol("k")
        rows = ex.data["rows"]
        bad = []
        for kv in range(-4, 5):
            M = sp.Matrix([[kv if x is None else x for x in r] for r in rows])
            if not M.is_diagonalizable():
                bad.append(kv)
        assert bad == [ans(ex).value]
    run("diagonalization", "param", param)

    def comp(ex):
        p = ex.data["p"]
        C = sm(ans(ex).M)
        assert [sq(c) for c in reversed(p.c)] == sp.Poly(C.charpoly(t).as_expr(), t).all_coeffs()
        # notes' convention: ones below the diagonal
        n = C.rows
        assert all(C[i, i - 1] == 1 for i in range(1, n))
    run("diagonalization", "companion", comp)


def test_notes_diag_examples():
    from gal.topics.ch4 import _diag_analysis
    ok, _ = _diag_analysis(Matrix([[3, -1, -2], [1, 0, -1], [0, 1, 1]]))
    assert not ok
    ok, _ = _diag_analysis(Matrix([[4, -3, -3], [1, 0, -1], [1, -1, 0]]))
    assert ok
    ok, _ = _diag_analysis(Matrix([[3, -2, -1], [0, 0, 1], [2, -2, -1]]))
    assert not ok  # (t² + 1)(t - 2): not over R
