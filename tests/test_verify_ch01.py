"""Chapters 0–1: every generator's answer recomputed with sympy."""
import random
from fractions import Fraction

import sympy as sp

from gal.core.matrix import Matrix

from .helpers import sm, smv, sq, to_frac
from .verify import affine_equal_sympy, ans, equations_define, run


def test_vectors():
    def v(ex):
        d = ex.data
        exp = d["a"] * smv(d["v"]) + d["b"] * smv(d["w"])
        assert smv(ans(ex).v) == exp
    run("vectors", "combo", v)

    def v2(ex):
        assert smv(ans(ex).v) == smv(ex.data["B"]) - smv(ex.data["A"])
    run("vectors", "AB", v2)


def test_lines_planes():
    def line2(ex):
        a = ans(ex)
        A, B = smv(ex.data["A"]), smv(ex.data["B"])
        # both points on the line, direction parallel to B-A
        assert sp.Matrix.hstack(smv(a.dirs[0]), B - A).rank() == 1
        assert sp.Matrix.hstack(smv(a.dirs[0]), A - smv(a.point)).rank() <= 1
    run("lines_planes", "line2pts", line2)

    def cart(ex):
        a = ans(ex)
        equations_define(a.shown, ex.data["P"], [ex.data["v"]], 3)
    run("lines_planes", "line_cart", cart)

    def plane(ex):
        a = ans(ex)
        (coef, rhs), = a.shown
        n = smv(coef)
        for k in "ABC":
            assert (n.T * smv(ex.data[k]))[0] == sq(rhs)
        assert any(coef)
    run("lines_planes", "plane3pts", plane)


def test_notes_plane_example():
    from gal.topics.ch0 import ex_plane_three_points
    ex = ex_plane_three_points(None, ([1, 1, 0], [1, -2, 1], [2, -1, 3]))
    assert ans(ex).check("7x - y - 3z = 6").ok
    assert not ans(ex).check("7x - y - 3z = 5").ok


def test_matrices():
    run("matrices", "lincomb", lambda ex: _eq(ans(ex).M, ex.data["a"] * sm(ex.data["A"]) + ex.data["b"] * sm(ex.data["B"])))
    run("matrices", "product", lambda ex: _eq(ans(ex).M, sm(ex.data["A"]) * sm(ex.data["B"])))
    run("matrices", "sqsum", lambda ex: _eq(ans(ex).M, (sm(ex.data["A"]) + sm(ex.data["B"])) ** 2))
    run("matrices", "transp", lambda ex: _eq(ans(ex).M, (sm(ex.data["A"]) * sm(ex.data["B"])).T))

    def ptype(ex):
        m, p, q, n = ex.data["dims"]
        a = ans(ex)
        txt = a.options[a.correct]
        assert txt == (f"AB ∈ M({m},{n})" if p == q else "AB is not defined")
    run("matrices", "ptype", ptype)


def _eq(M: Matrix, S):
    assert sm(M) == S


def test_systems():
    run("systems", "solve", lambda ex: affine_equal_sympy(sm(ex.data["A"]), smv(ex.data["b"]), ans(ex)))

    def free(ex):
        assert ans(ex).value == ex.data["A"].n - sm(ex.data["A"]).rank()
    run("systems", "free", free)

    def ech(ex):
        M = sm(ex.data["M"])
        # echelon iff leading positions strictly increase and zero rows last
        leads = []
        for i in range(M.rows):
            row = list(M.row(i))
            leads.append(next((j for j, x in enumerate(row) if x != 0), None))
        nz = [l for l in leads if l is not None]
        ok = nz == sorted(set(nz)) and all(l is None for l in leads[len(nz):])
        assert ans(ex).value == ok
    run("systems", "echelon", ech)


def test_rank():
    run("rank", "rank", lambda ex: _check(ans(ex).value == sm(ex.data["A"]).rank()))

    def rb(ex):
        A, b, i, a0 = sm(ex.data["A"]), smv(ex.data["b"]), ex.data["row"], ex.data["alpha0"]
        al = sp.Symbol("alpha")
        bb = b.copy()
        bb[i] = b[i] - sq(a0) + al
        # solvable exactly at alpha0
        for test in [a0, a0 + 1, a0 - 2]:
            M = bb.subs(al, sq(test))
            solvable = A.rank() == A.row_join(M).rank()
            assert solvable == (test == a0)
        assert ans(ex, 0).value == a0
        assert ans(ex, 1).value == A.cols - A.rank()
    run("rank", "rouche_b", rb)

    def rk(ex):
        A0 = sm(ex.data["A0"])
        i, j = ex.data["ij"]
        k = sp.Symbol("k")
        Ak = A0.copy()
        Ak[i, j] = A0[i, j] + k
        roots = sp.solve(sp.Eq(Ak.det(), 0), k)
        assert len(roots) == 1 and to_frac(roots[0]) == ans(ex, 0).value
        Akk = Ak.subs(k, roots[0])
        b = smv(ex.data["b"])
        none = Akk.row_join(b).rank() > Akk.rank()
        c = ans(ex, 1)
        assert c.options[c.correct] == ("no solutions" if none else "infinitely many solutions (∞^1)")
    run("rank", "rouche_k", rk)


def _check(b):
    assert b


def test_inverse_det():
    run("inverse", "inverse", lambda ex: _eq(ans(ex).M, sm(ex.data["A"]).inv()))
    run("inverse", "invertible", lambda ex: _check(ans(ex).value == (sm(ex.data["A"]).det() != 0)))
    run("determinant", "det", lambda ex: _check(ans(ex).value == to_frac(sm(ex.data["A"]).det())))

    def binet(ex):
        d = ex.data
        n = d["n"]
        # build actual matrices with the requested determinants and compute
        A = sp.diag(*([sq(d["dA"])] + [1] * (n - 1)))
        B = sp.diag(*([1] * (n - 1) + [sq(d["dB"])]))
        B = sp.Matrix(B)
        Bq = B ** d["q"] if d["q"] >= 0 else B.inv() ** (-d["q"])
        X = sq(d["lam"]) * A ** d["p"] * Bq
        if d["tr"]:
            X = X * A.T
        assert to_frac(X.det()) == ans(ex).value
    run("determinant", "binet", binet)

    def rowops(ex):
        d = ex.data
        n = 4
        # simulate the operations on a concrete matrix of determinant d
        M = sp.diag(sq(d["d"]), 1, 1, 1)
        for t in d["texts"]:
            if "↔" in t:
                i, j = [int(x.strip()[1:]) - 1 for x in t.split("↔")]
                M.row_swap(i, j)
            elif "R" in t.split("→")[1] and t.split("→")[1].strip().count("R") == 2:
                lhs, rhs = t.split("→")
                i = int(lhs.strip()[1:]) - 1
                rest = rhs.strip()[len(f"R{i + 1}"):].strip()
                sign = -1 if rest.startswith("-") else 1
                coef_j = rest[1:].strip()
                k, j = coef_j.split("R")
                k = sign * int(k or 1)
                M[i, :] = M[i, :] + k * M[int(j) - 1, :]
            else:
                lhs, rhs = t.split("→")
                i = int(lhs.strip()[1:]) - 1
                k = int(rhs.strip().split("R")[0])
                M[i, :] = k * M[i, :]
        assert to_frac(M.det()) == ans(ex).value
    run("determinant", "rowops", rowops)
