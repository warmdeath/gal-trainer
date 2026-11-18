"""Chapter 5 and conics answers recomputed with sympy (exact surds compared symbolically)."""
import sympy as sp

from gal.core.matrix import Matrix

from .helpers import sm, smv, sq, surd_to_sympy, to_frac
from .verify import ans, run


def cols(vs):
    return sp.Matrix.hstack(*[smv(v) for v in vs])


def same(a, b):
    assert sp.simplify(a - b) == 0, (a, b)


def test_bilinear():
    def g(ex):
        A = sm(ex.data["A"])
        n = A.rows
        x = sp.symbols(f"x0:{n}")
        y = sp.symbols(f"y0:{n}")
        f = (sp.Matrix([x]) * A * sp.Matrix(y))[0]
        # Gram matrix = second mixed derivatives
        G = sp.Matrix(n, n, lambda i, j: sp.diff(f, x[i], y[j]))
        assert sm(ans(ex, 0).M) == G
        assert ans(ex, 1).value == (G == G.T)
    run("bilinear", "gram", g)

    run("bilinear", "eval", lambda ex: _ok(ans(ex).value == to_frac((smv(ex.data["v"]).T * sm(ex.data["A"]) * smv(ex.data["w"]))[0])))

    def c(ex):
        S, P = sm(ex.data["S"]), sm(ex.data["P"])
        assert sm(ans(ex).M) == P.T * S * P
    run("bilinear", "congruent", c)


def _ok(b):
    assert b


def test_scalar_products():
    def pd(ex):
        S = sm(ex.data["S"])
        assert ans(ex).value == S.is_positive_definite
    run("scalar_products", "posdef", pd)

    def nrm(ex):
        v = smv(ex.data["v"])
        S = ex.data["S"]
        G = sp.eye(len(v)) if S is None else sm(S)
        same(surd_to_sympy(ans(ex).value), sp.sqrt((v.T * G * v)[0]))
    run("scalar_products", "norm", nrm)

    run("scalar_products", "distance",
        lambda ex: same(surd_to_sympy(ans(ex).value), (smv(ex.data["v"]) - smv(ex.data["w"])).norm()))

    def ang(ex):
        v, w = smv(ex.data["v"]), smv(ex.data["w"])
        same(surd_to_sympy(ans(ex).value), v.dot(w) / (v.norm() * w.norm()))
    run("scalar_products", "angle", ang)


def test_orthogonality_gs():
    def f(ex):
        B, v = ex.data["B"], ex.data["v"]
        assert cols(B) * smv(ans(ex).v) == smv(v)
    run("orthogonality", "fourier", f)

    def u(ex):
        v = smv(ex.data["v"])
        got = sp.Matrix([surd_to_sympy(x) for x in ans(ex).v])
        assert sp.simplify(got - v / v.norm()) == sp.zeros(len(v), 1)
    run("orthogonality", "unit", u)

    def o(ex):
        w = smv(ex.data["w"])
        B = ans(ex).basis
        assert len(B) == 2 and cols(B).rank() == 2 and (w.T * cols(B)).is_zero_matrix
    run("orthogonality", "orth_to", o)

    def gs(ex):
        B = [smv(b) for b in ex.data["B"]]
        expected = sp.GramSchmidt(B)
        got = [smv(c) for c in ans(ex).vs]
        for e, g in zip(expected, got):
            assert sp.Matrix.hstack(e, g).rank() == 1
    run("gram_schmidt", "gs", gs)

    def gson(ex):
        B = [smv(b) for b in ex.data["B"]]
        e = sp.GramSchmidt(B, orthonormal=True)[1]
        got = sp.Matrix([surd_to_sympy(x) for x in ans(ex).v])
        assert sp.simplify(got - e) == sp.zeros(3, 1)
    run("gram_schmidt", "gs_on", gson)


def _proj_sympy(ws):
    W = cols(ws)
    return W * (W.T * W).inv() * W.T  # standard orthogonal projection matrix


def test_projections():
    def comp(ex):
        W = cols(ex.data["ws"])
        B = ans(ex).basis
        assert len(B) == W.rows - W.rank() and (W.T * cols(B)).is_zero_matrix and cols(B).rank() == len(B)
    run("projections", "complement", comp)

    run("projections", "projection",
        lambda ex: _ok(smv(ans(ex).v) == _proj_sympy(ex.data["ws"]) * smv(ex.data["v"])))
    run("projections", "proj_matrix", lambda ex: _ok(sm(ans(ex).M) == _proj_sympy(ex.data["ws"])))

    def dist(ex):
        v = smv(ex.data["v"])
        r = v - _proj_sympy(ex.data["ws"]) * v
        same(surd_to_sympy(ans(ex).value), r.norm())
    run("projections", "dist_sub", dist)


def test_cross_distances():
    run("cross_distances", "cross", lambda ex: _ok(smv(ans(ex).v) == smv(ex.data["v"]).cross(smv(ex.data["w"]))))
    run("cross_distances", "area",
        lambda ex: same(surd_to_sympy(ans(ex).value), smv(ex.data["v"]).cross(smv(ex.data["w"])).norm()))

    x, y, z, t, s = sp.symbols("x y z t s", real=True)

    def pp(ex):
        P, a, b = smv(ex.data["P"]), smv(ex.data["a"]), ex.data["b"]
        # minimise squared distance over the plane by Lagrange-free parametrisation
        X = sp.Matrix([x, y, z])
        sol = sp.solve(list(X - P - sp.Symbol("k") * a) + [(a.T * X)[0] - b], [x, y, z, sp.Symbol("k")], dict=True)[0]
        foot = X.subs(sol)
        same(surd_to_sympy(ans(ex).value), (foot - P).norm())
    run("cross_distances", "pt_plane", pp)

    def pl(ex):
        P, Q, v = smv(ex.data["P"]), smv(ex.data["Q"]), smv(ex.data["v"])
        d2 = ((Q + t * v - P).T * (Q + t * v - P))[0]
        tmin = sp.solve(sp.diff(d2, t), t)[0]
        same(surd_to_sympy(ans(ex).value), sp.sqrt(d2.subs(t, tmin)))
    run("cross_distances", "pt_line", pl)

    def sk(ex):
        P1, v1, P2, v2 = (smv(ex.data[k]) for k in ("P1", "v1", "P2", "v2"))
        D = P1 + t * v1 - P2 - s * v2
        d2 = (D.T * D)[0]
        sol = sp.solve([sp.diff(d2, t), sp.diff(d2, s)], [t, s], dict=True)[0]
        same(surd_to_sympy(ans(ex).value), sp.sqrt(d2.subs(sol)))
    run("cross_distances", "skew", sk)


def test_isometries():
    def iso(ex):
        A = sm(ex.data["A"])
        assert ans(ex).value == (A.T * A == sp.eye(A.rows))
    run("isometries", "is_orth", iso)

    def rr(ex):
        A = sm(ex.data["A"])
        assert A.T * A == sp.eye(2)
        a = ans(ex)
        assert ("rotation" in a.options[a.correct]) == (A.det() == 1)
    run("isometries", "rot_refl", rr)

    def refl(ex):
        P0, d, Q = smv(ex.data["P0"]), smv(ex.data["d"]), smv(ex.data["Q"])
        R = smv(ans(ex).v)
        # midpoint on the line and Q - R orthogonal to the direction
        mid = (Q + R) / 2
        assert sp.Matrix.hstack(mid - P0, d).rank() <= 1
        assert (Q - R).dot(d) == 0
    run("isometries", "reflect", refl)


def test_conics():
    x, y = sp.symbols("x y", real=True)

    def cl(ex):
        M = sm(ex.data["M"])
        assert sm(ans(ex, 0).M) == M
        A = M[:2, :2]
        D3, D2 = M.det(), A.det()
        if D3 == 0:
            exp = "degenerate"
        elif D2 < 0:
            exp = "hyperbola"
        elif D2 == 0:
            exp = "parabola"
        else:
            # real ellipse iff it has real points: minimum of the form is negative
            X = sp.Matrix([x, y, 1])
            q = (X.T * M * X)[0]
            crit = sp.solve([sp.diff(q, x), sp.diff(q, y)], [x, y], dict=True)[0]
            val = q.subs(crit)
            exp = "real ellipse" if (val * A.trace() < 0) else "imaginary ellipse (no real points)"
        a = ans(ex, 1)
        assert a.options[a.correct] == exp
    run("conics", "classify", cl)

    def ce(ex):
        M = sm(ex.data["M"])
        X = sp.Matrix([x, y, 1])
        q = (X.T * M * X)[0]
        crit = sp.solve([sp.diff(q, x), sp.diff(q, y)], [x, y], dict=True)[0]
        assert smv(ans(ex).v) == sp.Matrix([crit[x], crit[y]])
    run("conics", "centre", ce)
