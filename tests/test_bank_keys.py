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


def _dirs(item_id):
    return [sp.Matrix([sp.Rational(str(x)) for x in d]) for d in BANK.items[item_id].data["dirs"]]


def _par(u, v):
    return sp.Matrix.hstack(u, v).rank() == 1


def test_F08_E1_matrix_of_f():
    a, b, c = sp.symbols("a b c")
    M = sp.Matrix([[0, 1], [-1, 0]])
    Y = M.T * sp.Matrix([[a, b], [b, c]]) * M
    img = sp.Matrix([Y[0, 0], Y[0, 1], Y[1, 1]])
    F = img.jacobian([a, b, c])
    assert F == sp.Matrix(BANK.items["F08-E1a"].data["A"])


def test_F08_E7_E8_geometry():
    x, y, z = sp.symbols("x y z")
    # E7 l = {x + y − z = 1, 2x − y = 2}: direction = cross of normals
    dl = sp.Matrix([1, 1, -1]).cross(sp.Matrix([2, -1, 0]))
    assert _par(_dirs("F08-E7ai")[0], dl)
    for iid, normal in [("F08-E7aii", dl), ("F08-E7bii", sp.Matrix([1, 2, 3])),
                        ("F08-E8a", sp.Matrix([1, 2, 3])), ("F08-E8b", sp.Matrix([0, 1, 1]).cross(sp.Matrix([2, -1, 1])))]:
        ds = _dirs(iid)
        assert len(ds) == 2 and all(d.dot(normal) == 0 for d in ds) and sp.Matrix.hstack(*ds).rank() == 2, iid
    assert _par(_dirs("F08-E7bi")[0], sp.Matrix([1, 2, 3]))
    assert _par(_dirs("F08-E8c")[0], sp.Matrix([1, 1, -1]))
    d1 = sp.Matrix([1, 1, 1]).cross(sp.Matrix([1, 1, 0]))
    d = _dirs("F08-E8d")[0]
    assert d.dot(d1) == 0 and d.dot(sp.Matrix([1, -1, -1])) == 0 and d != sp.zeros(3, 1)
    d = _dirs("F08-E8e")[0]
    assert d.dot(sp.Matrix([1, -1, -2])) == 0 and d.dot(sp.Matrix([3, 1, 1])) == 0
    # E8 f: the line through P meets both lines
    P = sp.Matrix([1, 0, 0])
    t = sp.Symbol("t")
    L = P + t * _dirs("F08-E8f")[0]
    for eqs in ([x - y + z - 2, x + z], [x + 2 * y - z - 2, x - z]):
        sub = [e.subs({x: L[0], y: L[1], z: L[2]}) for e in eqs]
        assert sp.solve(sub, t), eqs


def _key(iid):
    a = ans(iid)[0]
    return sorted(getattr(a, "correct", [])) if isinstance(getattr(a, "correct", None), list) else a.correct


def test_Q01_determinant_quiz():
    A = sp.Matrix([[2, 1, 0, 1], [0, 1, 3, 0], [1, 0, 1, 2], [0, 2, 0, 1]])
    A = A * sp.diag(sp.Rational(3) / A.det(), 1, 1, 1)
    c = [A[:, i] for i in range(4)]
    opts12 = [-3, 0, 3, 8, 18]
    d12 = sp.Matrix.hstack(c[0] + c[1], 2 * c[1], 3 * c[2], c[3]).det()
    assert _key("Q01-D12") == opts12.index(d12)
    B = A * sp.diag(sp.Rational(1, 3), 1, 1, 1)
    assert B.det() == 1
    c = [B[:, i] for i in range(4)]
    d13 = sp.Matrix.hstack(c[1], c[2], 2 * c[0] + 3 * c[3], c[3]).det()
    assert _key("Q01-D13") == [2, -2, 6, 0, 1].index(d13)
    M = sp.Matrix([[1, 2, 0], [0, 1, 4], [3, 0, 1]])
    M = M * sp.diag(sp.Rational(3) / M.det(), 1, 1)
    assert _key("Q01-D10") == [3, 6, 9, 18, 24].index((2 * M).det())
    assert sp.Matrix([[1, 2, 0], [0, 1, sp.pi], [0, 0, 1]]).rank() != sp.Matrix([[1, 2, sp.pi], [0, 0, 1], [0, 0, 0]]).rank()
    assert _key("Q01-D2") == 1


def _plane_ok(coef, rhs, point, dirs):
    n = sp.Matrix(coef)
    return n.dot(sp.Matrix(point)) == rhs and all(n.dot(sp.Matrix(d)) == 0 for d in dirs)


def test_Q02_geometry_keys():
    # D15: r meets l at (1,1,1); option e) plane contains both, option c) does not
    assert _key("Q02-D15") == 4
    assert _plane_ok([1, -1, 2], 2, [0, 2, 2], [[1, -1, -1], [1, 1, 0]])
    assert not _plane_ok([1, -1, 2], 2, [0, 2, 3], [])
    t, s = sp.symbols("t s")
    assert sp.solve(list(sp.Matrix([4, 3, 2]) + t * sp.Matrix([1, -1, -2]) - sp.Matrix([1, 2, 1]) - s * sp.Matrix([3, 1, 1])), [t, s])
    assert _key("Q02-D16") == 1
    opts17 = [(2, 3, -3), (1, -1, 1), (2, 2, 2), (0, 1, -2), (1, -1, -1)]
    good = [i for i, n in enumerate(opts17) if sp.Matrix(n).dot(sp.Matrix([0, 1, -1])) == 0 and sp.Matrix(n).dot(sp.Matrix([1, -1, 0])) == 0]
    assert good == [_key("Q02-D17")]
    opts18 = [((-6, 3, 2), 6), ((1, 1, 0), 1), ((2, -1, 2), 6), ((4, -2, 1), 3), ((-2, 1, 2), 6)]
    good = [i for i, (n, d) in enumerate(opts18) if _plane_ok(n, d, [1, 2, 3], [[1, 0, 1]]) and _plane_ok(n, d, [0, 0, 3], [])]
    assert good == [_key("Q02-D18")]
    opts19 = {0: ((1, 1, 1), 9), 2: ((2, 2, 0), 10), 3: ((1, 3, -2), 3), 4: ((1, -2, 2), 4)}
    good = [i for i, (n, d) in opts19.items()
            if sp.Matrix([[2, 2, 0, 2], [1, 2, -1, 2], list(n) + [d]]).rank() == 2 and sp.Matrix(n).dot(sp.Matrix([2, 3, 4])) == d]
    assert good == [_key("Q02-D19")]


def test_Q02_D2_dimension():
    a, b, c = sp.symbols("a b c")
    v = sp.Matrix([a + b + c + 1, b + c, a - b - c + 1, 0])
    J = v.jacobian([a, b, c])
    const = v.subs({a: 0, b: 0, c: 0})
    assert J.rank() == 2 and J.rank() == J.row_join(const).rank()  # affine part lies in the span ⇒ subspace
    assert _key("Q02-D2") == 1


def test_Q03_computational_keys():
    # D4: coordinates w.r.t. b1, b2, b3: f(b1) = b2, f(b2) = b1 + b3, f(b3) = 0
    F = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 1, 0]])
    img = F * sp.Matrix([-1, 1, 3])
    opts = [(1, -1, 1), (1, 1, 1), (1, 1, 0), (0, 1, -1), (0, 0, 0)]
    assert _key("Q03-D4") == opts.index(tuple(img))
    # D9: solve for the matrix of f
    X = sp.Matrix(3, 3, sp.symbols("x0:9"))
    e = sp.eye(3)
    eqs = list(X * e[:, 0] - e[:, 2]) + list(X * (e[:, 1] + e[:, 2]) - (e[:, 0] - e[:, 1])) + list(X * (e[:, 0] - e[:, 1]))
    sol = sp.solve(eqs, list(X))
    f3 = tuple(X.subs(sol) * e[:, 2])
    opts = [(-1, 1, -1), (1, -1, -1), (0, 1, 0), (1, 0, 1), (0, 0, 0)]
    assert _key("Q03-D9") == opts.index(f3)
    # D10: eigenvalues 2 on u, -1 on v ⇒ coefficients (4, -1)
    assert _key("Q03-D10") == [(2, -1), (1, 1), (4, 1), (4, -1), (1, -1)].index((2 ** 2 * 1, (-1) ** 2 * -1))


def test_Q04_computational_keys():
    k = sp.Symbol("k")
    A = sp.Matrix([[2, 2], [1, 3]])
    good = [i for i, kv in enumerate([-2, -1, 2, 1, 0])
            if sorted(sp.Matrix([[1, 0], [kv, 2 * kv]]).eigenvals()) == sorted(A.eigenvals())]
    assert good == [_key("Q04-D3")]
    pairs = [([[1, 1], [1, 1]], [[2, 0], [0, 0]]), ([[1, 0], [1, 1]], [[1, 0], [0, 1]]),
             ([[1, 1], [0, 0]], [[2, 0], [0, 0]]), ([[1, 1], [1, 1]], [[-1, -3], [1, 3]]),
             ([[1, 2], [1, 2]], [[2, 4], [2, 4]])]

    def similar(X, Y):
        X, Y = sp.Matrix(X), sp.Matrix(Y)
        return X.charpoly().as_expr() == Y.charpoly().as_expr() and \
            sorted(X.jordan_form()[1].tolist()) == sorted(Y.jordan_form()[1].tolist())
    good = [i for i, (X, Y) in enumerate(pairs) if similar(X, Y)]
    assert good == _key("Q04-D12")
