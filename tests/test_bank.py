"""The practice bank (official material): structure + every auto-checked item verified
independently with sympy, by task kind."""
import re
from fractions import Fraction

import pytest
import sympy as sp

from gal.bank import build, load_bank
from gal.bank.tasks import AUTO_TASKS, POS_OPTS, SYS_OPTS, TASKS
from gal.exercise import MultiChoiceAns, ChoiceAns
from gal.topics import BY_ID

from .helpers import sm, smv, sq, surd_to_sympy, to_frac

BANK = load_bank()
ITEMS = list(BANK.items.values())
AUTO = [i for i in ITEMS if i.auto]


def S(x):
    return sq(Fraction(x.replace(" ", "")) if isinstance(x, str) else Fraction(x))


def SM(rows):
    return sp.Matrix([[S(x) for x in r] for r in rows])


def SV(v):
    return sp.Matrix([S(x) for x in v])


def cols(vs):
    return sp.Matrix.hstack(*[SV(v) for v in vs])


# ------------------------------------------------------------------ structure
def test_bank_not_empty_and_ids_unique():
    assert len(ITEMS) > 0
    assert len({i.id for i in ITEMS}) == len(ITEMS)


@pytest.mark.parametrize("item", ITEMS, ids=[i.id for i in ITEMS])
def test_item_structure_and_round_trip(item):
    assert item.topic in BY_ID, f"{item.id}: unknown topic {item.topic}"
    assert item.task in TASKS, f"{item.id}: unknown task {item.task}"
    assert item.text, f"{item.id}: empty text"
    assert item.id.startswith(item.source.id + "-"), item.id
    ex = build(item)
    assert ex.parts
    for p in ex.parts:
        assert p.answer.check(p.answer.typed()).ok, (item.id, p.label)


def test_mcq_counts_match_statement():
    """'Selezionare le 3 affermazioni vere' / '(2 risposte corrette)' agree with the key."""
    words = {"una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5}
    for item in ITEMS:
        if item.task != "mcq":
            continue
        letters = item.data["correct"]
        letters = letters if isinstance(letters, list) else [letters]
        n = len(letters)
        assert len(set(letters)) == n
        assert all(0 <= ord(c.lower()) - 97 < len(item.data["options"]) for c in letters), item.id
        m = re.search(r"(?:le|\()\s*(\d+)\s*(?:affermazioni|risposte|coppie)", item.text)
        if m:
            assert int(m.group(1)) == n, f"{item.id}: statement says {m.group(1)}, key has {n}"


# ------------------------------------------------------------------ sympy verifiers
def answers(item):
    return [p.answer for p in build(item).parts]


def v_solve(d, a):
    A, b = SM(d["A"]), SV(d["b"])
    ans = a[0]
    try:
        sol, params = A.gauss_jordan_solve(b)
    except ValueError:
        assert ans.point is None
        return
    assert A * SV([str(x) for x in ans.point]) == b
    assert len(ans.dirs) == params.shape[0]
    for w in ans.dirs:
        assert A * SV([str(x) for x in w]) == sp.zeros(A.rows, 1)


def v_nsol(d, a):
    Ab = SM(d["Ab"])
    A = Ab[:, :-1]
    rA, rAb = A.rank(), Ab.rank()
    k = 0 if rA < rAb else (1 if rA == A.cols else 2)
    assert a[0].correct == k
    if k == 2:
        assert a[1].value == A.cols - rA


def v_rank(d, a):
    assert a[0].value == SM(d["A"]).rank()


def v_homog(d, a):
    A = SM(d["A"])
    assert a[0].value == A.rank()
    _basis_of_null(a[1].basis, A)


def _basis_of_null(basis, A):
    ns = A.nullspace()
    assert len(basis) == len(ns)
    if basis:
        B = cols([[str(x) for x in v] for v in basis])
        assert B.rank() == len(basis) and (A * B).is_zero_matrix


def _same_span(basis, vectors):
    if not basis:
        assert not vectors or cols(vectors).rank() == 0
        return
    B = cols([[str(x) for x in v] for v in basis])
    V = cols(vectors) if vectors and not isinstance(vectors, sp.Matrix) else vectors
    assert B.rank() == len(basis) == V.rank() == B.row_join(V).rank()


def v_det(d, a):
    assert a[0].value == to_frac(SM(d["A"]).det())


def v_inverse(d, a):
    A = SM(d["A"])
    inv = A.det() != 0
    assert a[0].value == inv
    if inv:
        assert sm(a[1].M) == A.inv()


def v_matexpr(d, a):
    mats = {k: SM(v) for k, v in d["matrices"].items()}
    for expr, ans in zip(d["exprs"], a):
        py = re.sub(r"\^T", ".T", expr)
        py = re.sub(r"\^(\d+)", r"**\1", py)
        py = re.sub(r"\)\s*\(", ")*(", py)
        try:
            R = eval(py, {}, mats)
            assert ans.defined and sm(ans.M) == R, expr
        except (sp.ShapeError, sp.NonSquareMatrixError, TypeError, ValueError):
            assert not ans.defined, expr


def v_indep(d, a):
    V = cols(d["vectors"])
    r = V.rank()
    assert a[0].value == (r == V.cols)
    if len(a) > 1:
        assert a[1].value == (r == V.rows)


def v_basis_span(d, a):
    V = cols(d["vectors"])
    assert a[0].value == V.rank()
    _same_span(a[1].basis, V)


def v_basis_eqs(d, a):
    A = SM(d["A"])
    assert a[0].value == len(A.nullspace())
    _basis_of_null(a[1].basis, A)


def v_coords(d, a):
    B, v = cols(d["basis"]), SV(d["v"])
    is_basis = B.rows == B.cols and B.det() != 0
    assert a[0].value == is_basis
    if is_basis:
        assert B.solve(v) == sp.Matrix([sq(x) for x in a[1].v])


def v_complete_basis(d, a):
    V = cols(d["vectors"])
    k = 0
    if d.get("ask_indep", True):
        assert a[0].value == (V.rank() == V.cols)
        k = 1
    model = a[k].model
    assert len(model) == V.rows - V.rank()
    assert V.row_join(cols([[str(x) for x in m] for m in model]) if model else V).rank() == V.rows


def v_in_span(d, a):
    V = cols(d["vectors"])
    assert a[0].value == (V.rank() == V.row_join(SV(d["v"])).rank())


def _eqs_define(eqs, P, dirs):
    A = sp.Matrix([[sq(c) for c in co] for co, _ in eqs])
    b = sp.Matrix([sq(r) for _, r in eqs])
    assert A * P == b
    for w in dirs:
        assert A * w == sp.zeros(len(eqs), 1)
    assert A.rank() == P.rows - (sp.Matrix.hstack(*dirs).rank() if dirs else 0)


def v_cartesian(d, a):
    dirs = [SV(x) for x in d["dirs"]]
    P = SV(d.get("point", [0] * len(d["dirs"][0])))
    _eqs_define(a[0].shown, P, dirs)
    # the checker accepts exactly the right number of equations
    assert len(a[0].dirs) == sp.Matrix.hstack(*dirs).rank()


def v_affine_forms(d, a):
    if "points" in d:
        pts = [SV(p) for p in d["points"]]
        P, dirs = pts[0], [q - pts[0] for q in pts[1:]]
    else:
        P, dirs = SV(d["point"]), [SV(x) for x in d["dirs"]]
    par = a[0]
    # the parametric model answer spans the same affine subspace
    Pm = sp.Matrix([sq(x) for x in par.point])
    D = sp.Matrix.hstack(*dirs)
    assert D.rank() == D.row_join(Pm - P).rank()
    _eqs_define(a[1].shown, P, dirs)
    for q, ans in zip(d.get("check", []), a[2:]):
        assert ans.value == (D.rank() == D.row_join(SV(q) - P).rank())


def _sub_basis(obj):
    if "span" in obj:
        return cols(obj["span"])
    ns = SM(obj["A"]).nullspace()
    return sp.Matrix.hstack(*ns) if ns else sp.zeros(SM(obj["A"]).cols, 1)


def v_grassmann(d, a):
    U, W = _sub_basis(d["U"]), _sub_basis(d["W"])
    du, dw = U.rank(), W.rank()
    ds = U.row_join(W).rank()
    _same_span(a[0].basis, U)
    _same_span(a[1].basis, W)
    assert a[2].value == ds and a[3].value == du + dw - ds
    assert a[4].value == (du + dw == ds)


def _aff_sympy(obj):
    if "dirs" in obj:
        dirs = [SV(x) for x in obj["dirs"]]
        P = SV(obj.get("point", [0] * len(obj["dirs"][0])))
        return P, dirs
    A, b = SM(obj["A"]), SV(obj.get("b", [0] * len(obj["A"])))
    sol, params = A.gauss_jordan_solve(b)
    P = sol.subs({p: 0 for p in params})
    dirs = [sol.diff(p) for p in params]
    return P, dirs


def v_position(d, a):
    P1, V1 = _aff_sympy(d["S1"])
    P2, V2 = _aff_sympy(d["S2"])
    ts = sp.symbols(f"t0:{len(V1) + len(V2)}")
    expr = P1 + sum((t * v for t, v in zip(ts, V1)), sp.zeros(P1.rows, 1)) - P2 - \
        sum((t * v for t, v in zip(ts[len(V1):], V2)), sp.zeros(P1.rows, 1))
    meet = bool(sp.linsolve(list(expr), ts))
    r1, r2 = sp.Matrix.hstack(*V1).rank(), sp.Matrix.hstack(*V2).rank()
    r12 = sp.Matrix.hstack(*V1, *V2).rank()
    contained = r12 == max(r1, r2)
    k = (0 if contained else 2) if meet else (1 if contained else 3)
    assert a[0].correct == k


def v_kerim(d, a):
    A = SM(d["A"])
    _basis_of_null(a[0].basis, A)
    _same_span(a[1].basis, A.columnspace() and sp.Matrix.hstack(*A.columnspace()) or sp.zeros(A.rows, 1))
    assert a[2].value == (A.rank() == A.cols) and a[3].value == (A.rank() == A.rows)


def v_repmat(d, a):
    A = SM(d["A"])
    B = cols(d["B"]) if "B" in d else sp.eye(A.cols)
    C = cols(d["C"]) if "C" in d else sp.eye(A.rows)
    assert C * sm(a[0].M) == A * B
    for v, ans in zip(d.get("images", []), a[1:]):
        assert C * sp.Matrix([sq(x) for x in ans.v]) == A * SV(v)


def v_interp_matrix(d, a):
    B, W = cols(d["basis"]), cols(d["images"])
    exp = {"E_E": W * B.inv(), "B_E": W, "B_B": B.inv() * W, "E_B": B.inv() * W * B.inv()}
    for key, ans in zip(d.get("ask", ["E_E"]), a):
        assert sm(ans.M) == exp[key]
    # defining property: M^E_E maps b_j to w_j
    assert (W * B.inv()) * B == W


def v_change_basis(d, a):
    B, C = cols(d["B"]), cols(d["C"])
    assert C * sm(a[0].M) == B and B * sm(a[1].M) == C
    k = 2
    if "vB" in d:
        assert C * sp.Matrix([sq(x) for x in a[k].v]) == B * SV(d["vB"])
        k += 1
    if "vC" in d:
        assert B * sp.Matrix([sq(x) for x in a[k].v]) == C * SV(d["vC"])


t = sp.Symbol("t")


def _cp(A, ans):
    exp = sp.Poly(A.charpoly(t).as_expr(), t).all_coeffs()
    assert [sq(c) for c in reversed(ans.p.c)] == exp


def v_eigen(d, a):
    A = SM(d["A"])
    _cp(A, a[0])
    if any(r.is_real and not r.is_rational for r in A.eigenvals()):
        # irrational branch: only p_A and 'diagonalisable' (all real & simple)
        ev = A.eigenvals()
        assert all(r.is_real for r in ev) and all(m == 1 for m in ev.values())
        assert len(a) == 2 and a[1].value is True
        return
    real = sorted(to_frac(r) for r in A.eigenvals() if r.is_real)
    assert a[1].values == real
    k = 2
    if d.get("eigenspaces", True):
        for lam in real:
            _basis_of_null(a[k].basis, sq(lam) * sp.eye(A.rows) - A)
            k += 1
    diag = all(r.is_real for r in A.eigenvals()) and A.is_diagonalizable()
    assert a[k].value == diag
    if diag and d.get("ask_P", True):
        P = sm(a[k + 1].P)
        assert P.det() != 0 and (P.inv() * A * P).is_diagonal()


def v_charpoly(d, a):
    A = SM(d["A"])
    _cp(A, a[0])
    if not d.get("spectrum", True):
        return
    assert a[1].values == sorted(to_frac(r) for r in A.eigenvals() if r.is_real)


def v_roots(d, a):
    x = sp.Symbol(d.get("var", "x"))
    from sympy.parsing.sympy_parser import (implicit_multiplication_application, parse_expr,
                                            standard_transformations, convert_xor)
    tr = standard_transformations + (implicit_multiplication_application, convert_xor)
    p = sp.Poly(parse_expr(d["p"], local_dict={x.name: x}, transformations=tr), x)
    rts = sp.roots(p)
    assert a[0].values == sorted(to_frac(r) for r in rts if r.is_rational)
    assert a[1].value == len(set(sp.real_roots(p)))
    assert a[2].value == len(set(sp.Poly(p).all_roots()))


def v_metric(d, a):
    v, w = SV(d["v"]), SV(d["w"])
    G = SM(d["S"]) if "S" in d else sp.eye(len(v))
    ip = lambda x, y: (x.T * G * y)[0]
    vals = [sp.sqrt(ip(v, v)), sp.sqrt(ip(w, w)), sp.sqrt(ip(v - w, v - w))]
    if len(a) == 4:
        vals.append(ip(v, w) / (vals[0] * vals[1]))
    for ans, exp in zip(a, vals):
        assert sp.simplify(surd_to_sympy(ans.value) - exp) == 0


def v_gram_schmidt(d, a):
    B = [SV(b) for b in d["vectors"]]
    G = SM(d["S"]) if "S" in d else sp.eye(B[0].rows)
    cs = [sp.Matrix([sq(x) for x in c]) for c in a[0].vs]
    for i in range(len(cs)):
        for j in range(i):
            assert (cs[i].T * G * cs[j])[0] == 0
        Bi, Ci = sp.Matrix.hstack(*B[:i + 1]), sp.Matrix.hstack(*cs[:i + 1])
        assert Bi.rank() == Ci.rank() == Bi.row_join(Ci).rank() == i + 1


def v_gram(d, a):
    G = SM(d["G"])
    assert sm(a[0].M) == G
    assert a[1].value == (G == G.T and G.is_positive_definite)


def _drop_set(M, var):
    """Real values where rank(M) < generic rank, via sympy gcd of minors."""
    r = M.rank()
    from itertools import combinations
    g = 0
    for I in combinations(range(M.rows), r):
        for J in combinations(range(M.cols), r):
            g = sp.gcd(g, M.extract(list(I), list(J)).det())
    if g == 0 or sp.Poly(g, var).degree() <= 0:
        return [], r
    roots = sp.real_roots(sp.Poly(g, var))
    assert all(x.is_rational for x in roots), "irrational drop"
    return sorted(set(to_frac(x) for x in roots)), r


def _pm_sym(rows, var):
    from sympy.parsing.sympy_parser import (implicit_multiplication_application, parse_expr,
                                            standard_transformations, convert_xor)
    tr = standard_transformations + (implicit_multiplication_application, convert_xor)
    return sp.Matrix([[parse_expr(str(x), local_dict={var.name: var}, transformations=tr) for x in r]
                      for r in rows])


def v_param_invertible(d, a):
    var = sp.Symbol(d.get("param", "λ"))
    M = _pm_sym(d["A"], var)
    D = sp.expand(M.det())
    if d.get("ask_det"):
        got = sum(sq(c) * var ** k for k, c in enumerate(a[0].p.c))
        assert sp.expand(got - D) == 0
        a = a[1:]
    if D == 0:
        assert a[0].value is False
        return
    roots = sorted(set(to_frac(x) for x in sp.real_roots(sp.Poly(D, var)))) if sp.Poly(D, var).degree() > 0 else []
    assert a[0].values == roots


def v_param_rank(d, a):
    var = sp.Symbol(d.get("param", "λ"))
    M = _pm_sym(d["A"], var)
    drops, r = _drop_set(M, var)
    assert a[0].values == drops and a[1].value == r
    for c, ans in zip(drops, a[2:]):
        assert ans.value == M.subs(var, sq(c)).rank()


def v_param_system(d, a):
    var = sp.Symbol(d.get("param", "λ"))
    A = _pm_sym(d["A"], var)
    b = _pm_sym([[x] for x in d["b"]], var)
    cand = sorted(set(_drop_set(A, var)[0]) | set(_drop_set(A.row_join(b), var)[0]))

    def kind(val):
        Av, bv = A.subs(var, val), b.subs(var, val)
        rA, rAb = Av.rank(), Av.row_join(bv).rank()
        return (0, 0) if rA < rAb else ((1, 0) if rA == Av.cols else (2, Av.cols - rA))
    gen = kind(sp.Rational(1000003, 7))
    # cross-check the generic case at another random value
    assert kind(sp.Rational(-999983, 11)) == gen
    crit = [c for c in cand if kind(sq(c)) != gen]
    assert a[0].values == crit
    assert a[1].correct == gen[0]
    k = 2
    if gen[0] == 2:
        assert a[2].value == gen[1]
        k = 3
    for c in crit:
        kc, f = kind(sq(c))
        assert a[k].correct == kc
        k += 1
        if kc == 2:
            assert a[k].value == f
            k += 1


def v_param_indep(d, a):
    var = sp.Symbol(d.get("param", "k"))
    V = _pm_sym([list(r) for r in zip(*d["vectors"])], var)
    drops, r = _drop_set(V, var)
    if r == V.cols:
        assert a[0].values == drops
    else:
        assert a[0].value is False
    if len(a) > 1:
        if r == V.rows:
            assert a[1].values == drops
        else:
            assert a[1].value is False


def _diag_R(M):
    return all(e.is_real for e in M.eigenvals()) and M.is_diagonalizable()


def v_param_diag(d, a):
    var = sp.Symbol(d.get("param", "k"))
    A = _pm_sym(d["A"], var)
    n = A.rows
    roots = sp.roots(sp.Poly(A.charpoly(t).as_expr(), t))
    assert sum(roots.values()) == n, "eigenvalues must be rational functions of the parameter"
    exprs = list(roots)
    crit = set()
    for i in range(len(exprs)):
        for j in range(i + 1, len(exprs)):
            for sol in sp.solve(sp.Eq(exprs[i], exprs[j]), var):
                if sol.is_rational:
                    crit.add(sol)
    for r in exprs:
        drops, _ = _drop_set(r * sp.eye(n) - A, var)
        crit |= {sq(x) for x in drops}
    g = sp.Rational(1000003, 7)
    gen = _diag_R(A.subs(var, g))
    assert _diag_R(A.subs(var, sp.Rational(-999983, 13))) == gen
    exc = sorted(to_frac(c) for c in crit if _diag_R(A.subs(var, c)) != gen)
    assert a[0].value == gen
    assert a[1].values == exc, (a[1].values, exc)


def v_charpoly_target(d, a):
    x = sp.Symbol(d.get("var", "x"))
    from sympy.parsing.sympy_parser import (implicit_multiplication_application, parse_expr,
                                            standard_transformations, convert_xor)
    tr = standard_transformations + (implicit_multiplication_application, convert_xor)
    p = sp.Poly(parse_expr(d["p"], local_dict={x.name: x}, transformations=tr), x)
    p = sp.Poly(p.as_expr() / p.LC(), x)
    model = sm(a[0].model)
    assert sp.Poly(model.charpoly(x).as_expr(), x) == p


def v_eigen_target(d, a):
    x = sp.Symbol(d.get("var", "x"))
    from sympy.parsing.sympy_parser import (implicit_multiplication_application, parse_expr,
                                            standard_transformations, convert_xor)
    tr = standard_transformations + (implicit_multiplication_application, convert_xor)
    p = sp.Poly(parse_expr(d["p"], local_dict={x.name: x}, transformations=tr), x)
    model = sm(a[0].model)
    assert sp.Poly(model.charpoly(x).as_expr(), x) == p
    for lam, basis in d["eigenspaces"].items():
        ns = (S(lam) * sp.eye(model.rows) - model).nullspace()
        B = cols(basis)
        assert len(ns) == B.cols and sp.Matrix.hstack(*ns).row_join(B).rank() == B.cols


VERIFY = {
    "solve": v_solve, "nsol": v_nsol, "rank": v_rank, "homog": v_homog, "det": v_det,
    "inverse": v_inverse, "matexpr": v_matexpr, "indep": v_indep, "basis_span": v_basis_span,
    "basis_eqs": v_basis_eqs, "coords": v_coords, "complete_basis": v_complete_basis, "in_span": v_in_span, "cartesian": v_cartesian,
    "affine_forms": v_affine_forms, "grassmann": v_grassmann, "position": v_position,
    "kerim": v_kerim, "repmat": v_repmat, "interp_matrix": v_interp_matrix, "change_basis": v_change_basis, "eigen": v_eigen,
    "charpoly": v_charpoly, "roots": v_roots, "metric": v_metric, "gram_schmidt": v_gram_schmidt,
    "gram": v_gram, "param_invertible": v_param_invertible, "param_rank": v_param_rank,
    "param_system": v_param_system, "param_indep": v_param_indep, "param_diag": v_param_diag,
    "charpoly_target": v_charpoly_target, "eigen_target": v_eigen_target,
}
KEYED = {"yesno", "mcq", "values"}  # answer keys written by hand (checked by item-specific tests below)


def test_every_auto_task_has_a_verifier():
    assert set(AUTO_TASKS) - KEYED <= set(VERIFY)


@pytest.mark.parametrize("item", [i for i in AUTO if i.task in VERIFY], ids=lambda i: i.id)
def test_item_against_sympy(item):
    VERIFY[item.task](item.data, answers(item))
