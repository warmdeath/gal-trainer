"""Capitolo 3 — Applicazioni lineari."""
from __future__ import annotations

import random
from fractions import Fraction

from ..core.fmt import flin, fm, fs, fv, fvs, hcat, indent, named, named_many, var_names
from ..core.gauss import (column_space_basis, inverse, kernel_basis, rank, row_reduce, solve,
                          span_rank)
from ..core.matrix import Matrix, vadd, vec, vscale
from ..core.poly import Poly
from ..exercise import (IntAns, MatrixAns, PolyAns, SpanAns, VectorAns, YesNoAns, mk)
from .base import Gen, Topic
from .rand import imat, invertible, ivec, nz, rank_r, unimodular

XY = ["x", "y"]
XYZ = ["x", "y", "z"]

# ================================================================== linear maps, ker, Im
MAPS_LESSON = [
    ("Intuition", """\
Linear maps are the functions that respect the structure: they send sums to sums and
multiples to multiples — so they send lines through 0 to lines (or to 0), planes to planes...
Every matrix A gives one (x ↦ Ax) and, after choosing bases, every linear map is a matrix."""),
    ("Definitions", """\
• f: V → W is an applicazione lineare if L1) f(v1 + v2) = f(v1) + f(v2) (additività) and
  L2) f(λv) = λf(v) (omogeneità) — equivalently f(λv1 + μv2) = λf(v1) + μf(v2).
  Examples: L_A(v) = Av, the transpose, the derivative on K[t]≤d, the trace.
  Not linear: det (n > 1), (x,y) ↦ x² + y², e^x. Always f(0_V) = 0_W (necessary, not sufficient!).
• Compositions and inverses of linear maps are linear; L_A ∘ L_B = L_{AB}.
• Nucleo (kernel) ker f = {v : f(v) = 0} ◁ V;  immagine Im f = f(V) ◁ W;  rango rg(f) = dim Im f.
  For L_A: ker L_A = Sol(A|0) = ker A, Im L_A = Col(A), rg(L_A) = rg(A).
• Hom(V,W) = linear maps V → W (a vector space); End(V) = Hom(V,V), endomorfismi."""),
    ("Injective, surjective, bijective (§3.2–3.3)", """\
• If V = Span(v1..vn) then Im f = Span(f(v1),...,f(vn)).
• Suriettiva ⇔ rg(f) = dim W.     Iniettiva ⇔ ker f = {0}  (Teoremi 3.11, 3.13).
  Injective maps preserve independence; surjective ones preserve generators; isomorfismi
  (bijective) preserve bases, so dim V = dim W.
• Teorema nullità più rango (3.17):  dim ker f + dim Im f = dim V.
  ⇒ dim V < dim W: never surjective; dim V > dim W: never injective;
    dim V = dim W: injective ⇔ surjective ⇔ bijective (Prop. 3.18).
  For A ∈ M_K(m,n): L_A injective ⇔ rg(A) = n; surjective ⇔ rg(A) = m."""),
]


def _is_linear_case(rng):
    a, b, c, d = (nz(rng, -4, 4) for _ in range(4))
    k = nz(rng, -3, 3)
    L1, L2 = flin([a, b], XY), flin([c, d], XY)
    cases = [
        (f"f(x, y) = ({L1}, {L2})", True, "Yes: each component is a homogeneous linear expression, f = L_A."),
        (f"f(x, y) = ({L1} {'+' if k > 0 else '-'} {abs(k)}, {L2})", False,
         f"No: f(0,0) = ({fs(k)}, 0) ≠ 0 (Lemma 3.3)."),
        (f"f(x, y) = ({L1}, xy)", False, "No: f(2·(1,1)) has second component 4, but 2·f(1,1) has 2 (L2 fails)."),
        (f"f(x, y) = (x² + y², {L2})", False, "No: f(2·(1,1)) = (8, ...) ≠ 2·f(1,1) = (4, ...)."),
        (f"f(x, y, z) = ({flin([a, 0, b], XYZ)}, {flin([0, c, d], XYZ)})", True, "Yes: f = L_A for a 2×3 matrix A."),
        ("f: M_R(2) → R, f(A) = tr(A)", True, "Yes: tr(A + B) = tr A + tr B and tr(λA) = λ tr A."),
        ("f: M_R(2) → R, f(A) = det(A)", False, "No: det(2A) = 4 det A ≠ 2 det A."),
        ("f: R[t]≤2 → R[t]≤1, f(p) = p'(t)", True, "Yes: the derivative is linear."),
        ("f: R[t]≤2 → R, f(p) = p(0)·p(1)", False, "No: it is not homogeneous: f(2p) = 4f(p)."),
        ("f: R[t]≤2 → R, f(p) = p(0) + p(1)", True, "Yes: evaluation is linear, sums of linear maps are linear."),
        (f"f(x, y) = ({L1}, |x|)", False, "No: f(-1·(1,0)) has second component 1, but -1·f(1,0) has -1."),
    ]
    return rng.choice(cases)


def ex_is_linear(rng, data=None):
    text, ok, why = data or _is_linear_case(rng)
    return mk("linear_maps", "is_linear", f"{text}\nIs f linear?", YesNoAns(ok, why), [why], text=text, ok=ok)


def ex_ker_im(rng, data=None):
    if data:
        A = data
    else:
        m, n = rng.choice([(2, 3), (3, 3), (3, 2), (3, 4), (4, 3)])
        A = rank_r(rng, m, n, rng.randint(max(1, min(m, n) - 1), min(m, n)))
    K = kernel_basis(A)
    C = column_space_basis(A)
    r = len(C)
    prompt = (named("A", A) + f"\n\nL_A: R^{A.n} → R^{A.m}, v ↦ Av. Find a basis of ker L_A and dim Im L_A;"
              " is L_A injective? surjective?")
    sol = solve(A, [0] * A.m)
    steps = ["ker L_A = Sol(A|0):"] + sol.steps + [
        f"ker L_A = Span({fvs(K)})" if K else "ker L_A = {0}",
        f"Im L_A = Col(A), dim Im = rg(A) = {r}; a basis: {fvs(C)}.",
        f"Check (nullità + rango): {len(K)} + {r} = {A.n} = dim R^{A.n}.",
        f"Injective ⇔ ker = {{0}}: {'yes' if not K else 'no'}.  Surjective ⇔ rg = {A.m}: {'yes' if r == A.m else 'no'}."]
    return mk("linear_maps", "ker_im", prompt,
              [("basis of ker L_A ('none' if {0})", SpanAns(K, A.n, "ker L_A")),
               ("dim Im L_A", IntAns(r, mistakes=[(len(K), "That's dim ker. dim Im = n - dim ker = rg(A).")])),
               ("injective?", YesNoAns(not K, "Injective ⇔ ker L_A = {0}.")),
               ("surjective?", YesNoAns(r == A.m, f"Surjective ⇔ rg(A) = dim R^{A.m} = {A.m}."))],
              steps, A=A)


def ex_nullity_rank(rng, data=None):
    kind = rng.choice(["im", "ker", "inj"])
    n, m = rng.randint(2, 6), rng.randint(2, 6)
    spaces = {2: "R²", 3: "R³", 4: "R[t]≤3", 5: "R⁵", 6: "M_R(2,3)"}
    V, W = spaces[n], spaces[m]
    if kind == "im":
        k = rng.randint(max(0, n - m), n)
        prompt = f"f: {V} → {W} is linear with dim ker f = {k}. Compute dim Im f."
        val = n - k
        steps = [f"dim {V} = {n}. Nullità più rango: dim Im f = dim V - dim ker f = {n} - {k} = {val}."]
        return mk("linear_maps", "nullity_rank", prompt, IntAns(val, mistakes=[(m - k, "Use dim V (the DOMAIN), not dim W.")]), steps, n=n, m=m, k=k, case=kind)
    if kind == "ker":
        r = rng.randint(0, min(n, m))
        prompt = f"f: {V} → {W} is linear with rg(f) = {r}. Compute dim ker f."
        val = n - r
        steps = [f"dim {V} = {n}. dim ker f = dim V - rg(f) = {n} - {r} = {val}."]
        return mk("linear_maps", "nullity_rank", prompt, IntAns(val), steps, n=n, m=m, r=r, case=kind)
    prompt = f"Can a linear map f: {V} → {W} be injective?"
    val = n <= m
    steps = [f"dim {V} = {n}, dim {W} = {m}.",
             (f"If injective then dim V = dim Im f ≤ dim W: here {n} > {m}, impossible." if not val else
              f"{n} ≤ {m}: yes, e.g. a map sending a basis of V to independent vectors of W.")]
    return mk("linear_maps", "nullity_rank", prompt, YesNoAns(val, steps[-1]), steps, n=n, m=m, case=kind)


def _poly_map(rng):
    """A linear map R[t]≤2 → R[t]≤k described by a formula, with its matrix (canonical bases)."""
    cases = [
        ("f(p) = t·p''(t) - p'(t)", 1, lambda p: _shift(_der(_der(p))) - _der(p)),
        ("f(p) = p'(t) + p(0)", 1, lambda p: _der(p) + Poly([p(0)])),
        ("f(p) = p(t + 1) - p(t)", 1, lambda p: _subst_shift(p, 1) - p),
        ("f(p) = (t + 1)·p'(t)", 2, lambda p: Poly([1, 1]) * _der(p)),
        ("f(p) = p'(t) - p''(t)", 1, lambda p: _der(p) - _der(_der(p))),
        ("f(p) = p(1) + p(-1)·t", 1, lambda p: Poly([p(1), p(-1)])),
        ("f(p) = p(1)·t² + p(0)", 2, lambda p: Poly([p(0), 0, p(1)])),
    ]
    return rng.choice(cases)


def _der(p: Poly) -> Poly:
    return Poly([k * p.c[k] for k in range(1, len(p.c))])


def _shift(p: Poly) -> Poly:
    return Poly([0] + list(p.c))


def _subst_shift(p: Poly, a) -> Poly:
    out = Poly()
    for k, c in enumerate(p.c):
        out = out + Poly([c]) * Poly([a, 1]) ** k
    return out


def poly_coords(p: Poly, n: int) -> list[Fraction]:
    return [p.c[k] if k < len(p.c) else Fraction(0) for k in range(n)]


def ex_poly_map(rng, data=None):
    text, k, f = data or _poly_map(rng)
    basis = [Poly([1]), Poly([0, 1]), Poly([0, 0, 1])]
    imgs = [f(b) for b in basis]
    A = Matrix.from_cols([poly_coords(q, k + 1) for q in imgs])
    K = kernel_basis(A)
    r = rank(A)
    kerpolys = [Poly(v) for v in K]
    prompt = (f"f: R[t]≤2 → R[t]≤{k},  {text}  (f is linear).\n"
              "Find dim ker f and dim Im f. Is f surjective?")
    steps = ["Apply f to the canonical basis {1, t, t²}:"] + [
        f"  f({b}) = {q}" for b, q in zip(basis, imgs)] + [
        f"Coordinates as columns (Prop. 3.10, Im f = Span(f(1), f(t), f(t²))):", indent(fm(A)),
        f"rg = {r} ⇒ dim Im f = {r}; dim ker f = 3 - {r} = {3 - r}" +
        (f", ker f = Span({', '.join(str(q) for q in kerpolys)})." if kerpolys else "."),
        f"dim R[t]≤{k} = {k + 1} ⇒ surjective: {'yes' if r == k + 1 else 'no'}."]
    return mk("linear_maps", "poly_map", prompt,
              [("dim ker f", IntAns(3 - r)), ("dim Im f", IntAns(r)),
               ("surjective?", YesNoAns(r == k + 1, f"Surjective ⇔ rg(f) = dim R[t]≤{k} = {k + 1}."))],
              steps, A=A, text=text, k=k)


LINEAR_MAPS = Topic(
    "linear_maps", "§3.1–3.3", "Linear maps, kernel and image, nullity + rank",
    "Applicazioni lineari, nucleo e immagine, teorema nullità più rango", "3 Applicazioni lineari",
    ["bases", "cart_param"], MAPS_LESSON,
    [lambda: ex_ker_im(None, Matrix([[1, 2, 3], [4, 5, 6]])),
     lambda: ex_poly_map(None, ("f(p) = t·p''(t) - p'(t)", 1, lambda p: _shift(_der(_der(p))) - _der(p)))],
    [Gen("is_linear", "is f linear?", ex_is_linear),
     Gen("ker_im", "ker, Im, injective, surjective of L_A", ex_ker_im, exam=True, weight=3),
     Gen("nullity_rank", "nullity + rank", ex_nullity_rank),
     Gen("poly_map", "maps on polynomials", ex_poly_map, exam=True)],
)

# ================================================================== representative matrix
REP_LESSON = [
    ("Intuition", """\
A linear map is completely decided by what it does on a basis. Write those images in
coordinates as columns and you get a matrix that "is" the map — a dictionary that turns
abstract questions (polynomials, matrices...) into matrix computations."""),
    ("Definitions and theorems (§3.4)", """\
• Teorema di interpolazione (3.20): given a basis B = {v1..vn} of V and any w1..wn ∈ W,
  there is exactly one linear f with f(vi) = wi:  f(Σλi vi) = Σλi wi.
• Every linear f: K^n → K^m is L_A, with columns A = (f(e1)|...|f(en)) (Lemma 3.21).
• dim V = n ⇒ V ≅ K^n via the coordinate map (mappa delle coordinate) X_B: v ↦ (v)_B; its
  inverse P_B is the mappa di parametrizzazione. V ≅ W ⇔ dim V = dim W.
• Matrice rappresentativa (Def. 3.25) of f w.r.t. B (of V) and C (of W): M^B_C(f) ∈ M_K(m,n),
  j-th column = (f(bj))_C.  Then M^B_C(f)·(v)_B = (f(v))_C,  rg(f) = rg(M^B_C(f)).
  (In this trainer: superscript = basis of the domain, subscript = basis of the codomain.)
• M^B_D(g∘f) = M^C_D(g)·M^B_C(f);   M^B_C(f)^-1 = M^C_B(f^-1);
  M^{B'}_{C'}(f) = M^C_{C'}(id)·M^B_C(f)·M^{B'}_B(id)."""),
    ("Change of basis (§3.4.2)", """\
• Matrice di cambio di base M^B_{B'}(id): its columns are the coordinates of the vectors of B
  w.r.t. B';  (v)_{B'} = M^B_{B'}(id)·(v)_B  and  M^{B'}_B(id) = M^B_{B'}(id)^-1.
  In K^n: M^B_E(id) = (b1|...|bn) — just the vectors of B as columns — so M^E_B(id) = (b1|...|bn)^-1.
• Endomorphisms: M^{B'}_{B'}(f) = M^B_{B'}(id)·M^B_B(f)·M^{B'}_B(id) = P^-1 A P with P = M^{B'}_B(id).
• A, B similar (simili, coniugate) if B = P^-1 A P: same rank, det, trace, char. polynomial."""),
]


def ex_canonical_matrix(rng, data=None):
    m, n = rng.choice([(2, 2), (2, 3), (3, 3), (3, 2)])
    A = data or imat(rng, m, n, -3, 3)
    names = var_names(n)
    comps = ", ".join(flin(A.row(i), names) for i in range(A.m))
    prompt = f"f: R^{n} → R^{m},  f({', '.join(names)}) = ({comps}).\nWrite the matrix A with f = L_A (i.e. M^E_E(f))."
    steps = ["Columns are f(e1), ..., f(en) (Lemma 3.21):"] + [
        f"  f(e{j + 1}) = {fv(A.col(j))}" for j in range(n)] + [fm(A)]
    return mk("rep_matrix", "canonical", prompt,
              MatrixAns(A, mistakes=[(A.T, "You wrote the images f(ej) as rows: they are the COLUMNS.")] if A.m == A.n else []),
              steps, A=A)


def ex_interpolation(rng, data=None):
    n = rng.choice([2, 3])
    m = rng.choice([2, 3])
    B = [list(c) for c in unimodular(rng, n).cols()]
    Ws = [ivec(rng, m, -3, 3, nonzero=False) for _ in range(n)]
    lam = ivec(rng, n, -2, 2)
    v = [sum(l * b[i] for l, b in zip(lam, B)) for i in range(n)]
    res = [sum(l * w[i] for l, w in zip(lam, Ws)) for i in range(m)]
    prompt = ("f: R^%d → R^%d is the linear map with\n" % (n, m) +
              "\n".join(f"   f{fv(b)} = {fv(w)}" for b, w in zip(B, Ws)) + f"\nCompute f{fv(v)}.")
    sol = solve(Matrix.from_cols(B), v, [f"λ{i + 1}" for i in range(n)])
    steps = ["Write v in the basis: v = λ1b1 + ... (solve (b1|...|bn)λ = v):"] + sol.steps + [
        "By linearity f(v) = λ1 f(b1) + ... + λn f(bn) = " +
        " + ".join(f"({fs(l)}){fv(w)}" for l, w in zip(sol.particular, Ws)) + f" = {fv(res)}"]
    return mk("rep_matrix", "interpolation", prompt, VectorAns(res), steps, B=B, Ws=Ws, v=v)


def ex_rep_matrix(rng, data=None):
    if data:
        A, B, C = data
    else:
        n, m = rng.choice([(2, 2), (3, 2), (2, 3)])
        A = imat(rng, m, n, -2, 2)
        B = [list(c) for c in unimodular(rng, n).cols()]
        C = [list(c) for c in unimodular(rng, m).cols()]
    Bm, Cm = Matrix.from_cols(B), Matrix.from_cols(C)
    M = inverse(Cm) * A * Bm
    prompt = (named("A", A) + f"\n\nf = L_A: R^{A.n} → R^{A.m}. Bases B = {{{', '.join(fv(b) for b in B)}}} of R^{A.n} and "
              f"C = {{{', '.join(fv(c) for c in C)}}} of R^{A.m}.\nCompute M^B_C(f).")
    steps = ["The j-th column of M^B_C(f) is (f(bj))_C."]
    for j, b in enumerate(B):
        fb = A.apply(b)
        coords = M.col(j)
        steps.append(f"  f(b{j + 1}) = A{fv(b)} = {fv(fb)} = " +
                     " + ".join(f"({fs(c)})c{i + 1}" for i, c in enumerate(coords)) + f"  →  column {fv(coords)}")
    steps.append(hcat("M^B_C(f) = ", fm(M), "   (= C^-1 · A · B with the bases as column matrices)"))
    return mk("rep_matrix", "rep", prompt,
              MatrixAns(M, mistakes=[(A * Bm, "These are the f(bj) in CANONICAL coordinates: express them in the basis C."),
                                     (Cm * A * Bm if Cm.n == A.m else A, "Use C^-1 (coordinates w.r.t. C), not C.")]),
              steps, A=A, B=B, C=C)


def ex_change_basis(rng, data=None):
    if data:
        Bp, v = data
    else:
        n = rng.choice([2, 3])
        Bp = [list(c) for c in unimodular(rng, n).cols()]
        v = ivec(rng, n, -4, 4)
    P = Matrix.from_cols(Bp)
    Minv = inverse(P)
    coords = Minv.apply(v)
    n = len(v)
    prompt = (f"In R^{n} let E be the canonical basis and B' = {{{', '.join(fv(b) for b in Bp)}}}.\n"
              "(1) Compute the change of basis matrix M^E_{B'}(id).  (2) Compute (v)_{B'} for v = " + fv(v) + ".")
    steps = [hcat("M^{B'}_E(id) = (vectors of B' as columns) = ", fm(P)),
             "M^E_{B'}(id) = M^{B'}_E(id)^-1:"] + inverse_steps_short(P) + [
        f"(v)_B' = M^E_B'(id)·v = {fv(coords)}"]
    return mk("rep_matrix", "change_basis", prompt,
              [("(1) M^E_B'(id)", MatrixAns(Minv, mistakes=[(P, "That's M^{B'}_E(id) (B' in canonical coordinates); you need its inverse.")])),
               ("(2) (v)_B'", VectorAns(coords))], steps, Bp=Bp, v=v)


def inverse_steps_short(P):
    from ..core.gauss import inverse_steps
    inv, steps = inverse_steps(P)
    return steps[-3:] if len(steps) > 3 else steps


def ex_endo_change(rng, data=None):
    n = 2
    A = data or imat(rng, 2, 2, -3, 3)
    P = unimodular(rng, 2, bound=3)
    D = inverse(P) * A * P
    B = [list(c) for c in P.cols()]
    prompt = (named("A", A) + f"\n\nf = L_A: R² → R². Compute M^B_B(f) for B = {{{fv(B[0])}, {fv(B[1])}}}.")
    steps = [hcat("P = M^B_E(id) = ", fm(P)), hcat("P^-1 = ", fm(inverse(P))),
             hcat("M^B_B(f) = P^-1 A P = ", fm(D)),
             f"Check: same trace ({fs(A.trace())}) and same det as A (similar matrices)."]
    return mk("rep_matrix", "endo_change", prompt,
              MatrixAns(D, mistakes=[(P * A * inverse(P), "The formula is P^-1 A P (P has the new basis as columns), not P A P^-1.")]),
              steps, A=A, P=P)


REP_MATRIX = Topic(
    "rep_matrix", "§3.4", "Representative matrix and change of basis",
    "Matrice rappresentativa e matrice di cambio di base", "3 Applicazioni lineari",
    ["linear_maps", "inverse"], REP_LESSON,
    [lambda: ex_change_basis(None, ([[1, -2], [2, -3]], [4, -7]))],
    [Gen("canonical", "matrix of f in the canonical bases", ex_canonical_matrix),
     Gen("interpolation", "f defined on a basis", ex_interpolation, exam=True),
     Gen("rep", "M^B_C(f)", ex_rep_matrix, exam=True, weight=2),
     Gen("change_basis", "change of basis", ex_change_basis, weight=2),
     Gen("endo_change", "P^-1 A P", ex_endo_change)],
)

TOPICS = [LINEAR_MAPS, REP_MATRIX]
