"""Capitolo 5 — Spazi euclidei e forme quadratiche."""
from __future__ import annotations

import random
from fractions import Fraction

from ..core.fmt import flin, fm, fparam, fs, fv, fvs, hcat, indent, named, named_many
from ..core.gauss import det, inverse, kernel_basis, rank, span_rank
from ..core.geom import (cartesian_from_param, cross, gram_inner, gram_schmidt, norm,
                         orth_complement, projection, projection_matrix, ratio)
from ..core.matrix import Matrix, clear_denominators, dot, vadd, vec, vscale, vsub
from ..core.surd import Surd
from ..exercise import (ChoiceAns, MatrixAns, RealAns, RealVectorAns, ScalarAns,
                        ScaledVectorsAns, SpanAns, VectorAns, YesNoAns, mk)
from .base import Gen, Topic
from .rand import imat, ivec, nz, unimodular

XY = ["x", "y"]
XYZ = ["x", "y", "z"]


# ================================================================== bilinear forms
BIL_LESSON = [
    ("Intuition", """\
A bilinear form eats two vectors and returns a number, linearly in each slot — like the dot
product, but with "weights". Once a basis is fixed it is just a matrix: f(v,w) = (v)^T A (w)."""),
    ("Definitions (§5.1)", """\
• Forma bilineare f: V × V → K, linear in each argument (B1–B4). Bil(V) is a vector space.
  f(0, w) = f(v, 0) = 0.  Every A ∈ M_K(n) gives f(v,w) = v^T A w on K^n.
• Matrice di Gram w.r.t. B = {b1..bn}: M_B(f) = (f(bi, bj))_{i,j};  f(v,w) = (v)_B^T M_B(f) (w)_B.
  Change of basis: M_C(f) = P^T M_B(f) P with P = M^C_B(id). A and P^T A P are congruenti.
• f is simmetrica if f(v,w) = f(w,v) ∀v,w ⇔ its Gram matrix is symmetric (in any basis).
Reading a form off its formula on R²: f((x1,y1),(x2,y2)) = a x1x2 + b x1y2 + c y1x2 + d y1y2
has Gram matrix (a b; c d): the coefficient of x_i y_j goes in row i, column j."""),
]


def _bil_formula(A: Matrix) -> str:
    n = A.n
    first = ["x1", "y1", "z1"][:n]
    second = ["x2", "y2", "z2"][:n]
    terms, names = [], []
    for i in range(n):
        for j in range(n):
            terms.append(A[i, j])
            names.append(f"{first[i]}{second[j]}")
    return flin(terms, names)


def ex_gram(rng, data=None):
    A = data or imat(rng, *(2 * [rng.choice([2, 2, 3])]), -3, 3)
    n = A.n
    v1 = "(x1, y1)" if n == 2 else "(x1, y1, z1)"
    v2 = "(x2, y2)" if n == 2 else "(x2, y2, z2)"
    prompt = (f"f: R^{n} × R^{n} → R,  f({v1}, {v2}) = {_bil_formula(A)}.\n"
              "Write the Gram matrix of f w.r.t. the canonical basis, and say whether f is symmetric.")
    steps = ["a_ij = f(e_i, e_j) = coefficient of (i-th coordinate of v)·(j-th coordinate of w):", fm(A),
             "Symmetric ⇔ A = A^T: " + ("yes." if A.is_symmetric() else "no.")]
    return mk("bilinear", "gram", prompt,
              [("Gram matrix", MatrixAns(A, mistakes=[(A.T, "Transposed: row i = first vector's coordinate, column j = second's.")])),
               ("symmetric?", YesNoAns(A.is_symmetric(), steps[-1]))], steps, A=A)


def ex_eval_form(rng, data=None):
    n = rng.choice([2, 3])
    A = imat(rng, n, n, -3, 3)
    v, w = ivec(rng, n, -3, 3), ivec(rng, n, -3, 3)
    val = dot(v, A.apply(w))
    prompt = named("A", A) + f"\n\nf(v, w) = v^T A w. Compute f(v, w) for v = {fv(v)}, w = {fv(w)}."
    steps = [f"A w = {fv(A.apply(w))}", f"v^T (A w) = {fs(val)}"]
    wrong = dot(w, A.apply(v))
    return mk("bilinear", "eval", prompt,
              ScalarAns(val, mistakes=[(wrong, "You computed f(w, v) = w^T A v; A is not symmetric so the order matters.")] if wrong != val else []),
              steps, A=A, v=v, w=w)


def ex_congruent(rng, data=None):
    S = data or _sym(rng, 2)
    P = unimodular(rng, 2, bound=2)
    B = [list(c) for c in P.cols()]
    R = P.T * S * P
    prompt = (named("A", S) + f"\n\nA is the Gram matrix of a symmetric bilinear form f w.r.t. the canonical basis of R².\n"
              f"Compute the Gram matrix of f w.r.t. B = {{{fv(B[0])}, {fv(B[1])}}}.")
    steps = [hcat("P = M^B_E(id) = ", fm(P)), "M_B(f) = P^T A P (entries f(bi, bj)):", fm(R)]
    return mk("bilinear", "congruent", prompt,
              MatrixAns(R, mistakes=[(inverse(P) * S * P, "For bilinear forms it is P^T A P (congruence), not P^-1 A P (similarity).")]),
              steps, S=S, P=P)


def _sym(rng, n, lo=-3, hi=3) -> Matrix:
    rows = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            rows[i][j] = rows[j][i] = rng.randint(lo, hi)
    return Matrix(rows)


BILINEAR = Topic(
    "bilinear", "§5.1", "Bilinear forms and Gram matrices", "Forme bilineari, matrice di Gram",
    "5 Spazi euclidei", ["rep_matrix"], BIL_LESSON,
    [lambda: ex_gram(None, Matrix([[1, -1], [0, 1]]))],
    [Gen("gram", "Gram matrix from a formula", ex_gram, weight=2),
     Gen("eval", "evaluate v^T A w", ex_eval_form),
     Gen("congruent", "change of basis P^T A P", ex_congruent)],
)

# ================================================================== scalar products
SP_LESSON = [
    ("Intuition", """\
A scalar product is a symmetric bilinear form that is positive on every non-zero vector:
exactly what you need to define lengths and angles. The standard one is the dot product."""),
    ("Definitions (§5.2)", """\
From now on K = R.
• Prodotto scalare ⟨·,·⟩: symmetric bilinear form, definita positiva: ⟨v,v⟩ ≥ 0 and
  ⟨v,v⟩ = 0 ⇔ v = 0. (V, ⟨·,·⟩) is a spazio euclideo. Standard product on R^n:
  ⟨v,w⟩ = v1w1 + ... + vnwn (Gram matrix I_n). Others: ⟨A,B⟩ = tr(A^T B) on M_R(m,n).
• Norma ||v|| = √⟨v,v⟩; versore (unit vector) if ||v|| = 1; v/||v|| is a unit vector.
  ||λv|| = |λ|·||v||;  polarizzazione ⟨v,w⟩ = (||v+w||² - ||v||² - ||w||²)/2.
• Cauchy-Schwarz |⟨v,w⟩| ≤ ||v||·||w|| (equality ⇔ dependent); triangolare ||v+w|| ≤ ||v|| + ||w||.
• Distanza d(v,w) = ||v - w||.  Angolo θ ∈ [0,π] with cos θ = ⟨v,w⟩/(||v||·||w||).
• Carnot: ||v+w||² = ||v||² + ||w||² + 2||v||·||w|| cos θ; Pitagora if ⟨v,w⟩ = 0."""),
    ("Is a symmetric matrix a scalar product? Complete the squares", """\
Example of the notes: (1 1 0; 1 2 0; 0 0 3) gives v1² + 2v1v2 + 2v2² + 3v3² = (v1+v2)² + v2² + 3v3²,
a sum of squares with positive coefficients that vanishes only at 0 ⇒ positive definite.
If some coefficient is ≤ 0 (or a diagonal entry a_ii ≤ 0: f(e_i,e_i) = a_ii) it is NOT."""),
]


def complete_squares(S: Matrix):
    """Symmetric elimination without swaps: f(v,v) = Σ d_k (v_k + ...)². Returns (d list, steps).
    Stops at the first d_k ≤ 0 (then f is not positive definite)."""
    n = S.n
    A = S.tolist()
    names = ["v1", "v2", "v3", "v4"][:n]
    ds, steps = [], []
    for k in range(n):
        d = A[k][k]
        ds.append(d)
        if d <= 0:
            steps.append(f"After completing the first {k} square(s) the coefficient of {names[k]}² is {fs(d)} ≤ 0.")
            return ds, steps
        lin = [Fraction(0)] * k + [Fraction(1)] + [A[k][j] / d for j in range(k + 1, n)]
        steps.append(f"{fs(d)}·({flin(lin, names)})²")
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] -= A[i][k] * A[k][j] / d
    return ds, steps


def ex_pos_def(rng, data=None):
    if data:
        S = data
    elif rng.random() < 0.5:
        n = rng.choice([2, 3])
        while True:
            L = unimodular(rng, n, steps=n, bound=2)
            S = L.T * Matrix.diag([rng.randint(1, 3) for _ in range(n)]) * L
            if all(abs(x) <= 6 for r in S.rows for x in r):
                break
    else:
        S = _sym(rng, rng.choice([2, 3]), -3, 3)
    ds, sq = complete_squares(S)
    ok = len(ds) == S.n and all(d > 0 for d in ds)
    q = _quadratic_text(S)
    prompt = named("S", S) + "\n\nIs ⟨v, w⟩ = v^T S w a scalar product on R^" + f"{S.n} (i.e. is S positive definite)?"
    steps = [f"f(v,v) = {q}"]
    if ok:
        steps.append("= " + " + ".join(sq) + "  — all coefficients > 0 ⇒ positive definite: a scalar product.")
    else:
        steps += sq + ["⇒ NOT positive definite (some non-zero v has f(v,v) ≤ 0): not a scalar product."]
    return mk("scalar_products", "posdef", prompt, YesNoAns(ok, steps[-1]), steps, S=S)


def _quadratic_text(S: Matrix) -> str:
    n = S.n
    names = ["v1", "v2", "v3", "v4"][:n]
    coeffs, mons = [], []
    for i in range(n):
        for j in range(i, n):
            coeffs.append(S[i, j] if i == j else 2 * S[i, j])
            mons.append(f"{names[i]}²" if i == j else f"{names[i]}{names[j]}")
    return flin(coeffs, mons)


def _maybe_gram(rng):
    if rng.random() < 0.6:
        return None
    L = unimodular(rng, 2, bound=2)
    return L.T * Matrix.diag([1, rng.randint(1, 2)]) * L


def ex_norm(rng, data=None):
    if data:
        S, v = None, data
    else:
        S = _maybe_gram(rng)
        v = ivec(rng, 2 if S is not None else rng.choice([2, 3, 4]), -4, 4)
    n = len(v)
    ip = gram_inner(S)
    nv = ip(v, v)
    val = Surd.sqrt(nv)
    where = "with the standard scalar product" if S is None else "with ⟨v,w⟩ = v^T S w,\n" + named("S", S)
    prompt = f"In R^{n} {where}\ncompute ||v|| for v = {fv(v)}."
    steps = [f"||v||² = ⟨v,v⟩ = {fs(nv)}" + ("" if S is not None else " = " + " + ".join(f"({fs(x)})²" for x in v)),
             f"||v|| = √{fs(nv)} = {val.pretty()}"]
    mistakes = [(Surd(nv), "That is ||v||²; take the square root.")]
    if S is not None:
        mistakes.append((Surd.sqrt(dot(v, v)), "You used the standard product; this space has Gram matrix S."))
    return mk("scalar_products", "norm", prompt, RealAns(val, mistakes=mistakes), steps, v=v, S=S)


def ex_distance(rng, data=None):
    n = rng.choice([2, 3, 4])
    v, w = ivec(rng, n, -4, 4), ivec(rng, n, -4, 4)
    d = vsub(v, w)
    val = Surd.sqrt(dot(d, d))
    prompt = f"In R^{n} (standard scalar product) compute d(v, w) for v = {fv(v)}, w = {fv(w)}."
    steps = [f"v - w = {fv(d)}", f"d(v,w) = ||v - w|| = √{fs(dot(d, d))} = {val.pretty()}"]
    return mk("scalar_products", "distance", prompt, RealAns(val), steps, v=v, w=w)


def ex_angle(rng, data=None):
    if data:
        v, w = data
    else:
        n = 3
        while True:
            v, w = ivec(rng, n, -3, 3), ivec(rng, n, -3, 3)
            if rank(Matrix([v, w])) == 2 or rng.random() < 0.1:
                break
    n = len(v)
    c = Surd(dot(v, w)) / (norm(v) * norm(w))
    prompt = f"In R^{n} (standard product) compute cos θ, θ = angle between v = {fv(v)} and w = {fv(w)}."
    steps = [f"⟨v,w⟩ = {fs(dot(v, w))},  ||v|| = {norm(v).pretty()},  ||w|| = {norm(w).pretty()}",
             f"cos θ = ⟨v,w⟩ / (||v||·||w||) = {c.pretty()}"]
    return mk("scalar_products", "angle", prompt, RealAns(c), steps, v=v, w=w)


SCALAR_PRODUCTS = Topic(
    "scalar_products", "§5.2", "Scalar products, norm, distance, angle",
    "Prodotti scalari, norma, distanza, angolo", "5 Spazi euclidei", ["bilinear"], SP_LESSON,
    [lambda: ex_angle(None, ([1, 2, 3], [0, 1, -1])),
     lambda: ex_pos_def(None, Matrix([[1, 1, 0], [1, 2, 0], [0, 0, 3]]))],
    [Gen("posdef", "is it a scalar product?", ex_pos_def, exam=True),
     Gen("norm", "norm of a vector", ex_norm, weight=2),
     Gen("distance", "distance of two vectors", ex_distance),
     Gen("angle", "cosine of the angle", ex_angle)],
)

# ================================================================== orthogonality, Fourier
ORTH_LESSON = [
    ("Intuition", """\
Orthogonal = "at right angles": ⟨v,w⟩ = 0. With an orthogonal basis, coordinates are free —
no system to solve, just one scalar product per coordinate."""),
    ("Definitions (§5.3, §5.3.1)", """\
• v ⊥ w if ⟨v,w⟩ = 0. 0 is orthogonal to everything and is the only vector orthogonal to itself.
  U, W are orthogonal subspaces if ⟨u,w⟩ = 0 ∀u, w; affine subspaces are orthogonal if their
  giaciture are. In R³ the vector (α,β,γ) is normal to the plane αx + βy + γz = δ.
• Base ortogonale: ⟨bi,bj⟩ = 0 for i ≠ j; base ortonormale: in addition ||bi|| = 1.
  Non-zero pairwise orthogonal vectors are linearly independent.
• Teorema 5.18 (coefficienti di Fourier): if {b1..bp} is an orthogonal basis of W, every v ∈ W is
     v = (⟨v,b1⟩/||b1||²) b1 + ... + (⟨v,bp⟩/||bp||²) bp;   orthonormal: v = Σ⟨v,bj⟩ bj.
  With an orthonormal basis the Gram matrix is I_n: ⟨v,w⟩ = (v)_B · (w)_B."""),
]


def _orth_basis(rng, n):
    """Orthogonal integer basis of R^n (Gram-Schmidt of a unimodular basis, denominators cleared)."""
    while True:
        cs, _ = gram_schmidt([list(c) for c in unimodular(rng, n, steps=n, bound=2).cols()])
        cs = [clear_denominators(c) for c in cs]
        if all(abs(x) <= 5 for c in cs for x in c):
            return cs


def ex_fourier(rng, data=None):
    if data:
        B, v = data
    else:
        n = rng.choice([2, 3])
        B = _orth_basis(rng, n)
        v = ivec(rng, n, -4, 4)
    coeffs = [Fraction(dot(v, b)) / dot(b, b) for b in B]
    prompt = ("B = {" + ", ".join(fv(b) for b in B) + "} is an orthogonal basis of R^" + f"{len(v)}"
              f" (standard product).\nFind the coordinates (v)_B of v = {fv(v)} using the Fourier coefficients.")
    steps = [f"λ{j + 1} = ⟨v,b{j + 1}⟩/||b{j + 1}||² = {fs(dot(v, b))}/{fs(dot(b, b))} = {fs(c)}" for j, (b, c) in enumerate(zip(B, coeffs))]
    wrong = [Fraction(dot(v, b)) for b in B]
    return mk("orthogonality", "fourier", prompt,
              VectorAns(coeffs, mistakes=[(wrong, "You forgot to divide by ||b_j||² (B is orthogonal, not orthonormal).")]),
              steps, B=B, v=v)


def ex_unit(rng, data=None):
    n = rng.choice([2, 3])
    v = data or ivec(rng, n, -3, 3)
    nv = norm(v)
    u = [Surd(x) / nv for x in v]
    prompt = f"Normalise v = {fv(v)} (standard product): give the unit vector v/||v||."
    steps = [f"||v|| = {nv.pretty()}", "v/||v|| = (" + ", ".join(x.pretty() for x in u) + ")"]
    return mk("orthogonality", "unit", prompt, RealVectorAns(u), steps, v=v)


def ex_orthogonal_to(rng, data=None):
    """W = Span(w) in R^3 (standard): W⊥ is a plane, find a basis."""
    w = data or ivec(rng, 3, -3, 3)
    K = orth_complement([w], 3)
    prompt = f"In R³ (standard product) find a basis of the vectors orthogonal to w = {fv(w)}."
    steps = [f"v ⊥ w ⇔ ⟨v,w⟩ = 0 ⇔ {flin(w, XYZ)} = 0: a plane.", f"Basis: {fvs(K)}"]
    return mk("orthogonality", "orth_to", prompt, SpanAns(K, 3, "w⊥"), steps, w=w)


ORTHOGONALITY = Topic(
    "orthogonality", "§5.3–5.3.1", "Orthogonality, orthogonal and orthonormal bases",
    "Ortogonalità, basi ortogonali e ortonormali", "5 Spazi euclidei", ["scalar_products"], ORTH_LESSON,
    [lambda: ex_fourier(None, ([[1, -2], [2, 1]], [4, 3]))],
    [Gen("fourier", "Fourier coefficients", ex_fourier, weight=2),
     Gen("unit", "normalise a vector", ex_unit),
     Gen("orth_to", "vectors orthogonal to w", ex_orthogonal_to)],
)

# ================================================================== Gram-Schmidt
GS_LESSON = [
    ("Intuition", """\
Take the vectors one at a time and remove from each its "shadow" (projection) on the
previous ones: what remains is perpendicular to all of them. The span of the first j vectors
never changes."""),
    ("Teorema 5.20 (Ortonormalizzazione di Gram-Schmidt)", """\
From a basis {b1..bp} of W:
   c1 = b1
   c2 = b2 - (⟨b2,c1⟩/||c1||²) c1
   c3 = b3 - (⟨b3,c1⟩/||c1||²) c1 - (⟨b3,c2⟩/||c2||²) c2
   cj = bj - Σ_{k<j} (⟨bj,ck⟩/||ck||²) ck
{c1..cp} is an orthogonal basis; dj = cj/||cj|| gives an orthonormal one, and
Span(b1..bj) = Span(c1..cj) = Span(d1..dj) for every j.
Tip: you may rescale each cj (e.g. clear denominators) before using it in the next step —
the coefficients ⟨bj,ck⟩/||ck||² · ck do not change. The trainer accepts rescaled cj."""),
]


def ex_gram_schmidt(rng, data=None):
    if data:
        B = data
    else:
        n = rng.choice([3, 3, 4])
        p = rng.choice([2, 3]) if n == 3 else 3
        while True:
            B = [ivec(rng, n, -2, 2) for _ in range(p)]
            if span_rank(B) < p:
                continue
            cs, _ = gram_schmidt(B)
            if all(abs(x.numerator) <= 12 and x.denominator <= 6 for c in cs for x in c):
                break
    cs, steps = gram_schmidt(B)
    prompt = ("Apply Gram-Schmidt (standard product) to\n   " +
              ",  ".join(f"b{i + 1} = {fv(b)}" for i, b in enumerate(B)) +
              "\nand give the orthogonal basis c1, c2, ... (each cj may be rescaled).")
    return mk("gram_schmidt", "gs", prompt, ScaledVectorsAns(cs), steps, B=B)


def ex_gs_orthonormal(rng, data=None):
    while True:
        B = [ivec(rng, 3, -2, 2) for _ in range(2)]
        if span_rank(B) == 2:
            break
    cs, steps = gram_schmidt(B)
    d2 = [Surd(x) / norm(cs[1]) for x in cs[1]]
    prompt = (f"Gram-Schmidt on b1 = {fv(B[0])}, b2 = {fv(B[1])} (standard product).\n"
              "Give the second vector d2 of the ORTHONORMAL basis.")
    steps = steps + [f"||c2|| = {norm(cs[1]).pretty()}", "d2 = c2/||c2|| = (" + ", ".join(x.pretty() for x in d2) + ")"]
    return mk("gram_schmidt", "gs_on", prompt, RealVectorAns(d2), steps, B=B)


GRAM_SCHMIDT = Topic(
    "gram_schmidt", "§5.3.2", "Gram-Schmidt orthonormalisation", "Ortonormalizzazione di Gram-Schmidt",
    "5 Spazi euclidei", ["orthogonality"], GS_LESSON,
    [lambda: ex_gram_schmidt(None, [[0, 1, 0, 1], [2, 1, 0, 1], [-1, 0, 0, 1], [0, 0, 1, 0]])],
    [Gen("gs", "orthogonal basis by Gram-Schmidt", ex_gram_schmidt, exam=True, weight=2),
     Gen("gs_on", "orthonormal vector", ex_gs_orthonormal)],
)

# ================================================================== complement & projections
PROJ_LESSON = [
    ("Intuition", """\
Every vector splits uniquely as "a part inside W" plus "a part perpendicular to W". The part
in W is the orthogonal projection: the point of W closest to v."""),
    ("Definitions (§5.3.3)", """\
• Complemento ortogonale W⊥ = {v : ⟨v,w⟩ = 0 ∀w ∈ W} ◁ V (enough to test a basis of W).
  V = W ⊕ W⊥, dim W + dim W⊥ = dim V, (W⊥)⊥ = W.
  In R^n (standard): W = Sol(A|0) ⇒ W⊥ = Rig(A) (rows of A);  W = Span(v1..vp) ⇒ W⊥ = Sol(M|0)
  with M having v1^T..vp^T as rows.
• Proiezione ortogonale on W, with an ORTHOGONAL basis {b1..bp} of W:
     π_W(v) = Σ (⟨v,bk⟩/||bk||²) bk;   π_{W⊥} = id - π_W;   ker π_W = W⊥;  π_W∘π_W = π_W.
  If the basis of W is not orthogonal, run Gram-Schmidt first! (common mistake)
• Best approximation: min_{w∈W} d(v,w) = ||v - π_W(v)|| = ||π_{W⊥}(v)||."""),
    ("Matrices of projections (§5.3.4)", """\
Teorema 5.24: with an orthonormal basis {b1..bp} of W and Gram matrix S = M_C(⟨·,·⟩),
   M^C_C(π_W) = Σ (bi)_C (bi)_C^T S;   standard product, C canonical: Σ bi bi^T.
With an orthogonal (not normalised) basis: Σ ci ci^T / ||ci||². π_W is diagonalisable with
eigenvalues 1 (on W) and 0 (on W⊥)."""),
]


def ex_complement(rng, data=None):
    n = 4 if rng.random() < 0.6 else 3
    ws = data or _indep(rng, n, rng.choice([1, 2]))
    K = orth_complement(ws, n)
    prompt = f"In R^{n} (standard product) let W = Span({', '.join(fv(w) for w in ws)}). Find a basis of W⊥."
    M = Matrix(ws)
    steps = [hcat("W⊥ = Sol(M|0) with the generators of W as ROWS: M = ", fm(M)),
             f"dim W⊥ = {n} - {len(ws)} = {len(K)};  basis: {fvs(K)}"]
    return mk("projections", "complement", prompt, SpanAns(K, n, "W⊥"), steps, ws=ws)


def _indep(rng, n, k):
    while True:
        vs = [ivec(rng, n, -2, 2) for _ in range(k)]
        if span_rank(vs) == k:
            return vs


def ex_projection(rng, data=None):
    if data:
        v, ws = data
    else:
        n = rng.choice([3, 4])
        while True:
            ws = _indep(rng, n, rng.choice([1, 2]))
            v = ivec(rng, n, -3, 3)
            cs, _ = gram_schmidt(ws)
            p = projection(v, cs)
            if all(x.denominator <= 7 for x in p):
                break
    cs, gs_steps = gram_schmidt(ws)
    p = projection(v, cs)
    prompt = (f"In R^{len(v)} (standard product) let W = Span({', '.join(fv(w) for w in ws)}).\n"
              f"Compute the orthogonal projection π_W(v) of v = {fv(v)}.")
    steps = []
    orth = len(ws) == 1 or dot(ws[0], ws[1]) == 0
    if not orth:
        steps += ["The generators of W are NOT orthogonal: Gram-Schmidt first."] + gs_steps
    steps += [f"π_W(v) = Σ (⟨v,ck⟩/||ck||²)ck = " + " + ".join(f"({ratio(dot(v, c), dot(c, c))})·{fv(c)}" for c in cs) + f" = {fv(p)}",
              f"Check: v - π_W(v) = {fv(vsub(v, p))} is orthogonal to W."]
    mistakes = []
    if not orth:
        naive = [Fraction(0)] * len(v)
        for w in ws:
            naive = vadd(naive, vscale(Fraction(dot(v, w)) / dot(w, w), w))
        mistakes.append((naive, "You used the Fourier formula with a NON-orthogonal basis of W: apply Gram-Schmidt first."))
    mistakes.append((vsub(v, p), "That's π_{W⊥}(v) = v - π_W(v)."))
    return mk("projections", "projection", prompt, VectorAns(p, mistakes=mistakes), steps, v=v, ws=ws)


def ex_proj_matrix(rng, data=None):
    if data:
        ws = data
    else:
        ws = _indep(rng, 3, rng.choice([1, 1, 2]))
        while len(ws) == 2 and any(x.denominator > 10 for r in projection_matrix(ws).rows for x in r):
            ws = _indep(rng, 3, 2)
    P = projection_matrix(ws)
    cs, gs_steps = gram_schmidt(ws)
    n = len(ws[0])
    prompt = f"In R³ (standard product) let W = Span({', '.join(fv(w) for w in ws)}).\nWrite the matrix M^E_E(π_W)."
    steps = (gs_steps if len(ws) > 1 else []) + [
        "M = Σ c c^T / ||c||² over an orthogonal basis {c} of W (Teorema 5.24, Cor. 5.25):", fm(P),
        f"Check: M² = M, M = M^T, tr(M) = dim W = {len(ws)}."]
    return mk("projections", "proj_matrix", prompt,
              MatrixAns(P, mistakes=[(Matrix.identity(n) - P, "That's the projection on W⊥ (I - M).")]), steps, ws=ws)


def ex_dist_to_subspace(rng, data=None):
    n = 3
    while True:
        ws = _indep(rng, n, rng.choice([1, 2]))
        v = ivec(rng, n, -3, 3)
        cs, _ = gram_schmidt(ws)
        r = vsub(v, projection(v, cs))
        if any(r):
            break
    val = Surd.sqrt(dot(r, r))
    prompt = f"In R³ let W = Span({', '.join(fv(w) for w in ws)}). Compute the distance of v = {fv(v)} from W."
    steps = [f"π_W(v) = {fv(projection(v, cs))}", f"v - π_W(v) = π_W⊥(v) = {fv(r)}",
             f"d(v, W) = ||π_W⊥(v)|| = {val.pretty()}"]
    return mk("projections", "dist_sub", prompt, RealAns(val), steps, v=v, ws=ws)


PROJECTIONS = Topic(
    "projections", "§5.3.3–5.3.4", "Orthogonal complement and orthogonal projections",
    "Complemento ortogonale e proiezioni ortogonali", "5 Spazi euclidei", ["gram_schmidt"], PROJ_LESSON,
    [lambda: ex_projection(None, ([2, 3, 0], [[1, 0, 1], [0, 1, 0]]))],
    [Gen("complement", "basis of W⊥", ex_complement, exam=True),
     Gen("projection", "π_W(v)", ex_projection, exam=True, weight=2),
     Gen("proj_matrix", "matrix of π_W", ex_proj_matrix, exam=True),
     Gen("dist_sub", "distance from a subspace", ex_dist_to_subspace)],
)

# ================================================================== cross product, distances
CROSS_LESSON = [
    ("Intuition", """\
In R³ two vectors determine a third one perpendicular to both, long as the area of their
parallelogram: the cross product. Distances between lines and planes are lengths of
"perpendicular drops" — projections onto the orthogonal complement."""),
    ("Prodotto vettoriale (§5.4)", """\
v × w = ||v||·||w||·sin θ · n (n unit, right-hand rule), computed as the "mixed determinant"
   v × w = (v2w3 - v3w2,  -(v1w3 - v3w1),  v1w2 - v2w1).
v × w ⊥ v, w;  ||v × w|| = area of the parallelogram;  v × w = 0 ⇔ v, w dependent;
bilinear and alternating (v × w = -w × v); NOT associative."""),
    ("Distances (§5.5)", """\
d(S1, S2) = min d(P,Q) = ||π_{W⊥}(P2 - P1)|| with W = S1,0 + S2,0 (Prop. 5.28).
• Point–hyperplane (Cor. 5.29): d(P, {a·x = b}) = |a·P - b| / ||a||.
• Point–line P0 + vR:  ||P - P0 - (⟨P-P0,v⟩/||v||²) v||.  Parallel lines: point–line.
• Skew lines Pi + viR: d = |⟨P2 - P1, n⟩| / ||n||, n ⊥ v1, v2 (e.g. n = v1 × v2)."""),
]


def ex_cross(rng, data=None):
    v, w = data or (ivec(rng, 3, -3, 3), ivec(rng, 3, -3, 3))
    c = cross(v, w)
    prompt = f"Compute v × w for v = {fv(v)}, w = {fv(w)}."
    steps = ["v × w = (v2w3 - v3w2, -(v1w3 - v3w1), v1w2 - v2w1)",
             f"= ({fs(v[1])}·{fs(w[2])} - {fs(v[2])}·{fs(w[1])}, -({fs(v[0])}·{fs(w[2])} - {fs(v[2])}·{fs(w[0])}), "
             f"{fs(v[0])}·{fs(w[1])} - {fs(v[1])}·{fs(w[0])}) = {fv(c)}",
             f"Check: ⟨v×w, v⟩ = {fs(dot(c, v))}, ⟨v×w, w⟩ = {fs(dot(c, w))}."]
    return mk("cross_distances", "cross", prompt,
              VectorAns(c, mistakes=[(cross(w, v), "That's w × v = -(v × w): the order matters."),
                                     ([c[0], -c[1], c[2]], "Sign of the middle component: it is -(v1w3 - v3w1).")]),
              steps, v=v, w=w)


def ex_area(rng, data=None):
    while True:
        v, w = ivec(rng, 3, -3, 3), ivec(rng, 3, -3, 3)
        if any(cross(v, w)):
            break
    c = cross(v, w)
    val = norm(c)
    prompt = f"Area of the parallelogram with sides v = {fv(v)} and w = {fv(w)}."
    steps = [f"v × w = {fv(c)}", f"Area = ||v × w|| = {val.pretty()}"]
    return mk("cross_distances", "area", prompt, RealAns(val), steps, v=v, w=w)


def ex_dist_point_plane(rng, data=None):
    if data:
        P, a, b = data
    else:
        P, a, b = ivec(rng, 3, -3, 3, nonzero=False), ivec(rng, 3, -3, 3), rng.randint(-8, 8)
    num = abs(dot(a, P) - b)
    val = Surd(num) / norm(a)
    prompt = f"Distance of P = {fv(P)} from the plane π: {flin(a, XYZ)} = {b}."
    steps = [f"d(P,π) = |a·P - b| / ||a|| = |{fs(dot(a, P))} - ({b})| / {norm(a).pretty()} = {ratio(num, norm(a))} = {val.pretty()}"]
    return mk("cross_distances", "pt_plane", prompt,
              RealAns(val, mistakes=[(Surd(num), "Divide by ||(a,b,c)||, the norm of the normal vector.")]), steps, P=P, a=a, b=b)


def ex_dist_point_line(rng, data=None):
    P, Q, v = ivec(rng, 3, -3, 3, nonzero=False), ivec(rng, 3, -3, 3, nonzero=False), ivec(rng, 3, -2, 2)
    d = vsub(P, Q)
    perp = vsub(d, vscale(Fraction(dot(d, v)) / dot(v, v), v))
    val = Surd.sqrt(dot(perp, perp))
    prompt = f"Distance of P = {fv(P)} from the line r = {fparam(Q, [v])}."
    steps = [f"P - Q = {fv(d)}",
             f"Remove its component along v: P - Q - (⟨P-Q,v⟩/||v||²)v = {fv(d)} - ({fs(dot(d, v))}/{fs(dot(v, v))}){fv(v)} = {fv(perp)}",
             f"d(P, r) = ||{fv(perp)}|| = {val.pretty()}"]
    return mk("cross_distances", "pt_line", prompt, RealAns(val), steps, P=P, Q=Q, v=v)


def ex_dist_skew(rng, data=None):
    while True:
        P1, v1 = ivec(rng, 3, -3, 3, nonzero=False), ivec(rng, 3, -2, 2)
        P2, v2 = ivec(rng, 3, -3, 3, nonzero=False), ivec(rng, 3, -2, 2)
        n = cross(v1, v2)
        if any(n) and dot(vsub(P2, P1), n) != 0:
            break
    val = Surd(abs(dot(vsub(P2, P1), n))) / norm(n)
    prompt = f"The lines r1 = {fparam(P1, [v1])} and r2 = {fparam(P2, [v2])} are skew. Compute d(r1, r2)."
    steps = [f"n = v1 × v2 = {fv(n)} is orthogonal to both directions.",
             f"P2 - P1 = {fv(vsub(P2, P1))};  d = |⟨P2 - P1, n⟩| / ||n|| = {ratio(abs(dot(vsub(P2, P1), n)), norm(n))} = {val.pretty()}"]
    return mk("cross_distances", "skew", prompt, RealAns(val), steps, P1=P1, v1=v1, P2=P2, v2=v2)


CROSS_DISTANCES = Topic(
    "cross_distances", "§5.4–5.5", "Cross product and distances", "Prodotto vettoriale, sottospazi affini e distanze",
    "5 Spazi euclidei", ["projections", "affine"], CROSS_LESSON,
    [lambda: ex_dist_point_plane(None, ([1, 2, 3], [1, -3, 1], 10)),
     lambda: ex_cross(None, ([1, 2, 3], [4, 5, 6]))],
    [Gen("cross", "v × w", ex_cross),
     Gen("area", "area of a parallelogram", ex_area),
     Gen("pt_plane", "point–plane distance", ex_dist_point_plane, exam=True),
     Gen("pt_line", "point–line distance", ex_dist_point_line, exam=True),
     Gen("skew", "distance of skew lines", ex_dist_skew, exam=True)],
)

# ================================================================== orthogonal matrices & isometries
ISO_LESSON = [
    ("Intuition", """\
Isometries are the rigid motions: they keep lengths and angles. The linear ones are
rotations and reflections; adding a translation gives all rigid motions."""),
    ("Orthogonal matrices (§5.6.1)", """\
• A ∈ M_R(n) is ortogonale if A^T = A^-1 (i.e. A^T A = I); O(n) is closed under products,
  inverses and transposes; det A = ±1.
• A orthogonal ⇔ its columns are an orthonormal basis of R^n (Lemma 5.31).
  The change of basis between two orthonormal bases is orthogonal.
• O(2) (Teorema 5.33): (cos θ  sin θ; sin θ  -cos θ) — reflection across the line of
  inclination θ/2 (det -1); (cos θ  -sin θ; sin θ  cos θ) — rotation by θ (det +1)."""),
    ("Isometries (§5.6.2)", """\
• Isometria lineare: ⟨f(v),f(w)⟩ = ⟨v,w⟩; it preserves norms and angles, is bijective, and
  S(f) ⊂ {-1, 1}. With an ORTHONORMAL basis B: f isometry ⇔ M^B_B(f) orthogonal.
• Trasformazione affine T(v) = f(v) + v0 (f isomorphism); it is an isometria iff f is a linear
  isometry ("isometria = isometria lineare + traslazione").
• Riflessione across S = P0 + W:  σ_S(v) = v - 2π_{W⊥}(v - P0); for a point P0: σ(v) = 2P0 - v.
• Rotazione: ρ(v) = f(v - P0) + P0 with f an isometry fixing W⊥ (axis P0 + W⊥)."""),
]


def _householder(v) -> Matrix:
    v = vec(v)
    n = len(v)
    vv = dot(v, v)
    return Matrix([[Fraction(int(i == j)) - 2 * v[i] * v[j] / vv for j in range(n)] for i in range(n)])


def _rational_orthogonal(rng, n, det_sign=None):
    H = _householder(ivec(rng, n, -2, 2))
    if det_sign == 1 or (det_sign is None and rng.random() < 0.5):
        H = H * _householder(ivec(rng, n, -2, 2))
    return H


def ex_is_orthogonal(rng, data=None):
    n = rng.choice([2, 3])
    A = _rational_orthogonal(rng, n)
    ok = True
    if rng.random() < 0.5:
        rows = A.tolist()
        i, j = rng.randrange(n), rng.randrange(n)
        rows[i][j] += rng.choice([1, -1, Fraction(1, 2)])
        A = Matrix(rows)
        ok = A.T * A == Matrix.identity(n)
    prompt = named("A", A) + "\n\nIs A orthogonal?"
    steps = [hcat("A^T A = ", fm(A.T * A)), ("= I ⇒ orthogonal (columns orthonormal)." if ok else "≠ I ⇒ not orthogonal.")]
    return mk("isometries", "is_orth", prompt, YesNoAns(ok, steps[-1]), steps, A=A)


TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (0, 1, 1), (1, 0, 1)]


def ex_rot_or_refl(rng, data=None):
    a, b, c = rng.choice(TRIPLES)
    co, si = Fraction(a, c) * rng.choice([1, -1]), Fraction(b, c) * rng.choice([1, -1])
    rot = rng.random() < 0.5
    A = Matrix([[co, -si], [si, co]]) if rot else Matrix([[co, si], [si, -co]])
    ans = ChoiceAns(["a rotation", "a reflection across a line"], 0 if rot else 1)
    prompt = named("A", A) + "\n\nA ∈ O(2). Does L_A describe a rotation or a reflection?\n" + ans.render()
    steps = [f"det A = {fs(det(A))}: " + ("+1 ⇒ rotation (cos θ -sin θ; sin θ cos θ)." if rot else
                                         "-1 ⇒ reflection (cos θ sin θ; sin θ -cos θ), across the line of inclination θ/2.")]
    return mk("isometries", "rot_refl", prompt, ans, steps, A=A)


def ex_reflection_line(rng, data=None):
    P0, d = ivec(rng, 2, -3, 3, nonzero=False), rng.choice([[1, 1], [1, -1], [1, 0], [0, 1], [1, 2], [2, 1]])
    Q = ivec(rng, 2, -4, 4, nonzero=False)
    nrm = [-d[1], d[0]]
    diff = vsub(Q, P0)
    pperp = vscale(Fraction(dot(diff, nrm)) / dot(nrm, nrm), nrm)
    img = vsub(Q, vscale(2, pperp))
    prompt = f"r = {fparam(P0, [d])} ⊂ R². Compute the reflection σ_r(Q) of Q = {fv(Q)} across r."
    steps = [f"W = giacitura = Span({fv(d)}), W⊥ = Span({fv(nrm)}).",
             f"Q - P0 = {fv(diff)};  π_W⊥(Q - P0) = ({fs(dot(diff, nrm))}/{fs(dot(nrm, nrm))}){fv(nrm)} = {fv(pperp)}",
             f"σ_r(Q) = Q - 2π_W⊥(Q - P0) = {fv(img)}"]
    return mk("isometries", "reflect", prompt,
              VectorAns(img, mistakes=[(vsub(Q, pperp), "That's the projection of Q on r: the reflection subtracts TWICE π_W⊥.")]),
              steps, P0=P0, d=d, Q=Q)


ISOMETRIES = Topic(
    "isometries", "§5.6", "Orthogonal matrices and isometries", "Matrici ortogonali, isometrie e trasformazioni affini",
    "5 Spazi euclidei", ["projections"], ISO_LESSON,
    [lambda: ex_rot_or_refl(random.Random(0))],
    [Gen("is_orth", "is A orthogonal?", ex_is_orthogonal),
     Gen("rot_refl", "rotation or reflection?", ex_rot_or_refl),
     Gen("reflect", "reflection across a line", ex_reflection_line, exam=True)],
)

TOPICS = [BILINEAR, SCALAR_PRODUCTS, ORTHOGONALITY, GRAM_SCHMIDT, PROJECTIONS, CROSS_DISTANCES, ISOMETRIES]
