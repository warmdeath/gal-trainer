"""Capitolo 2 — Spazi vettoriali."""
from __future__ import annotations

import random
from fractions import Fraction

from ..core.fmt import flin, fm, fparam, fs, fsys, fv, fvs, hcat, indent, named, var_names
from ..core.gauss import (column_space_basis, det, in_span, kernel_basis, rank, row_reduce,
                          same_span, solve, span_rank)
from ..core.geom import (cartesian_from_param, line_line_position, line_plane_position,
                         plane_plane_position)
from ..core.matrix import Matrix, vadd, vec, vscale, vsub
from ..core.parse import ParseError, parse_vector
from ..core.poly import Poly
from ..exercise import (Answer, Check, ChoiceAns, EquationsAns, IntAns, ScalarAns, SpanAns,
                        VectorAns, YesNoAns, mk)
from .base import Gen, Topic
from .rand import imat, invertible, ivec, nz, rank_r, unimodular

XYZ = ["x", "y", "z"]
XYZW = ["x", "y", "z", "w"]


def _cols(vs) -> str:
    return fm(Matrix.from_cols(vs))


# ================================================================== spaces & subspaces
SPACES_LESSON = [
    ("Intuition", """\
A vector space is any set where you can add elements and multiply them by scalars with
the usual rules: arrows, n-tuples, matrices, polynomials, functions... A subspace is a
"flat" subset through the origin that is closed under these operations — in R³: {0},
lines through 0, planes through 0, R³ itself."""),
    ("Definitions", """\
• Spazio vettoriale over K (Definizione 2.1): a set V with sum and product by scalars
  satisfying SV1–SV8 (associativity, commutativity, neutral element 0, opposite, (cd)v = c(dv),
  1·v = v, two distributive laws). Elements: vettori. Examples: K^n, M_K(m,n), K[t],
  C⁰(R), Sol(A|0). Consequences: 0 is unique, 0·v = 0, λ·0 = 0, λv = 0 ⇒ λ = 0 or v = 0.
• Sottospazio vettoriale W ⊂ V (written W ◁ V, Definizione 2.2):
    S1) 0 ∈ W;   S2) w1 + w2 ∈ W (chiuso rispetto alla somma);   S3) λw ∈ W.
• Examples: {0}, V, K[t]≤d ◁ K[t], C¹(R) ◁ C⁰(R), Sol(A|0) ◁ K^n.
  NOT subspaces: Sol(A|b) with b ≠ 0 (0 ∉, it is an affine subspace), {x ≥ 0, y ≥ 0},
  {xy ≥ 0} (not closed under sum)."""),
    ("Quick test", """\
1) Is 0 in W? If not → not a subspace.  2) Are the conditions linear and homogeneous
(every term has degree exactly 1, no constants)? Then yes. Products of coordinates,
squares, inequalities, det(A) = 0 usually break S2 or S3: find a counterexample."""),
]


def _subspace_case(rng):
    a, b, c = nz(rng, -4, 4), nz(rng, -4, 4), nz(rng, -4, 4)
    d = nz(rng, -5, 5)
    lin = flin([a, b, c], XYZ)
    cases = [
        (f"W = {{(x,y,z)^T ∈ R³ | {lin} = 0}}", True,
         "Yes: it is Sol(A|0) of a homogeneous linear system, always a subspace."),
        (f"W = {{(x,y,z)^T ∈ R³ | {lin} = {d}}}", False,
         f"No: 0 ∉ W since {a}·0 + {b}·0 + {c}·0 = 0 ≠ {d} (S1 fails). It is an affine subspace."),
        ("W = {(x,y,z)^T ∈ R³ | xy = 0}", False,
         "No: (1,0,0) and (0,1,0) are in W but their sum (1,1,0) is not (S2 fails)."),
        ("W = {(x,y,z)^T ∈ R³ | x ≥ 0}", False,
         "No: (1,0,0) ∈ W but (-1)·(1,0,0) ∉ W (S3 fails)."),
        (f"W = {{(x,y,z)^T ∈ R³ | x = {fs(a)}y², z = 0}}" if a else "", False,
         "No: the condition is not linear; e.g. (a,1,0) ∈ W but 2·(a,1,0) = (2a,2,0) ∉ W."),
        (f"W = Span({fv(ivec(rng, 3))}, {fv(ivec(rng, 3))})", True,
         "Yes: a span is always a subspace (Proposizione 2.4)."),
        ("W = {p(t) ∈ R[t]≤2 | p(1) = 0}", True,
         "Yes: p(1) = 0 and q(1) = 0 ⇒ (p+q)(1) = 0 and (λp)(1) = 0; the zero polynomial is in W."),
        ("W = {p(t) ∈ R[t]≤2 | p(0) = 1}", False,
         "No: the zero polynomial has p(0) = 0 ≠ 1."),
        ("W = {A ∈ M_R(2) | A = A^T} (symmetric matrices)", True,
         "Yes: O is symmetric, (A+B)^T = A^T + B^T = A + B, (λA)^T = λA."),
        ("W = {A ∈ M_R(2) | det(A) = 0}", False,
         "No: diag(1,0) and diag(0,1) have det 0, their sum I₂ has det 1 (S2 fails)."),
        ("W = {A ∈ M_R(2) | tr(A) = 0}", True,
         "Yes: the trace is linear, so W = ker(tr) is a subspace."),
        (f"W = {{(x,y)^T ∈ R² | {flin([a, b], ['x', 'y'])} = 0}} ∪ {{(x,y)^T | x = y}}", False,
         "No: the union of two different lines is not closed under sum (Osservazione p.41)."),
    ]
    return rng.choice([c for c in cases if c[0]])


def ex_subspace(rng, data=None):
    text, ok, why = data or _subspace_case(rng)
    prompt = f"{text}\nIs W a vector subspace (sottospazio vettoriale)?"
    return mk("vector_spaces", "subspace", prompt, YesNoAns(ok, why), [why], text=text, ok=ok)


VECTOR_SPACES = Topic(
    "vector_spaces", "§2.1–2.2", "Vector spaces and subspaces", "Spazi vettoriali e sottospazi vettoriali",
    "2 Spazi vettoriali", ["matrices"], SPACES_LESSON,
    [lambda: ex_subspace(None, ("W = {(x,y,z)^T ∈ R³ | x + 2y - z = 1}", False,
                                "No: 0 ∉ W, since 0 + 0 - 0 ≠ 1. It is an affine subspace (§2.7)."))],
    [Gen("subspace", "is W a subspace?", ex_subspace)],
)

# ================================================================== span & independence
SPAN_LESSON = [
    ("Intuition", """\
Span = everything you can reach with the given vectors. Linearly independent = none of them
is redundant: no vector can be reached using the others."""),
    ("Definitions", """\
• Combinazione lineare of v1..vn: λ1v1 + ... + λnvn.  Span(v1,...,vn) = set of all of them;
  it is a subspace (Prop. 2.4).
• {v1..vn} generates V (insieme di generatori) if Span(v1..vn) = V.
• Linearly independent (linearmente indipendenti): λ1v1 + ... + λnvn = 0 ⇒ λ1 = ... = λn = 0.
  Otherwise linearly dependent (linearmente dipendenti).
• Lemma di scarto/aggiunta (2.7): dependent ⇔ one vector is a combination of the others;
  if v1..vn are independent, v1..vn,v_{n+1} are dependent ⇔ v_{n+1} ∈ Span(v1..vn).
• Lemma di Steinitz (2.8): #independent vectors ≤ #generators."""),
    ("Everything is a linear system", """\
In K^m put the vectors as COLUMNS of A = (c1|...|cp): then λ1c1 + ... + λpcp = A·λ.
• b ∈ Span(c1..cp) ⇔ Ax = b is solvable ⇔ rg(A) = rg(A|b).
• c1..cp independent ⇔ Aλ = 0 has only λ = 0 ⇔ rg(A) = p.
• c1..cp generate K^m ⇔ rg(A) = m.  (So you need at least m generators, at most m independent.)
For polynomials or matrices, use their coordinates w.r.t. the canonical basis."""),
]


def _indep(rng, n, k):
    while True:
        vs = [ivec(rng, n, -3, 3) for _ in range(k)]
        if span_rank(vs) == k:
            return vs


def ex_combination(rng, data=None):
    if data:
        vs, b = data
        lam = None
    else:
        n, k = rng.choice([(3, 2), (3, 3), (4, 2), (4, 3)])
        vs = _indep(rng, n, k) if k < n else [list(c) for c in unimodular(rng, n).cols()][:k]
        lam = ivec(rng, k, -3, 3)
        b = [sum(l * v[i] for l, v in zip(lam, vs)) for i in range(n)]
    A = Matrix.from_cols(vs)
    sol = solve(A, b, [f"λ{i + 1}" for i in range(len(vs))])
    lam = sol.particular
    prompt = (f"Write b = {fv(b)} as a linear combination λ1v1 + ... of\n   "
              + ",  ".join(f"v{i + 1} = {fv(v)}" for i, v in enumerate(vs))
              + "\nAnswer with the vector of coefficients (λ1, ..., λk).")
    steps = ["Put v1, ..., vk as columns of A and solve A·λ = b:"] + sol.steps
    return mk("span_independence", "combination", prompt, VectorAns(lam), steps, vs=vs, b=b)


def ex_in_span(rng, data=None):
    n = rng.choice([3, 4])
    vs = _indep(rng, n, 2)
    if rng.random() < 0.5:
        b = vadd(vscale(nz(rng, -2, 2), vs[0]), vscale(nz(rng, -2, 2), vs[1]))
    else:
        b = ivec(rng, n, -4, 4)
    inside = in_span(b, vs)
    A = Matrix.from_cols(vs)
    sol = solve(A, b, ["λ1", "λ2"])
    prompt = f"Is b = {fv(b)} in Span({fv(vs[0])}, {fv(vs[1])})?"
    steps = ["b ∈ Span(c1, c2) ⇔ the system (c1|c2)·λ = b is solvable:"] + sol.steps
    return mk("span_independence", "in_span", prompt, YesNoAns(inside, steps[-1]), steps, vs=vs, b=b)


def ex_independent(rng, data=None):
    n = rng.choice([3, 4])
    k = rng.choice([2, 3, 3]) if n == 3 else rng.choice([3, 4])
    if rng.random() < 0.5:
        vs = _indep(rng, n, k)
    else:
        base = _indep(rng, n, k - 1)
        coeffs = [nz(rng, -2, 2) for _ in base]
        extra = [sum(c * v[i] for c, v in zip(coeffs, base)) for i in range(n)]
        vs = base + [extra]
        rng.shuffle(vs)
    A = Matrix.from_cols(vs)
    red = row_reduce(A)
    r = red.rank
    prompt = "Are these vectors linearly independent?\n   " + ",  ".join(f"v{i + 1} = {fv(v)}" for i, v in enumerate(vs))
    steps = ["They are independent ⇔ rg(v1|...|vk) = k. Reduce the matrix with the vectors as columns:",
             indent(_cols(vs))] + red.steps + [
        f"rg = {r} {'=' if r == k else '<'} k = {k} ⇒ " + ("independent." if r == k else "dependent.")]
    return mk("span_independence", "independent", prompt, YesNoAns(r == k, steps[-1]), steps, vs=vs)


def ex_dependence(rng, data=None):
    n = rng.choice([3, 4])
    base = _indep(rng, n, 2)
    a, b = nz(rng, -3, 3), nz(rng, -3, 3)
    third = vadd(vscale(a, base[0]), vscale(b, base[1]))
    vs = base + [third]
    rng.shuffle(vs)
    A = Matrix.from_cols(vs)
    K = kernel_basis(A)
    sol = solve(A, [0] * n, ["λ1", "λ2", "λ3"])
    prompt = ("The vectors\n   " + ",  ".join(f"v{i + 1} = {fv(v)}" for i, v in enumerate(vs)) +
              "\nare linearly dependent. Find (λ1, λ2, λ3) ≠ 0 with λ1v1 + λ2v2 + λ3v3 = 0.")
    steps = ["Solve the homogeneous system (v1|v2|v3)·λ = 0:"] + sol.steps + [
        f"For example λ = {fv(K[0])}: " + " + ".join(f"({fs(l)})v{i + 1}" for i, l in enumerate(K[0])) + " = 0."]
    return mk("span_independence", "dependence", prompt, VectorAns(K[0], up_to_scalar=True), steps, vs=vs)


def ex_generators(rng, data=None):
    k = rng.choice([3, 4])
    if rng.random() < 0.5:
        vs = [list(c) for c in unimodular(rng, 3).cols()]
        if k == 4:
            vs.append(ivec(rng, 3, -2, 2))
    else:
        vs = [list(c) for c in rank_r(rng, 3, k, 2).cols()]
    rng.shuffle(vs)
    r = span_rank(vs)
    prompt = "Do these vectors generate R³?\n   " + ",  ".join(fv(v) for v in vs)
    red = row_reduce(Matrix.from_cols(vs))
    steps = ["They generate R³ ⇔ (v1|...|vk)x = b is solvable for every b ⇔ rg = 3."] + red.steps + [
        f"rg = {r} ⇒ " + ("they generate R³." if r == 3 else "they do NOT generate R³.")]
    return mk("span_independence", "generators", prompt, YesNoAns(r == 3, steps[-1]), steps, vs=vs)


def ex_poly_indep(rng, data=None):
    if data:
        ps = data
    else:
        while True:
            ps = [Poly(ivec(rng, 3, -2, 2)) for _ in range(3)]
            if all(p.degree >= 0 for p in ps):
                break
    coords = [list(p.c) + [Fraction(0)] * (3 - len(p.c)) for p in ps]
    A = Matrix.from_cols(coords)
    d = det(A)
    prompt = ("In R[t]≤2, are the polynomials\n   " + ",  ".join(f"p{i + 1} = {p}" for i, p in enumerate(ps)) +
              "\nlinearly independent (equivalently: a basis of R[t]≤2, since dim = 3)?")
    steps = ["Take coordinates w.r.t. the canonical basis {1, t, t²} and put them as columns:",
             indent(fm(A)), f"Independent ⇔ det ≠ 0. det = {fs(d)} ⇒ " +
             ("independent (a basis, Corollario 2.18)." if d else "dependent.")]
    return mk("span_independence", "poly_indep", prompt, YesNoAns(d != 0, steps[-1]), steps, coords=coords)


SPAN_INDEP = Topic(
    "span_independence", "§2.3", "Linear combinations, span, linear independence",
    "Combinazioni lineari, generatori, indipendenza lineare", "2 Spazi vettoriali",
    ["vector_spaces", "rank"], SPAN_LESSON,
    [lambda: ex_combination(None, ([[2, 0, 2], [1, -1, 0]], [1, 1, 2])),
     lambda: ex_poly_indep(None, [Poly([1, 1]), Poly([0, 1, 1]), Poly([1, 0, 1])])],
    [Gen("combination", "write b as a combination", ex_combination),
     Gen("in_span", "is b in the span?", ex_in_span),
     Gen("independent", "are they independent?", ex_independent, weight=2),
     Gen("dependence", "find a dependence relation", ex_dependence),
     Gen("generators", "do they generate R³?", ex_generators),
     Gen("poly_indep", "polynomials: independent?", ex_poly_indep)],
)

# ================================================================== bases & dimension
BASES_LESSON = [
    ("Intuition", """\
A basis is a "coordinate system": enough vectors to reach everything (generators), and no
redundancy (independent), so every vector has ONE list of coordinates. All bases of V have
the same number of vectors: the dimension."""),
    ("Definitions and theorems", """\
• Base (Def. 2.10): an ORDERED set of independent generators. Canonical basis (base canonica)
  e1..en of K^n; E_{ij} of M_K(m,n); {1, t, ..., t^d} of K[t]≤d.
• Coordinates (coordinate, Def. 2.12): v = λ1b1 + ... + λnbn, (v)_B = (λ1..λn)^T, unique
  (Prop. 2.11) and linear: (k1v1 + k2v2)_B = k1(v1)_B + k2(v2)_B.
• Teorema della base: every finitely generated V has a basis. Teorema della dimensione:
  all bases have the same cardinality = dim V. dim{0} = 0, dim K^n = n,
  dim M_K(m,n) = mn, dim K[t]≤d = d + 1, dim Sol(A|0) = n - rg(A).
• W ◁ V ⇒ dim W ≤ dim V, with equality ⇔ W = V.
• Every independent set can be completed to a basis; from every set of generators one can
  extract a basis. In dim n, n vectors are generators ⇔ they are independent (Cor. 2.18):
  {v1..vn} basis of K^n ⇔ det(v1|...|vn) ≠ 0."""),
    ("Row space, column space, kernel (§2.4.1)", """\
Rig(A) = Span(rows), Col(A) = Span(columns), ker A = Sol(A|0).
Teorema 2.20: rg(A) = dim Rig(A) = dim Col(A) = rg(A^T).
Prop. 2.21: the columns of A in the pivot positions of an echelon form U form a basis of
Col(A). Attenzione: Col(A) ≠ Col(U) in general — take the columns of A, not of U!"""),
]


def ex_dim_span(rng, data=None):
    if data:
        vs = data
    else:
        n, k = rng.choice([(3, 3), (3, 4), (4, 3), (4, 4)])
        r = rng.randint(1, min(n, k) - (0 if rng.random() < 0.3 else 1)) if min(n, k) > 1 else 1
        r = max(r, 2) if min(n, k) >= 2 else r
        vs = [list(c) for c in rank_r(rng, n, k, r).cols()]
    A = Matrix.from_cols(vs)
    red = row_reduce(A)
    basis = column_space_basis(A)
    prompt = "Let W = Span(" + ", ".join(fv(v) for v in vs) + ").\nFind dim W and a basis of W."
    piv = [j for _, j in red.pivots]
    steps = ["Put the generators as columns and reduce (Prop. 2.21):", indent(_cols(vs))] + red.steps + [
        f"Pivots in columns {', '.join(str(j + 1) for j in piv)} ⇒ dim W = {red.rank};",
        "a basis: the ORIGINAL vectors in those positions: " + fvs(basis)]
    return mk("bases", "dim_span", prompt,
              [("dim W", IntAns(red.rank, mistakes=[(len(vs), "That's the number of generators; remove the dependent ones.")])),
               ("a basis of W", SpanAns(basis, len(vs[0])))], steps, vs=vs)


def ex_coordinates(rng, data=None):
    if data:
        B, v = data
    else:
        n = rng.choice([2, 3, 3])
        B = [list(c) for c in unimodular(rng, n).cols()]
        lam = ivec(rng, n, -3, 3)
        v = [sum(l * b[i] for l, b in zip(lam, B)) for i in range(n)]
    A = Matrix.from_cols(B)
    sol = solve(A, v, [f"λ{i + 1}" for i in range(len(B))])
    prompt = ("B = {" + ", ".join(fv(b) for b in B) + f"}} is a basis of R^{len(B)}.\n"
              f"Find the coordinates (v)_B of v = {fv(v)}.")
    steps = ["(v)_B = (λ1..λn) with λ1b1 + ... + λnbn = v: solve (b1|...|bn)·λ = v."] + sol.steps
    return mk("bases", "coordinates", prompt, VectorAns(sol.particular,
              mistakes=[(v, "Those are the canonical coordinates of v, not the ones w.r.t. B.")]), steps, B=B, v=v)


class CompleteBasisAns(Answer):
    fmt_hint = "one vector"

    def __init__(self, vs, model):
        super().__init__()
        self.vs, self.model = [vec(v) for v in vs], vec(model)

    def check(self, text):
        x = parse_vector(text, len(self.vs[0]))
        if span_rank(self.vs + [x]) == len(self.vs) + 1:
            return Check.right()
        return Check.wrong(f"{fv(x)} ∈ Span of the given vectors (det = 0): it adds nothing new.")

    def show(self):
        return fv(self.model) + "  (any vector outside the span works)"

    def typed(self):
        return fv(self.model)


def ex_complete(rng, data=None):
    vs = data or _indep(rng, 3, 2)
    e = next(ei for ei in ([1, 0, 0], [0, 1, 0], [0, 0, 1]) if span_rank(vs + [ei]) == 3)
    prompt = (f"v1 = {fv(vs[0])}, v2 = {fv(vs[1])} are linearly independent in R³.\n"
              "Complete them to a basis of R³: give a vector v3.")
    steps = ["We need v3 ∉ Span(v1, v2), i.e. det(v1|v2|v3) ≠ 0 (Lemma 2.7, Prop. 2.17).",
             "Try the canonical vectors e1, e2, e3: at least one works."]
    for ei in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        d = det(Matrix.from_cols(vs + [ei]))
        steps.append(f"  det(v1|v2|{fv(ei)}) = {fs(d)}")
    steps.append(f"E.g. v3 = {fv(e)}.")
    return mk("bases", "complete", prompt, CompleteBasisAns(vs, e), steps, vs=vs)


def ex_col_space(rng, data=None):
    A = data or rank_r(rng, 3, 4, rng.choice([2, 2, 3]))
    red = row_reduce(A)
    basis = column_space_basis(A)
    piv = [j for _, j in red.pivots]
    prompt = named("A", A) + "\n\nFind a basis of the column space Col(A) and dim Rig(A)."
    steps = red.steps + [f"Pivot columns: {', '.join(str(j + 1) for j in piv)} → basis of Col(A): the columns of A "
                         f"in those positions: {fvs(basis)}",
                         f"dim Rig(A) = dim Col(A) = rg(A) = {red.rank} (Teorema 2.20)."]
    return mk("bases", "col_space", prompt,
              [("basis of Col(A)", SpanAns(basis, A.m, "Col(A)")), ("dim Rig(A)", IntAns(red.rank))], steps, A=A)


BASES = Topic(
    "bases", "§2.4", "Bases, dimension, coordinates", "Basi e dimensione", "2 Spazi vettoriali",
    ["span_independence"], BASES_LESSON,
    [lambda: ex_dim_span(None, [[1, 1, 0, 0], [1, 0, 1, 0], [2, 1, 1, 0]])],
    [Gen("dim_span", "dimension and basis of a span", ex_dim_span, exam=True, weight=2),
     Gen("coordinates", "coordinates w.r.t. a basis", ex_coordinates, weight=2),
     Gen("complete", "complete to a basis", ex_complete),
     Gen("col_space", "Col(A) and rank", ex_col_space)],
)

# ================================================================== cartesian / parametric
CARTPAR_LESSON = [
    ("Intuition", """\
Two ways to describe a subspace W ◁ K^n: implicitly by the equations its vectors satisfy
(Cartesian), or explicitly by vectors that generate it (parametric). If dim W = r you need
r parameters, or n - r independent equations."""),
    ("Cartesian → parametric", """\
Solve the homogeneous system: W = Sol(A|0) = Span(w1..w_{n-r}) — one vector per free variable."""),
    ("Parametric → Cartesian", """\
W = Span(w1..wr) = Col(M) with M = (w1|...|wr). Then x = (x1..xn) ∈ W ⇔ rg(M) = rg(M|x).
Reduce (M|x) with row operations (the last column becomes linear expressions in x1..xn):
the entries of the last column in the zero rows of M must vanish → Cartesian equations.
(Alternative: eliminate the parameters as in Chapter 0.)"""),
]


def ex_param_to_cart(rng, data=None):
    if data:
        ws = data
    else:
        n = rng.choice([3, 4, 4])
        ws = _indep(rng, n, rng.choice([1, 2]) if n == 3 else 2)
    n = len(ws[0])
    names = var_names(n)
    eqs, steps = cartesian_from_param([0] * n, ws, names)
    prompt = (f"W = Span({', '.join(fv(w) for w in ws)}) ◁ R^{n}.\n"
              f"Find Cartesian equations of W (variables {', '.join(names)}; separate equations with ';').")
    return mk("cart_param", "param_to_cart", prompt, EquationsAns([0] * n, ws, names, eqs), steps, ws=ws)


def ex_cart_to_basis(rng, data=None):
    if data:
        A = data
    else:
        n = rng.choice([3, 4, 4])
        m = rng.randint(1, n - 1)
        A = rank_r(rng, m, n, m)
    names = var_names(A.n)
    K = kernel_basis(A)
    sol = solve(A, [0] * A.m, names)
    prompt = ("W = {x ∈ R^" + str(A.n) + " :\n" + indent(fsys(A, [0] * A.m, names), 3) +
              "\n}.  Find dim W and a basis of W.")
    steps = sol.steps + [f"dim W = n - rg(A) = {A.n} - {A.n - len(K)} = {len(K)}; basis: {fvs(K)}"]
    return mk("cart_param", "cart_to_basis", prompt,
              [("dim W", IntAns(len(K), mistakes=[(A.m, "dim W = n - rg(A), not the number of equations.")])),
               ("a basis of W", SpanAns(K, A.n))], steps, A=A)


CART_PARAM = Topic(
    "cart_param", "§2.5", "Cartesian and parametric equations of subspaces",
    "Equazioni cartesiane e parametriche", "2 Spazi vettoriali", ["bases"], CARTPAR_LESSON,
    [lambda: ex_param_to_cart(None, [[2, -2, 1, 0], [1, 0, -1, 1]])],
    [Gen("param_to_cart", "parametric → Cartesian", ex_param_to_cart, exam=True),
     Gen("cart_to_basis", "Cartesian → basis", ex_cart_to_basis, exam=True)],
)

# ================================================================== sum & intersection
SUM_LESSON = [
    ("Intuition", """\
Two subspaces always meet at least in 0. U ∩ W is what they share; U + W is the smallest
subspace containing both. Grassmann's formula is the "inclusion-exclusion" of dimensions."""),
    ("Definitions", """\
• U ∩ W is a subspace (Lemma 2.22); U ∪ W in general is NOT.
• Somma U + W = {u + w : u ∈ U, w ∈ W} (Def. 2.23), generated by a basis of U together with
  a basis of W. Direct sum (somma diretta) U ⊕ W if U ∩ W = {0}: every vector of U + W is
  u + w in a unique way.
• Formula di Grassmann (2.26): dim U + dim W = dim(U ∩ W) + dim(U + W).
  Corollario: the sum is direct ⇔ dim(U+W) = dim U + dim W ⇔ B_U ∪ B_W is a basis of U + W."""),
    ("Method", """\
dim(U+W) = rank of the matrix with the generators of U and of W as columns.
U ∩ W: if W = Span(w1..ws) and U = Sol(A|0), a vector a1w1 + ... + asws is in U iff its
components satisfy the equations of U → a homogeneous system in a1..as.
Then check with Grassmann."""),
]


def _nice_U(rng, n=4):
    """U = Sol(A|0) with small sparse integer equations."""
    while True:
        m = rng.choice([1, 2, 2])
        A = Matrix([[rng.choice([0, 0, 1, -1, 1, 2, -2]) for _ in range(n)] for _ in range(m)])
        if rank(A) == m:
            K = kernel_basis(A)
            if all(abs(x) <= 4 for v in K for x in v):
                return A, K


def _sum_data(rng):
    """(A_U, U basis, W generators) with small numbers everywhere."""
    while True:
        A_U, us = _nice_U(rng)
        dw = rng.choice([1, 2, 2])
        ws = [ivec(rng, 4, -2, 2) for _ in range(dw)]
        if rng.random() < 0.6:  # force a common vector
            c = us[0] if len(us) == 1 else vadd(vscale(rng.choice([1, -1]), us[0]), vscale(rng.choice([0, 1, -1]), us[1]))
            ws[0] = c
        if span_rank(ws) < dw:
            continue
        inter = intersection_basis(A_U, ws)
        if all(abs(x) <= 9 for v in inter for x in v):
            return A_U, us, ws


def intersection_basis(A_U: Matrix, ws) -> list[list[Fraction]]:
    W = Matrix.from_cols(ws)
    K = kernel_basis(A_U * W)
    from ..core.matrix import clear_denominators
    out = [clear_denominators(W.apply(a)) for a in K]
    # remove dependent ones (if ws were dependent)
    basis = []
    for v in out:
        if span_rank(basis + [v]) > len(basis):
            basis.append(v)
    return basis


def ex_grassmann(rng, data=None):
    if data:
        A_U, ws = data
        us = kernel_basis(A_U)
    else:
        A_U, us, ws = _sum_data(rng)
    dU, dW = len(us), span_rank(ws)
    dsum = span_rank(us + ws)
    inter = intersection_basis(A_U, ws)
    dint = len(inter)
    assert dU + dW == dint + dsum
    prompt = ("In R⁴ let\n   U = {x ∈ R⁴ : " + ";  ".join(f"{flin(A_U.row(i), XYZW)} = 0" for i in range(A_U.m)) + "}\n"
              f"   W = Span({', '.join(fv(w) for w in ws)}).\n"
              "Find dim U, dim W, dim(U + W), dim(U ∩ W), and say whether the sum is direct.")
    solU = solve(A_U, [0] * A_U.m, XYZW)
    steps = ["U = Sol(A|0):"] + solU.steps + [f"dim U = {dU}, basis {fvs(us)}",
             f"dim W = rank of its generators = {dW}",
             "U + W is generated by the basis of U together with the generators of W:",
             indent(_cols(us + ws)), f"its rank is {dsum} ⇒ dim(U+W) = {dsum}.",
             f"Grassmann: dim(U∩W) = dim U + dim W - dim(U+W) = {dU} + {dW} - {dsum} = {dint}."]
    if dint:
        steps.append(f"Directly: a1w1 + ... ∈ U ⇔ (A·W)a = 0; U ∩ W = Span({fvs(inter)}).")
    steps.append("The sum is direct." if dint == 0 else "The sum is NOT direct (U ∩ W ≠ {0}).")
    return mk("sum_intersection", "grassmann", prompt,
              [("dim U", IntAns(dU)), ("dim W", IntAns(dW)),
               ("dim(U + W)", IntAns(dsum, mistakes=[(dU + dW, "dim U + dim W only if the sum is direct: compute the rank of all generators together.")])),
               ("dim(U ∩ W)", IntAns(dint)),
               ("is the sum direct?", YesNoAns(dint == 0, "Direct ⇔ U ∩ W = {0}."))],
              steps, A_U=A_U, ws=ws)


def ex_intersection(rng, data=None):
    while True:
        A_U, us, ws = _sum_data(rng)
        inter = intersection_basis(A_U, ws)
        if inter or rng.random() < 0.2:
            break
    prompt = ("In R⁴ let U = {x : " + ";  ".join(f"{flin(A_U.row(i), XYZW)} = 0" for i in range(A_U.m)) + "}\n"
              f"and W = Span({', '.join(fv(w) for w in ws)}). Find a basis of U ∩ W ('none' if U ∩ W = {{0}}).")
    W = Matrix.from_cols(ws)
    AW = A_U * W
    sol = solve(AW, [0] * AW.m, [f"a{i + 1}" for i in range(len(ws))])
    steps = ["A vector of W is a1w1 + ... + asws = W·a. It is in U iff A·(W·a) = 0:",
             hcat("A·W = ", fm(AW))] + sol.steps + [
        ("U ∩ W = Span(" + fvs(inter) + ")") if inter else "Only a = 0: U ∩ W = {0}."]
    return mk("sum_intersection", "intersection", prompt, SpanAns(inter, 4, "U ∩ W"), steps, A_U=A_U, ws=ws)


SUM_INTERSECTION = Topic(
    "sum_intersection", "§2.6", "Intersection and sum of subspaces, Grassmann",
    "Intersezione e somma di sottospazi, formula di Grassmann", "2 Spazi vettoriali",
    ["cart_param"], SUM_LESSON,
    [lambda: ex_grassmann(None, (Matrix([[1, 1, -1, 0], [1, -1, 0, 1]]), [[1, 1, 0, 0], [1, 0, 1, 0], [2, 1, 1, 0]]))],
    [Gen("grassmann", "dimensions and Grassmann", ex_grassmann, exam=True, weight=2),
     Gen("intersection", "basis of U ∩ W", ex_intersection, exam=True)],
)

# ================================================================== affine subspaces
AFFINE_LESSON = [
    ("Intuition", """\
Solutions of Ax = b with b ≠ 0 are a subspace "shifted away from the origin": a line or a
plane not through 0. Its direction (the subspace you shift) is the giacitura."""),
    ("Definitions (§2.7)", """\
• Sottospazio affine S = v + S0 with S0 ◁ V; S0 is the giacitura (direction), dim S := dim S0.
  Elements are called points. Sol(A|b) = v_P + ker(A) is affine with giacitura ker(A).
  dim 1: retta;  dim 2: piano;  dim n - 1: iperpiano.  P, Q ∈ S ⇒ P - Q ∈ S0.
• Line through P, Q: P + (Q - P)t.   Plane through non-collinear P, Q, R: P + (Q-P)t1 + (R-P)t2.
• Positions (Def. 2.30): S1, S2 are paralleli if S1,0 ⊆ S2,0 or S2,0 ⊆ S1,0; incidenti if they
  meet and neither contains the other; sghembi if they do not meet and S1,0 ∩ S2,0 = {0}."""),
    ("Relative positions in R³ (parametric data: l_i = P_i + v_i R, π = P2 + w1 R + w2 R)", """\
Line–line:   rg(v1|v2) = 1 & rg(v1|v2|P2-P1) = 1 → coincidenti;  1 & 2 → parallele;
             rg(v1|v2) = 2 & rg(v1|v2|P2-P1) = 2 → incidenti;    2 & 3 → sghembe.
Line–plane:  rg(v1|w1|w2) = 3 → incidenti (one point);  = 2 → parallel: l ⊂ π if
             P2 - P1 ∈ Span(w1,w2), otherwise parallela e disgiunta.
Plane–plane: rg(v1|v2|w1|w2) = 3 → incidenti (a line); = 2 → coincidenti or paralleli.
With Cartesian equations stack them: (A|b) and compare rg(A), rg(A|b) (Rouché-Capelli).
Fascio di piani through a line {α1x+β1y+γ1z=δ1, α2x+β2y+γ2z=δ2}:
   λ(α1x+β1y+γ1z-δ1) + μ(α2x+β2y+γ2z-δ2) = 0."""),
]


def _line_text(P, v, name, cart: bool):
    if cart:
        eqs, _ = cartesian_from_param(P, [v], XYZ)
        return f"{name}: " + ",  ".join(f"{flin(a, XYZ)} = {fs(b)}" for a, b in eqs)
    return f"{name} = {fparam(P, [v])}"


LL_OPTS = ["coincident (coincidenti)", "parallel (parallele, distinct)", "incident (incidenti)", "skew (sghembe)"]
LL_KEYS = ["coincidenti", "parallele", "incidenti", "sghembe"]


def _line_pair(rng, case):
    P1, v1 = ivec(rng, 3, -3, 3, nonzero=False), ivec(rng, 3, -3, 3)
    while True:
        if case in ("coincidenti", "parallele"):
            v2 = vscale(rng.choice([1, -1, 2, -2]), v1)
            P2 = vadd(P1, vscale(nz(rng, -2, 2), v1)) if case == "coincidenti" else vadd(P1, ivec(rng, 3, -3, 3))
        else:
            v2 = ivec(rng, 3, -3, 3)
            if span_rank([v1, v2]) < 2:
                continue
            if case == "incidenti":
                X = vadd(P1, vscale(nz(rng, -2, 2), v1))
                P2 = vsub(X, vscale(nz(rng, -2, 2), v2))
            else:
                P2 = vadd(P1, ivec(rng, 3, -3, 3))
        if line_line_position(P1, v1, P2, v2) == case:
            return P1, v1, P2, v2


def ex_line_line(rng, data=None):
    if data:
        P1, v1, P2, v2 = data
        case = line_line_position(P1, v1, P2, v2)
    else:
        case = rng.choice(LL_KEYS + ["incidenti", "sghembe"])
        P1, v1, P2, v2 = _line_pair(rng, case)
    cart = rng.random() < 0.5 if rng else False
    ans = ChoiceAns(LL_OPTS, LL_KEYS.index(case))
    prompt = ("Relative position (posizione reciproca) of the lines\n   " + _line_text(P1, v1, "l1", cart) +
              "\n   " + _line_text(P2, v2, "l2", False) + "\n" + ans.render())
    d = vsub(P2, P1)
    r12, r3 = span_rank([v1, v2]), span_rank([v1, v2, d])
    steps = []
    if cart:
        steps.append(f"Parametric form of l1: {fparam(P1, [v1])}.")
    steps += [f"Directions v1 = {fv(v1)}, v2 = {fv(v2)}; P2 - P1 = {fv(d)}.",
              f"rg(v1|v2) = {r12}, rg(v1|v2|P2-P1) = {r3} ⇒ {case}."]
    parts = [("position", ans)]
    if case == "incidenti":
        A = Matrix.from_cols([v1, vscale(-1, v2)])
        s = solve(A, d, ["t", "s"])
        X = vadd(P1, vscale(s.particular[0], v1))
        steps.append(f"Intersection: P1 + t v1 = P2 + s v2 ⇒ t = {fs(s.particular[0])}, s = {fs(s.particular[1])}: point {fv(X)}.")
        parts.append(("intersection point", VectorAns(X)))
    return mk("affine", "line_line", prompt, parts, steps, P1=P1, v1=v1, P2=P2, v2=v2)


LP_OPTS = ["l is contained in π (l ⊂ π)", "parallel, no common points", "incident in one point"]
LP_KEYS = ["contenuta", "parallele", "incidenti"]


def ex_line_plane(rng, data=None):
    if data:
        P1, v1, P2, w1, w2 = data
    else:
        case = rng.choice(LP_KEYS + ["incidenti"])
        while True:
            P2 = ivec(rng, 3, -3, 3, nonzero=False)
            w1, w2 = _indep(rng, 3, 2)
            if case == "incidenti":
                v1 = ivec(rng, 3, -3, 3)
                X = vadd(P2, vadd(vscale(nz(rng, -2, 2), w1), vscale(nz(rng, -1, 1), w2)))
                P1 = vsub(X, vscale(nz(rng, -2, 2), v1))
            else:
                v1 = vadd(vscale(nz(rng, -1, 1), w1), vscale(rng.randint(-1, 1), w2))
                P1 = vadd(P2, vscale(nz(rng, -1, 1), w2)) if case == "contenuta" else vadd(P2, ivec(rng, 3, -2, 2))
            if line_plane_position(P1, v1, P2, w1, w2) == case:
                break
    case = line_plane_position(P1, v1, P2, w1, w2)
    eqs, _ = cartesian_from_param(P2, [w1, w2], XYZ)
    (nrm, d0), = eqs
    ans = ChoiceAns(LP_OPTS, LP_KEYS.index(case))
    prompt = (f"Line l = {fparam(P1, [v1])} and plane π: {flin(nrm, XYZ)} = {fs(d0)}.\n"
              "What is their relative position?\n" + ans.render())
    # substitute the line into the plane: n·(P1 + t v1) = d0
    a = sum(x * y for x, y in zip(nrm, v1))
    c = sum(x * y for x, y in zip(nrm, P1))
    steps = [f"Substitute the line into the equation of π: {fs(c)} + ({fs(a)})t = {fs(d0)}."]
    parts = [("position", ans)]
    if a != 0:
        t = (d0 - c) / a
        X = vadd(P1, vscale(t, v1))
        steps.append(f"Unique solution t = {fs(t)} ⇒ incident in {fv(X)}.")
        parts.append(("intersection point", VectorAns(X)))
    elif c == d0:
        steps.append("0 = 0 for every t: the whole line lies in π.")
    else:
        steps.append(f"{fs(c)} = {fs(d0)} is impossible: no common points, l is parallel to π.")
    return mk("affine", "line_plane", prompt, parts, steps, P1=P1, v1=v1, nrm=nrm, d0=d0)


PP_OPTS = ["coincident", "parallel (distinct)", "incident along a line"]


def ex_plane_plane(rng, data=None):
    n1 = ivec(rng, 3, -3, 3)
    d1 = rng.randint(-5, 5)
    case = rng.choice(["coincidenti", "paralleli", "incidenti", "incidenti"])
    if case == "incidenti":
        while True:
            n2 = ivec(rng, 3, -3, 3)
            if span_rank([n1, n2]) == 2:
                break
        d2 = rng.randint(-5, 5)
    else:
        k = rng.choice([1, -1, 2, -2, 3])
        n2 = vscale(k, n1)
        d2 = k * d1 if case == "coincidenti" else k * d1 + nz(rng, -3, 3)
    A = Matrix([n1, n2])
    Ab = A.hstack(Matrix.col_vector([d1, d2]))
    rA, rAb = rank(A), rank(Ab)
    key = "coincidenti" if (rA, rAb) == (1, 1) else "paralleli" if rA == 1 else "incidenti"
    ans = ChoiceAns(PP_OPTS, ["coincidenti", "paralleli", "incidenti"].index(key))
    prompt = (f"π1: {flin(n1, XYZ)} = {d1},   π2: {flin(n2, XYZ)} = {fs(d2)}.\n"
              "Relative position?\n" + ans.render())
    steps = [f"Stack the equations: rg(A) = {rA}, rg(A|b) = {rAb}.",
             {"coincidenti": "rg(A) = rg(A|b) = 1: same plane.",
              "paralleli": "rg(A) = 1 < rg(A|b) = 2: same giacitura, no common points.",
              "incidenti": "rg(A) = rg(A|b) = 2: they meet in a line."}[key]]
    return mk("affine", "plane_plane", prompt, ans, steps, n1=n1, d1=d1, n2=n2, d2=d2)


AFFINE = Topic(
    "affine", "§2.7", "Affine subspaces and their relative positions in R³",
    "Sottospazi affini e posizioni reciproche", "2 Spazi vettoriali",
    ["cart_param", "lines_planes"], AFFINE_LESSON,
    [lambda: ex_line_line(None, ([0, 1, 2], [1, 1, 1], [0, -1, -2], [1, 2, 3]))],
    [Gen("line_line", "line–line", ex_line_line, exam=True, weight=2),
     Gen("line_plane", "line–plane", ex_line_plane, exam=True),
     Gen("plane_plane", "plane–plane", ex_plane_plane)],
)

TOPICS = [VECTOR_SPACES, SPAN_INDEP, BASES, CART_PARAM, SUM_INTERSECTION, AFFINE]
