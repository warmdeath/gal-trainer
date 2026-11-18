"""Capitolo 4 — Endomorfismi e diagonalizzazione."""
from __future__ import annotations

import random
from fractions import Fraction

from ..core.fmt import fm, fs, fv, fvs, hcat, indent, named
from ..core.gauss import det_steps, inverse, kernel_basis, rank, row_reduce, solve
from ..core.matrix import Matrix, vscale
from ..core.poly import Poly, charpoly, factored
from ..exercise import (ChoiceAns, DiagonalizerAns, IntAns, MatrixAns, PolyAns, ScalarAns,
                        ScalarSetAns, SpanAns, YesNoAns, mk)
from .base import Gen, Topic
from .rand import imat, ivec, nz, unimodular

EIGEN_LESSON = [
    ("Intuition", """\
An eigenvector is a direction that f does not rotate: f only stretches it by a factor λ, the
eigenvalue. If we find a whole basis of such directions, in that basis f is just "stretch
axis i by λi": a diagonal matrix. Example: a reflection across a line l has eigenvalue 1 on l
and -1 on the perpendicular direction."""),
    ("Definitions (§4.1)", """\
• λ ∈ K is an autovalore of f ∈ End(V) if f(v) = λv for some v ≠ 0; such v ≠ 0 is an
  autovettore (with eigenvalue λ). Spettro S(f) = set of eigenvalues.
• 0 is an eigenvalue ⇔ ker f ≠ {0} ⇔ f not injective.
• λ eigenvalue ⇔ ker(λ·id - f) ≠ {0} ⇔ rg(λI - A) < n ⇔ det(λI - A) = 0  (A = M^B_B(f)).
  The eigenvectors of λ are the non-zero solutions of (λI - A)x = 0."""),
    ("Characteristic polynomial (§4.2)", """\
• Polinomio caratteristico p_A(t) = det(t·I_n - A): monic of degree n,
     p_A(t) = t^n - tr(A) t^{n-1} + ... + (-1)^n det(A);   2×2: t² - (a+d)t + (ad - bc).
  Similar matrices have the same p_A (Lemma 4.6), so p_f is well defined.
• Eigenvalues = roots of p_A IN K. Attenzione al campo: t² + 1 has no real roots, t² - 2 has
  no rational roots. (Some authors use det(A - tI) = (-1)^n p_A(t).)"""),
]


def _small(A: Matrix, bound: int = 7) -> bool:
    """Small integer entries and not diagonal (diagonal matrices make trivial exercises)."""
    off = any(A[i, j] != 0 for i in range(A.n) for j in range(A.n) if i != j)
    return off and all(abs(x) <= bound and x.denominator == 1 for r in A.rows for x in r)


def lamI(lam) -> str:
    """'I', '-I', '3I', '(1/2)I' — λ·I written nicely."""
    lam = Fraction(lam)
    if lam == 1:
        return "I"
    if lam == -1:
        return "-I"
    return f"({fs(lam)})I" if lam.denominator != 1 else f"{fs(lam)}I"


def tfactor(lam) -> str:
    lam = Fraction(lam)
    return "t" if lam == 0 else f"(t {'-' if lam > 0 else '+'} {fs(abs(lam))})"


def _nice_char_matrix(rng, n):
    """Small integer matrix with integer eigenvalues (A = P T P^-1, T triangular)."""
    for _ in range(500):
        P = unimodular(rng, n, steps=n + 1, bound=2)
        T = [[0] * n for _ in range(n)]
        for i in range(n):
            T[i][i] = rng.randint(-3, 3)
            for j in range(i + 1, n):
                T[i][j] = rng.choice([0, 0, 1, -1])
        A = P * Matrix(T) * inverse(P)
        if _small(A):
            return A
    return A


def ex_charpoly(rng, data=None):
    A = data or (_nice_char_matrix(rng, rng.choice([2, 3, 3])) if rng.random() < 0.8 else imat(rng, 3, 3, -2, 2))
    p = charpoly(A)
    n = A.n
    tI_A = "tI - A"
    steps = [f"p_A(t) = det(tI - A), expanding the {n}×{n} determinant with entries t - a_ii on the diagonal:"]
    if n == 2:
        steps.append(f"= {tfactor(A[0, 0])}{tfactor(A[1, 1])} - ({fs(A[0, 1])})({fs(A[1, 0])}) "
                     f"= t² - tr(A)t + det(A) = t² - ({fs(A.trace())})t + ({fs(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])})")
    steps.append(f"p_A(t) = {p}  =  {factored(p)}")
    steps.append(f"Check: coefficient of t^{n - 1} is -tr(A) = {fs(-A.trace())}, constant term (-1)^n det(A) = {fs(p.c[0])}.")
    mistakes = [(-p, "You computed det(A - tI) = (-1)^n p_A(t); the notes use det(tI - A).")] if n % 2 else []
    return mk("eigen", "charpoly", named("A", A) + "\n\nCompute the characteristic polynomial p_A(t).",
              PolyAns(p, mistakes=mistakes), steps, A=A)


def ex_eigenvalues(rng, data=None):
    if data:
        A = data
    elif rng.random() < 0.25:
        # a real root plus an irreducible quadratic: (t - a)(t² + b)
        a, b = rng.randint(-3, 3), rng.randint(1, 4)
        p = Poly([-a, 1]) * Poly([b, 0, 1])
        A = _companion(p)
        P = unimodular(rng, 3, bound=2)
        A = P * A * inverse(P)
    else:
        A = _nice_char_matrix(rng, rng.choice([2, 3]))
    p = charpoly(A)
    roots, rest = p.rational_roots()
    prompt = named("A", A) + "\n\nFind the spectrum S(A) over R (the real eigenvalues; each value once)."
    steps = [f"p_A(t) = det(tI - A) = {p} = {factored(p)}",
             "Real roots: " + (", ".join(f"{fs(r)} (ma = {k})" for r, k in sorted(roots.items())) or "none")]
    if rest.degree >= 1:
        steps.append(f"The factor {rest} has no real roots (its discriminant is negative): no further real eigenvalues.")
    return mk("eigen", "eigenvalues", prompt, ScalarSetAns(list(roots), "S(A)"), steps, A=A)


def ex_eigenspace(rng, data=None):
    if data:
        A, lam = data
    else:
        A = _nice_char_matrix(rng, rng.choice([2, 3, 3]))
        roots, _ = charpoly(A).rational_roots()
        lam = rng.choice(sorted(roots))
    M = Matrix.identity(A.n).scale(lam) - A
    K = kernel_basis(M)
    sol = solve(M, [0] * A.n)
    prompt = named("A", A) + f"\n\nλ = {fs(lam)} is an eigenvalue of A. Find a basis of the eigenspace (autospazio) V_λ."
    steps = [hcat(f"V_{fs(lam)} = ker({lamI(lam)} - A),   {lamI(lam)} - A = ", fm(M))] + sol.steps + [
        f"V_{fs(lam)} = Span({fvs(K)}),  mg({fs(lam)}) = {len(K)}"]
    return mk("eigen", "eigenspace", prompt, SpanAns(K, A.n, f"V_{fs(lam)}"), steps, A=A, lam=lam)


def ex_is_eigenvector(rng, data=None):
    A = _nice_char_matrix(rng, 3)
    roots, _ = charpoly(A).rational_roots()
    if rng.random() < 0.6:
        lam = rng.choice(sorted(roots))
        v = kernel_basis(Matrix.identity(3).scale(lam) - A)[0]
    else:
        v = ivec(rng, 3, -2, 2)
    Av = A.apply(v)
    is_eig = rank(Matrix([Av, v])) == 1
    lam = next((a / b for a, b in zip(Av, v) if b != 0), None) if is_eig else None
    prompt = named("A", A) + f"\n\nIs v = {fv(v)} an eigenvector of A? If so, with which eigenvalue?"
    steps = [f"Av = {fv(Av)}"]
    if is_eig:
        steps.append(f"Av = {fs(lam)}·v ⇒ eigenvector with eigenvalue {fs(lam)}.")
        parts = [("eigenvector?", YesNoAns(True, steps[-1])), ("eigenvalue", ScalarAns(lam))]
    else:
        steps.append("Av is not a multiple of v ⇒ not an eigenvector.")
        parts = [("eigenvector?", YesNoAns(False, steps[-1]))]
    return mk("eigen", "is_eigvec", prompt, parts, steps, A=A, v=v)


EIGEN = Topic(
    "eigen", "§4.1–4.2", "Eigenvalues and eigenvectors, characteristic polynomial",
    "Autovettori ed autovalori, polinomio caratteristico", "4 Endomorfismi e diagonalizzazione",
    ["rep_matrix", "determinant"], EIGEN_LESSON,
    [lambda: ex_charpoly(None, Matrix([[3, -2, -1], [0, 0, 1], [2, -2, -1]])),
     lambda: ex_eigenspace(None, (Matrix([[4, -3, -3], [1, 0, -1], [1, -1, 0]]), 1))],
    [Gen("charpoly", "characteristic polynomial", ex_charpoly, weight=2),
     Gen("eigenvalues", "spectrum over R", ex_eigenvalues, exam=True, weight=2),
     Gen("eigenspace", "basis of an eigenspace", ex_eigenspace, exam=True, weight=2),
     Gen("is_eigvec", "is v an eigenvector?", ex_is_eigenvector)],
)

# ================================================================== diagonalization
DIAG_LESSON = [
    ("Intuition", """\
Diagonalisable = there are enough independent eigen-directions to make a basis. Each
eigenvalue λ "promises" ma(λ) directions (its multiplicity as a root) but only delivers
mg(λ) = dim V_λ of them. Diagonalisable ⇔ every promise is kept and the roots are all in K."""),
    ("Definitions (§4.3)", """\
• f is diagonalizzabile if V has a basis of eigenvectors of f; equivalently (1° criterio,
  Teorema 4.4) A = M^B_B(f) is similar to a diagonal matrix: D = P^-1 A P.
• Autospazio V_λ = ker(λ·id - f).  Molteplicità geometrica mg(λ) = dim V_λ = n - rg(λI - A).
  Molteplicità algebrica ma(λ) = exponent of (t - λ) in p_f(t).
• Dominazione (Teorema 4.10): 1 ≤ mg(λ) ≤ ma(λ) ≤ n.  λ regolare if ma = mg, semplice if ma = 1
  (simple ⇒ regular).
• Eigenvectors of distinct eigenvalues are independent (Prop. 4.13).
• 2° criterio (Teorema 4.14): f diagonalisable ⇔ Σ ma(λi) = n and ma(λi) = mg(λi) ∀i ⇔ Σ mg(λi) = n.
  Corollario 4.15: all roots in K and all simple ⇒ diagonalisable."""),
    ("Algorithm (§4.4 Sintesi)", """\
0) A = M^B_B(f). 1) p_A(t). 2) Factor it: eigenvalues and ma. Check 1: Σ ma < n ⇒ NOT diag.
3) For each λ: basis of V_λ, mg(λ). Check 2: mg < ma for some λ ⇒ NOT diag.
Output: P = (eigenvectors as COLUMNS, grouped by eigenvalue), D = diag(λ1, ..., λn) in the
same order, and D = P^-1 A P.
Complementi (§4.5): the matrice compagna of p(t) = t^n + a_{n-1}t^{n-1} + ... + a0 is
C(p) with 1's under the diagonal and last column (-a0, -a1, ..., -a_{n-1})^T; p_{C(p)} = p."""),
]


def _diag_matrix(rng, n, diagonalizable=True):
    for _ in range(500):
        P = unimodular(rng, n, steps=n + 1, bound=2)
        vals = [rng.randint(-3, 3) for _ in range(n)]
        if (rng.random() < 0.5 or not diagonalizable) and n >= 2:
            vals[1] = vals[0]  # repeated eigenvalue
        T = Matrix.diag(vals).tolist()
        if not diagonalizable:
            T[0][1] = 1
        A = P * Matrix(T) * inverse(P)
        if _small(A):
            return A, P, vals
    return A, P, vals


def _companion(p: Poly) -> Matrix:
    n = p.degree
    rows = [[Fraction(0)] * n for _ in range(n)]
    for i in range(1, n):
        rows[i][i - 1] = Fraction(1)
    for i in range(n):
        rows[i][n - 1] = -p.c[i]
    return Matrix(rows)


def multiplicities(A: Matrix):
    p = charpoly(A)
    roots, rest = p.rational_roots()
    mg = {lam: len(kernel_basis(Matrix.identity(A.n).scale(lam) - A)) for lam in roots}
    return p, roots, rest, mg


def _diag_analysis(A: Matrix) -> tuple[bool, list[str]]:
    p, roots, rest, mg = multiplicities(A)
    n = A.n
    steps = [f"p_A(t) = {p} = {factored(p)}"]
    steps.append("Eigenvalues: " + (", ".join(f"{fs(l)} (ma = {k})" for l, k in sorted(roots.items())) or "none in R"))
    if sum(roots.values()) < n:
        steps.append(f"Check 1 fails: Σ ma = {sum(roots.values())} < {n} (p_A has non-real roots) ⇒ NOT diagonalisable over R.")
        return False, steps
    steps.append(f"Check 1 OK: Σ ma = {n}.")
    for lam in sorted(roots):
        M = Matrix.identity(n).scale(lam) - A
        steps.append(f"mg({fs(lam)}) = {n} - rg({lamI(lam)} - A) = {n} - {rank(M)} = {mg[lam]}")
    bad = [l for l in roots if mg[l] < roots[l]]
    if bad:
        steps.append(f"Check 2 fails: mg({fs(bad[0])}) = {mg[bad[0]]} < ma = {roots[bad[0]]} ⇒ NOT diagonalisable.")
        return False, steps
    steps.append("Every eigenvalue is regular ⇒ diagonalisable (2° criterio).")
    return True, steps


def ex_diagonalizable(rng, data=None):
    if data:
        A = data
    else:
        n = 3
        kind = rng.choice(["yes", "yes", "no", "complex"])
        if kind == "complex":
            p = Poly([-rng.randint(-2, 2), 1]) * Poly([rng.randint(1, 4), rng.choice([0, 0, 1]) , 1])
            if p.rational_roots()[1].degree < 2:
                p = Poly([-1, 1]) * Poly([1, 0, 1])
            P = unimodular(rng, 3, bound=2)
            A = P * _companion(p) * inverse(P)
        else:
            A, _, _ = _diag_matrix(rng, n, kind == "yes")
    ok, steps = _diag_analysis(A)
    return mk("diagonalization", "diagonalizable", named("A", A) + "\n\nIs A diagonalisable over R?",
              YesNoAns(ok, steps[-1]), steps, A=A)


def ex_multiplicities(rng, data=None):
    A = data or _diag_matrix(rng, 3, rng.random() < 0.5)[0]
    p, roots, rest, mg = multiplicities(A)
    lam = max(roots, key=lambda l: (roots[l], l))
    M = Matrix.identity(A.n).scale(lam) - A
    prompt = named("A", A) + f"\n\np_A(t) = {factored(p)}. For λ = {fs(lam)} compute ma(λ) and mg(λ)."
    red = row_reduce(M)
    steps = [f"ma({fs(lam)}) = exponent of {tfactor(lam)} = {roots[lam]}.",
             hcat(f"{lamI(lam)} - A = ", fm(M))] + red.steps + [
        f"rg = {red.rank} ⇒ mg = {A.n} - {red.rank} = {mg[lam]}."]
    return mk("diagonalization", "mult", prompt,
              [("ma", IntAns(roots[lam])),
               ("mg", IntAns(mg[lam], mistakes=[(red.rank, "That's rg(λI - A); mg = n - rg(λI - A).")]))],
              steps, A=A, lam=lam)


def ex_find_P(rng, data=None):
    if data:
        A = data
    else:
        while True:
            A, _, vals = _diag_matrix(rng, rng.choice([2, 3, 3]), True)
            if len(set(vals)) >= 2:
                break
    p, roots, rest, mg = multiplicities(A)
    cols, diag = [], []
    steps = [f"p_A(t) = {factored(p)}"]
    for lam in sorted(roots):
        K = kernel_basis(Matrix.identity(A.n).scale(lam) - A)
        steps.append(f"V_{fs(lam)} = ker({lamI(lam)} - A) = Span({fvs(K)})")
        cols += K
        diag += [lam] * len(K)
    P = Matrix.from_cols(cols)
    steps.append(hcat("P = ", fm(P), "   D = P^-1 A P = ", fm(Matrix.diag(diag))))
    return mk("diagonalization", "find_P", named("A", A) +
              "\n\nA is diagonalisable. Find an invertible P such that P^-1 A P is diagonal.",
              DiagonalizerAns(A, P), steps, A=A)


def ex_param_diag(rng, data=None):
    a, b = rng.sample(range(-3, 4), 2)
    # A_k = [[a, 1, 0], [0, k, 0], [0, 0, b]], conjugated by a permutation for variety
    perm = rng.choice([(0, 1, 2), (2, 0, 1), (1, 2, 0)])
    base = [[a, 1, 0], [0, None, 0], [0, 0, b]]
    rows = [[base[perm[i]][perm[j]] for j in range(3)] for i in range(3)]
    txt = Matrix([[0 if x is None else x for x in r] for r in rows])
    shown = "\n".join("   [ " + "  ".join(("k" if x is None else fs(x)).rjust(2) for x in r) + " ]" for r in rows)
    prompt = (f"For k ∈ R let\nA_k =\n{shown}\n"
              "A_k is diagonalisable for all k except one value. Which?")
    steps = [f"The matrix is (up to reordering the basis) triangular: eigenvalues {a}, k, {b}.",
             f"k ∉ {{{a}, {b}}}: three distinct eigenvalues ⇒ diagonalisable (Cor. 4.15).",
             f"k = {b}: λ = {b} has ma = 2 and rg({b}I - A) = 1 ⇒ mg = 2: diagonalisable.",
             f"k = {a}: λ = {a} has ma = 2 but rg({a}I - A) = 2 ⇒ mg = 1 < 2: NOT diagonalisable.",
             f"Answer: k = {a}."]
    return mk("diagonalization", "param", prompt,
              ScalarAns(a, mistakes=[(b, f"For k = {b} the double eigenvalue {b} is regular (mg = 2).")]),
              steps, rows=rows, a=a, b=b)


def ex_companion(rng, data=None):
    n = rng.choice([2, 3, 3])
    p = data or Poly(ivec(rng, n, -4, 4, nonzero=False) + [1])
    C = _companion(p)
    prompt = f"Write the companion matrix (matrice compagna) C(p) of p(t) = {p}."
    steps = ["C(p): 1's just below the diagonal, last column = (-a0, -a1, ..., -a_{n-1})^T:",
             fm(C), f"Check: p_C(t) = {charpoly(C)}."]
    return mk("diagonalization", "companion", prompt,
              MatrixAns(C, mistakes=[(C.T, "Transposed: the notes put the 1's below the diagonal and the coefficients in the last COLUMN.")]),
              steps, p=p)


DIAGONALIZATION = Topic(
    "diagonalization", "§4.3–4.5", "Multiplicities and diagonalisation",
    "Molteplicità e diagonalizzazione", "4 Endomorfismi e diagonalizzazione", ["eigen"], DIAG_LESSON,
    [lambda: ex_find_P(None, Matrix([[4, -3, -3], [1, 0, -1], [1, -1, 0]])),
     lambda: ex_diagonalizable(None, Matrix([[3, -1, -2], [1, 0, -1], [0, 1, 1]]))],
    [Gen("diagonalizable", "is A diagonalisable?", ex_diagonalizable, exam=True, weight=2),
     Gen("mult", "ma and mg", ex_multiplicities, weight=2),
     Gen("find_P", "find P with P^-1AP diagonal", ex_find_P, exam=True, weight=2),
     Gen("param", "diagonalisable for which k?", ex_param_diag, exam=True),
     Gen("companion", "companion matrix", ex_companion)],
)

TOPICS = [EIGEN, DIAGONALIZATION]
