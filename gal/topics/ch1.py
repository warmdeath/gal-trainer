"""Capitolo 1 — Matrici e sistemi lineari, rango, inversa, determinante."""
from __future__ import annotations

import random
from fractions import Fraction

from ..core.fmt import fm, fs, fsys, hcat, indent, named, named_many, var_names
from ..core.gauss import (det, det_steps, inverse_steps, rank, row_reduce, solve)
from ..core.matrix import Matrix
from ..exercise import (AffineAns, ChoiceAns, IntAns, MatrixAns, ScalarAns, YesNoAns, mk)
from .base import Gen, Topic
from .rand import imat, invertible, ivec, nz, rank_r, unimodular

# ================================================================== matrices
MATRICES_LESSON = [
    ("Intuition", """\
A matrix is a table of numbers: m rows (righe) and n columns (colonne). It stores the
coefficients of a linear system, and — Chapter 3 — it *is* a linear map. Products of
matrices are compositions of maps, which is why the product is "rows times columns"."""),
    ("Definitions", """\
• A field (campo) K: a set with + and · satisfying C1–C10 (Q, R, C; also {0,1}).
  Its elements are scalars (scalari). N and Z are NOT fields.
• An m×n matrix (matrice di tipo (m,n)) over K; the set is M_K(m,n). Entry a_{i,j}:
  row i, column j. Square matrix of order n (matrice quadrata): M_K(n).
  Row vector (vettore riga) M_K(1,n) ≅ K^n, column vector (vettore colonna) M_K(m,1) ≅ K^m,
  zero matrix (matrice nulla) O, identity (matrice identità) I_n, diagonal (diagonale)."""),
    ("Operations (operazioni fra matrici)", """\
• Sum (somma) A + B: only for matrices of the SAME type, entry by entry.
• Scalar product (prodotto per scalare) kA: multiply every entry by k.
• Transpose (trasposta) A^T ∈ M_K(n,m): (A^T)_{i,j} = a_{j,i}. A is symmetric (simmetrica) if A = A^T.
  (A^T)^T = A, (A+B)^T = A^T + B^T, (kA)^T = kA^T, (AB)^T = B^T A^T  ← order reverses!
• Product rows by columns (prodotto righe per colonne): A ∈ M_K(m,p), B ∈ M_K(p,n),
     (AB)_{i,j} = a_{i,1}b_{1,j} + ... + a_{i,p}b_{p,j}   (row i of A · column j of B)
  defined only if #columns of A = #rows of B; AB has as many rows as A and columns as B.
  Associative, distributive, A·I_n = A = I_m·A — but NOT commutative: in general AB ≠ BA,
  and AB = O is possible with A, B ≠ O. Hence (A+B)² = A² + AB + BA + B² ≠ A² + 2AB + B².
• Powers of a square matrix: A⁰ = I_n, A^p = A^{p-1}·A."""),
]


def ex_lincomb(rng, data=None):
    m, n = (2, 3) if data else (rng.randint(2, 3), rng.randint(2, 3))
    A, B, a, b = data or (imat(rng, m, n), imat(rng, m, n), nz(rng, -3, 3), nz(rng, -3, 3))
    R = A.scale(a) + B.scale(b)
    prompt = named_many(("A", A), ("B", B)) + f"\n\nCompute {fs(a)}A {'+' if b > 0 else '-'} {fs(abs(b))}B."
    sol = [hcat(f"{fs(a)}A = ", fm(A.scale(a))), hcat(f"{fs(b)}B = ", fm(B.scale(b))),
           "Sum entry by entry:", fm(R)]
    return mk("matrices", "lincomb", prompt,
              MatrixAns(R, mistakes=[(A + B, "You forgot the scalars."),
                                     (A.scale(a) - B.scale(b), "Sign of the second coefficient is wrong.")]),
              sol, A=A, B=B, a=a, b=b)


def ex_product(rng, data=None):
    if data:
        A, B = data
    else:
        m, p, n = rng.randint(2, 3), rng.randint(2, 3), rng.randint(2, 3)
        A, B = imat(rng, m, p, -3, 3), imat(rng, p, n, -3, 3)
    C = A * B
    prompt = named_many(("A", A), ("B", B)) + f"\n\nA ∈ M(" \
        f"{A.m},{A.n}), B ∈ M({B.m},{B.n}). Compute the product AB (rows by columns)."
    sol = [f"AB is defined: A has {A.n} columns and B has {B.m} rows; AB ∈ M({A.m},{B.n}).",
           "(AB)_{i,j} = (row i of A)·(column j of B):"]
    for i in range(A.m):
        for j in range(B.n):
            terms = " + ".join(f"({fs(A[i, k])})({fs(B[k, j])})" for k in range(A.n))
            sol.append(f"  (AB)_{i + 1},{j + 1} = {terms} = {fs(C[i, j])}")
    sol.append(fm(C))
    mistakes = []
    if B.n == A.m and (B * A).shape == C.shape:
        mistakes.append((B * A, "You computed BA instead of AB: the product is not commutative."))
    if A.shape == B.shape:
        mistakes.append((Matrix([[A[i, j] * B[i, j] for j in range(A.n)] for i in range(A.m)]),
                         "You multiplied entry by entry: the product is rows by columns."))
    if C.shape[::-1] == C.shape:
        mistakes.append((C.T, "You got (AB)^T: you used columns of A with rows of B."))
    return mk("matrices", "product", prompt, MatrixAns(C, mistakes=mistakes), sol, A=A, B=B)


def ex_product_type(rng, data=None):
    dims = [rng.randint(1, 4) for _ in range(4)]
    m, p, q, n = dims
    if rng.random() < 0.6:
        q = p
    defined = p == q
    opts = [f"AB ∈ M({m},{n})", f"AB ∈ M({n},{m})", f"AB ∈ M({p},{q})", "AB is not defined"]
    # remove duplicates while keeping the right one
    correct_txt = opts[0] if defined else opts[3]
    uniq = []
    for o in opts:
        if o not in uniq:
            uniq.append(o)
    rng.shuffle(uniq)
    c = uniq.index(correct_txt)
    why = {i: ("The product needs #columns(A) = #rows(B)." if o == "AB is not defined" or not defined else
               "AB has as many rows as A and as many columns as B.") for i, o in enumerate(uniq)}
    ans = ChoiceAns(uniq, c, why)
    prompt = f"A ∈ M_R({m},{p}), B ∈ M_R({q},{n}). Which is true?\n" + ans.render()
    sol = [f"AB is defined iff the number of columns of A ({p}) equals the number of rows of B ({q}).",
           (f"Here {p} = {q}, so AB is defined and AB ∈ M({m},{n})." if defined else f"Here {p} ≠ {q}: AB is not defined.")]
    return mk("matrices", "ptype", prompt, ans, sol, dims=(m, p, q, n))


def ex_square_of_sum(rng, data=None):
    while True:
        A, B = data or (imat(rng, 2, 2, -2, 2), imat(rng, 2, 2, -2, 2))
        if data or A * B != B * A:
            break
    S = (A + B) * (A + B)
    wrong = A * A + (A * B).scale(2) + B * B
    prompt = named_many(("A", A), ("B", B)) + "\n\nCompute (A + B)²."
    sol = [hcat("A + B = ", fm(A + B)), hcat("(A + B)² = (A + B)(A + B) = ", fm(S)),
           hcat("Note: A² + 2AB + B² = ", fm(wrong), "  is different, because AB ≠ BA:"),
           hcat("AB = ", fm(A * B), "   BA = ", fm(B * A))]
    return mk("matrices", "sqsum", prompt,
              MatrixAns(S, mistakes=[(wrong, "You used (A+B)² = A² + 2AB + B², which needs AB = BA. "
                                             "Matrices do not commute: (A+B)² = A² + AB + BA + B².")]),
              sol, A=A, B=B)


def ex_transpose_product(rng, data=None):
    A, B = data or (imat(rng, 2, 3, -3, 3), imat(rng, 3, 2, -3, 3))
    R = (A * B).T
    prompt = named_many(("A", A), ("B", B)) + "\n\nCompute (AB)^T."
    sol = ["(AB)^T = B^T A^T (the order reverses). Directly:", hcat("AB = ", fm(A * B)),
           hcat("(AB)^T = ", fm(R))]
    return mk("matrices", "transp", prompt,
              MatrixAns(R, mistakes=[(A * B, "You forgot to transpose."),
                                     ((A.T * B.T) if A.T.n == B.T.m and (A.T * B.T).shape == R.shape else A * B,
                                      "(AB)^T = B^T A^T, not A^T B^T.")]),
              sol, A=A, B=B)


MATRICES = Topic(
    "matrices", "§1.1", "Matrices and operations", "Matrici e operazioni fra matrici",
    "1 Matrici e sistemi lineari", ["vectors"], MATRICES_LESSON,
    [lambda: ex_product(None, (Matrix([[1, 3, 2], [0, 2, -4]]), Matrix([[-1, 1], [3, 0], [6, 0]]))),
     lambda: ex_square_of_sum(None, (Matrix([[1, 1], [0, 0]]), Matrix([[1, 0], [1, 0]])))],
    [Gen("lincomb", "aA + bB", ex_lincomb),
     Gen("product", "product AB", ex_product, weight=2),
     Gen("ptype", "when is AB defined?", ex_product_type),
     Gen("sqsum", "(A+B)² and non-commutativity", ex_square_of_sum),
     Gen("transp", "(AB)^T", ex_transpose_product)],
)

# ================================================================== systems
SYSTEMS_LESSON = [
    ("Intuition", """\
A linear system asks: which vectors x satisfy all equations at once? Gaussian elimination
transforms it into an equivalent system (same solutions) that is "triangular" and solvable
by back-substitution. First goal of the course: decide whether a system has solutions
and describe all of them."""),
    ("Definitions", """\
• Linear system (sistema lineare) Ax = b, A ∈ M_K(m,n) coefficient matrix (matrice dei
  coefficienti), b ∈ M_K(m,1) column of constants (colonna dei termini noti), x the unknowns
  (incognite). Complete matrix (matrice completa) (A|b).
• Sol(A|b) = {s ∈ M_K(n,1) : As = b}. Solvable (risolubile) if Sol(A|b) ≠ ∅.
• Echelon form (forma a scala): every non-zero row starts with more zeros than the previous
  one, and zero rows are at the bottom. The first non-zero entry of each row is a pivot.
• Elementary row operations (operazioni elementari sulle righe) — they do not change Sol:
    1. Ri ↔ Rj    2. Ri → kRi (k ≠ 0)    3. Ri → Ri + kRj.
  Shortcut used in the solutions to avoid fractions: Ri → aRi + kRj (a ≠ 0) = move 2 then move 3.
  (It multiplies det by a — irrelevant for solving systems and computing ranks.)"""),
    ("Gaussian elimination (MEG — metodo di eliminazione di Gauss)", """\
Clean the columns one at a time, left to right:
  Step 2: take the first non-zero column j, swap a row with a_{i,j} ≠ 0 to the top (pivot p1);
  Step 3: for each row l below, Rl → Rl - (b_{l,j}/p1)·R1, so the column under the pivot is 0;
  Step 4: restart on the submatrix without the first row.
The echelon form obtained is NOT unique. Gauss-Jordan then makes every pivot 1 and clears the
entries above the pivots (forma a scala ridotta).
Pivot variables are dependent (variabili dipendenti), the others are free (variabili libere):
  Sol(A|b) = v_P + t1·w1 + ... (one parameter per free variable)."""),
]


def _system_data(rng, kind=None):
    """A, b with integer-friendly elimination. kind ∈ unique / inf / none."""
    kind = kind or rng.choice(["unique", "unique", "inf", "inf", "none"])
    m = rng.choice([2, 3, 3, 4]) if kind != "unique" else rng.choice([2, 3, 3])
    n = m if kind == "unique" else rng.choice([m, m + 1]) if kind == "inf" else m
    if kind == "unique":
        A = unimodular(rng, n) if rng.random() < 0.7 else invertible(rng, n, (1, -1, 2, -2, 3))
        x0 = ivec(rng, n, -3, 3, nonzero=False)
        return A, A.apply(x0), kind
    r = rng.randint(1, min(m, n) - 1) if min(m, n) > 1 else 1
    if kind == "inf" and n > m:
        r = rng.randint(max(1, m - 1), m)
    A = rank_r(rng, m, n, r)
    x0 = ivec(rng, n, -2, 2, nonzero=False)
    b = A.apply(x0)
    if kind == "none":
        while True:
            b2 = list(b)
            i = rng.randrange(m)
            b2[i] += nz(rng, -3, 3)
            if rank(A.hstack(Matrix.col_vector(b2))) > rank(A):
                return A, b2, kind
    return A, b, kind


def ex_solve(rng, data=None):
    if data:
        A, b = data
    else:
        A, b, _ = _system_data(rng)
    sol = solve(A, b)
    names = var_names(A.n)
    prompt = ("Solve the linear system with Gaussian elimination:\n" + indent(fsys(A, b, names), 3)
              + f"\nUnknowns ({', '.join(names)}). Write 'none', a vector, or (v_P) + t(w1) + s(w2)...")
    ans = AffineAns(sol.particular, sol.basis, A.n)
    return mk("systems", "solve", prompt, ans, sol.steps, A=A, b=b)


def ex_echelon(rng, data=None):
    m, n = rng.randint(2, 4), rng.randint(3, 4)
    U = row_reduce(rank_r(rng, m, n, rng.randint(1, min(m, n)))).result
    is_ech = True
    if rng.random() < 0.55:
        rows = [list(r) for r in U.rows]
        choice = rng.choice(["swap", "below", "zerotop"])
        if choice == "swap" and m >= 2:
            i = rng.randrange(m - 1)
            rows[i], rows[i + 1] = rows[i + 1], rows[i]
        elif choice == "below":
            i = rng.randrange(1, m)
            j = next((c for c in range(n) if rows[i - 1][c] != 0), 0)
            rows[i][j] = nz(rng, -3, 3)
        else:
            rows = [[0] * n] + rows[:-1]
        M = Matrix(rows)
        is_ech = _is_echelon(M)
    else:
        M = U
    prompt = named("M", M) + "\n\nIs M in echelon form (forma a scala)?"
    sol = (["Yes: each non-zero row starts with more zeros than the previous one and the zero rows are at the bottom.",
            "Pivots: " + ", ".join(fs(M[i, j]) for i, j in _pivots(M))]
           if is_ech else [_why_not_echelon(M)])
    return mk("systems", "echelon", prompt, YesNoAns(is_ech, sol[0]), sol, M=M)


def _leading(row):
    return next((j for j, x in enumerate(row) if x != 0), None)


def _is_echelon(M: Matrix) -> bool:
    prev = -1
    seen_zero = False
    for r in M.rows:
        lead = _leading(r)
        if lead is None:
            seen_zero = True
            continue
        if seen_zero or lead <= prev:
            return False
        prev = lead
    return True


def _pivots(M):
    return [(i, _leading(r)) for i, r in enumerate(M.rows) if _leading(r) is not None]


def _why_not_echelon(M: Matrix) -> str:
    prev = -1
    zero_row = None
    for i, r in enumerate(M.rows):
        lead = _leading(r)
        if lead is None:
            zero_row = zero_row if zero_row is not None else i
            continue
        if zero_row is not None:
            return f"No: row {zero_row + 1} is zero but row {i + 1} below it is not (zero rows go at the bottom)."
        if lead <= prev:
            return f"No: row {i + 1} does not start with more zeros than row {i} (its first non-zero entry is in column {lead + 1})."
        prev = lead
    return "?"


def ex_free_vars(rng, data=None):
    A, b, _ = _system_data(rng, rng.choice(["inf", "unique"]))
    s = solve(A, b)
    prompt = "How many free variables (variabili libere) does this system have?\n" + indent(fsys(A, b), 3)
    return mk("systems", "free", prompt,
              IntAns(A.n - s.rank_A, mistakes=[(s.rank_A, "That is the number of pivots (dependent variables) = rg(A); free = n - rg(A).")]),
              s.steps, A=A, b=b)


SYSTEMS = Topic(
    "systems", "§1.2–1.2.2", "Linear systems and Gaussian elimination",
    "Sistemi lineari, matrici a scala, metodo di eliminazione di Gauss (MEG)",
    "1 Matrici e sistemi lineari", ["matrices"], SYSTEMS_LESSON,
    [lambda: ex_solve(None, (Matrix([[0, -3, -5], [2, -4, 2], [1, -1, 3]]), [1, 6, 5])),
     lambda: ex_solve(None, (Matrix([[2, -6, 2, 2], [-3, 9, 0, 3]]), [6, -9]))],
    [Gen("solve", "solve Ax = b", ex_solve, exam=True, weight=3),
     Gen("echelon", "is it in echelon form?", ex_echelon),
     Gen("free", "number of free variables", ex_free_vars)],
)

# ================================================================== rank & Rouché-Capelli
RANK_LESSON = [
    ("Intuition", """\
The rank counts how many equations are "really different" — how many pivots survive
elimination. It decides everything about a system."""),
    ("Definitions and theorems", """\
• Rank (rango) rg(A): number of pivots of an echelon form of A (= number of non-zero rows).
  It does not depend on the reduction; 0 ≤ rg(A) ≤ min{m, n}; row operations preserve it.
• Teorema di Rouché-Capelli (1.12): Ax = b is solvable ⇔ rg(A) = rg(A|b). If rg(A) = rg(A|b) = r
  the solutions depend on n - r parameters; unique solution ⇔ rg(A) = rg(A|b) = n.
• Homogeneous system (sistema omogeneo) Ax = 0: always the trivial solution (soluzione banale) 0.
• Teorema di struttura (1.14): Sol(A|b) = v_P + Sol(A|0) for any particular solution v_P.
• Lemma 1.15: if m < n the solution is never unique.  Teorema di Cramer (1.16): A square,
  rg(A) = n ⇔ Ax = b has a unique solution for one (hence every) b.
• Teorema degli orlati (1.29): rg(A) = r ⇔ some r×r minor ≠ 0 and every (r+1)×(r+1) minor
  "bordering" it is 0."""),
    ("Systems with a parameter", """\
Reduce (A|b) treating the parameter as a number, and watch the entries that depend on it.
Example of the notes: the last row becomes (0 0 0 0 | 5 + α): solvable iff α = -5."""),
]


def ex_rank(rng, data=None):
    A = data or rank_r(rng, rng.randint(3, 4), rng.randint(3, 4), rng.randint(1, 3))
    red = row_reduce(A)
    r = red.rank
    prompt = named("A", A) + "\n\nCompute rg(A)."
    sol = ["Gaussian elimination:"] + red.steps + [f"{r} pivot(s) ⇒ rg(A) = {r}."]
    mistakes = [(r - 1, "One pivot too few: a row became zero by an arithmetic slip? A row vanishes only if it is a "
                         "combination of the others."),
                (r + 1, "One pivot too many: one of your rows should have become zero (it is a combination of the others)."),
                (A.m, "That's the number of rows; dependent rows vanish in the echelon form.")]
    return mk("rank", "rank", prompt, IntAns(r, mistakes=[m for m in mistakes if m[0] != r]), sol, A=A)


def ex_rouche_b(rng, data=None):
    """b depends on α (notes p.18): solvable iff α = α0."""
    m, n = 3, rng.choice([3, 4])
    A = rank_r(rng, m, n, 2)
    x0 = ivec(rng, n, -2, 2, nonzero=False)
    b = A.apply(x0)
    a0 = nz(rng, -6, 6)
    # find a row i such that removing it keeps rank 2; put α there: b_i = b_i - a0 + α
    for i in range(m):
        rest = Matrix([A.row(k) for k in range(m) if k != i])
        if rank(rest) == 2:
            break
    names = var_names(n)
    shown_b = [fs(b[k]) if k != i else (f"{fs(b[k] - a0)} + α" if b[k] - a0 else "α") for k in range(m)]
    lines = [f"{_lhs(A, k, names)} = {shown_b[k]}" for k in range(m)]
    prompt = ("For which value of the parameter α ∈ R is the system solvable?\n"
              + "\n".join("   " + s for s in lines))
    sol_steps = [f"Row {i + 1} is a combination of the other rows (rg(A) = 2). Reducing (A|b), "
                 f"that row becomes (0 ... 0 | α - ({fs(a0)})).",
                 f"Rouché-Capelli: solvable iff rg(A|b) = rg(A) = 2 iff α - ({fs(a0)}) = 0, i.e. α = {fs(a0)}.",
                 f"For α = {fs(a0)} the solutions depend on n - rg(A) = {n} - 2 = {n - 2} parameter(s)."]
    red = row_reduce(A.hstack(Matrix.col_vector(b)), ncols=n, aug=n)
    sol_steps += ["Check with α = " + fs(a0) + ":"] + red.steps
    return mk("rank", "rouche_b", prompt,
              [("α", ScalarAns(a0, mistakes=[(-a0, "Sign: the last entry is α - α0, it vanishes for α = α0.")])),
               ("number of free parameters for that α", IntAns(n - 2))],
              sol_steps, A=A, b=b, row=i, alpha0=a0)


def _lhs(A, k, names):
    from ..core.fmt import flin
    return flin(A.row(k), names)


def ex_rouche_k(rng, data=None):
    """A_k = A0 + k·E_ij: unique solution iff k ≠ k0; at k0 none or infinitely many."""
    for _ in range(4000):
        A0 = imat(rng, 3, 3, -3, 3)
        i, j = rng.randrange(3), rng.randrange(3)
        C = (-1) ** (i + j) * det(A0.minor(i, j))
        if C == 0:
            continue
        d0 = det(A0)
        if d0 % C:
            continue
        k0 = -d0 / C
        if abs(k0) > 5 or k0 == 0:
            continue
        Ak0 = Matrix([[A0[r, c] + (k0 if (r, c) == (i, j) else 0) for c in range(3)] for r in range(3)])
        if rank(Ak0) != 2:
            continue
        b = ivec(rng, 3, -4, 4)
        if rng.random() < 0.5:  # make it solvable at k0
            b = Ak0.apply(ivec(rng, 3, -2, 2))
        break
    rA = rank(Ak0)
    rAb = rank(Ak0.hstack(Matrix.col_vector(b)))
    names = ["x", "y", "z"]
    rows = []
    for r in range(3):
        s = ""
        for c in range(3):
            if (r, c) == (i, j):
                base = A0[r, c]
                term = f"(k {'+' if base >= 0 else '-'} {fs(abs(base))}){names[c]}" if base else f"k{names[c]}"
                s += (" + " if s else "") + term
            elif A0[r, c]:
                a = A0[r, c]
                coef = "" if abs(a) == 1 else fs(abs(a))
                if s:
                    s += (" - " if a < 0 else " + ") + coef + names[c]
                else:
                    s = ("-" if a < 0 else "") + coef + names[c]
        rows.append(f"{s} = {fs(b[r])}")
    prompt = ("Consider the system depending on k ∈ R:\n" + "\n".join("   " + s for s in rows) +
              "\n(1) For which k does it NOT have a unique solution?\n(2) For that k, what happens?")
    opts = ["no solutions", "infinitely many solutions (∞^1)", "infinitely many solutions (∞^2)"]
    correct = 0 if rAb > rA else 1
    sol = [f"Square system: unique solution ⇔ det(A_k) ≠ 0 (Cramer + Teorema 1.28).",
           f"Expanding along the entry ({i + 1},{j + 1}) that contains k: det(A_k) = {fs(d0)} + k·({fs(C)}).",
           f"det(A_k) = 0 ⇔ k = {fs(k0)}.",
           f"For k = {fs(k0)}: rg(A) = {rA}, rg(A|b) = {rAb} ⇒ " +
           ("no solutions (Rouché-Capelli)." if rAb > rA else f"∞^{3 - rA} solutions.")]
    red = row_reduce(Ak0.hstack(Matrix.col_vector(b)), ncols=3, aug=3)
    sol += red.steps
    return mk("rank", "rouche_k", prompt,
              [("(1) k", ScalarAns(k0)), ("(2)", ChoiceAns(opts, correct))],
              sol, A0=A0, ij=(i, j), b=b, k0=k0)


RANK = Topic(
    "rank", "§1.2.3", "Rank and the Rouché-Capelli theorem",
    "Rango, Teorema di Rouché-Capelli e Teorema di struttura", "1 Matrici e sistemi lineari",
    ["systems"], RANK_LESSON,
    [lambda: ex_rank(None, Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))],
    [Gen("rank", "rank of a matrix", ex_rank, weight=2),
     Gen("rouche_b", "parameter in the constants", ex_rouche_b, exam=True),
     Gen("rouche_k", "parameter in the coefficients", ex_rouche_k, exam=True)],
)

# ================================================================== inverse
INVERSE_LESSON = [
    ("Intuition", """\
A^-1 "undoes" A, like 1/k undoes k. Not every non-zero matrix can be undone: if A squashes
some vector to 0, nothing can bring it back."""),
    ("Definition and theorem", """\
• A ∈ M_K(n) is invertible (invertibile) if ∃B with AB = BA = I_n; B = A^-1 (matrice inversa) is unique.
  Non-square matrices are never invertible.
• Teorema 1.18/1.28: A invertible ⇔ rg(A) = n ⇔ Ax = b has a unique solution ∀b ⇔ det(A) ≠ 0.
• (AB)^-1 = B^-1 A^-1,   (A^T)^-1 = (A^-1)^T."""),
    ("How to compute it", """\
Solve the n systems AX = I_n at once with Gauss-Jordan on (A | I_n):
   (A | I_n) → (I_n | B)  ⇒ A^-1 = B;   if a zero row appears on the left, A is NOT invertible.
For 2×2: (a b; c d)^-1 = 1/(ad - bc) · (d -b; -c a). Always check AB = I."""),
]


def ex_inverse(rng, data=None):
    A = data or invertible(rng, rng.choice([2, 3, 3]))
    inv, steps = inverse_steps(A)
    n = A.n
    mistakes = [(inv.T, "You got the transpose of A^-1."),
                (inv.scale(det(A)), "You forgot to divide by det(A).")]
    prompt = named("A", A) + "\n\nCompute A^-1."
    return mk("inverse", "inverse", prompt, MatrixAns(inv, mistakes=mistakes),
              steps + ["Check: A·A^-1 = I."], A=A)


def ex_invertible(rng, data=None):
    n = rng.choice([2, 3])
    if rng.random() < 0.5:
        A = rank_r(rng, n, n, n - 1, bound=6)
    else:
        A = invertible(rng, n, (1, -1, 2, -2, 3, -3, 4, 5))
    d = det(A)
    prompt = named("A", A) + "\n\nIs A invertible?"
    sol = det_steps(A) + [f"det(A) = {fs(d)} " + ("≠ 0 ⇒ A is invertible." if d else "= 0 ⇒ A is NOT invertible (rg(A) < n).")]
    return mk("inverse", "invertible", prompt, YesNoAns(d != 0, sol[-1]), sol, A=A)


INVERSE = Topic(
    "inverse", "§1.2.4", "Inverse matrix", "Matrice inversa", "1 Matrici e sistemi lineari",
    ["rank"], INVERSE_LESSON,
    [lambda: ex_inverse(None, Matrix([[1, 2], [3, 4]])),
     lambda: ex_inverse(None, Matrix([[1, 0, 1], [2, 1, 6], [3, 0, 2]]))],
    [Gen("inverse", "compute A^-1", ex_inverse, exam=True, weight=2),
     Gen("invertible", "is A invertible?", ex_invertible)],
)

# ================================================================== determinant
DET_LESSON = [
    ("Intuition", """\
det(A) is the signed area (n = 2) or signed volume (n = 3) of the parallelogram /
parallelepiped spanned by the columns of A. Volume 0 means the columns are flat (dependent):
exactly the non-invertible case."""),
    ("Definition (Definizione 1.20)", """\
det: M_K(n) → K, recursively: det(a) = a;  for n ≥ 2, Laplace expansion (espansione di Laplace)
along a row i:   det(A) = Σ_j (-1)^{i+j} a_{i,j} det(Â_{i,j}),
Â_{i,j} = A without row i and column j. It does not depend on the row chosen, and
det(A) = det(A^T), so you can expand along a column too. Choose the line with most zeros!
  2×2: det(a b; c d) = ad - bc.     Triangular (triangolare) A: det = a11·a22···ann."""),
    ("Properties", """\
• Row operations: Ri ↔ Rj changes the sign; Ri → kRi multiplies det by k;
  Ri → Ri + kRj does not change det.  Hence det(λA) = λ^n det(A).
• Multilinear (multilineare) and alternating (alternante) in the columns.
• Formula di Binet: det(AB) = det(A)·det(B);  det(A^-1) = 1/det(A).
• A invertible ⇔ det(A) ≠ 0 (Teorema 1.28)."""),
]


def _sparse_matrix(rng, n):
    while True:
        A = imat(rng, n, n, -3, 3)
        rows = [list(r) for r in A.rows]
        z = n * n // 3
        for _ in range(z):
            rows[rng.randrange(n)][rng.randrange(n)] = 0
        M = Matrix(rows)
        if det(M) != 0 or rng.random() < 0.15:
            return M


def ex_det(rng, data=None):
    A = data or _sparse_matrix(rng, rng.choice([3, 3, 4]))
    d = det(A)
    no_sign = sum(A[0, j] * det(A.minor(0, j)) for j in range(A.n)) if A.n > 1 else d
    mistakes = [(no_sign, "You forgot the signs (-1)^(i+j) in the Laplace expansion (pattern + - + ...).")] if no_sign != d else []
    prompt = named("A", A) + "\n\nCompute det(A)."
    return mk("determinant", "det", prompt, ScalarAns(d, mistakes=mistakes), det_steps(A), A=A)


def ex_det_rowops(rng, data=None):
    n = rng.choice([3, 4])
    d = nz(rng, -6, 6)
    ops, factor, texts = [], Fraction(1), []
    for _ in range(rng.randint(2, 3)):
        kind = rng.choice(["swap", "scale", "add"])
        i, j = rng.sample(range(1, n + 1), 2)
        if kind == "swap":
            factor *= -1
            texts.append(f"R{i} ↔ R{j}")
        elif kind == "scale":
            k = nz(rng, -3, 3)
            factor *= k
            texts.append(f"R{i} → {fs(k)}R{i}")
        else:
            k = nz(rng, -3, 3)
            texts.append(f"R{i} → R{i} {'+' if k > 0 else '-'} {fs(abs(k))}R{j}")
    res = factor * d
    prompt = (f"A ∈ M_R({n}) has det(A) = {fs(d)}. The matrix B is obtained from A with the row operations\n"
              f"   {',  '.join(texts)}   (in this order). Compute det(B).")
    sol = ["Ri ↔ Rj: det changes sign;  Ri → kRi: det is multiplied by k;  Ri → Ri + kRj: det unchanged."]
    sol.append(f"det(B) = {fs(factor)}·det(A) = {fs(res)}")
    return mk("determinant", "rowops", prompt, ScalarAns(res), sol, d=d, factor=factor, texts=texts)


def ex_det_binet(rng, data=None):
    n = rng.choice([2, 3, 4])
    dA, dB = (data or (nz(rng, -3, 3), nz(rng, -3, 3)))
    lam = rng.choice([2, -1, 3, Fraction(1, 2), -2])
    p, q = rng.randint(1, 3), rng.choice([-2, -1, 1, 2])
    tr = rng.random() < 0.5
    val = Fraction(lam) ** n * Fraction(dA) ** p * Fraction(dB) ** q * (Fraction(dA) if tr else 1)
    pw = lambda X, e: X if e == 1 else f"{X}^{e}" if e > 0 else f"{X}^({e})"
    expr = f"{fs(lam) if Fraction(lam).denominator == 1 else '(' + fs(lam) + ')'}·{pw('A', p)}·{pw('B', q)}" + ("·A^T" if tr else "")
    prompt = (f"A, B ∈ M_R({n}) are invertible with det(A) = {dA}, det(B) = {dB}.\n"
              f"Compute det({expr}).")
    wrong = Fraction(lam) * Fraction(dA) ** p * Fraction(dB) ** q * (Fraction(dA) if tr else 1)
    sol = ["Binet: det(XY) = det(X)det(Y);  det(B^q) = det(B)^q (also for q < 0);  det(A^T) = det(A);",
           f"det(λX) = λ^n det(X) with n = {n}.",
           f"= ({fs(lam)})^{n} · ({dA})^{p} · ({dB})^{q}" + (f" · ({dA})" if tr else "") + f" = {fs(val)}"]
    return mk("determinant", "binet", prompt,
              ScalarAns(val, mistakes=[(wrong, f"det(λA) = λ^n det(A): the scalar must be raised to n = {n}.")]),
              sol, dA=dA, dB=dB, lam=Fraction(lam), p=p, q=q, tr=tr, n=n)


DETERMINANT = Topic(
    "determinant", "§1.3", "Determinant", "Determinante", "1 Matrici e sistemi lineari",
    ["inverse"], DET_LESSON,
    [lambda: ex_det(None, Matrix([[4, 0, 3], [1, 2, 0], [7, 3, 0]])),
     lambda: ex_det(None, Matrix([[0, 0, -3, 5], [1, 8, 6, 7], [2, 0, 4, 7], [3, 0, 6, 9]]))],
    [Gen("det", "det by Laplace expansion", ex_det, exam=True, weight=3),
     Gen("rowops", "det and row operations", ex_det_rowops),
     Gen("binet", "Binet and det(λA)", ex_det_binet)],
)

TOPICS = [MATRICES, SYSTEMS, RANK, INVERSE, DETERMINANT]
