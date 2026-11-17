"""Gaussian elimination (MEG), Gauss-Jordan and everything built on them.

Every routine can return the list of steps in the notation of the notes:
R1 ↔ R3,  R2 → -1/2·R2,  R3 → R3 + 3R2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

from .fmt import fm, fs, hcat, indent
from .matrix import Matrix, clear_denominators, vec


def _op_swap(i, j):
    return f"R{i + 1} ↔ R{j + 1}"


def _op_scale(i, k):
    ks = fs(k)
    if "/" in ks or ks.startswith("-"):
        ks = f"({ks})"
    return f"R{i + 1} → {ks}·R{i + 1}"


def _op_add(i, k, j):
    a = abs(k)
    coef = "" if a == 1 else (f"({fs(a)})" if a.denominator != 1 else fs(a))
    sign = "-" if k < 0 else "+"
    return f"R{i + 1} → R{i + 1} {sign} {coef}R{j + 1}"


@dataclass
class Reduction:
    result: Matrix
    pivots: list[tuple[int, int]]  # (row, col) of each pivot
    steps: list[str] = field(default_factory=list)
    swaps: int = 0
    scale: Fraction = Fraction(1)  # product of the scalars used in Ri → kRi

    @property
    def rank(self) -> int:
        return len(self.pivots)


def _op_combo(i, a, k, j):
    """Ri → a·Ri + k·Rj (a ≠ 0): move 2 followed by move 3, avoids fractions."""
    sign = "-" if k < 0 else "+"
    kk = abs(k)
    return f"R{i + 1} → {fs(a)}R{i + 1} {sign} {'' if kk == 1 else fs(kk)}R{j + 1}"


def row_reduce(M: Matrix, *, jordan: bool = False, ncols: int | None = None,
               aug: int | None = None, integer: bool = True) -> Reduction:
    """MEG on M. Pivots are searched only in the first `ncols` columns.

    jordan=True continues to the reduced echelon form (forma a scala ridotta).
    `aug` is only used to draw the bar of a complete matrix (A|b) in the steps.
    integer=True avoids fractions in the forward phase when the entries are integers,
    using Rl → p·Rl - b·Rr (allowed: it is Rl → pRl followed by Rl → Rl - bRr). This
    changes det by the factor p, so det() calls it with integer=False.
    """
    A = M.tolist()
    m, n = M.m, M.n
    ncols = n if ncols is None else ncols
    steps: list[str] = []
    pivots: list[tuple[int, int]] = []
    swaps = 0
    scale = Fraction(1)

    def snap(ops: list[str]):
        if ops:
            steps.append(",  ".join(ops) + "\n" + indent(fm(A, aug)))

    r = 0
    for j in range(ncols):
        if r >= m:
            break
        cand = [i for i in range(r, m) if A[i][j] != 0]
        if not cand:
            continue

        def key(i):
            x = A[i][j]
            # prefer pivot 1, then -1, then integers of small size, then current row
            return (x != 1, x != -1, x.denominator != 1, abs(x), i)

        p = min(cand, key=key)
        ops = []
        if p != r:
            A[r], A[p] = A[p], A[r]
            swaps += 1
            ops.append(_op_swap(r, p))
            snap(ops)
            ops = []
        piv = A[r][j]
        for i in range(r + 1, m):
            if A[i][j] != 0:
                k = -A[i][j] / piv
                all_int = all(x.denominator == 1 for x in A[i] + A[r])
                if integer and all_int and k.denominator != 1:
                    from math import gcd
                    g = gcd(int(piv), int(A[i][j]))
                    a, c = Fraction(int(piv) // g), Fraction(-int(A[i][j]) // g)
                    if a < 0:
                        a, c = -a, -c
                    A[i] = [a * x + c * y for x, y in zip(A[i], A[r])]
                    scale *= a
                    ops.append(_op_combo(i, a, c, r))
                else:
                    A[i] = [a + k * b for a, b in zip(A[i], A[r])]
                    ops.append(_op_add(i, k, r))
        snap(ops)
        pivots.append((r, j))
        r += 1

    if jordan:
        for (r, j) in reversed(pivots):
            ops = []
            piv = A[r][j]
            if piv != 1:
                k = 1 / piv
                A[r] = [k * a for a in A[r]]
                scale *= k
                ops.append(_op_scale(r, k))
            for i in range(r):
                if A[i][j] != 0:
                    k = -A[i][j]
                    A[i] = [a + k * b for a, b in zip(A[i], A[r])]
                    ops.append(_op_add(i, k, r))
            snap(ops)
    return Reduction(Matrix(A), pivots, steps, swaps, scale)


def rref(M: Matrix) -> Matrix:
    return row_reduce(M, jordan=True).result


def rank(M: Matrix) -> int:
    return row_reduce(M).rank


def det(M: Matrix) -> Fraction:
    """Determinant via MEG: (-1)^swaps · product of the pivots."""
    if not M.is_square:
        raise ValueError("det needs a square matrix")
    red = row_reduce(M, integer=False)
    if red.rank < M.n:
        return Fraction(0)
    d = Fraction(-1 if red.swaps % 2 else 1)
    for i in range(M.n):
        d *= red.result[i, i]
    return d


def laplace(entries: Sequence[Sequence]):
    """Determinant by Laplace expansion along the first row, for ANY commutative
    ring (Fractions, polynomials, ...). Used for det(tI - A)."""
    n = len(entries)
    if n == 1:
        return entries[0][0]
    if n == 2:
        return entries[0][0] * entries[1][1] - entries[0][1] * entries[1][0]
    total = None
    for j in range(n):
        a = entries[0][j]
        if a == 0:
            continue
        sub = [[row[c] for c in range(n) if c != j] for row in entries[1:]]
        term = a * laplace(sub)
        if j % 2:
            term = -term
        total = term if total is None else total + term
    return total if total is not None else entries[0][0] * 0


def det_steps(M: Matrix, name: str = "A", depth: int = 0) -> list[str]:
    """Explain det(M) by Laplace expansion along the row/column with most zeros."""
    n = M.n
    pad = "  " * depth
    if n == 1:
        return [f"{pad}det({name}) = {fs(M[0, 0])}"]
    if n == 2:
        a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        return [f"{pad}det({name}) = ({fs(a)})·({fs(d)}) - ({fs(b)})·({fs(c)}) = {fs(a * d - b * c)}"]
    best = None
    for i in range(n):
        z = sum(1 for x in M.row(i) if x == 0)
        if best is None or z > best[0]:
            best = (z, "row", i)
    for j in range(n):
        z = sum(1 for x in M.col(j) if x == 0)
        if z > best[0]:
            best = (z, "col", j)
    _, kind, idx = best
    what = "riga" if kind == "row" else "colonna"
    out = [f"{pad}Laplace expansion of det({name}) along {what} {idx + 1} "
           f"({'row' if kind == 'row' else 'column'} with the most zeros):"]
    terms, total = [], Fraction(0)
    for k in range(n):
        i, j = (idx, k) if kind == "row" else (k, idx)
        a = M[i, j]
        if a == 0:
            continue
        sub = M.minor(i, j)
        d = det(sub)
        sign = -1 if (i + j) % 2 else 1
        sub_name = f"Â{i + 1},{j + 1}"
        out.append(hcat(f"{pad}  {sub_name} = ", fm(sub)))
        out.extend(det_steps(sub, sub_name, depth + 2))
        terms.append(f"(-1)^({i + 1}+{j + 1})·({fs(a)})·({fs(d)})")
        total += sign * a * d
    if not terms:
        out.append(f"{pad}  the whole {('row' if kind == 'row' else 'column')} is zero → det = 0")
    out.append(f"{pad}det({name}) = " + (" + ".join(terms) if terms else "0") + f" = {fs(total)}")
    return out


def inverse(M: Matrix) -> Matrix | None:
    r = inverse_steps(M)
    return r[0]


def inverse_steps(M: Matrix) -> tuple[Matrix | None, list[str]]:
    """Gauss-Jordan on (A | I_n). Returns (A^-1 or None, steps)."""
    if not M.is_square:
        raise ValueError("only square matrices can be invertible")
    n = M.n
    red = row_reduce(M.hstack(Matrix.identity(n)), jordan=True, ncols=n, aug=n)
    steps = ["Gauss-Jordan on (A | I):\n" + indent(fm(M.hstack(Matrix.identity(n)), n))] + red.steps
    if red.rank < n:
        steps.append(f"The left block has a zero row: rg(A) = {red.rank} < {n}, so A is NOT invertible.")
        return None, steps
    inv = red.result.submatrix(range(n), range(n, 2 * n))
    steps.append("The left block is I, so the right block is A^-1.")
    return inv, steps


def kernel_basis(M: Matrix) -> list[list[Fraction]]:
    """Basis of ker A = Sol(A|0): one vector per free variable (integer entries)."""
    red = row_reduce(M, jordan=True)
    R = red.result
    pivot_cols = [j for _, j in red.pivots]
    basis = []
    for f in range(M.n):
        if f in pivot_cols:
            continue
        v = [Fraction(0)] * M.n
        v[f] = Fraction(1)
        for (r, j) in red.pivots:
            v[j] = -R[r, f]
        basis.append(clear_denominators(v))
    return basis


def column_space_basis(M: Matrix) -> list[list[Fraction]]:
    """Columns of A at the pivot positions of an echelon form (Proposizione 2.21)."""
    return [M.col(j) for _, j in row_reduce(M).pivots]


def row_space_basis(M: Matrix) -> list[list[Fraction]]:
    red = row_reduce(M, jordan=True)
    return [red.result.row(r) for r, _ in red.pivots]


def span_rank(vectors: Sequence[Sequence]) -> int:
    vectors = [v for v in vectors]
    if not vectors:
        return 0
    return rank(Matrix(vectors))


def in_span(v: Sequence, basis: Sequence[Sequence]) -> bool:
    if not basis:
        return all(x == 0 for x in v)
    return span_rank(list(basis) + [v]) == span_rank(basis)


def same_span(a: Sequence[Sequence], b: Sequence[Sequence]) -> bool:
    ra, rb = span_rank(a), span_rank(b)
    return ra == rb and span_rank(list(a) + list(b)) == ra


@dataclass
class Solution:
    kind: str  # "none" | "unique" | "inf"
    particular: list[Fraction] | None
    basis: list[list[Fraction]]
    rank_A: int
    rank_Ab: int
    steps: list[str]


def solve(A: Matrix, b: Sequence, names: Sequence[str] | None = None) -> Solution:
    """Solve Ax = b with Gauss-Jordan; Sol(A|b) = v_P + Sol(A|0) (Teorema 1.14)."""
    from .fmt import var_names

    b = vec(b)
    n = A.n
    names = names or var_names(n)
    Ab = A.hstack(Matrix.col_vector(b))
    red = row_reduce(Ab, jordan=True, ncols=n, aug=n)
    R = red.result
    rA = red.rank
    steps = ["Complete matrix (A|b):\n" + indent(fm(Ab, n))] + red.steps
    bad = [i for i in range(rA, A.m) if R[i, n] != 0]
    if bad:
        steps.append(
            f"Row {bad[0] + 1} reads 0 = {fs(R[bad[0], n])}: impossible. "
            f"rg(A) = {rA} < rg(A|b) = {rA + 1} → no solutions (Rouché-Capelli)."
        )
        return Solution("none", None, [], rA, rA + 1, steps)
    pivot_cols = [j for _, j in red.pivots]
    part = [Fraction(0)] * n
    for (r, j) in red.pivots:
        part[j] = R[r, n]
    basis = kernel_basis(A)
    free = [names[j] for j in range(n) if j not in pivot_cols]
    steps.append(f"rg(A) = rg(A|b) = {rA}: the system is solvable (Rouché-Capelli), "
                 f"with n - rg(A) = {n} - {rA} = {n - rA} free variable(s).")
    if free:
        steps.append("Dependent variables (pivots): " + ", ".join(names[j] for j in pivot_cols)
                     + ";  free variables: " + ", ".join(free) + ".")
        for (r, j) in red.pivots:
            rest = [(-R[r, f], names[f]) for f in range(n) if f not in pivot_cols and R[r, f] != 0]
            from .fmt import flin
            expr = flin([c for c, _ in rest], [nm for _, nm in rest], zero="")
            c0 = fs(R[r, n])
            if not expr:
                expr = c0
            elif R[r, n] != 0:
                expr = c0 + (" - " + expr[1:] if expr.startswith("-") else " + " + expr)
            steps.append(f"  {names[j]} = {expr}")
        from .fmt import fparam
        steps.append("Sol(A|b) = v_P + Sol(A|0) = " + fparam(part, basis))
        return Solution("inf", part, basis, rA, rA, steps)
    steps.append("Unique solution: " + ", ".join(f"{names[j]} = {fs(part[j])}" for j in range(n)))
    return Solution("unique", part, [], rA, rA, steps)
