"""Exercises, answer checkers and diagnosis of the likely error.

An Exercise has a prompt, one or more Parts (each with an Answer checker) and a
full step-by-step solution. Answer.check(text) parses what the student typed and
returns a Check; on a wrong answer `msg` names the most likely mistake.
Unreadable input raises ParseError (the UI then asks again, without penalty).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Sequence

from .core.fmt import fm, fm_inline, fparam, fs, fv, fvs
from .core.gauss import in_span, rank, same_span, span_rank
from .core.matrix import Matrix, dot, frac, vec, vsub
from .core.parse import (ParseError, parse_equations, parse_expr, parse_int, parse_matrix,
                         parse_param, parse_poly, parse_real, parse_scalar, parse_vector,
                         parse_vectors, parse_yesno, split_top, _entries)
from .core.poly import Poly
from .core.surd import Surd


@dataclass
class Check:
    ok: bool
    msg: str = ""
    score: float = 0.0

    @staticmethod
    def right(msg: str = "") -> "Check":
        return Check(True, msg, 1.0)

    @staticmethod
    def wrong(msg: str = "", score: float = 0.0) -> "Check":
        return Check(False, msg, score)


class Answer:
    fmt_hint = ""

    def __init__(self, mistakes: Sequence[tuple[object, str]] = ()):
        # known wrong answers produced by typical errors: (value, explanation)
        self.mistakes = list(mistakes)

    def check(self, text: str) -> Check:  # pragma: no cover - abstract
        raise NotImplementedError

    def show(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _known(self, value, eq=lambda a, b: a == b) -> str | None:
        for wrong, why in self.mistakes:
            try:
                if eq(value, wrong):
                    return why
            except (TypeError, ValueError):
                continue
        return None


# ------------------------------------------------------------------ scalars
class ScalarAns(Answer):
    fmt_hint = "a number or fraction, e.g. -3/2"

    def __init__(self, value, mistakes=()):
        super().__init__(mistakes)
        self.value = frac(value)

    def check(self, text):
        x = parse_scalar(text)
        if x == self.value:
            return Check.right()
        return Check.wrong(self._known(x) or diagnose_scalar(x, self.value))

    def show(self):
        return fs(self.value)

    def typed(self):
        return fs(self.value)


def diagnose_scalar(x: Fraction, c: Fraction) -> str:
    if c != 0 and x == -c:
        return "Right magnitude, wrong sign: re-check the signs (a (-1)^(i+j) factor, a row swap, a minus in a row operation)."
    if c != 0 and x != 0 and x == 1 / c:
        return "You gave the reciprocal of the correct value."
    if c != 0 and x != 0 and (x / c).denominator == 1 and abs(x / c) in (2, 3, 4):
        return f"Your value is {fs(x / c)} times the correct one: a factor was counted twice or not divided out."
    if c != 0 and x != 0 and (c / x).denominator == 1 and abs(c / x) in (2, 3, 4):
        return f"The correct value is {fs(c / x)} times yours: a factor got lost."
    if x == 0:
        return "It is not zero: some term you dropped does not vanish."
    return "Arithmetic slip somewhere: compare your computation with the steps below."


class RealAns(Answer):
    fmt_hint = "exact value, e.g. 12/sqrt(11) or 2*sqrt(3)/3"

    def __init__(self, value: Surd, mistakes=()):
        super().__init__(mistakes)
        self.value = Surd.lift(value)

    def check(self, text):
        x = parse_real(text)
        if x == self.value:
            return Check.right()
        why = self._known(x)
        if why:
            return Check.wrong(why)
        if x.square() == self.value.square():
            return Check.wrong("Right magnitude, wrong sign.")
        if x.coef != 0 and x.is_rational() and x.coef == self.value.square():
            return Check.wrong("You gave the square of the value: remember the square root (norm = sqrt(<v,v>)).")
        if x.coef != 0 and self.value.coef != 0 and x.square() == self.value.square() ** 2:
            return Check.wrong("You did not take the square root.")
        return Check.wrong("Not this value; compare with the steps below.")

    def show(self):
        return self.value.pretty()

    def typed(self):
        return str(self.value)


class IntAns(ScalarAns):
    fmt_hint = "an integer"

    def check(self, text):
        x = parse_int(text)
        if x == self.value:
            return Check.right()
        why = self._known(x)
        if not why:
            why = "Too small: " if x < self.value else "Too large: "
            why += "redo the count following the steps below."
        return Check.wrong(why)


class YesNoAns(Answer):
    fmt_hint = "yes / no"

    def __init__(self, value: bool, why_wrong: str = ""):
        super().__init__()
        self.value = value
        self.why_wrong = why_wrong

    def check(self, text):
        x = parse_yesno(text)
        return Check.right() if x == self.value else Check.wrong(self.why_wrong)

    def show(self):
        return "yes" if self.value else "no"

    def typed(self):
        return self.show()


class ChoiceAns(Answer):
    """Multiple choice. Accepts the letter, the number, or the exact option text."""

    def __init__(self, options: Sequence[str], correct: int, why: dict[int, str] | None = None):
        super().__init__()
        self.options = list(options)
        self.correct = correct
        self.why = why or {}
        self.fmt_hint = "letter: " + "/".join(chr(97 + i) for i in range(len(options)))

    def render(self) -> str:
        return "\n".join(f"  {chr(97 + i)}) {o}" for i, o in enumerate(self.options))

    def check(self, text):
        t = text.strip().lower().rstrip(").")
        idx = None
        if len(t) == 1 and "a" <= t <= chr(96 + len(self.options)):
            idx = ord(t) - 97
        elif t.isdigit() and 1 <= int(t) <= len(self.options):
            idx = int(t) - 1
        else:
            for i, o in enumerate(self.options):
                if t == o.lower():
                    idx = i
        if idx is None:
            raise ParseError(f"choose one of {self.fmt_hint}")
        if idx == self.correct:
            return Check.right()
        return Check.wrong(self.why.get(idx, ""))

    def show(self):
        return f"{chr(97 + self.correct)}) {self.options[self.correct]}"

    def typed(self):
        return chr(97 + self.correct)


class MultiChoiceAns(Answer):
    """Select ALL the true options (quiz style 'Selezionare le 3 affermazioni vere').
    Accepts 'a c d', 'A,C,D', 'acd'. Partial credit: (right picks - wrong picks)/#correct."""

    def __init__(self, options: Sequence[str], correct: Sequence[int], why: dict[int, str] | None = None):
        super().__init__()
        self.options = list(options)
        self.correct = sorted(set(correct))
        self.why = why or {}
        self.fmt_hint = f"all the correct letters, e.g. 'a c' ({len(self.correct)} correct)"

    def render(self) -> str:
        return "\n".join(f"  {chr(97 + i)}) {o}" for i, o in enumerate(self.options))

    def check(self, text):
        t = text.strip().lower().replace(",", " ").replace(";", " ").replace(")", " ")
        letters = t.split() if " " in t.strip() else list(t.strip())
        picked = set()
        for x in letters:
            if len(x) != 1 or not ("a" <= x <= chr(96 + len(self.options))):
                raise ParseError(f"use letters a–{chr(96 + len(self.options))}, e.g. 'a c'")
            picked.add(ord(x) - 97)
        if not picked:
            raise ParseError("pick at least one letter")
        good = set(self.correct)
        if picked == good:
            return Check.right()
        msgs = []
        for i in sorted(picked - good):
            msgs.append(f"{chr(97 + i)}) is false" + (f": {self.why[i]}" if i in self.why else "."))
        for i in sorted(good - picked):
            msgs.append(f"{chr(97 + i)}) is true" + (f": {self.why[i]}" if i in self.why else "."))
        score = max(0.0, (len(picked & good) - len(picked - good)) / len(good))
        return Check.wrong(" ".join(msgs), score)

    def show(self):
        return ", ".join(f"{chr(97 + i)})" for i in self.correct)

    def typed(self):
        return " ".join(chr(97 + i) for i in self.correct)


class OpenAns(Answer):
    """Proofs, examples, 'dimostrare o confutare': the student writes an answer, then compares
    with the model answer and grades themself (y = right, p = partly, n = wrong)."""
    fmt_hint = "after reading the model answer: y (correct) / p (partly) / n (wrong)"
    self_assessed = True

    def __init__(self, model: str):
        super().__init__()
        self.model = model.strip()

    def check(self, text):
        w = text.strip().lower()
        if w in ("y", "yes", "s", "si", "sì", "1", "ok"):
            return Check.right()
        if w in ("p", "partly", "partial", "parz", "0.5"):
            return Check.wrong("Partly right: compare once more with the model answer.", 0.5)
        if w in ("n", "no", "0"):
            return Check.wrong("Study the model answer and retry later.")
        raise ParseError("grade yourself: y / p / n")

    def show(self):
        return self.model

    def typed(self):
        return "y"


# ------------------------------------------------------------------ vectors / matrices
class VectorAns(Answer):
    fmt_hint = "a vector, e.g. (1, -2, 3/2)"

    def __init__(self, v, up_to_scalar: bool = False, mistakes=()):
        super().__init__(mistakes)
        self.v = vec(v)
        self.up_to_scalar = up_to_scalar
        if up_to_scalar:
            self.fmt_hint = "a non-zero vector (any non-zero multiple is accepted)"

    def check(self, text):
        x = parse_vector(text, len(self.v))
        if self.up_to_scalar:
            if all(a == 0 for a in x):
                return Check.wrong("The zero vector is never an eigenvector / generator.")
            if rank(Matrix([x, self.v])) == 1:
                return Check.right()
        elif x == self.v:
            return Check.right()
        why = self._known(x)
        if why:
            return Check.wrong(why)
        return Check.wrong(diagnose_vector(x, self.v, self.up_to_scalar))

    def show(self):
        return fv(self.v)

    def typed(self):
        return fv(self.v)


def diagnose_vector(x, v, up_to_scalar=False) -> str:
    if not up_to_scalar and any(a != 0 for a in v) and rank(Matrix([x, v])) == 1 and any(x):
        k = next(a / b for a, b in zip(x, v) if b != 0)
        if k == -1:
            return "Every component has the wrong sign."
        return f"Your vector is {fs(k)} times the correct one."
    diff = [i for i in range(len(v)) if x[i] != v[i]]
    if len(diff) == 1 and not up_to_scalar:
        i = diff[0]
        return f"Only component {i + 1} is wrong (you wrote {fs(x[i])}, it is {fs(v[i])})."
    return "Compare with the solution step by step."


class RealVectorAns(Answer):
    """Vector with entries q·sqrt(n) (unit vectors, orthonormal bases)."""
    fmt_hint = "a vector with exact entries, e.g. (1/sqrt(14), 2/sqrt(14), 3/sqrt(14))"

    def __init__(self, v: Sequence, mistakes=()):
        super().__init__(mistakes)
        self.v = [Surd.lift(x) for x in v]

    def check(self, text):
        x = [parse_real(e) for e in _entries(text)]
        if len(x) != len(self.v):
            raise ParseError(f"expected {len(self.v)} components")
        if x == self.v:
            return Check.right()
        sq = sum((a.square() for a in x), Fraction(0))
        if sq != 1:
            return Check.wrong(f"Your vector has squared norm {fs(sq)}, not 1: divide by its norm.")
        return Check.wrong("Unit vector, but not the right one (direction or sign).")

    def show(self):
        return "(" + ", ".join(a.pretty() for a in self.v) + ")"

    def typed(self):
        return "(" + ", ".join(str(a) for a in self.v) + ")"


class MatrixAns(Answer):
    fmt_hint = "rows separated by ';', e.g. 1 2; 3 4"

    def __init__(self, M: Matrix, mistakes=()):
        super().__init__(mistakes)
        self.M = M

    def check(self, text):
        X = parse_matrix(text, self.M.shape)
        if X == self.M:
            return Check.right()
        why = self._known(X)
        if why:
            return Check.wrong(why)
        return Check.wrong(diagnose_matrix(X, self.M))

    def show(self):
        return "\n" + fm(self.M) + f"\n(type: {fm_inline(self.M)})"

    def typed(self):
        return fm_inline(self.M)


def diagnose_matrix(X: Matrix, M: Matrix) -> str:
    if X.shape == M.T.shape and X == M.T and M != M.T:
        return "You wrote the transpose of the correct matrix (rows and columns swapped)."
    if X == -M:
        return "Every entry has the wrong sign."
    diffs = [(i, j) for i in range(M.m) for j in range(M.n) if X[i, j] != M[i, j]]
    if len(diffs) == 1:
        i, j = diffs[0]
        return (f"Only entry ({i + 1},{j + 1}) is wrong: you wrote {fs(X[i, j])}, it is {fs(M[i, j])} "
                f"(row {i + 1} of the first factor times column {j + 1} of the second, if it is a product).")
    rows = {i for i, _ in diffs}
    if len(rows) == 1:
        return f"Only row {next(iter(rows)) + 1} is wrong."
    cols = {j for _, j in diffs}
    if len(cols) == 1:
        return f"Only column {next(iter(cols)) + 1} is wrong."
    flat_x = [x for r in X.rows for x in r]
    flat_m = [x for r in M.rows for x in r]
    if not M.is_zero() and any(flat_x) and rank(Matrix([flat_x, flat_m])) == 1:
        a = next(X[i, j] / M[i, j] for i in range(M.m) for j in range(M.n) if M[i, j] != 0)
        return f"Your matrix is {fs(a)} times the correct one: a scalar factor is off."
    return f"{len(diffs)} entries differ; compare with the solution."


UNDEFINED_WORDS = {"undefined", "not defined", "non definito", "non definita", "nd", "n.d.",
                   "impossibile", "impossible", "none", "-"}


class MaybeMatrixAns(MatrixAns):
    """A matrix, or 'not defined' (e.g. a product of incompatible matrices)."""
    fmt_hint = "rows separated by ';' — or 'not defined'"

    def __init__(self, M: Matrix | None, mistakes=(), why_undefined: str = ""):
        super().__init__(M if M is not None else Matrix([[0]]), mistakes)
        self.defined = M is not None
        self.why_undefined = why_undefined

    def check(self, text):
        if text.strip().lower() in UNDEFINED_WORDS:
            return Check.right() if not self.defined else Check.wrong("It IS defined: the sizes are compatible.")
        if not self.defined:
            parse_matrix(text)  # still reject garbage politely
            return Check.wrong(self.why_undefined or "It is not defined (incompatible sizes).")
        return super().check(text)

    def show(self):
        return super().show() if self.defined else "not defined"

    def typed(self):
        return super().typed() if self.defined else "not defined"


class SpanAns(Answer):
    """A basis of a given subspace W of K^n (any basis is accepted)."""
    fmt_hint = "basis vectors separated by ';', e.g. (1,0,1); (0,1,1)   ('none' for {0})"

    def __init__(self, basis: Sequence[Sequence], n: int, what: str = "W"):
        super().__init__()
        self.basis = [vec(b) for b in basis]
        self.n = n
        self.what = what

    def check(self, text):
        xs = parse_vectors(text, self.n)
        d = len(self.basis)
        if d == 0:
            if not xs or all(all(a == 0 for a in x) for x in xs):
                return Check.right()
            return Check.wrong(f"{self.what} = {{0}}: its basis is empty (answer 'none').")
        if not xs:
            return Check.wrong(f"{self.what} is not {{0}}: it has dimension {d}.")
        outside = [x for x in xs if not in_span(x, self.basis)]
        if outside:
            return Check.wrong(f"{fv(outside[0])} does not belong to {self.what}.")
        r = span_rank(xs)
        if r < len(xs):
            return Check.wrong("Your vectors are linearly dependent, so they are not a basis.",
                               0.5 if r == d else 0.0)
        if r < d:
            return Check.wrong(f"Your vectors are in {self.what} and independent, but {self.what} has dimension {d}: "
                               f"you need {d} vectors.", 0.5)
        return Check.right()

    def show(self):
        return fvs(self.basis) if self.basis else "none (the space is {0})"

    def typed(self):
        return fvs(self.basis) if self.basis else "none"


class ScaledVectorsAns(Answer):
    """An ordered list of vectors, each accepted up to a non-zero multiple
    (e.g. the Gram-Schmidt orthogonal basis c1, ..., cp)."""
    fmt_hint = "vectors separated by ';' (each may be rescaled)"

    def __init__(self, vectors: Sequence[Sequence]):
        super().__init__()
        self.vs = [vec(v) for v in vectors]

    def check(self, text):
        xs = parse_vectors(text, len(self.vs[0]))
        if len(xs) != len(self.vs):
            raise ParseError(f"expected {len(self.vs)} vectors")
        for k, (x, v) in enumerate(zip(xs, self.vs)):
            if all(a == 0 for a in x) or rank(Matrix([x, v])) != 1:
                return Check.wrong(f"Vector number {k + 1} is wrong (the earlier ones are fine)." if k else
                                   "The first vector is wrong.", k / len(self.vs))
        return Check.right()

    def show(self):
        return fvs(self.vs)

    def typed(self):
        return fvs(self.vs)


class AffineAns(Answer):
    """An affine subspace P + Span(dirs) of K^n given in parametric form;
    None = empty set. Used for Sol(A|b)."""
    fmt_hint = "'none', or a vector, or (p) + t(v) + s(w) ..."

    def __init__(self, point, dirs, n: int, what="Sol(A|b)"):
        super().__init__()
        self.point = None if point is None else vec(point)
        self.dirs = [vec(d) for d in dirs]
        self.n = n
        self.what = what

    def check(self, text):
        x = parse_param(text, self.n)
        if self.point is None:
            return Check.right() if x is None else Check.wrong(
                "There are no solutions: rg(A) < rg(A|b) (a row 0 = c with c ≠ 0 appears).")
        if x is None:
            return Check.wrong("The system IS solvable: rg(A) = rg(A|b).")
        p, ds = x
        if not in_span(vsub(p, self.point), self.dirs):
            return Check.wrong(f"The point {fv(p)} does not belong to {self.what} "
                               f"(the particular solution v_P is wrong).", 0.25 if same_span(ds, self.dirs) and len(ds) == len(self.dirs) else 0)
        d = len(self.dirs)
        if len(ds) != d:
            return Check.wrong(f"The solution set depends on {d} parameter(s) (n - rg(A)), you used {len(ds)}.", 0.5)
        if d and (span_rank(ds) < d or not same_span(ds, self.dirs)):
            return Check.wrong("Particular solution OK, but the directions do not span Sol(A|0).", 0.5)
        return Check.right()

    def show(self):
        if self.point is None:
            return "none (no solutions)"
        return fparam(self.point, self.dirs)

    def typed(self):
        return "none" if self.point is None else fparam(self.point, self.dirs)


class EquationsAns(Answer):
    """Cartesian equations of the affine subspace P + Span(dirs) of R^n."""

    def __init__(self, point, dirs, names: Sequence[str], shown: Sequence[tuple] = ()):
        super().__init__()
        self.point = vec(point)
        self.dirs = [vec(d) for d in dirs]
        self.names = list(names)
        self.shown = list(shown)  # model answer: list of (coeffs, rhs)
        self.fmt_hint = f"equations separated by ';', in the variables {', '.join(names)}"

    def check(self, text):
        eqs = parse_equations(text, self.names)
        n = len(self.names)
        need = n - len(self.dirs)
        for a, b in eqs:
            if all(x == 0 for x in a):
                return Check.wrong("An equation without variables is not allowed.")
            if dot(a, self.point) != b:
                return Check.wrong(f"The point {fv(self.point)} of the subspace does not satisfy one of your equations.")
            if any(dot(a, d) != 0 for d in self.dirs):
                return Check.wrong("One of your equations is not satisfied by all points (a direction vector violates it).")
        r = rank(Matrix([a for a, _ in eqs]))
        if r < need:
            return Check.wrong(f"Correct but not enough: you need {need} independent equations (n - dim).", 0.5)
        return Check.right()

    def show(self):
        from .core.fmt import flin
        return ";  ".join(f"{flin(a, self.names)} = {fs(b)}" for a, b in self.shown)

    def typed(self):
        return self.show()


class PolyAns(Answer):
    fmt_hint = "a polynomial in t, e.g. t^3 - 4t^2 + 5t - 2"

    def __init__(self, p: Poly, mistakes=(), var: str = "t"):
        super().__init__(mistakes)
        self.p = p
        self.var = var
        if var != "t":
            self.fmt_hint = f"a polynomial in {var} (you may also write it with l, k, x or t)"

    def _parse(self, text):
        for v in (self.var, "t", "l", "k", "x", "a", "λ"):
            try:
                return parse_poly(text, v)
            except ParseError:
                continue
        return parse_poly(text, self.var)

    def check(self, text):
        q = self._parse(text)
        if q == self.p:
            return Check.right()
        why = self._known(q)
        if why:
            return Check.wrong(why)
        if q == -self.p:
            return Check.wrong("Opposite sign: the notes use p_A(t) = det(tI - A), which is monic.")
        if q.degree != self.p.degree:
            return Check.wrong(f"p_A must have degree n = {self.p.degree}.")
        return Check.wrong("Not the right polynomial: check the Laplace expansion of det(tI - A).")

    def show(self):
        return self.p.show(self.var)

    def typed(self):
        return self.p.show(self.var)


class ScalarSetAns(Answer):
    """A set of scalars, e.g. the spectrum S(f). 'none' for the empty set."""
    fmt_hint = "values separated by ',' (each once), or 'none'"

    def __init__(self, values: Sequence, what="S(f)"):
        super().__init__()
        self.values = sorted(set(frac(v) for v in values))
        self.what = what

    def check(self, text):
        t = text.strip().strip("{}")
        if t.lower() in ("none", "empty", "nessuno", "∅", ""):
            xs = []
        else:
            xs = [parse_scalar(p) for p in split_top(t, ",;")]
        s = sorted(set(xs))
        if s == self.values:
            return Check.right()
        miss = [v for v in self.values if v not in s]
        extra = [v for v in s if v not in self.values]
        if extra:
            return Check.wrong(f"{fs(extra[0])} is not in {self.what}.")
        return Check.wrong(f"You missed {fs(miss[0])}.", 0.5 if s else 0)

    def show(self):
        return ", ".join(fs(v) for v in self.values) if self.values else "none"

    def typed(self):
        return self.show()


class DiagonalizerAns(Answer):
    """An invertible P with P^-1 A P diagonal (any such P is accepted)."""
    fmt_hint = "the matrix P (eigenvectors as COLUMNS), rows separated by ';'"

    def __init__(self, A: Matrix, P_model: Matrix):
        super().__init__()
        self.A = A
        self.P = P_model

    def check(self, text):
        from .core.gauss import det, inverse
        P = parse_matrix(text, self.A.shape)
        if det(P) == 0:
            return Check.wrong("P must be invertible: its columns must be a basis of eigenvectors.")
        D = inverse(P) * self.A * P
        n = self.A.n
        if all(D[i, j] == 0 for i in range(n) for j in range(n) if i != j):
            return Check.right(f"Indeed P⁻¹AP = diag({', '.join(fs(D[i, i]) for i in range(n))}).")
        AP = self.A * P
        bad = [j for j in range(n) if rank(Matrix([AP.col(j), P.col(j)])) > 1]
        rows_ok = all(rank(Matrix([AP.T.col(j), P.T.col(j)])) == 1 for j in range(n))
        if rows_ok and bad:
            return Check.wrong("It looks like you put the eigenvectors in the ROWS: they go in the columns of P.")
        return Check.wrong(f"Column {bad[0] + 1} of P is not an eigenvector of A." if bad else "")

    def show(self):
        return "\n" + fm(self.P) + f"\n(type: {fm_inline(self.P)}; any invertible P whose columns are eigenvectors works)"

    def typed(self):
        return fm_inline(self.P)


# ------------------------------------------------------------------ exercise
@dataclass
class Part:
    label: str
    answer: Answer


@dataclass
class Exercise:
    topic: str
    kind: str
    prompt: str
    parts: list[Part]
    solution: list[str]
    data: dict = field(default_factory=dict)
    choices_inline: bool = True

    def answer_text(self) -> str:
        return "\n".join(f"{p.label}: {p.answer.show()}" for p in self.parts)


def mk(topic: str, kind: str, prompt: str, parts, solution, **data) -> Exercise:
    """Shortcut: parts may be a single Answer or a list of (label, Answer)."""
    if isinstance(parts, Answer):
        parts = [Part("Answer", parts)]
    else:
        parts = [Part(lbl, a) for lbl, a in parts]
    return Exercise(topic, kind, prompt, parts, list(solution), data)
