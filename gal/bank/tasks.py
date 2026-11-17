"""Task builders: turn a transcribed official exercise (TOML `data`) into a checked Exercise.

Each builder receives (data, text) and returns (prompt_extra, parts, solution_steps).
`text` is the exercise statement as written in the official sheet; `prompt_extra` adds the
exact data (matrices, vectors) rendered by the engine, so the student always sees precise
numbers even if the statement is abbreviated.

Number formats in data: integers, or strings like "3/2" / "-1/3". Parametric entries are
strings in the parameter, e.g. "λ+1", "k^2-1" (declare it with  param = "λ").
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Callable

from ..core.fmt import (flin, fm, fparam, fs, fsys, fv, fvs, hcat, indent, named, named_many,
                        var_names)
from ..core.gauss import (column_space_basis, det, det_steps, in_span, inverse, inverse_steps,
                          kernel_basis, rank, row_reduce, same_span, solve, span_rank)
from ..core.geom import cartesian_from_param, gram_inner, gram_schmidt, norm, ratio
from ..core.matrix import Matrix, clear_denominators, dot, frac, vec, vsub
from ..core.param import PMatrix, fmt_pmatrix, generic_value, rank_profile, values_text
from ..core.parse import parse_poly
from ..core.poly import Poly, charpoly, factored, real_root_count
from ..core.surd import Surd
from ..exercise import (AffineAns, Answer, ChoiceAns, EquationsAns, IntAns, MatrixAns,
                        MaybeMatrixAns, MultiChoiceAns, OpenAns, PolyAns, RealAns, ScalarAns,
                        ScalarSetAns, ScaledVectorsAns, SpanAns, VectorAns, YesNoAns,
                        DiagonalizerAns)


class TaskError(ValueError):
    """The data cannot be turned into an auto-checked exercise (e.g. irrational parameters)."""


def num(x) -> Fraction:
    if isinstance(x, str):
        return Fraction(x.replace(" ", ""))
    return frac(x)


def M_(rows) -> Matrix:
    return Matrix([[num(x) for x in r] for r in rows])


def V_(v) -> list[Fraction]:
    return [num(x) for x in v]


def VS_(vs) -> list[list[Fraction]]:
    return [V_(v) for v in vs]


TASKS: dict[str, Callable] = {}


def task(name):
    def deco(fn):
        TASKS[name] = fn
        return fn
    return deco


def _names(d, n):
    return d.get("names") or var_names(n)


# ------------------------------------------------------------------ free text / keys
@task("open")
def t_open(d, text):
    """Proofs, examples, discussions: self-assessed against a model answer."""
    sol = d.get("solution", "")
    return "", [("self-check", OpenAns(sol or "No model answer yet: compare with the handwritten "
                                             "solution of the esercitazione, if any."))], []


@task("yesno")
def t_yesno(d, text):
    why = d.get("why", "")
    return "", [("yes / no", YesNoAns(bool(d["answer"]), why))], [why] if why else []


@task("values")
def t_values(d, text):
    """Scalar answers keyed by hand: answers = [[label, value], ...]  (value '3/2', or a
    list of values for a set). Verified by item-specific tests in tests/test_bank_keys.py."""
    parts = []
    for label, val in d["answers"]:
        if isinstance(val, list):
            parts.append((label, ScalarSetAns([num(x) for x in val])))
        elif isinstance(val, str) and ("sqrt" in val or "√" in val):
            from ..core.parse import parse_real
            parts.append((label, RealAns(parse_real(val))))      # exact roots: 'sqrt(2)', '2*sqrt(3)/3'
        else:
            parts.append((label, ScalarAns(num(val))))
    why = d.get("why", "")
    return "", parts, [why] if why else []


@task("mcq")
def t_mcq(d, text):
    opts = d["options"]
    letters = d["correct"] if isinstance(d["correct"], list) else [d["correct"]]
    idx = [ord(c.lower()) - 97 for c in letters]
    raw = d.get("why", {})
    # `why`: per-option explanations {a = "..."}, or one explanation of the whole question
    why = {ord(k.lower()) - 97: v for k, v in raw.items()} if isinstance(raw, dict) else {}
    multi = d.get("multi", len(idx) > 1)
    ans = MultiChoiceAns(opts, idx, why) if multi else ChoiceAns(opts, idx[0], why)
    steps = [f"{chr(97 + i)}) {'TRUE ' if i in idx else 'false'} — {why[i]}" for i in sorted(why)]
    if isinstance(raw, str) and raw:
        steps = [f"Correct: {', '.join(chr(97 + i) + ')' for i in idx)}  {opts[idx[0]] if len(idx) == 1 else ''}".rstrip(),
                 raw]
    return ans.render(), [("answer", ans)], steps


# ------------------------------------------------------------------ chapter 1
@task("solve")
def t_solve(d, text):
    A, b = M_(d["A"]), V_(d["b"])
    names = _names(d, A.n)
    s = solve(A, b, names)
    return indent(fsys(A, b, names), 3), [("Sol(A|b)", AffineAns(s.particular, s.basis, A.n))], s.steps


@task("nsol")
def t_nsol(d, text):
    """How many solutions? (complete matrix given)"""
    Ab = M_(d["Ab"])
    A, b = Ab.submatrix(range(Ab.m), range(Ab.n - 1)), Ab.col(Ab.n - 1)
    s = solve(A, b)
    opts = ["no solutions", "exactly one solution", "infinitely many solutions"]
    k = {"none": 0, "unique": 1, "inf": 2}[s.kind]
    parts = [("how many", ChoiceAns(opts, k))]
    if s.kind == "inf":
        parts.append(("number of free parameters", IntAns(A.n - s.rank_A)))
    return named("(A|b)", Ab, aug=A.n), parts, s.steps


@task("rank")
def t_rank(d, text):
    A = M_(d["A"])
    red = row_reduce(A)
    return named(d.get("name", "A"), A), [(f"rg({d.get('name', 'A')})", IntAns(red.rank))], red.steps + [
        f"{red.rank} pivots ⇒ rg = {red.rank}"]


@task("homog")
def t_homog(d, text):
    """Rank and solution space of the homogeneous system Ax = 0."""
    A = M_(d["A"])
    K = kernel_basis(A)
    s = solve(A, [0] * A.m)
    nm = d.get("name", "A")
    return named(nm, A), [(f"rg({nm})", IntAns(A.n - len(K))),
                          ("basis of Sol(A|0) ('none' if {0})", SpanAns(K, A.n, "Sol(A|0)"))], s.steps


@task("det")
def t_det(d, text):
    A = M_(d["A"])
    nm = d.get("name", "A")
    return named(nm, A), [(f"det({nm})", ScalarAns(det(A)))], det_steps(A, nm)


@task("inverse")
def t_inverse(d, text):
    A = M_(d["A"])
    nm = d.get("name", "A")
    inv, steps = inverse_steps(A)
    parts = [("invertible?", YesNoAns(inv is not None, steps[-1]))]
    if inv is not None:
        parts.append((f"{nm}^-1", MatrixAns(inv)))
    return named(nm, A), parts, steps


# --- matrix expressions: A + C, A*B, B*(A+C), (E*A)*B, A*E + D^T, E^T*D, A^2026, 2*A - B
_TOK = re.compile(r"\s*(?:(\d+)|([A-Za-z][0-9]?)|(\^T|\^-1|\^\d+|[-+*()]))")


class _Undefined(Exception):
    pass


def eval_matrix_expr(expr: str, mats: dict[str, Matrix]):
    toks, pos = [], 0
    while pos < len(expr):
        m = _TOK.match(expr, pos)
        if not m or m.end() == pos:
            if expr[pos:].strip() == "":
                break
            raise TaskError(f"bad expression {expr!r}")
        toks.append(m.group(1) or m.group(2) or m.group(3))
        pos = m.end()
    i = 0

    def peek():
        return toks[i] if i < len(toks) else None

    def take():
        nonlocal i
        i += 1
        return toks[i - 1]

    def expr_():
        v = term()
        while peek() in ("+", "-"):
            op = take()
            w = term()
            if v.shape != w.shape:
                raise _Undefined(f"{op}: different sizes {v.m}×{v.n} and {w.m}×{w.n}")
            v = v + w if op == "+" else v - w
        return v

    def term():
        v = factor()
        while peek() == "*" or (peek() is not None and (peek() == "(" or peek()[0].isalpha())):
            if peek() == "*":
                take()
            w = factor()
            if isinstance(v, Fraction):
                v = w.scale(v)
                continue
            if v.n != w.m:
                raise _Undefined(f"product {v.m}×{v.n} · {w.m}×{w.n}: columns ≠ rows")
            v = v * w
        return v

    def factor():
        t = take()
        if t == "-":
            return factor().scale(-1)
        if t is None:
            raise TaskError("unexpected end")
        if t.isdigit():
            return Fraction(int(t))
        if t == "(":
            v = expr_()
            if take() != ")":
                raise TaskError("missing )")
        else:
            v = mats[t]
        while peek() and peek().startswith("^"):
            p = take()
            if p == "^T":
                v = v.T
            elif p == "^-1":
                if not v.is_square:
                    raise _Undefined("only square matrices can be inverted")
                inv = inverse(v)
                if inv is None:
                    raise _Undefined("the matrix is singular (det = 0): no inverse")
                v = inv
            else:
                k = int(p[1:])
                if not v.is_square:
                    raise _Undefined("powers need a square matrix")
                v = _fast_pow(v, k)
        return v

    v = expr_()
    if i != len(toks):
        raise TaskError(f"trailing tokens in {expr!r}")
    return v


def _fast_pow(A: Matrix, k: int) -> Matrix:
    R, B = Matrix.identity(A.n), A
    while k:
        if k & 1:
            R = R * B
        B = B * B
        k >>= 1
    return R


@task("matexpr")
def t_matexpr(d, text):
    mats = {k: M_(v) for k, v in d["matrices"].items()}
    parts, steps = [], []
    labels = d.get("labels") or d["exprs"]     # labels: hide the formula (e.g. 'X' for M^-1*N)
    for expr, lbl in zip(d["exprs"], labels):
        try:
            R = eval_matrix_expr(expr, mats)
            parts.append((lbl, MaybeMatrixAns(R)))
            steps.append(hcat(f"{lbl} = {expr} = " if lbl != expr else f"{expr} = ", fm(R)))
        except _Undefined as e:
            parts.append((lbl, MaybeMatrixAns(None, why_undefined=f"Not defined: {e}.")))
            steps.append(f"{expr}: not defined ({e}).")
    shown = named_many(*[(k, v) for k, v in mats.items()]) if d.get("show", True) else ""
    return shown, parts, steps


# ------------------------------------------------------------------ chapter 2
def _space_vectors(d):
    vs = VS_(d["vectors"])
    labels = d.get("labels") or [fv(v) for v in vs]
    return vs, labels


@task("indep")
def t_indep(d, text):
    """Independent? generators of V? Vectors given by coordinates (dim V = len)."""
    vs, labels = _space_vectors(d)
    n = len(vs[0])
    r = span_rank(vs)
    red = row_reduce(Matrix.from_cols(vs))
    steps = ["Coordinates as columns:", indent(fm(Matrix.from_cols(vs)))] + red.steps + [
        f"rank = {r}; {len(vs)} vectors; dim V = {n}.",
        f"independent ⇔ rank = #vectors: {'yes' if r == len(vs) else 'no'};  "
        f"generators ⇔ rank = dim V: {'yes' if r == n else 'no'}."]
    parts = [("linearly independent?", YesNoAns(r == len(vs), steps[-1]))]
    if d.get("ask_generators", True):
        parts.append(("generators of V?", YesNoAns(r == n, steps[-1])))
    extra = "Vectors: " + ",  ".join(labels)
    if d.get("coords_note"):
        extra += "\n(" + d["coords_note"] + ")"
    return extra, parts, steps


@task("basis_span")
def t_basis_span(d, text):
    vs, labels = _space_vectors(d)
    n = len(vs[0])
    basis = column_space_basis(Matrix.from_cols(vs)) if vs else []
    red = row_reduce(Matrix.from_cols(vs))
    steps = red.steps + [f"dim = {len(basis)}; basis (pivot columns): {fvs(basis)}"]
    shown = "" if d.get("hide") else "W = Span(" + ", ".join(labels) + ")"
    return (shown + (f"\n({d['coords_note']})" if d.get("coords_note") else "")).strip(), \
        [("dim W", IntAns(len(basis))), ("a basis of W (coordinates)", SpanAns(basis, n))], steps


@task("basis_eqs")
def t_basis_eqs(d, text):
    A = M_(d["A"])
    names = _names(d, A.n)
    K = kernel_basis(A)
    s = solve(A, [0] * A.m, names)
    eqs = ";  ".join(f"{flin(A.row(i), names)} = 0" for i in range(A.m))
    return ("" if d.get("hide") else f"W = {{ {eqs} }}") + (f"\n({d['coords_note']})" if d.get("coords_note") else ""), \
        [("dim W", IntAns(len(K))), ("a basis of W (coordinates)", SpanAns(K, A.n))], s.steps


@task("coords")
def t_coords(d, text):
    B, v = VS_(d["basis"]), V_(d["v"])
    shown = (f"B = {{{', '.join(d.get('labels') or [fv(b) for b in B])}}},  v = {d.get('vlabel') or fv(v)}" +
             (f"\n({d['coords_note']})" if d.get("coords_note") else ""))
    if len(B) != len(B[0]) or span_rank(B) < len(B):
        r = span_rank(B)
        why = f"rank of the coordinate vectors = {r} < {len(B[0])}: not a basis."
        return shown, [("is it a basis?", YesNoAns(False, why))], [why]
    s = solve(Matrix.from_cols(B), v, [f"λ{i + 1}" for i in range(len(B))])
    return shown, [("is it a basis?", YesNoAns(True)), ("coordinates of v", VectorAns(s.particular))], s.steps


class CompleteAns(Answer):
    """Vectors that, added to the given ones, give a set generating K^n
    (a basis when the given vectors are independent)."""

    def __init__(self, given, model, n):
        super().__init__()
        self.given, self.model, self.n = given, model, n
        self.need = n - span_rank(given)
        self.fmt_hint = f"{self.need} vector(s) separated by ';'"

    def check(self, text):
        from ..core.parse import ParseError, parse_vectors
        from ..exercise import Check
        xs = parse_vectors(text, self.n)
        if len(xs) != self.need:
            raise ParseError(f"give exactly {self.need} vector(s)")
        if span_rank(self.given + xs) == self.n:
            return Check.right()
        return Check.wrong("Together with the given vectors they do not span K^n: some vector you added "
                           "is in the span of the others.")

    def show(self):
        return fvs(self.model) + "  (many other choices work)"

    def typed(self):
        return fvs(self.model)


def complete_with_canonical(vs, n):
    out = []
    for i in range(n):
        e = [Fraction(int(i == j)) for j in range(n)]
        if span_rank(vs + out + [e]) > span_rank(vs + out):
            out.append(e)
    return out


@task("complete_basis")
def t_complete_basis(d, text):
    vs, labels = _space_vectors(d)
    n = len(vs[0])
    r = span_rank(vs)
    extra = complete_with_canonical(vs, n)
    steps = [f"rank of the given vectors = {r}" + (" (independent)" if r == len(vs) else " (dependent!)"),
             "Add canonical vectors e_i one at a time, keeping those that increase the rank:",
             f"e.g. {fvs(extra)}"]
    parts = []
    if d.get("ask_indep", True):
        parts.append(("are the given vectors independent?", YesNoAns(r == len(vs))))
    parts.append((f"{n - r} vector(s) completing them to span K^{n}", CompleteAns(vs, extra, n)))
    return "Vectors: " + ",  ".join(labels) + (f"\n({d['coords_note']})" if d.get("coords_note") else ""), parts, steps


@task("in_span")
def t_in_span(d, text):
    vs, v = VS_(d["vectors"]), V_(d["v"])
    ok = in_span(v, vs)
    s = solve(Matrix.from_cols(vs), v, [f"λ{i + 1}" for i in range(len(vs))])
    return f"v = {fv(v)} ∈ Span({', '.join(fv(x) for x in vs)}) ?", [("in the span?", YesNoAns(ok))], s.steps


@task("cartesian")
def t_cartesian(d, text):
    dirs = VS_(d["dirs"])
    n = len(dirs[0]) if dirs else len(d["point"])
    P = V_(d.get("point", [0] * n))
    names = _names(d, n)
    basis = column_space_basis(Matrix.from_cols(dirs))
    eqs, steps = cartesian_from_param(P, basis, names)
    if len(basis) < len(dirs):
        steps = [f"The generators are dependent: a basis is {fvs(basis)} (dim {len(basis)})."] + steps
    shown = fparam(P, dirs) if any(P) else f"Span({', '.join(fv(x) for x in dirs)})"
    parts = [("Cartesian equations", EquationsAns(P, basis, names, eqs))]
    if d.get("ask_dim"):
        parts.append(("dim", IntAns(len(basis))))
    return ("" if d.get("hide") else f"S = {shown}"), parts, steps


def _affine(obj):
    """Affine subspace from {'point','dirs'} or {'A','b'} (Cartesian) → (P, dirs)."""
    if "dirs" in obj:
        dirs = VS_(obj["dirs"])
        return V_(obj.get("point", [0] * len(dirs[0]))), [x for x in dirs]
    A, b = M_(obj["A"]), V_(obj.get("b", [0] * len(obj["A"])))
    s = solve(A, b)
    if s.kind == "none":
        raise TaskError("empty affine subspace")
    return s.particular, s.basis


def _affine_text(obj, names):
    if "dirs" in obj:
        P, dirs = V_(obj.get("point", [0] * len(obj["dirs"][0]))), VS_(obj["dirs"])
        return fparam(P, dirs) if any(P) else "Span(" + ", ".join(fv(x) for x in dirs) + ")"
    A, b = M_(obj["A"]), V_(obj.get("b", [0] * len(obj["A"])))
    return "{ " + ";  ".join(f"{flin(A.row(i), names)} = {fs(b[i])}" for i in range(A.m)) + " }"


@task("affine_forms")
def t_affine_forms(d, text):
    """Through given points (or point + directions): parametric AND Cartesian equations,
    plus optional membership questions."""
    if "points" in d:
        pts = VS_(d["points"])
        P = pts[0]
        dirs = [vsub(q, P) for q in pts[1:]]
    else:
        P, dirs = V_(d["point"]), VS_(d["dirs"])
    n = len(P)
    names = _names(d, n)
    if span_rank(dirs) < len(dirs):
        raise TaskError("points not in general position")
    eqs, steps = cartesian_from_param(P, dirs, names)
    parts = [("parametric equation", AffineAns(P, dirs, n, "S")),
             ("Cartesian equation(s)", EquationsAns(P, dirs, names, eqs))]
    for q in d.get("check", []):
        q = V_(q)
        inside = in_span(vsub(q, P), dirs)
        parts.append((f"does {fv(q)} lie on it?", YesNoAns(inside)))
        steps.append(f"{fv(q)}: {'satisfies' if inside else 'does not satisfy'} the equations.")
    steps = [f"Parametric: {fparam(P, dirs)}"] + steps
    return "", parts, steps


def grassmann_keys(d) -> list[str]:
    """Which parts a grassmann item asks, in order. Keys: U, W (bases), dim_sum, dim_int,
    direct, int_basis, sum_basis. `parts = [...]` chooses explicitly; the older flags
    ask_bases / only_bases are still understood."""
    if "parts" in d:
        return list(d["parts"])
    if d.get("only_bases"):
        return ["int_basis", "sum_basis"]
    keys = ["U", "W", "dim_sum", "dim_int", "direct"]
    return keys + (["int_basis", "sum_basis"] if d.get("ask_bases") else [])


@task("grassmann")
def t_grassmann(d, text):
    def basis(obj):
        if "span" in obj:
            vs = VS_(obj["span"])
            return column_space_basis(Matrix.from_cols(vs))
        return kernel_basis(M_(obj["A"]))
    U, W = basis(d["U"]), basis(d["W"])
    n = len((U or W)[0])
    dsum = span_rank(U + W) if U + W else 0
    dint = len(U) + len(W) - dsum
    names = _names(d, n)
    show = lambda o: ("Span(" + ", ".join(fv(x) for x in VS_(o["span"])) + ")") if "span" in o else \
        "{ " + ";  ".join(f"{flin(r, names)} = 0" for r in M_(o["A"]).rows) + " }"
    ident = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    eqU = kernel_basis(Matrix(U)) if U else ident
    eqW = kernel_basis(Matrix(W)) if W else ident
    inter = kernel_basis(Matrix(eqU + eqW)) if eqU + eqW else ident
    ssum = column_space_basis(Matrix.from_cols(U + W)) if U + W else []
    steps = [f"basis of U: {fvs(U)}  (dim {len(U)})", f"basis of W: {fvs(W)}  (dim {len(W)})",
             f"dim(U+W) = rank of all basis vectors together = {dsum}; a basis of U + W: {fvs(ssum)}",
             f"Grassmann: dim(U∩W) = {len(U)} + {len(W)} - {dsum} = {dint}",
             f"U ∩ W: solve the Cartesian equations of U and of W together → basis {fvs(inter)}",
             "direct sum ⇔ U ∩ W = {0}: " + ("yes" if dint == 0 else "no")]
    allp = {"U": ("a basis of U", SpanAns(U, n, "U")), "W": ("a basis of W", SpanAns(W, n, "W")),
            "dim_sum": ("dim(U + W)", IntAns(dsum)), "dim_int": ("dim(U ∩ W)", IntAns(dint)),
            "direct": ("is the sum direct?", YesNoAns(dint == 0, steps[-1])),
            "int_basis": ("a basis of U ∩ W ('none' if {0})", SpanAns(inter, n, "U ∩ W")),
            "sum_basis": ("a basis of U + W", SpanAns(ssum, n, "U + W"))}
    return f"U = {show(d['U'])}\nW = {show(d['W'])}", [allp[k] for k in grassmann_keys(d)], steps


POS_OPTS = ["coincident / one contained in the other", "parallel (no common points)",
            "incident (they meet, neither contains the other)", "skew (sghembi)"]


def classify_position(P1, V1, P2, V2) -> int:
    meet = in_span(vsub(P2, P1), V1 + V2)
    r1, r2, r12 = span_rank(V1), span_rank(V2), span_rank(V1 + V2)
    contained = r12 == max(r1, r2)  # one giacitura inside the other
    if meet:
        return 0 if contained else 2
    if contained:
        return 1
    return 3


@task("position")
def t_position(d, text):
    names = _names(d, len(_affine(d["S1"])[0]))
    P1, V1 = _affine(d["S1"])
    P2, V2 = _affine(d["S2"])
    k = classify_position(P1, V1, P2, V2)
    steps = [f"S1 = {fparam(P1, V1)}", f"S2 = {fparam(P2, V2)}",
             f"giaciture: dim {span_rank(V1)}, {span_rank(V2)}; dim(sum of giaciture) = {span_rank(V1 + V2)}",
             f"P2 - P1 = {fv(vsub(P2, P1))} {'∈' if in_span(vsub(P2, P1), V1 + V2) else '∉'} S1,0 + S2,0 "
             f"⇒ they {'do' if in_span(vsub(P2, P1), V1 + V2) else 'do not'} meet.",
             f"⇒ {POS_OPTS[k]}"]
    a1 = d.get("name1", "S1")
    a2 = d.get("name2", "S2")
    shown = f"{a1} = {_affine_text(d['S1'], names)}\n{a2} = {_affine_text(d['S2'], names)}"
    ans = ChoiceAns(POS_OPTS, k)
    return shown + "\n" + ans.render(), [("position", ans)], steps


# ------------------------------------------------------------------ chapter 3
@task("kerim")
def t_kerim(d, text):
    """f with matrix A (w.r.t. the given bases, default canonical): ker, Im, inj., surj."""
    A = M_(d["A"])
    K = kernel_basis(A)
    C = column_space_basis(A)
    s = solve(A, [0] * A.m)
    steps = s.steps + [f"ker: {fvs(K) if K else '{0}'} (dim {len(K)});  Im = Col: {fvs(C)} (dim {len(C)})",
                       f"nullità + rango: {len(K)} + {len(C)} = {A.n}"]
    shown = named(d.get("name", "A"), A) + (f"\n({d['coords_note']})" if d.get("coords_note") else "")
    return shown, [("basis of ker (coordinates; 'none' if {0})", SpanAns(K, A.n, "ker")),
                   ("basis of Im (coordinates)", SpanAns(C, A.m, "Im")),
                   ("injective?", YesNoAns(not K, "Injective ⇔ ker = {0}.")),
                   ("surjective?", YesNoAns(len(C) == A.m, "Surjective ⇔ rank = dim of the codomain."))], steps


@task("repmat")
def t_repmat(d, text):
    """M^B_C(f) = C^-1 · A · B, with A = matrix of f in the reference bases and B, C given by
    coordinates (columns) in those reference bases. Default B, C = canonical."""
    A = M_(d["A"])
    B = Matrix.from_cols(VS_(d["B"])) if "B" in d else Matrix.identity(A.n)
    C = Matrix.from_cols(VS_(d["C"])) if "C" in d else Matrix.identity(A.m)
    Ci = inverse(C)
    if Ci is None or det(B) == 0:
        raise TaskError("B or C is not a basis")
    R = Ci * A * B
    steps = [hcat("f(b_j) in reference coordinates = A·B = ", fm(A * B)),
             hcat("coordinates w.r.t. C: C^-1·(A·B) = ", fm(R))]
    shown = "" if d.get("hide") else named(d.get("name", "A"), A)
    shown += (f"\n({d['coords_note']})" if d.get("coords_note") else "")
    parts = [(d.get("label", "M^B_C(f)"), MatrixAns(R))]
    for v in d.get("images", []):
        w = Ci.apply(A.apply(V_(v)))
        parts.append((f"(f({fv(V_(v))}))_C", VectorAns(w)))
        steps.append(f"(f({fv(V_(v))}))_C = C^-1·A·v = {fv(w)}")
    return shown.strip(), parts, steps


@task("interp_matrix")
def t_interp_matrix(d, text):
    """f defined on a basis B = {b_j} (columns, canonical coordinates) by images w_j:
    M^B_E = (w_1|...|w_n), M^E_E = W·B^-1, M^B_B = B^-1·W, M^E_B = B^-1·W·B^-1."""
    B, W = Matrix.from_cols(VS_(d["basis"])), Matrix.from_cols(VS_(d["images"]))
    Bi = inverse(B)
    if Bi is None:
        raise TaskError("not a basis")
    allm = {"E_E": W * Bi, "B_E": W, "B_B": Bi * W, "E_B": Bi * W * Bi}
    names = {"E_E": "M^E_E(f)", "B_E": "M^B_E(f)", "B_B": "M^B_B(f)", "E_B": "M^E_B(f)"}
    parts, steps = [], [hcat("B (basis as columns) = ", fm(B), "   W (images as columns) = ", fm(W))]
    for key in d.get("ask", ["E_E"]):
        parts.append((names[key], MatrixAns(allm[key])))
        steps.append(hcat(f"{names[key]} = ", fm(allm[key])))
    steps.insert(1, "M^E_E(f)·B = W  ⇒  M^E_E(f) = W·B^-1;  M^B_E(f) = W;  M^B_B(f) = B^-1·W;  M^E_B(f) = B^-1·W·B^-1")
    return "", parts, steps


@task("change_basis")
def t_change_basis(d, text):
    """Both M^B_C(id) (= C^-1 B) and M^C_B(id), bases given by coordinates in a reference basis."""
    B, C = Matrix.from_cols(VS_(d["B"])), Matrix.from_cols(VS_(d["C"]))
    MBC = inverse(C) * B
    MCB = inverse(MBC)
    parts = [("M^B_C(id)", MatrixAns(MBC)), ("M^C_B(id)", MatrixAns(MCB))]
    steps = [hcat("M^B_C(id) = C^-1·B = ", fm(MBC)), hcat("M^C_B(id) = (M^B_C(id))^-1 = ", fm(MCB))]
    if "vB" in d:
        v = MBC.apply(V_(d["vB"]))
        parts.append((f"(v)_C for (v)_B = {fv(V_(d['vB']))}", VectorAns(v)))
        steps.append(f"(v)_C = M^B_C(id)·(v)_B = {fv(v)}")
    if "vC" in d:
        v = MCB.apply(V_(d["vC"]))
        parts.append((f"(v)_B for (v)_C = {fv(V_(d['vC']))}", VectorAns(v)))
        steps.append(f"(v)_B = M^C_B(id)·(v)_C = {fv(v)}")
    note = f"\n({d['coords_note']})" if d.get("coords_note") else ""
    return f"B = {{{', '.join(fv(c) for c in B.cols())}}}\nC = {{{', '.join(fv(c) for c in C.cols())}}}{note}", parts, steps


# ------------------------------------------------------------------ chapter 4
@task("eigen")
def t_eigen(d, text):
    """p_A, real spectrum, eigenspaces, diagonalisable over R (+ P)."""
    A = M_(d["A"])
    n = A.n
    p = charpoly(A)
    roots, rest = p.rational_roots()
    if rest.degree > 0 and real_root_count(rest) > 0:
        # irrational real eigenvalues: still checkable when all eigenvalues are real and simple
        from ..core.poly import poly_gcd
        squarefree = poly_gcd(p, Poly([k * p.c[k] for k in range(1, len(p.c))])).degree == 0
        if not (squarefree and real_root_count(p) == n):
            raise TaskError("irrational eigenvalues with multiplicity")
        steps = [f"p_A(t) = {p} = {factored(p)}",
                 f"p_A has {n} distinct real roots (some irrational) ⇒ all eigenvalues simple ⇒ diagonalisable over R."]
        return named(d.get("name", "A"), A), [("p_A(t)", PolyAns(p)),
                                              ("diagonalisable over R?", YesNoAns(True, steps[-1]))], steps
    parts = [("p_A(t)", PolyAns(p)), ("spectrum over R", ScalarSetAns(list(roots), "S(A)"))]
    steps = [f"p_A(t) = {p} = {factored(p)}"]
    cols, ok = [], sum(roots.values()) == n
    for lam in sorted(roots):
        K = kernel_basis(Matrix.identity(n).scale(lam) - A)
        steps.append(f"λ = {fs(lam)}: ma = {roots[lam]}, V_λ = Span({fvs(K)}), mg = {len(K)}")
        if d.get("eigenspaces", True):
            parts.append((f"basis of V_{fs(lam)}", SpanAns(K, n, f"V_{fs(lam)}")))
        cols += K
        ok = ok and len(K) == roots[lam]
    steps.append("diagonalisable over R: " + ("yes" if ok else "no"))
    parts.append(("diagonalisable over R?", YesNoAns(ok, steps[-1])))
    if ok and d.get("ask_P", True):
        parts.append(("P with P^-1AP diagonal", DiagonalizerAns(A, Matrix.from_cols(cols))))
    return named(d.get("name", "A"), A) + (f"\n({d['coords_note']})" if d.get("coords_note") else ""), parts, steps


@task("charpoly")
def t_charpoly(d, text):
    A = M_(d["A"])
    p = charpoly(A)
    roots, rest = p.rational_roots()
    if rest.degree > 0 and real_root_count(rest) > 0:
        if d.get("spectrum", True):
            raise TaskError("irrational real eigenvalues: set spectrum = false")
        return named(d.get("name", "A"), A), [("p_A(t)", PolyAns(p))], [
            f"p_A(t) = {p}; its real roots are irrational: {factored(p)} — use the quadratic formula."]
    return named(d.get("name", "A"), A), [("p_A(t)", PolyAns(p)),
                                          ("spectrum over R", ScalarSetAns(list(roots), "S(A)"))], \
        [f"p_A(t) = {p} = {factored(p)}"]


def _diag_over_R(A: Matrix) -> bool:
    p = charpoly(A)
    roots, rest = p.rational_roots()
    if rest.degree > 0:
        if real_root_count(rest) > 0:
            raise TaskError("irrational eigenvalues at this parameter value")
        return False
    return all(len(kernel_basis(Matrix.identity(A.n).scale(l) - A)) == m for l, m in roots.items())


@task("param_diag")
def t_param_diag(d, text):
    """Diagonalisable over R as the parameter varies. `candidates` (author-supplied) are the
    values where the answer may change; the sympy test re-derives them independently."""
    pm = _pm(d)
    var = pm.var
    cand = sorted(num(c) for c in d["candidates"])
    g = generic_value(cand)
    gen = _diag_over_R(pm.at(g))
    exc = [c for c in cand if _diag_over_R(pm.at(c)) != gen]
    steps = [f"for generic {var} the matrix is {'' if gen else 'NOT '}diagonalisable over R"]
    for c in cand:
        Ac = pm.at(c)
        steps.append(f"{var} = {fs(c)}: p = {factored(charpoly(Ac))}; diagonalisable: {'yes' if _diag_over_R(Ac) else 'no'}")
    if d.get("why"):
        steps.insert(0, d["why"])
    return fmt_pmatrix(pm, d.get("name", "A")), [
        (f"diagonalisable over R for generic {var}?", YesNoAns(gen)),
        (f"exceptional values of {var} (where the answer is the opposite; 'none' if none)", ScalarSetAns(exc))], steps


class CharpolyMatrixAns(Answer):
    """Any square matrix whose characteristic polynomial is p (and optionally with prescribed
    eigenspaces V_λ = Span(...))."""

    def __init__(self, p: Poly, model: Matrix, eigenspaces: dict | None = None):
        super().__init__()
        self.p, self.model, self.eig = p, model, eigenspaces or {}
        self.fmt_hint = f"a {p.degree}×{p.degree} matrix, rows separated by ';'"

    def check(self, text):
        from ..core.parse import parse_matrix
        from ..exercise import Check
        X = parse_matrix(text, (self.p.degree, self.p.degree))
        q = charpoly(X)
        if q != self.p:
            return Check.wrong(f"Your matrix has p(t) = {q}, not {self.p}.")
        for lam, basis in self.eig.items():
            K = kernel_basis(Matrix.identity(X.n).scale(lam) - X)
            if not (len(K) == len(basis) and same_span(K, basis)):
                return Check.wrong(f"Right characteristic polynomial, but V_{fs(lam)} = Span({fvs(K)}), not Span({fvs(basis)}).", 0.5)
        return Check.right()

    def show(self):
        from ..core.fmt import fm_inline
        return "\n" + fm(self.model) + f"\n(type: {fm_inline(self.model)}; any matrix with these properties is accepted)"

    def typed(self):
        from ..core.fmt import fm_inline
        return fm_inline(self.model)


def _companion(p: Poly) -> Matrix:
    n = p.degree
    return Matrix([[(1 if i == j + 1 else 0) if j < n - 1 else -p.c[i] for j in range(n)] for i in range(n)])


@task("charpoly_target")
def t_charpoly_target(d, text):
    p = parse_poly(d["p"], d.get("var", "x"))
    p = p / p.c[-1]
    C = _companion(p)
    steps = [f"p = {p.show(d.get('var', 'x'))} = {factored(p).replace('t', d.get('var', 'x'))}",
             "One answer: the matrice compagna C(p) (§4.5), with p_{C(p)} = p:", fm(C),
             "If p splits into rational roots, a diagonal (or triangular) matrix with those roots also works."]
    return "", [("a matrix A with p_A = p", CharpolyMatrixAns(p, C))], steps


@task("eigen_target")
def t_eigen_target(d, text):
    """A matrix with characteristic polynomial p and eigenspaces V_λ = given spans. Model:
    P·J·P^-1 with J in Jordan form (one 2×2 block when an eigenvalue is defective)."""
    p = parse_poly(d["p"], d.get("var", "x"))
    eig = {num(k): VS_(v) for k, v in d["eigenspaces"].items()}
    Pcols = [V_(c) for c in d["P"]]
    J = M_(d["J"])
    P = Matrix.from_cols(Pcols)
    model = P * J * inverse(P)
    if charpoly(model) != p:
        raise TaskError("model matrix does not have the requested p")
    for lam, basis in eig.items():
        if not same_span(kernel_basis(Matrix.identity(model.n).scale(lam) - model), basis):
            raise TaskError("model matrix has the wrong eigenspace")
    steps = ["Choose a basis adapted to the eigenspaces and build A = P·J·P⁻¹:", hcat("P = ", fm(P), "   J = ", fm(J)),
             hcat("A = ", fm(model))]
    return "", [("a matrix A with these properties", CharpolyMatrixAns(p, model, eig))], steps


@task("roots")
def t_roots(d, text):
    """Factorisation questions: rational roots, number of distinct real and complex roots."""
    p = parse_poly(d["p"], d.get("var", "x"))
    roots, rest = p.rational_roots()
    nreal = len(roots) + real_root_count(rest)
    from ..core.poly import poly_gcd
    sqfree_deg = p.degree - poly_gcd(p, Poly([k * p.c[k] for k in range(1, len(p.c))])).degree
    steps = [f"p = {p.show(d.get('var', 'x'))} = {factored(p).replace('t', d.get('var', 'x'))}",
             f"distinct roots in C: {sqfree_deg}; real: {nreal}; rational: {len(roots)}"]
    return f"p({d.get('var', 'x')}) = {p.show(d.get('var', 'x'))}", [
        ("rational roots", ScalarSetAns(list(roots), "the rational roots")),
        ("number of distinct real roots", IntAns(nreal)),
        ("number of distinct complex roots", IntAns(sqfree_deg))], steps


# ------------------------------------------------------------------ chapter 5
@task("metric")
def t_metric(d, text):
    v, w = V_(d["v"]), V_(d["w"])
    S = M_(d["S"]) if "S" in d else None
    ip = gram_inner(S)
    nv, nw = norm(v, S), norm(w, S)
    dist = Surd.sqrt(ip(vsub(v, w), vsub(v, w)))
    parts = [("||v||", RealAns(nv)), ("||w||", RealAns(nw)), ("d(v, w)", RealAns(dist))]
    if any(v) and any(w):
        c = Surd(ip(v, w)) / (nv * nw)
        parts.append(("cos θ", RealAns(c)))
    steps = [f"⟨v,v⟩ = {fs(ip(v, v))}, ⟨w,w⟩ = {fs(ip(w, w))}, ⟨v,w⟩ = {fs(ip(v, w))}",
             f"||v|| = {nv.pretty()}, ||w|| = {nw.pretty()}, d = ||v - w|| = {dist.pretty()}"]
    if len(parts) == 4:
        steps.append(f"cos θ = ⟨v,w⟩/(||v||·||w||) = {parts[3][1].value.pretty()}")
    shown = f"v = {fv(v)},  w = {fv(w)}" + ("" if S is None else "\n" + named("Gram matrix", S))
    return shown, parts, steps


@task("gram_schmidt")
def t_gs(d, text):
    B = VS_(d["vectors"])
    S = M_(d["S"]) if "S" in d else None
    cs, steps = gram_schmidt(B, S)
    return ("vectors: " + ", ".join(fv(b) for b in B) + ("" if S is None else "\n" + named("Gram matrix", S))), \
        [("orthogonal basis c1, c2, ... (rescaling allowed)", ScaledVectorsAns(cs))], steps


@task("gram")
def t_gram(d, text):
    """Gram matrix of a bilinear form (given in canonical coordinates) + is it a scalar product?"""
    G = M_(d["G"])
    from ..topics.ch5 import complete_squares
    ds, sq = complete_squares(G) if G.is_symmetric() else ([], ["not symmetric"])
    ok = G.is_symmetric() and len(ds) == G.n and all(x > 0 for x in ds)
    return "", [("Gram matrix", MatrixAns(G)), ("is it a scalar product?", YesNoAns(ok))], \
        [fm(G)] + sq + [("positive definite and symmetric ⇒ scalar product" if ok else "not a scalar product")]


@task("orthobasis")
def t_orthobasis(d, text):
    """Orthogonal / orthonormal basis questions (standard product). Each basis vector is
    scale · dir with scale a string like '1/sqrt(3)' (default 1) and dir rational; exact."""
    from ..core.parse import parse_real
    from ..exercise import RealVectorAns
    vecs = []
    for b in d["basis"]:
        sc = parse_real(str(b.get("scale", "1")))
        vecs.append((sc, V_(b["dir"])))
    v = V_(d["v"])
    n = len(vecs)
    ortho = all(dot(vecs[i][1], vecs[j][1]) == 0 for i in range(n) for j in range(i + 1, n))
    norms2 = [sc.square() * dot(dr, dr) for sc, dr in vecs]
    normal = ortho and all(x == 1 for x in norms2)
    show = lambda sc, dr: ("(" + ", ".join((sc * x).pretty() for x in dr) + ")")
    steps = ["⟨b_i, b_j⟩ for i ≠ j: " + ", ".join((sc1 * sc2 * Surd(dot(d1, d2))).pretty()
                                                  for k, (sc1, d1) in enumerate(vecs) for (sc2, d2) in vecs[k + 1:]),
             "||b_i||² = " + ", ".join(fs(x) for x in norms2)]
    parts = [("orthogonal?", YesNoAns(ortho)), ("orthonormal?", YesNoAns(normal))]
    if ortho:
        if normal:
            coords = [sc * dot(v, dr) for sc, dr in vecs]
            steps.append("Orthonormal: (v)_B = (⟨v,b1⟩, ..., ⟨v,bn⟩) = (" + ", ".join(c.pretty() for c in coords) + ")")
            parts.append(("(v)_B", RealVectorAns(coords)))
        else:
            coords = [Surd(dot(v, dr)) / Surd.sqrt(dot(dr, dr)) for _, dr in vecs]
            steps.append("Normalise b'_i = b_i/||b_i||; then (v)_B' = (⟨v,b'_i⟩) = (" + ", ".join(c.pretty() for c in coords) + ")")
            parts.append(("(v)_B' (B' = normalised B)", RealVectorAns(coords)))
    else:
        steps.append("Some ⟨b_i, b_j⟩ ≠ 0: not orthogonal (so not orthonormal).")
    return "B = {" + ", ".join(show(sc, dr) for sc, dr in vecs) + "},  v = " + fv(v), parts, steps


# ------------------------------------------------------------------ parameters (al variare di λ)
def _pm(d, key="A"):
    return PMatrix(d[key], d.get("param", "λ"))


def _drop_values(pm: PMatrix):
    prof = rank_profile(pm)
    if prof.irrational is not None:
        raise TaskError(f"rank drops at irrational values (roots of {prof.irrational})")
    return prof


@task("param_invertible")
def t_param_invertible(d, text):
    pm = _pm(d)
    p = pm.det()
    roots, rest = p.rational_roots() if p.degree > 0 else ({}, p)
    if p.degree > 0 and rest.degree > 0 and real_root_count(rest) > 0:
        raise TaskError("irrational roots")
    if not p.c:
        roots = None
    var = pm.var
    det_part = [(f"det as a polynomial in {var}", PolyAns(p, var=var))] if d.get("ask_det") else []
    steps = [f"det = {p.show(var)}" + (f" = {factored(p).replace('t', var)}" if p.degree > 0 else ""),
             ("never invertible (det ≡ 0)" if roots is None else
              f"not invertible exactly for {var} ∈ {values_text(sorted(roots))}")]
    if roots is None:
        return fmt_pmatrix(pm, d.get("name", "A")), det_part + [("invertible for some value?", YesNoAns(False))], steps
    return fmt_pmatrix(pm, d.get("name", "A")), det_part + [
        (f"values of {var} where it is NOT invertible ('none' if always invertible)",
         ScalarSetAns(list(roots), f"the set of bad {var}"))], steps


@task("param_rank")
def t_param_rank(d, text):
    pm = _pm(d)
    prof = _drop_values(pm)
    var = pm.var
    parts = [(f"values of {var} where the rank drops ('none' if never)", ScalarSetAns(prof.drops)),
             (f"rank for the other values of {var}", IntAns(prof.generic))]
    steps = [f"generic rank {prof.generic}; the rank drops for {var} ∈ {values_text(prof.drops)}"]
    for c in prof.drops:
        r = rank(pm.at(c))
        parts.append((f"rank for {var} = {fs(c)}", IntAns(r)))
        steps.append(f"{var} = {fs(c)}: rank {r}")
    return fmt_pmatrix(pm, d.get("name", "A")), parts, steps


SYS_OPTS = ["no solutions", "exactly one solution", "infinitely many solutions"]


def _sys_kind(A: Matrix, b) -> tuple[int, int]:
    rA = rank(A)
    rAb = rank(A.hstack(Matrix.col_vector(b)))
    if rA < rAb:
        return 0, 0
    return (1, 0) if rA == A.n else (2, A.n - rA)


def _param_eq_text(pm: PMatrix, pb: PMatrix, names) -> str:
    var = pm.var
    lines = []
    for i in range(pm.m):
        out = ""
        for j in range(pm.n):
            e = pm.rows[i][j]
            if not e.c:
                continue
            if e.degree <= 0:
                c = e.const_value()
                mag = "" if abs(c) == 1 else fs(abs(c))
                if out:
                    out += (" - " if c < 0 else " + ") + mag + names[j]
                else:
                    out = ("-" if c < 0 else "") + mag + names[j]
            else:
                out += (" + " if out else "") + f"({e.show(var)}){names[j]}"
        lines.append(f"   {out or '0'} = {pb.rows[i][0].show(var)}")
    return "\n".join(lines)


@task("param_system")
def t_param_system(d, text):
    """Discussion of a linear system with a parameter. 'Critical values' are the values
    where the outcome (no / one / ∞^k solutions) differs from the generic one."""
    pm = _pm(d)
    pb = PMatrix([[x] for x in d["b"]], pm.var)
    p1, p2 = _drop_values(pm), _drop_values(pm.hstack(pb))
    cand = sorted(set(p1.drops) | set(p2.drops))
    var = pm.var
    g = generic_value(cand)
    gen = _sys_kind(pm.at(g), pb.at(g).col(0))
    crit = [c for c in cand if _sys_kind(pm.at(c), pb.at(c).col(0)) != gen]
    parts = [(f"values of {var} where the answer changes ('none' if it never changes)", ScalarSetAns(crit)),
             (f"for the other values of {var}", ChoiceAns(SYS_OPTS, gen[0]))]
    if gen[0] == 2:
        parts.append((f"free parameters for the other values of {var}", IntAns(gen[1])))
    steps = [f"rg(A) drops for {var} ∈ {values_text(p1.drops)};  rg(A|b) drops for {var} ∈ {values_text(p2.drops)}",
             f"generic {var}: rg(A) = {p1.generic}, rg(A|b) = {p2.generic} ⇒ {SYS_OPTS[gen[0]]}" +
             (f" (∞^{gen[1]})" if gen[0] == 2 else "")]
    for c in cand:
        A, b = pm.at(c), pb.at(c).col(0)
        kc, fc = _sys_kind(A, b)
        steps.append(f"{var} = {fs(c)}: rg(A) = {rank(A)}, rg(A|b) = {rank(A.hstack(Matrix.col_vector(b)))} ⇒ "
                     f"{SYS_OPTS[kc]}" + (f" (∞^{fc})" if kc == 2 else "") +
                     ("" if c in crit else "  — same as the generic case"))
        if c in crit:
            parts.append((f"for {var} = {fs(c)}", ChoiceAns(SYS_OPTS, kc)))
            if kc == 2:
                parts.append((f"free parameters for {var} = {fs(c)}", IntAns(fc)))
    return _param_eq_text(pm, pb, _names(d, pm.n)), parts, steps


@task("param_indep")
def t_param_indep(d, text):
    """Vectors with a parameter (as columns): when dependent / generators."""
    cols = d["vectors"]
    rows = [list(r) for r in zip(*cols)]
    pm = PMatrix(rows, d.get("param", "k"))
    prof = _drop_values(pm)
    nvec, dim = pm.n, pm.m
    var = pm.var
    indep_generic = prof.generic == nvec
    gen_generic = prof.generic == dim
    bad_indep = prof.drops if indep_generic else None
    parts = []
    steps = [f"rank of (v1|...|v{nvec}) is {prof.generic} except for {var} ∈ {values_text(prof.drops)}"]
    for c in prof.drops:
        steps.append(f"{var} = {fs(c)}: rank {rank(pm.at(c))}")
    if indep_generic:
        parts.append((f"values of {var} where they are DEPENDENT ('none' if never)", ScalarSetAns(prof.drops)))
    else:
        parts.append((f"independent for some {var}?", YesNoAns(False, "Too many vectors / rank never maximal.")))
    if d.get("ask_generators", True):
        if gen_generic:
            parts.append((f"values of {var} where they do NOT generate ('none' if never)", ScalarSetAns(prof.drops)))
        else:
            parts.append((f"generators for some {var}?", YesNoAns(False, f"The rank is at most {prof.generic} < {dim}.")))
    labels = d.get("labels") or ["(" + ", ".join(pm.entry_text(i, j) for i in range(dim)) + ")" for j in range(nvec)]
    return "Vectors: " + ",  ".join(labels) + (f"\n({d['coords_note']})" if d.get("coords_note") else ""), parts, steps


# ------------------------------------------------------------------ registry helpers
def build_parts(task_name: str, data: dict, text: str):
    if task_name not in TASKS:
        raise TaskError(f"unknown task '{task_name}'")
    return TASKS[task_name](data, text)


AUTO_TASKS = sorted(t for t in TASKS if t != "open")


@task("perp")
def t_perp(d, text):
    """W⊥ for the standard scalar product (or Gram matrix S): dim and a basis (parametric form).
    W given as span = [[..]] or by Cartesian equations A = [[..]] (rows)."""
    if "span" in d:
        W = column_space_basis(Matrix.from_cols(VS_(d["span"])))
        n = len(VS_(d["span"])[0])
    else:
        W = kernel_basis(M_(d["A"]))
        n = len(d["A"][0])
    S = M_(d["S"]) if "S" in d else None
    rows = [S.apply(w) if S is not None else w for w in W]
    perp = kernel_basis(Matrix(rows)) if rows else [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    steps = [f"basis of W: {fvs(W)} (dim {len(W)})",
             "v ∈ W⊥ ⇔ ⟨v, w⟩ = 0 for every basis vector w: solve the homogeneous system with rows "
             + "; ".join(fv(r) for r in rows),
             f"W⊥ = Span({', '.join(fv(x) for x in perp)}),  dim W⊥ = {n} - {len(W)} = {len(perp)}"]
    nm = d.get("name", "W")
    steps = [x.replace("W⊥", f"{nm}⊥").replace("basis of W:", f"basis of {nm}:") for x in steps]
    if d.get("orthogonal"):
        ob = _orth(perp, S)
        steps.append(f"Gram-Schmidt → orthogonal basis {fvs(ob)}")
        basis_ans = OrthSpanAns(ob, n, f"{nm}⊥", S)
        lbl = f"an ORTHOGONAL basis of {nm}⊥"
    else:
        basis_ans, lbl = SpanAns(perp, n, f"{nm}⊥"), f"a basis of {nm}⊥ ('none' if {{0}})"
    parts = [(f"dim {nm}⊥", IntAns(len(perp))), (lbl, basis_ans)]
    if d.get("only_basis"):
        parts = parts[1:]
    return "", parts, steps


class OrthSpanAns(SpanAns):
    """A basis of the subspace that must also be pairwise orthogonal (standard product or S)."""

    def __init__(self, basis, n, what="W", S=None):
        super().__init__(basis, n, what)
        self.S = S

    def check(self, text):
        res = super().check(text)
        if not res.ok:
            return res
        from ..core.parse import parse_vectors
        xs = parse_vectors(text, self.n)
        ip = (lambda a, b: dot(a, self.S.apply(b))) if self.S is not None else dot
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                if ip(xs[i], xs[j]) != 0:
                    from ..exercise import Check
                    return Check.wrong(f"A basis of {self.what}, but not orthogonal: ⟨{fv(xs[i])}, {fv(xs[j])}⟩ ≠ 0 "
                                       "(apply Gram-Schmidt).", 0.5)
        return res

    def show(self):
        return fvs(self.basis) if self.basis else "none (the space is {0})"


def _orth(vs, S=None):
    """Gram-Schmidt without normalising, rescaled to integers."""
    out = []
    ip = (lambda a, b: dot(a, S.apply(b))) if S is not None else dot
    for v in vs:
        w = list(v)
        for c in out:
            k = ip(v, c) / ip(c, c)
            w = [a - k * b for a, b in zip(w, c)]
        out.append(clear_denominators(w))
    return out


@task("affine_meet")
def t_affine_meet(d, text):
    """Intersection of two affine subspaces S1, S2 ({point, dirs} or {A, b}): a parametric
    description, or 'none' if they do not meet."""
    P1, V1 = _affine(d["S1"])
    P2, V2 = _affine(d["S2"])
    n = len(P1)
    # P1 + V1 a = P2 + V2 c  ⇔  [V1 | -V2] (a, c) = P2 - P1
    cols = [list(v) for v in V1] + [[-x for x in v] for v in V2]
    rhs = vsub(P2, P1)
    if cols:
        M = Matrix.from_cols(cols)
        s = solve(M, rhs)
    else:
        s = None
    names = _names(d, n)
    n1, n2 = d.get("name1", "S"), d.get("name2", "T")
    if s is None or s.kind == "none":
        steps = [f"{n1} ∩ {n2}: P1 + Σ a_i v_i = P2 + Σ c_j w_j has no solution (Rouché-Capelli): "
                 f"{n1} ∩ {n2} = ∅."]
        return "", [(f"{n1} ∩ {n2} ('none' if empty)", AffineAns(None, [], n, f"{n1} ∩ {n2}"))], steps
    k = len(V1)
    pt = [P1[i] + sum(s.particular[j] * V1[j][i] for j in range(k)) for i in range(n)]
    dirs = []
    for b in s.basis:
        dirs.append([sum(b[j] * V1[j][i] for j in range(k)) for i in range(n)])
    dirs = column_space_basis(Matrix.from_cols(dirs)) if dirs and any(any(x) for x in dirs) else []
    steps = [f"Solve P1 + Σ a_i v_i = P2 + Σ c_j w_j:  {n1} ∩ {n2} = {fparam(pt, dirs)}"]
    return "", [(f"{n1} ∩ {n2} ('none' if empty)", AffineAns(pt, dirs, n, f"{n1} ∩ {n2}"))], steps


@task("proj")
def t_proj(d, text):
    """Orthogonal projections of v on W and on W⊥ (standard product), W = span or equations A."""
    if "span" in d:
        W = column_space_basis(Matrix.from_cols(VS_(d["span"])))
    else:
        W = kernel_basis(M_(d["A"]))
    v = V_(d["v"])
    n = len(v)
    ob = _orth(W)
    pw = [Fraction(0)] * n
    for c in ob:
        k = dot(v, c) / dot(c, c)
        pw = [a + k * b for a, b in zip(pw, c)]
    pp = vsub(v, pw)
    nm = d.get("name", "W")
    steps = [f"orthogonal basis of {nm} (Gram-Schmidt): {fvs(ob)}",
             f"π_{nm}(v) = Σ ⟨v, c_i⟩/⟨c_i, c_i⟩ c_i = {fv(pw)}",
             f"π_{nm}⊥(v) = v − π_{nm}(v) = {fv(pp)}"]
    ask = d.get("ask", ["W", "perp"])
    parts = []
    for a in ask:
        if a == "W":
            parts.append((f"π_{nm}(v)", VectorAns(pw)))
        elif a == "perp":
            parts.append((f"π_{nm}⊥(v)", VectorAns(pp)))
        elif a == "dist":
            parts.append((f"d(v, {nm}) = ‖π_{nm}⊥(v)‖", RealAns(norm(pp))))
    return "", parts, steps
