"""Parser for what the student types.

Expressions: integers, decimals (read exactly: 0.5 = 1/2), + - * / ^ (or **),
parentheses, sqrt(...) (also √), implicit multiplication (2t, 3x, 2sqrt(3), 2(1+t)).
Values are either an exact real constant (Surd) or a polynomial in some variables
with rational coefficients (MPoly).
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Sequence

from .matrix import Matrix
from .poly import Poly
from .surd import Surd


class ParseError(ValueError):
    pass


# --------------------------------------------------------------------------- values
class MPoly:
    """Multivariate polynomial: {monomial: coeff}, monomial = sorted tuple of names."""
    __slots__ = ("t",)

    def __init__(self, terms: dict | None = None):
        self.t = {k: v for k, v in (terms or {}).items() if v != 0}

    @staticmethod
    def var(name: str) -> "MPoly":
        return MPoly({(name,): Fraction(1)})

    @staticmethod
    def const(c) -> "MPoly":
        return MPoly({(): Fraction(c)})

    def __add__(self, o: "MPoly") -> "MPoly":
        out = dict(self.t)
        for k, v in o.t.items():
            out[k] = out.get(k, Fraction(0)) + v
        return MPoly(out)

    def __neg__(self):
        return MPoly({k: -v for k, v in self.t.items()})

    def __mul__(self, o: "MPoly") -> "MPoly":
        out: dict = {}
        for k1, v1 in self.t.items():
            for k2, v2 in o.t.items():
                k = tuple(sorted(k1 + k2))
                out[k] = out.get(k, Fraction(0)) + v1 * v2
        return MPoly(out)

    def variables(self) -> set[str]:
        return {x for k in self.t for x in k}

    def is_const(self) -> bool:
        return all(k == () for k in self.t)

    def const_value(self) -> Fraction:
        return self.t.get((), Fraction(0))

    def degree(self) -> int:
        return max((len(k) for k in self.t), default=0)


def _to_mpoly(v) -> MPoly:
    if isinstance(v, MPoly):
        return v
    if not v.is_rational():
        raise ParseError("square roots can only appear in purely numerical answers")
    return MPoly.const(v.coef)


def _norm(v):
    """MPoly that is constant → Surd."""
    if isinstance(v, MPoly) and v.is_const():
        return Surd(v.const_value())
    return v


def _add(a, b):
    a, b = _norm(a), _norm(b)
    if isinstance(a, Surd) and isinstance(b, Surd):
        try:
            return a + b
        except ValueError as e:
            raise ParseError(str(e)) from None
    return _to_mpoly(a) + _to_mpoly(b)


def _mul(a, b):
    a, b = _norm(a), _norm(b)
    if isinstance(a, Surd) and isinstance(b, Surd):
        return a * b
    return _to_mpoly(a) * _to_mpoly(b)


def _div(a, b):
    a, b = _norm(a), _norm(b)
    if not isinstance(b, Surd):
        raise ParseError("division by an expression containing variables is not allowed")
    if b.coef == 0:
        raise ParseError("division by zero")
    if isinstance(a, Surd):
        return a / b
    if not b.is_rational():
        raise ParseError("cannot divide a polynomial by a square root")
    return a * MPoly.const(1 / b.coef)


def _pow(a, b):
    b = _norm(b)
    if not (isinstance(b, Surd) and b.is_rational() and b.coef.denominator == 1 and b.coef >= 0):
        raise ParseError("exponents must be non-negative integers")
    k = int(b.coef)
    if k > 64:
        raise ParseError("exponent too large")
    out = Surd(1)
    for _ in range(k):
        out = _mul(out, a)
    return out


# --------------------------------------------------------------------------- tokenizer
_TOKEN = re.compile(r"\s*(?:(\d+\.\d*|\.\d+|\d+)|(sqrt|√)|([^\W\d_][^\W\d_]*\d*)|(\*\*|[-+*/^()]))")


def _tokenize(s: str) -> list[tuple[str, str]]:
    s = s.replace("−", "-").replace("·", "*").replace("×", "*").replace("²", "^2").replace("³", "^3")
    out, pos = [], 0
    s = s.strip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m or m.end() == pos:
            raise ParseError(f"I can't read '{s[pos:]}'")
        num, sq, name, op = m.groups()
        if num is not None:
            out.append(("num", num))
        elif sq is not None:
            out.append(("sqrt", sq))
        elif name is not None:
            out.append(("name", name))
        else:
            out.append(("op", "^" if op == "**" else op))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, s: str, variables: Sequence[str]):
        self.toks = _tokenize(s)
        self.i = 0
        self.vars = set(variables)

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else ("eof", "")

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self):
        if not self.toks:
            raise ParseError("empty answer")
        v = self.expr()
        if self.peek()[0] != "eof":
            raise ParseError(f"unexpected '{self.peek()[1]}'")
        return _norm(v)

    def expr(self):
        v = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            w = self.term()
            v = _add(v, w) if op == "+" else _add(v, _mul(Surd(-1), w))
        return v

    def _starts_factor(self) -> bool:
        k, val = self.peek()
        return k in ("num", "sqrt", "name") or (k, val) == ("op", "(")

    def term(self):
        v = self.unary()
        while True:
            if self.peek() in (("op", "*"), ("op", "/")):
                op = self.take()[1]
                w = self.unary()
                v = _mul(v, w) if op == "*" else _div(v, w)
            elif self._starts_factor():  # implicit multiplication
                v = _mul(v, self.power())
            else:
                return v

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            return _mul(Surd(-1), self.unary())
        if self.peek() == ("op", "+"):
            self.take()
            return self.unary()
        return self.power()

    def power(self):
        base = self.atom()
        if self.peek() == ("op", "^"):
            self.take()
            exp = self.unary()
            return _pow(base, exp)
        return base

    def atom(self):
        k, val = self.take()
        if k == "num":
            return Surd(Fraction(val))
        if k == "sqrt":
            if self.peek() == ("op", "("):
                self.take()
                inner = _norm(self.expr())
                if self.take() != ("op", ")"):
                    raise ParseError("missing ')'")
            else:
                inner = _norm(self.atom())
            if not isinstance(inner, Surd) or not inner.is_rational():
                raise ParseError("sqrt(...) must contain a rational number")
            if inner.coef < 0:
                raise ParseError("square root of a negative number")
            return Surd.sqrt(inner.coef)
        if k == "name":
            if val in self.vars:
                return MPoly.var(val)
            raise ParseError(f"unknown symbol '{val}'" + (
                f" (allowed variables: {', '.join(sorted(self.vars))})" if self.vars else ""))
        if (k, val) == ("op", "("):
            v = self.expr()
            if self.take() != ("op", ")"):
                raise ParseError("missing ')'")
            return v
        raise ParseError(f"unexpected '{val or 'end of input'}'")


def parse_expr(s: str, variables: Sequence[str] = ()):
    return _Parser(s, variables).parse()


def parse_real(s: str) -> Surd:
    v = parse_expr(s)
    if not isinstance(v, Surd):
        raise ParseError("expected a number")
    return v


def parse_scalar(s: str) -> Fraction:
    v = parse_real(s)
    if not v.is_rational():
        raise ParseError("expected a rational number (fraction), not a square root")
    return v.coef


def parse_int(s: str) -> int:
    q = parse_scalar(s)
    if q.denominator != 1:
        raise ParseError("expected an integer")
    return int(q)


def parse_poly(s: str, var: str = "t") -> Poly:
    v = parse_expr(s, [var])
    if isinstance(v, Surd):
        if not v.is_rational():
            raise ParseError("coefficients must be rational")
        return Poly([v.coef])
    coeffs: dict[int, Fraction] = {}
    for mono, c in v.t.items():
        coeffs[len(mono)] = c
    deg = max(coeffs) if coeffs else 0
    return Poly([coeffs.get(k, Fraction(0)) for k in range(deg + 1)])


# --------------------------------------------------------------------------- structures
def split_top(s: str, seps: str) -> list[str]:
    """Split at separators that are not inside parentheses/brackets."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and ch in seps:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [x.strip() for x in out]


def _strip_brackets(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] in "([{" and s[-1] in ")]}" and _matching(s):
        s = s[1:-1].strip()
    return s


def _matching(s: str) -> bool:
    """True if the first bracket closes at the very end."""
    depth = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return False
    return depth == 0


def _entries(s: str) -> list[str]:
    s = _strip_brackets(s)
    if not s:
        raise ParseError("empty vector")
    if "," in s:
        parts = split_top(s, ",")
    else:
        # whitespace separated; glue a lone sign to the next token ("- 1")
        raw = s.split()
        parts, carry = [], ""
        for tok in raw:
            if tok in ("+", "-"):
                carry += tok
                continue
            parts.append(carry + tok)
            carry = ""
        if carry:
            raise ParseError("dangling sign")
    if any(p == "" for p in parts):
        raise ParseError("empty entry in vector")
    return parts


def parse_vector(s: str, n: int | None = None) -> list[Fraction]:
    v = [parse_scalar(p) for p in _entries(s)]
    if n is not None and len(v) != n:
        raise ParseError(f"expected a vector with {n} components, got {len(v)}")
    return v


def parse_vectors(s: str, n: int | None = None) -> list[list[Fraction]]:
    """Several vectors: '(1,0,1); (0,1,1)'  or  '(1,0,1), (0,1,1)'  or 'none'."""
    s = s.strip()
    if s.lower() in ("", "none", "empty", "{}", "0 vectors", "nessuno", "vuota", "∅"):
        return []
    s2 = s
    low = s2.lower()
    if low.startswith("span"):
        s2 = s2[4:].strip()
        s2 = _strip_brackets(s2) if _matching(s2) else s2
    parts = split_top(s2, ";")
    if len(parts) == 1:
        inner = split_top(s2, ",")
        if len(inner) > 1 and all(p.startswith("(") or p.startswith("[") for p in inner):
            parts = inner
    return [parse_vector(p, n) for p in parts if p]


def parse_matrix(s: str, shape: tuple[int, int] | None = None) -> Matrix:
    s = s.strip()
    if s.startswith("[[") or s.startswith("(("):
        rows = split_top(_strip_brackets(s), ",;")
    else:
        rows = split_top(s, ";\n")
        if len(rows) == 1 and shape and shape[0] > 1:
            # maybe '[1 2][3 4]' style
            rows = [r for r in re.split(r"\]\s*\[", s.strip("[] ")) if r]
    rows = [r for r in rows if r.strip()]
    try:
        M = Matrix([parse_vector(r) for r in rows])
    except ValueError as e:
        if isinstance(e, ParseError):
            raise
        raise ParseError("all rows must have the same length (separate rows with ';')") from None
    if shape and M.shape != shape:
        raise ParseError(f"expected a {shape[0]}x{shape[1]} matrix, got {M.m}x{M.n}")
    return M


NONE_WORDS = {"none", "no solution", "no solutions", "nessuna", "impossible", "impossibile",
              "empty", "vuoto", "∅", "{}"}


def parse_param(s: str, n: int) -> tuple[list[Fraction], list[list[Fraction]]] | None:
    """Parametric form '(1,2,0) + t(1,0,1) + s(0,1,1)'  or  '(1,2,0) + span((1,0,1),(0,1,1))'.
    Returns (point, directions); None for 'no solutions'."""
    st = s.strip()
    if st.lower() in NONE_WORDS:
        return None
    terms = split_top(_minus_to_plus(st.replace("−", "-")), "+")
    # also split on top-level '-' that precede a parameter: (1,2) - t(1,1)
    point = [Fraction(0)] * n
    dirs: list[list[Fraction]] = []
    have_point = False
    for term in terms:
        if not term:
            continue
        sign = 1
        while term.startswith("-"):
            sign = -sign
            term = term[1:].strip()
        low = term.lower()
        if low.startswith("span"):
            for v in parse_vectors(term, n):
                dirs.append(v)
            continue
        m = re.match(r"^([A-Za-z]\w*)\s*\*?\s*(\(.*\)|\[.*\])$", term)
        m2 = re.match(r"^(\(.*\)|\[.*\])\s*\*?\s*([A-Za-z]\w*)$", term)
        if m or m2:
            body = m.group(2) if m else m2.group(1)
            dirs.append([sign * x for x in parse_vector(body, n)])
            continue
        if have_point:
            raise ParseError("only one constant vector is allowed; write directions as t(...)")
        point = [sign * x for x in parse_vector(term, n)]
        have_point = True
    return point, dirs


def _minus_to_plus(s: str) -> str:
    """'a - b' at top level → 'a + -b' so that splitting on '+' keeps the sign."""
    out, depth = [], 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "-" and depth == 0 and s[:i].strip() and s[:i].strip()[-1] not in "+-":
            out.append("+")
        out.append(ch)
    return "".join(out)


def parse_equations(s: str, names: Sequence[str]) -> list[tuple[list[Fraction], Fraction]]:
    """Linear equations 'x - 2z = 0; y + w = 3'. Returns (coefficients, rhs) pairs."""
    eqs = []
    for part in split_top(s, ";\n,"):
        if not part:
            continue
        if part.count("=") != 1:
            raise ParseError(f"each equation needs exactly one '=': '{part}'")
        lhs, rhs = part.split("=")
        v = _add(_to_mpoly(_norm(parse_expr(lhs, names))),
                 _mul(Surd(-1), _to_mpoly(_norm(parse_expr(rhs, names)))))
        v = _to_mpoly(_norm(v))
        if v.degree() > 1:
            raise ParseError("equations must be linear")
        coeffs = [v.t.get((x,), Fraction(0)) for x in names]
        eqs.append((coeffs, -v.const_value()))
    if not eqs:
        raise ParseError("no equations found")
    return eqs


YES = {"y", "yes", "s", "si", "sì", "true", "vero", "1"}
NO = {"n", "no", "false", "falso", "0"}


def parse_yesno(s: str) -> bool:
    w = s.strip().lower().rstrip(".!")
    if w in YES:
        return True
    if w in NO:
        return False
    raise ParseError("answer yes or no (y/n)")
