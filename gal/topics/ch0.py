"""Capitolo 0 — Introduzione: vettori geometrici, rette e piani."""
from __future__ import annotations

import random
from fractions import Fraction

from ..core.fmt import fparam, fs, fv
from ..core.geom import cartesian_from_param
from ..core.matrix import vadd, vscale, vsub
from ..exercise import AffineAns, EquationsAns, VectorAns, mk
from .base import Gen, Topic
from .rand import ivec, nz

# ------------------------------------------------------------------ vectors
VECTORS_LESSON = [
    ("Intuition", """\
A vector is a *movement*. The applied vector AB (vettore applicato) goes from the point A
to the point B. If we forget where it starts and keep only the movement, we get a
free vector (vettore libero) v = [AB], determined by direction (direzione),
orientation (verso) and length (modulo). The zero vector (vettore nullo) 0 = [AA] has
length 0 and no direction."""),
    ("Operations", """\
• Scalar multiple (prodotto per uno scalare) c·v: same direction, length |c|·|v|,
  same orientation if c > 0, opposite if c < 0; 0·v = 0 and (-1)·v = -v.
• Sum (somma) v + w: parallelogram rule (regola del parallelogramma) — apply w at the
  tip of v; or the diagonal of the parallelogram with sides v and w.
Properties: the sum is associative and commutative, has neutral element 0 and opposites;
(cd)·v = c·(d·v), 1·v = v, (c+d)·v = c·v + d·v, c·(v+w) = c·v + c·w.
(These eight rules will become the axioms SV1–SV8 of a vector space in Chapter 2.)"""),
    ("Coordinates", """\
Fix a Cartesian reference system (sistema di riferimento cartesiano) (O, i, j, k).
A point P, the applied vector OP and the free vector [OP] are identified with the
coordinates (x_P, y_P, z_P): v = x_P·i + y_P·j + z_P·k.
In coordinates the operations are component-wise:
   v = 2i + 3j - 2k, w = 3i - 3j + 3k  ⇒  v + w = 5i + 0j + k,  2v = 4i + 6j - 4k."""),
]


def ex_vec_combo(rng: random.Random, data=None):
    v, w, a, b = data or (ivec(rng, 3), ivec(rng, 3), nz(rng, -3, 3), nz(rng, -3, 3))
    res = vadd(vscale(a, v), vscale(b, w))
    prompt = (f"Let v = {fv(v)} and w = {fv(w)} (coordinates w.r.t. i, j, k).\n"
              f"Compute {fs(a)}·v {'+' if b > 0 else '-'} {fs(abs(b))}·w.")
    sol = [f"{fs(a)}·v = {fv(vscale(a, v))}",
           f"{fs(b)}·w = {fv(vscale(b, w))}",
           f"sum, component by component: {fv(res)}"]
    mistakes = [(vadd(v, w), "You added v and w without multiplying by the scalars."),
                (vadd(vscale(a, v), vscale(-b, w)), "Sign of the second scalar is wrong.")]
    return mk("vectors", "combo", prompt, VectorAns(res, mistakes=mistakes), sol, v=v, w=w, a=a, b=b)


def ex_vec_AB(rng, data=None):
    A, B = data or (ivec(rng, 3, nonzero=False), ivec(rng, 3, nonzero=False))
    if A == B:
        B = vadd(B, [1, 0, 0])
    res = vsub(B, A)
    prompt = f"A = {fv(A)}, B = {fv(B)}. Write the coordinates of the free vector [AB]."
    sol = ["[AB] is the movement from A to B: coordinates B - A.", f"B - A = {fv(res)}"]
    return mk("vectors", "AB", prompt,
              VectorAns(res, mistakes=[(vsub(A, B), "You computed A - B = [BA]: same direction and length, opposite orientation.")]),
              sol, A=A, B=B)


VECTORS = Topic(
    "vectors", "§0.1", "Geometric vectors in the plane and in space",
    "Vettori geometrici nel piano e nello spazio", "0 Introduzione", [],
    VECTORS_LESSON,
    [lambda: ex_vec_combo(None, ([2, 3, -2], [3, -3, 3], 1, 1))],
    [Gen("combo", "linear combination a·v + b·w", ex_vec_combo),
     Gen("AB", "free vector [AB] from two points", ex_vec_AB)],
)

# ------------------------------------------------------------------ lines and planes
LINES_LESSON = [
    ("Intuition", """\
A line is "a point plus all the multiples of one direction"; a plane is "a point plus all
combinations of two different directions". Moving along a parameter t ∈ R we sweep the line."""),
    ("Parametric equations (equazioni parametriche)", """\
Line through P = (x_P, y_P, z_P) with direction v = (v_x, v_y, v_z) ≠ 0:
   x = x_P + t v_x,   y = y_P + t v_y,   z = z_P + t v_z      (t ∈ R)
Through two distinct points A, B: use the direction [AB] = B - A.
Plane through P with directions v, w (not one multiple of the other):
   (x, y, z) = P + t·v + s·w   (t, s ∈ R)
Through three non-collinear points (non collineari) A, B, C: directions B - A and C - A.
Attenzione! Parametric equations are NOT unique (any point, any non-zero direction)."""),
    ("Cartesian equations (equazioni cartesiane)", """\
Eliminate the parameters ("esplicitare il parametro e sostituire"):
   x = 1 + 2t, y = 2 - 5t, z = 1 + t   →  t = z - 1  →  x - 2z = -1,  y + 5z = 7.
• a line in R² is ax + by = c;  a line in R³ needs TWO equations
     a1 x + b1 y + c1 z = d1,  a2 x + b2 y + c2 z = d2;
• a plane in R³ is ONE linear equation (equazione lineare)  ax + by + cz = d.
Cartesian equations are not unique either. In this trainer you may write equations in any
equivalent form: they are checked against the true set of points."""),
]


def ex_line_two_points(rng, data=None):
    A, B = data or (ivec(rng, 3, -3, 3, nonzero=False), ivec(rng, 3, -3, 3, nonzero=False))
    if A == B:
        B = vadd(B, [1, 1, 0])
    d = vsub(B, A)
    prompt = (f"Write parametric equations of the line r ⊂ R³ through A = {fv(A)} and B = {fv(B)}.\n"
              f"Answer in the form (point) + t(direction).")
    sol = [f"Direction [AB] = B - A = {fv(d)}", f"r: {fparam(A, [d])}   (t ∈ R)",
           "Any point of r and any non-zero multiple of the direction give an equally correct answer."]
    return mk("lines_planes", "line2pts", prompt, AffineAns(A, [d], 3, "r"), sol, A=A, B=B)


def _coord(name, p, v, par="t"):
    out = f"{name} = "
    if p or not v:
        out += fs(p)
    if v:
        a = abs(v)
        coef = "" if a == 1 else fs(a)
        if p:
            out += (" - " if v < 0 else " + ") + coef + par
        else:
            out += ("-" if v < 0 else "") + coef + par
    return out


def ex_line_cartesian(rng, data=None):
    P, v = data or (ivec(rng, 3, -3, 3, nonzero=False), ivec(rng, 3, -3, 3))
    eqs, steps = cartesian_from_param(P, [v], ["x", "y", "z"])
    prompt = (f"The line r has parametric equations {fparam(P, [v])} (t ∈ R), i.e.\n"
              "   " + ",  ".join(_coord(nm, P[i], v[i]) for i, nm in enumerate("xyz")) + ".\n"
              f"Find Cartesian equations of r (two equations, separated by ';').")
    return mk("lines_planes", "line_cart", prompt,
              EquationsAns(P, [v], ["x", "y", "z"], eqs), steps + [
                  "Equivalent answer: eliminate t from one equation and substitute in the others."],
              P=P, v=v)


def _three_points(rng):
    while True:
        A, B, C = (ivec(rng, 3, -3, 3, nonzero=False) for _ in range(3))
        u, w = vsub(B, A), vsub(C, A)
        from ..core.gauss import span_rank
        if span_rank([u, w]) == 2:
            return A, B, C


def ex_plane_three_points(rng, data=None):
    A, B, C = data or _three_points(rng)
    u, w = vsub(B, A), vsub(C, A)
    eqs, steps = cartesian_from_param(A, [u, w], ["x", "y", "z"])
    prompt = (f"Find a Cartesian equation of the plane π through\n"
              f"   P = {fv(A)},  Q = {fv(B)},  R = {fv(C)}.")
    sol = [f"Directions: Q - P = {fv(u)},  R - P = {fv(w)}  (independent: the points are not collinear).",
           f"Parametric form: π = {fparam(A, [u, w])}"] + steps
    return mk("lines_planes", "plane3pts", prompt, EquationsAns(A, [u, w], ["x", "y", "z"], eqs),
              sol, A=A, B=B, C=C)


LINES_PLANES = Topic(
    "lines_planes", "§0.2", "Lines and planes (first look)", "Rette e piani", "0 Introduzione",
    ["vectors"], LINES_LESSON,
    [lambda: ex_line_cartesian(None, ([1, 2, 1], [2, -5, 1])),
     lambda: ex_plane_three_points(None, ([1, 1, 0], [1, -2, 1], [2, -1, 3]))],
    [Gen("line2pts", "line through two points", ex_line_two_points),
     Gen("line_cart", "line: parametric → Cartesian", ex_line_cartesian),
     Gen("plane3pts", "plane through three points", ex_plane_three_points, exam=True)],
)

TOPICS = [VECTORS, LINES_PLANES]
