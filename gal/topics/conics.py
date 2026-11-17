"""Extra — conics basics. NOT in Note_GAL26.pdf (the notes stop at §5.6 and announce
quadratic forms and quadrics for the next chapter). Kept short and clearly marked."""
from __future__ import annotations

import random
from fractions import Fraction

from ..core.fmt import flin, fm, fs, fv, hcat, named
from ..core.gauss import det, inverse
from ..core.matrix import Matrix
from ..exercise import ChoiceAns, MatrixAns, VectorAns, mk
from .base import Gen, Topic
from .rand import nz

CONICS_LESSON = [
    ("Not yet in the notes", """\
Note_GAL26.pdf ends with §5.6 and announces quadratic forms and quadrics for the next
chapter. This short extra topic prepares you with the plane case, using only tools you
already have (symmetric matrices, determinants, eigenvalues). Follow the official notes
when they are published if notation differs."""),
    ("Intuition", """\
A conic is the set of points of R² satisfying a degree-2 equation
   a x² + 2b xy + c y² + 2d x + 2e y + f = 0.
The quadratic part decides the shape: like a bowl (ellipse), a saddle (hyperbola), or a
valley that is flat in one direction (parabola)."""),
    ("Matrices and invariants", """\
Write it as (x y 1) Ã (x y 1)^T with the symmetric 3×3 matrix
   Ã = ( a b d ; b c e ; d e f )   and its quadratic part   A = ( a b ; b c ).
• det Ã ≠ 0: non-degenerate conic.  Then: det A > 0 ellipse (real if tr(A)·det Ã < 0,
  otherwise imaginary — no real points), det A < 0 hyperbola, det A = 0 parabola.
• det Ã = 0: degenerate (pairs of lines, a double line, a point...).
• Centre of an ellipse/hyperbola: solve A·(x0, y0)^T = -(d, e)^T.
• Diagonalising A with an orthogonal matrix (spectral theorem) rotates the axes so that the
  xy term disappears; translating to the centre removes the linear terms (canonical form)."""),
]

TYPES = ["real ellipse", "imaginary ellipse (no real points)", "hyperbola", "parabola", "degenerate"]


def classify(M: Matrix) -> str:
    A = M.submatrix([0, 1], [0, 1])
    D3, D2 = det(M), det(A)
    if D3 == 0:
        return "degenerate"
    if D2 > 0:
        return "real ellipse" if A.trace() * D3 < 0 else "imaginary ellipse (no real points)"
    return "hyperbola" if D2 < 0 else "parabola"


def _conic_text(M: Matrix) -> str:
    a, b, c, d, e, f = M[0, 0], M[0, 1], M[1, 1], M[0, 2], M[1, 2], M[2, 2]
    s = flin([a, 2 * b, c, 2 * d, 2 * e], ["x²", "xy", "y²", "x", "y"], zero="")
    if f:
        s = (s + (" - " if f < 0 else " + ") + fs(abs(f))) if s else fs(f)
    return f"{s} = 0"


def _random_conic(rng, kind=None):
    for _ in range(1000):
        b = rng.choice([0, 0, 1, -1])
        a, c = rng.randint(-3, 3), rng.randint(-3, 3)
        if kind == "parabola":
            c = Fraction(b * b, a) if a else c
            if a == 0 and b == 0:
                continue
        d, e = rng.randint(-3, 3), rng.randint(-3, 3)
        f = rng.randint(-6, 6)
        M = Matrix([[a, b, d], [b, c, e], [d, e, f]])
        if any(x.denominator != 1 for r in M.rows for x in r) or (a == 0 and b == 0 and c == 0):
            continue
        if kind is None or classify(M) == kind:
            return M
    return M


def ex_classify(rng, data=None):
    M = data or _random_conic(rng, rng.choice(["real ellipse", "hyperbola", "hyperbola", "parabola", None]))
    kind = classify(M)
    A = M.submatrix([0, 1], [0, 1])
    ans = ChoiceAns(TYPES, TYPES.index(kind))
    prompt = f"Classify the conic  {_conic_text(M)}\n" + ans.render()
    steps = [hcat("Ã = ", fm(M), "   A = ", fm(A)),
             f"det Ã = {fs(det(M))},  det A = {fs(det(A))},  tr A = {fs(A.trace())}",
             f"⇒ {kind}."]
    return mk("conics", "classify", prompt,
              [("matrix Ã", MatrixAns(M, mistakes=[(Matrix([[M[0, 0], 2 * M[0, 1], 2 * M[0, 2]], [2 * M[0, 1], M[1, 1], 2 * M[1, 2]],
                                                            [2 * M[0, 2], 2 * M[1, 2], M[2, 2]]]),
                                                    "Off-diagonal entries are HALF the coefficients of xy, x, y.")])),
               ("type", ans)], steps, M=M)


def ex_centre(rng, data=None):
    while True:
        M = data or _random_conic(rng, rng.choice(["real ellipse", "hyperbola"]))
        A = M.submatrix([0, 1], [0, 1])
        c = inverse(A).apply([-M[0, 2], -M[1, 2]])
        if data or all(x.denominator <= 3 for x in c):
            break
    prompt = f"Find the centre of the conic  {_conic_text(M)}."
    steps = [hcat("A = ", fm(A)), f"Solve A·(x0, y0) = -(d, e) = {fv([-M[0, 2], -M[1, 2]])}:  centre {fv(c)}"]
    return mk("conics", "centre", prompt, VectorAns(c), steps, M=M)


CONICS = Topic(
    "conics", "extra", "Conics (basics) — not yet in the notes", "Coniche (cenni) — extra",
    "Extra", ["diagonalization", "scalar_products"], CONICS_LESSON,
    [lambda: ex_classify(None, Matrix([[1, 0, 0], [0, 4, 0], [0, 0, -4]]))],
    [Gen("classify", "classify a conic", ex_classify, weight=2),
     Gen("centre", "centre of a conic", ex_centre)],
    extra=True,
)

TOPICS = [CONICS]
