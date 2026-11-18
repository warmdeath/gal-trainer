"""Bonus chapter — why this matters for AI. Exercises are exact (Fractions) and verified
against sympy like every other topic; the numerical training demo lives in gal/ai/."""
from __future__ import annotations

import random
from fractions import Fraction

from ..core.fmt import fm, fs, fv, hcat, named, named_many
from ..core.gauss import solve
from ..core.matrix import Matrix
from ..exercise import MatrixAns, VectorAns, mk
from .base import Gen, Topic
from .rand import imat, ivec

AI_LESSON = [
    ("Why this matters", """\
Modern AI is, to a large extent, linear algebra done fast. A neural network is a chain of
affine maps x ↦ Wx + b (Chapter 3 + translations, §5.6.2) separated by simple non-linear
functions. Training it means computing products, transposes and scalar products millions
of times. This bonus chapter builds one from scratch with exactly the operations you learned."""),
    ("1. A neuron is a scalar product (§5.2)", """\
A neuron with weights w and bias b computes ⟨w, x⟩ + b and then a non-linearity, e.g.
ReLU(s) = max(s, 0). Geometrically ⟨w, x⟩ + b = 0 is a hyperplane (Cor. 5.29): the neuron
asks "on which side of this hyperplane is x, and how far?" """),
    ("2. A layer is a matrix product (§1.1, §3)", """\
Stack k neurons: their weights are the ROWS of W ∈ M_R(k, n). The layer computes
   h = ReLU(W x + b),  W x = L_W(x).
A batch of examples is a matrix X whose columns are the inputs: W X processes all of them
at once — that's why GPUs are matrix-multiplication machines."""),
    ("3. Why the non-linearity is essential (Prop. 3.4)", """\
Without it, two layers give W2(W1 x) = (W2 W1) x = L_{W2W1}(x): a composition of linear maps
is linear, so a "deep" linear network is just ONE matrix. And one affine map cannot compute
XOR: for any affine f(x1,x2) = a x1 + b x2 + c we have f(0,0) + f(1,1) = f(1,0) + f(0,1),
but XOR needs 0 + 0 on the left and 1 + 1 on the right. Contradiction!"""),
    ("4. An exact XOR network (computed with this trainer's Matrix class)", """\
W1 = (1 1; 1 1), b1 = (0, -1),  h = ReLU(W1 x + b1);   y = (1, -2)·h.
   x = (0,0): h = (0, 0) → y = 0      x = (1,0): h = (1, 0) → y = 1
   x = (0,1): h = (1, 0) → y = 1      x = (1,1): h = (2, 1) → y = 2 - 2 = 0
Two hyperplanes x1 + x2 = 0 and x1 + x2 = 1 cut the plane; the second layer combines them."""),
    ("5. Learning = minimising a norm (§5.2) with transposes (§1.1)", """\
Loss L = ||Y - T||²/N (squared norm of the error matrix, ⟨A,B⟩ = tr(A^T B)).
Gradient descent moves every weight a little against ∂L/∂W. The chain rule in matrix form is
backpropagation:  dW2 = dZ2·H^T,   dH = W2^T·dZ2   ← the error travels back through W2^T,
                  dW1 = dZ1·X^T.
The 'ai' command trains a 2-4-1 network on XOR live, with these exact formulas."""),
    ("6. More links with the course", """\
• Linear regression = orthogonal projection (§5.3.3): the best b̂ ∈ Col(A) approximating b is
  π_{Col(A)}(b); its coefficients solve the normal equations A^T A x = A^T b.
• PCA (data compression) = eigenvectors of the symmetric matrix X X^T (Chapters 4–5).
• Gram-Schmidt = the QR factorisation used by numerical solvers.
• "Attention" in language models = scalar products ⟨q, k⟩ between query and key vectors.
• Embeddings: words as vectors; similarity = cos θ (§5.2.2)."""),
]


def relu_vec(v):
    return [x if x > 0 else Fraction(0) for x in v]


def ex_layer(rng, data=None):
    W, b, x = data or (imat(rng, 3, 2, -2, 2), ivec(rng, 3, -2, 2, nonzero=False), ivec(rng, 2, -2, 3))
    z = [a + bb for a, bb in zip(W.apply(x), b)]
    h = relu_vec(z)
    prompt = (named_many(("W", W)) + f"\n\nb = {fv(b)},  x = {fv(x)}.\n"
              "Compute the layer output h = ReLU(Wx + b)  (ReLU(s) = max(s, 0), entry by entry).")
    steps = [f"Wx = {fv(W.apply(x))}  (row i of W · x: a scalar product per neuron)",
             f"Wx + b = {fv(z)}", f"ReLU → h = {fv(h)}"]
    return mk("ai", "layer", prompt,
              VectorAns(h, mistakes=[(z, "You forgot the ReLU: negative entries become 0."),
                                     (W.apply(x), "You forgot the bias b.")]),
              steps, W=W, b=b, x=x)


def ex_collapse(rng, data=None):
    W1, W2 = data or (imat(rng, 3, 2, -2, 2), imat(rng, 2, 3, -2, 2))
    M = W2 * W1
    prompt = (named_many(("W1", W1), ("W2", W2)) +
              "\n\nA 'deep' network without non-linearity computes x ↦ W2(W1 x).\n"
              "Write the single matrix M with W2(W1 x) = M x for every x.")
    steps = ["Associativity of the product: W2(W1 x) = (W2 W1)x, i.e. L_W2 ∘ L_W1 = L_{W2W1} (Prop. 3.4).",
             hcat("M = W2 W1 = ", fm(M)), "Two linear layers = one linear layer: depth needs non-linearities."]
    return mk("ai", "collapse", prompt,
              MatrixAns(M, mistakes=[((W1 * W2) if (W1 * W2).shape == M.shape else M, "Order! W2 acts AFTER W1: M = W2·W1.")]),
              steps, W1=W1, W2=W2)


def ex_least_squares(rng, data=None):
    """Fit y = m x + q to 3-4 points: normal equations A^T A (m, q) = A^T y."""
    if data:
        pts = data
    else:
        while True:
            xs = sorted(rng.sample(range(-3, 4), rng.choice([3, 4])))
            pts = [(x, rng.randint(-3, 4)) for x in xs]
            A = Matrix([[x, 1] for x, _ in pts])
            sol = solve(A.T * A, (A.T).apply([y for _, y in pts]))
            if all(c.denominator <= 10 for c in sol.particular):
                break
    A = Matrix([[x, 1] for x, _ in pts])
    y = [y for _, y in pts]
    N = A.T * A
    rhs = A.T.apply(y)
    sol = solve(N, rhs, ["m", "q"])
    prompt = ("Least squares = orthogonal projection (§5.3.3). Find the line y = m x + q that best fits\n   "
              + ",  ".join(f"({x}, {yy})" for x, yy in pts) + "\nAnswer with the vector (m, q).")
    steps = [hcat("A = ", fm(A), f"   y = {fv(y)}"),
             "The best Ax̂ is the projection of y on Col(A): (y - Ax̂) ⊥ Col(A) ⇔ A^T A x̂ = A^T y.",
             hcat("A^T A = ", fm(N), f"   A^T y = {fv(rhs)}")] + sol.steps[-1:] + [
        f"(m, q) = {fv(sol.particular)}"]
    return mk("ai", "lstsq", prompt, VectorAns(sol.particular), steps, pts=pts)


AI = Topic(
    "ai", "bonus", "Why this matters for AI: a tiny neural network", "Perché serve all'IA (bonus)",
    "Bonus", ["matrices", "linear_maps", "projections"], AI_LESSON,
    [lambda: ex_layer(None, (Matrix([[1, 1], [1, 1]]), [0, -1], [1, 1]))],
    [Gen("layer", "forward pass of a ReLU layer", ex_layer),
     Gen("collapse", "linear layers collapse", ex_collapse),
     Gen("lstsq", "least squares line", ex_least_squares)],
    extra=True,
)

TOPICS = [AI]
