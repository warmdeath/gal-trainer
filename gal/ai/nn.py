"""A tiny neural network from scratch, written only with the matrix operations of the course.

This is the ONE place of the trainer that uses floats: training is numerical by nature.
Conventions follow the notes: vectors are columns, a layer is the affine map x ↦ Wx + b
(a linear map L_W plus a translation, §3 and §5.6.2), batches are matrices whose columns
are the examples.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

Mat = list[list[float]]


# ---------------------------------------------------------------- matrix operations (§1.1)
def matmul(A: Mat, B: Mat) -> Mat:
    """Rows by columns: (AB)_ij = Σ_l a_il b_lj. Needs #cols(A) = #rows(B)."""
    if len(A[0]) != len(B):
        raise ValueError(f"product not defined: {len(A)}x{len(A[0])} times {len(B)}x{len(B[0])}")
    Bt = transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def transpose(A: Mat) -> Mat:
    return [list(r) for r in zip(*A)]


def add(A: Mat, B: Mat) -> Mat:
    return [[a + b for a, b in zip(r, s)] for r, s in zip(A, B)]


def sub(A: Mat, B: Mat) -> Mat:
    return [[a - b for a, b in zip(r, s)] for r, s in zip(A, B)]


def scale(k: float, A: Mat) -> Mat:
    return [[k * a for a in r] for r in A]


def hadamard(A: Mat, B: Mat) -> Mat:
    """Entry-wise product (NOT the matrix product — it is used only for derivatives)."""
    return [[a * b for a, b in zip(r, s)] for r, s in zip(A, B)]


def add_col(A: Mat, b: Sequence[float]) -> Mat:
    """A + b·(1, ..., 1): add the column vector b to every column (the bias)."""
    return [[a + bi for a in row] for row, bi in zip(A, b)]


def row_sums(A: Mat) -> list[float]:
    return [sum(r) for r in A]


def apply(f, A: Mat) -> Mat:
    return [[f(a) for a in r] for r in A]


def sq_norm(A: Mat) -> float:
    """||A||² = ⟨A, A⟩ = tr(A^T A): the scalar product of §5.2 on matrices."""
    return sum(a * a for r in A for a in r)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def relu(x: float) -> float:
    return x if x > 0 else 0.0


# ---------------------------------------------------------------- the network
XOR_X: Mat = [[0, 1, 0, 1],   # first coordinate of the 4 inputs (columns)
              [0, 0, 1, 1]]   # second coordinate
XOR_T: Mat = [[0, 1, 1, 0]]   # targets


class TinyNet:
    """2 → H → 1 network:  h = tanh(W1 x + b1),  y = sigmoid(W2 h + b2)."""

    def __init__(self, hidden: int = 4, seed: int = 0, n_in: int = 2):
        rng = random.Random(seed)
        self.W1 = [[rng.uniform(-1, 1) for _ in range(n_in)] for _ in range(hidden)]
        self.b1 = [rng.uniform(-1, 1) for _ in range(hidden)]
        self.W2 = [[rng.uniform(-1, 1) for _ in range(hidden)]]
        self.b2 = [0.0]

    def forward(self, X: Mat):
        Z1 = add_col(matmul(self.W1, X), self.b1)       # affine map, one column per example
        H = apply(math.tanh, Z1)                          # non-linearity
        Z2 = add_col(matmul(self.W2, H), self.b2)
        Y = apply(sigmoid, Z2)
        return Z1, H, Z2, Y

    def predict(self, X: Mat) -> list[float]:
        return self.forward(X)[3][0]

    def loss(self, X: Mat, T: Mat) -> float:
        Y = self.forward(X)[3]
        return sq_norm(sub(Y, T)) / len(X[0])           # mean squared error = ||Y - T||²/N

    def step(self, X: Mat, T: Mat, lr: float) -> float:
        """One gradient-descent step (backpropagation = chain rule + TRANSPOSES)."""
        N = len(X[0])
        Z1, H, Z2, Y = self.forward(X)
        dY = scale(2.0 / N, sub(Y, T))                    # ∂L/∂Y
        dZ2 = hadamard(dY, apply(lambda y: y * (1 - y), Y))  # sigmoid' = y(1 - y)
        dW2 = matmul(dZ2, transpose(H))                   # ∂L/∂W2 = dZ2 · H^T
        db2 = row_sums(dZ2)
        dH = matmul(transpose(self.W2), dZ2)              # the error flows back through W2^T
        dZ1 = hadamard(dH, apply(lambda h: 1 - h * h, H))  # tanh' = 1 - tanh²
        dW1 = matmul(dZ1, transpose(X))
        db1 = row_sums(dZ1)
        self.W2 = sub(self.W2, scale(lr, dW2))
        self.b2 = [b - lr * g for b, g in zip(self.b2, db2)]
        self.W1 = sub(self.W1, scale(lr, dW1))
        self.b1 = [b - lr * g for b, g in zip(self.b1, db1)]
        return sq_norm(sub(Y, T)) / N


def train_xor(epochs: int = 4000, lr: float = 2.0, hidden: int = 4, seed: int = 0, every: int = 500):
    """Train on XOR; returns (net, [(epoch, loss), ...])."""
    net = TinyNet(hidden, seed)
    log = []
    for e in range(epochs + 1):
        L = net.step(XOR_X, XOR_T, lr)
        if e % every == 0:
            log.append((e, L))
    return net, log


def numerical_grad_check(net: TinyNet, X: Mat, T: Mat, eps: float = 1e-6) -> float:
    """Max |analytic - numerical| on W1 (used by the tests to validate backprop)."""
    N = len(X[0])
    Z1, H, Z2, Y = net.forward(X)
    dY = scale(2.0 / N, sub(Y, T))
    dZ2 = hadamard(dY, apply(lambda y: y * (1 - y), Y))
    dH = matmul(transpose(net.W2), dZ2)
    dZ1 = hadamard(dH, apply(lambda h: 1 - h * h, H))
    dW1 = matmul(dZ1, transpose(X))
    worst = 0.0
    for i in range(len(net.W1)):
        for j in range(len(net.W1[0])):
            old = net.W1[i][j]
            net.W1[i][j] = old + eps
            lp = net.loss(X, T)
            net.W1[i][j] = old - eps
            lm = net.loss(X, T)
            net.W1[i][j] = old
            worst = max(worst, abs((lp - lm) / (2 * eps) - dW1[i][j]))
    return worst
