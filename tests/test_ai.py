import sympy as sp

from gal.ai.chapter import exact_xor_network
from gal.ai.nn import XOR_T, XOR_X, TinyNet, matmul, numerical_grad_check, train_xor, transpose
from gal.core.matrix import Matrix

from .helpers import sm, smv
from .verify import ans, run


def test_exact_xor_network():
    for x, z, h, y in exact_xor_network():
        assert y == x[0] ^ x[1]


def test_float_ops_match_exact_matrix():
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[1, 0], [0, 1], [2, -1]]
    assert matmul(A, B) == [[float(v) for v in r] for r in (Matrix(A) * Matrix(B)).tolist()]
    assert transpose(A) == [list(r) for r in zip(*A)]


def test_backprop_matches_numerical_gradient():
    for seed in range(3):
        assert numerical_grad_check(TinyNet(4, seed), XOR_X, XOR_T) < 1e-6


def test_xor_training_converges():
    for seed in range(5):
        net, log = train_xor(seed=seed)
        assert log[-1][1] < 0.01
        assert [round(p) for p in net.predict(XOR_X)] == XOR_T[0]
        assert log[-1][1] < log[0][1]


def test_ai_exercises_against_sympy():
    def layer(ex):
        W, b, x = sm(ex.data["W"]), smv(ex.data["b"]), smv(ex.data["x"])
        z = W * x + b
        assert smv(ans(ex).v) == z.applyfunc(lambda s: sp.Max(s, 0))
    run("ai", "layer", layer)

    run("ai", "collapse", lambda ex: _ok(sm(ans(ex).M) == sm(ex.data["W2"]) * sm(ex.data["W1"])))

    def ls(ex):
        pts = ex.data["pts"]
        A = sp.Matrix([[x, 1] for x, _ in pts])
        y = sp.Matrix([v for _, v in pts])
        sol = (A.T * A).solve(A.T * y)
        assert smv(ans(ex).v) == sol
        # and it is the orthogonal projection: residual ⊥ Col(A)
        assert (A.T * (y - A * sol)).is_zero_matrix
    run("ai", "lstsq", ls)


def _ok(b):
    assert b
