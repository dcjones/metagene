import numpy as np
import pytest
import scipy.sparse as sp

import metagene


def random_counts(m=300, n=100, k=4, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.gamma(0.5, 1.0, (m, k))
    H = rng.gamma(0.3, 1.0, (k, n))
    return sp.csr_matrix(rng.poisson(W @ H).astype(np.float32))


def kl(X, W, H):
    X = X.toarray().astype(np.float64)
    U = W.astype(np.float64) @ H.astype(np.float64)
    nz = X > 0
    return np.sum(X[nz] * np.log(X[nz] / U[nz])) - X.sum() + U.sum()


def test_shapes_and_nonnegativity():
    X = random_counts()
    res = metagene.nmf(X, 4, max_iter=50, seed=1)
    assert res.W.shape == (300, 4)
    assert res.H.shape == (4, 100)
    assert (res.W > 0).all() and (res.H > 0).all()


def test_objective_decreases_and_matches_numpy():
    X = random_counts()
    res = metagene.nmf(X, 4, method="mu", max_iter=100, tol=-np.inf, eval_every=1, seed=1)
    losses = [l for _, _, l in res.loss]
    times = [t for _, t, _ in res.loss]
    assert all(b >= a for a, b in zip(times, times[1:]))
    assert res.n_iter == 100
    assert all(b <= a * (1 + 1e-6) for a, b in zip(losses, losses[1:]))
    assert losses[-1] == pytest.approx(kl(X, res.W, res.H), rel=1e-4)


def test_given_init_is_deterministic():
    X = random_counts()
    rng = np.random.default_rng(2)
    W0 = rng.random((300, 4)).astype(np.float32)
    H0 = rng.random((4, 100)).astype(np.float32)
    a = metagene.nmf(X, 4, max_iter=20, W0=W0, H0=H0, n_threads=1)
    b = metagene.nmf(X, 4, max_iter=20, W0=W0, H0=H0, n_threads=1)
    np.testing.assert_array_equal(a.W, b.W)
    np.testing.assert_array_equal(a.H, b.H)


def test_accepts_other_formats():
    X = random_counts(m=50, n=30)
    for Y in (X.tocsc(), X.toarray(), sp.csr_array(X)):
        res = metagene.nmf(Y, 3, max_iter=5, seed=0)
        assert res.W.shape == (50, 3)


def reference_mu(X, W, H, iters, eps=1e-6, fit_H=True):
    """Dense float64 alternating KL multiplicative updates (W then H), clamping values at eps."""
    X = X.toarray().astype(np.float64)
    W, H = W.astype(np.float64), H.astype(np.float64)
    for _ in range(iters):
        W = np.maximum(W * ((X / (W @ H)) @ H.T) / H.sum(1), eps)
        if fit_H:
            H = np.maximum(H * (W.T @ (X / (W @ H))) / W.sum(0)[:, None], eps)
    return W, H


def test_matches_reference_mu():
    X = random_counts()
    rng = np.random.default_rng(3)
    W0 = rng.random((300, 4)).astype(np.float32)
    H0 = rng.random((4, 100)).astype(np.float32)
    res = metagene.nmf(X, 4, method="mu", max_iter=30, tol=-np.inf, eval_every=0, W0=W0, H0=H0)
    Wr, Hr = reference_mu(X, W0, H0, 30)
    np.testing.assert_allclose(res.W, Wr, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(res.H, Hr, rtol=1e-3, atol=1e-5)


def test_max_time():
    X = random_counts()
    res = metagene.nmf(X, 4, max_iter=10**9, max_time=0.05, tol=-np.inf, seed=0)
    assert res.n_iter < 10**9
    assert 0.05 <= res.loss[-1][1] < 1.0


def reference_bmme(X, W, H, iters, eps=1e-6):
    """Dense float64 port of MUe_KLNMF.m (Hien, Leplat & Gillis), updating W before H."""
    X = X.toarray().astype(np.float64)
    W, H = W.astype(np.float64), H.astype(np.float64)
    W_prev, H_prev = W.copy(), H.copy()
    t = 1.0
    for _ in range(iters):
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        beta = (t - 1) / t_next
        t = t_next
        W_ex = W + beta * np.maximum(W - W_prev, 0)
        W_prev = W
        W = np.maximum(W_ex * ((X / (W_ex @ H)) @ H.T) / H.sum(1), eps)
        H_ex = H + beta * np.maximum(H - H_prev, 0)
        H_prev = H
        H = np.maximum(H_ex * (W.T @ (X / (W @ H_ex))) / W.sum(0)[:, None], eps)
    return W, H


def test_matches_reference_bmme():
    X = random_counts()
    rng = np.random.default_rng(3)
    W0 = rng.random((300, 4)).astype(np.float32)
    H0 = rng.random((4, 100)).astype(np.float32)
    res = metagene.nmf(X, 4, method="bmme", max_iter=30, tol=-np.inf, eval_every=0, W0=W0, H0=H0)
    Wr, Hr = reference_bmme(X, W0, H0, 30)
    np.testing.assert_allclose(res.W, Wr, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(res.H, Hr, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("restart", [False, True])
def test_bmme_loss_is_evaluated_at_iterates(restart):
    X = random_counts()
    res = metagene.nmf(X, 4, method="bmme", restart=restart, max_iter=50, tol=-np.inf, eval_every=1, seed=1)
    final = res.loss[-1][2]
    assert final == pytest.approx(kl(X, res.W, res.H), rel=1e-4)
    assert final < 0.5 * res.loss[0][2]


def test_fixed_H_matches_reference():
    X = random_counts()
    rng = np.random.default_rng(4)
    W0 = rng.random((300, 4)).astype(np.float32)
    H0 = rng.random((4, 100)).astype(np.float32)
    res = metagene.nmf(X, 4, method="mu", fit_H=False, max_iter=20,
                       tol=-np.inf, eval_every=1, W0=W0, H0=H0)
    Wr, Hr = reference_mu(X, W0, H0, 20, fit_H=False)
    np.testing.assert_allclose(res.W, Wr, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(res.H, Hr, rtol=1e-3, atol=1e-5)
    np.testing.assert_array_equal(res.H, H0)
    losses = [l for _, _, l in res.loss]
    assert all(b <= a * (1 + 1e-6) for a, b in zip(losses, losses[1:]))


def test_fixed_H_without_W0():
    X = random_counts()
    fit = metagene.nmf(X, 4, max_iter=100, seed=0)
    proj = metagene.nmf(X, 4, H0=fit.H, fit_H=False, max_iter=200, tol=0)
    np.testing.assert_array_equal(proj.H, fit.H)
    assert proj.loss[-1][2] <= fit.loss[-1][2] * (1 + 1e-3)


def test_warm_start():
    X = random_counts(m=600)
    res = metagene.nmf(X, 4, warm_start=True, warm_start_fraction=0.2, max_iter=50, tol=-np.inf,
                       eval_every=1, seed=0)
    assert res.warm_start_time > 0
    assert res.loss[0][1] >= res.warm_start_time
    assert res.loss[-1][2] == pytest.approx(kl(X, res.W, res.H), rel=1e-4)
    cold = metagene.nmf(X, 4, warm_start=False, max_iter=50, tol=-np.inf, eval_every=1, seed=0)
    assert res.loss[0][2] < cold.loss[0][2]


def test_warm_start_auto():
    X = random_counts()
    assert metagene.nmf(X, 4, max_iter=5, seed=0).warm_start_time == 0
    with pytest.raises(ValueError):
        metagene.nmf(X, 4, warm_start=True, warm_start_fraction=0.001, max_iter=5, seed=0)
