from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .metagene import _nmf, _SparseMatrix

__all__ = ["nmf", "NMFResult"]


@dataclass
class NMFResult:
    W: np.ndarray  # [m, k]
    H: np.ndarray  # [k, n]
    loss: list = field(default_factory=list)  # (iteration, elapsed seconds, objective) tuples
    n_iter: int = 0
    init_time: float = 0.0  # seconds spent on NNDSVD and the warm start, included in loss times


# The warm start is enabled automatically when the subsample would have at least this many cells.
WARM_START_MIN_CELLS = 20_000

# The solver's default floor on factor values (EPS in src/nmf.rs).
EPS = 1e-6


def _as_csr(X) -> sp.csr_matrix:
    if not sp.issparse(X):
        X = sp.csr_matrix(np.asarray(X))
    X = sp.csr_matrix(X)  # handles csr_array, csc, coo, ...; no copy if already csr_matrix
    if not X.has_canonical_format:
        X = X.copy()
        X.sum_duplicates()
    return X


def _as_u32(a: np.ndarray) -> np.ndarray:
    # scipy indices are int32/int64 and non-negative, so int32 can be reinterpreted for free.
    if a.dtype == np.int32 or a.dtype == np.uint32:
        return np.ascontiguousarray(a).view(np.uint32)
    if a.size and a.max() > np.iinfo(np.uint32).max:
        raise ValueError("matrix has too many nonzeros for 32-bit indices")
    return a.astype(np.uint32)


def _randomized_svd(X, k, rng, n_oversamples=10, n_iter=4, n_threads=None):
    """Rank-k truncated SVD (U [m, k], S [k], Vt [k, n]) of a sparse [m, n] matrix, by a randomized
    range finder with power iterations (Halko, Martinsson & Tropp 2011, algorithm 4.4).

    The range finder works on the gene side (n ≪ m), so the orthonormalizations are of [n, l]
    matrices and only the products with X scale with m. The left singular vectors then come from the
    eigendecomposition of the small Gram matrix of X V, which is accurate enough for initialization.
    X must be a canonical CSR matrix; the products with it run in parallel in Rust.
    """
    l = min(k + n_oversamples, *X.shape)
    data = np.ascontiguousarray(X.data, dtype=np.float32)
    A = _SparseMatrix(data, _as_u32(X.indices), _as_u32(X.indptr), X.shape[1], l, n_threads)
    V = rng.standard_normal((X.shape[1], l), dtype=np.float32)  # [n, l]
    for _ in range(n_iter + 1):
        V, _ = np.linalg.qr(A.rmatmul(A.matmul(V)))
    XV = A.matmul(V)  # [m, l]
    λ, E = np.linalg.eigh(XV.T.astype(np.float64) @ XV)  # XV = U S Eᵀ, ascending
    λ, E = λ[::-1][:k], E[:, ::-1][:, :k]
    S = np.sqrt(np.maximum(λ, 0.0))
    U = (XV @ E.astype(np.float32)) / np.maximum(S, 1e-30).astype(np.float32)
    return U, S, (V @ E.astype(np.float32)).T


def _nndsvd(X, k, rng, n_threads=None, eps=EPS):
    """NNDSVD initialization (Boutsidis & Gallopoulos 2008), with zeros raised to the solver's floor.

    Each singular triplet's sign is chosen so its positive (or negative) parts carry the most mass,
    which are then used as a nonnegative factor pair.
    """
    U, S, Vt = _randomized_svd(X, k, rng, n_threads=n_threads)
    V = Vt.T
    Up, Un = np.maximum(U, 0), np.maximum(-U, 0)
    Vp, Vn = np.maximum(V, 0), np.maximum(-V, 0)
    upn, vpn = np.linalg.norm(Up, axis=0), np.linalg.norm(Vp, axis=0)
    unn, vnn = np.linalg.norm(Un, axis=0), np.linalg.norm(Vn, axis=0)
    pos = upn * vpn >= unn * vnn
    pos[0] = True  # the leading pair is single-signed; abs() below guards against round-off
    Up[:, 0], Vp[:, 0] = np.abs(U[:, 0]), np.abs(V[:, 0])
    upn[0], vpn[0] = 1.0, 1.0
    u = np.where(pos, Up / np.maximum(upn, 1e-30), Un / np.maximum(unn, 1e-30))
    v = np.where(pos, Vp / np.maximum(vpn, 1e-30), Vn / np.maximum(vnn, 1e-30))
    scale = np.sqrt(S * np.where(pos, upn * vpn, unn * vnn))
    W = np.maximum(u * scale, eps).astype(np.float32)
    H = np.maximum(v * scale, eps).astype(np.float32).T
    return W, H


def _fit(X, w, ht, *, max_iter, tol, eval_every, verbose, n_threads, max_time, method, restart, fit_H,
         eps, h_shape, h_rate):
    """Run the Rust solver on a canonical CSR matrix. Returns (w, ht, loss, n_iter)."""
    data = np.ascontiguousarray(X.data, dtype=np.float32)
    return _nmf(
        data, _as_u32(X.indices), _as_u32(X.indptr), w, ht, max_iter, tol, eval_every, verbose,
        n_threads, max_time, method == "bmme", restart, fit_H, eps, h_shape, h_rate,
    )


def nmf(
    X,
    k: int,
    *,
    method: str = "bmme",
    init: str = "nndsvd",
    restart: bool = False,
    fit_H: bool = True,
    h_pseudocount: float = 1.0,
    h_rate: float | None = None,
    h_prior: str = "gene-rate",
    eps: float = EPS,
    warm_start: bool | None = None,
    warm_start_fraction: float = 0.1,
    warm_start_iter: int = 200,
    warm_start_w_passes: int = 5,
    max_iter: int = 200,
    max_time: float | None = None,
    tol: float = 1e-4,
    eval_every: int = 10,
    W0: np.ndarray | None = None,
    H0: np.ndarray | None = None,
    seed: int | None = None,
    n_threads: int | None = None,
    verbose: bool = False,
) -> NMFResult:
    """KL-NMF of a non-negative [m, n] matrix X ≈ W H, using multiplicative updates.

    With method="bmme", each factor is extrapolated before its multiplicative update, following

        Hien, L.T.K., Leplat, V. and Gillis, N. (2025) Block Majorization Minimization with
        extrapolation and application to β-NMF. SIAM J. Math. Data Sci., 7, 1292–1314.

    Parameters
    ----------
    X : scipy sparse matrix or array-like, shape [m, n]
        Count matrix, typically cells × genes. Converted to CSR float32.
    k : int
        Number of factors.
    method : {"bmme", "mu"}
        Multiplicative updates with extrapolation (BMMe, the default), or plain multiplicative
        updates. BMMe costs the same per iteration and typically needs 2-4x fewer iterations to
        reach a given objective. Its objective isn't guaranteed to be monotone, and evaluating it
        costs an extra pass over the data (so about 1/eval_every extra W-pass work).
    init : {"nndsvd", "random"}
        How to initialize the factors when W0 and H0 aren't given: NNDSVD (Boutsidis & Gallopoulos
        2008, from a randomized truncated SVD of X, or of the warm start's subsample), or uniformly
        random. NNDSVD costs an SVD up front, but finds substantially better solutions when k is a
        sizable fraction of the number of genes n (e.g. k=100 on a few-hundred-gene panel), where
        random starts tend to settle in worse local minima. With n much larger than k they reach
        similar solutions. Its time counts toward max_time and is included in loss times.
    restart : bool
        With method="bmme", reset the extrapolation whenever an evaluated objective increases.
    fit_H : bool
        If False, H is held fixed at H0 (which must be given) and only W is fit, e.g. to project
        new cells onto existing factors.
    h_pseudocount, h_rate, h_prior
        A Gamma(a_j + 1, b_j) prior on each entry h_kj of H, fit by MAP: each update of h_kj
        becomes (h_kj Σ_i w_ik x_ij / (W H)_ij + a_j) / (Σ_i w_ik + b_j), and the reported
        objective adds Σ_kj b_j h_kj - a_j log h_kj. Without it, fits overfit the noise when there
        are few cells per gene: rarely seen genes get near-zero loadings, which predict near-zero
        means for counts not seen in training. By default every gene gets h_pseudocount = 1
        pseudocount per factor (a_j = h_pseudocount), and with h_prior="gene-rate" the rate is
        b_j = h_rate / f_j, where f_j = (c_j + 1) / mean(c + 1) is gene j's frequency from its total
        count c_j in X. The prior's mode is then proportional to gene frequency, so loadings the
        data doesn't support shrink towards the rank-1 (depth × frequency) model, and factors the
        data doesn't support at all collapse onto it, making k an upper bound. With
        h_prior="flat", b_j = h_rate for all genes, so the mode is a flat gene profile; this
        over-predicts rare genes and prunes many more factors. Since W H is unchanged by rescaling
        W and H in opposite directions, h_rate only sets the scale of H; by default it's
        h_pseudocount / mean(H0), keeping H at its initial scale. The strength is set by
        h_pseudocount (0 disables the prior and gives plain KL-NMF).
    eps : float
        Floor on factor values, applied after each update.
    warm_start : bool, optional
        Initialize by fitting a random subsample of cells (warm_start_iter iterations on a
        warm_start_fraction of the rows), then fitting W for all cells with that H held fixed
        (warm_start_w_passes iterations), before fitting everything. This is much cheaper than full
        iterations and substantially speeds up convergence on large datasets. By default it's used
        when no initial factors are given and the subsample would have at least
        WARM_START_MIN_CELLS cells. Its time counts toward max_time and is included in loss times.
    max_iter : int
        Maximum number of iterations.
    max_time : float, optional
        Stop after this many seconds, not counting time spent only on evaluating the objective.
    tol : float
        Stop when the relative decrease in KL divergence between evaluations is below this.
    eval_every : int
        Compute the KL divergence every this many iterations (0 disables it, and early stopping).
    W0, H0 : arrays of shape [m, k] and [k, n], optional
        Initial factors. Both or neither must be given (except H0 alone with fit_H=False).
        Chosen according to init if omitted.
    seed : int, optional
        Seed for random initialization, the randomized SVD, and the warm start's subsample.
    n_threads : int, optional
        Number of threads. Defaults to rayon's global pool (all cores, or RAYON_NUM_THREADS).
    verbose : bool
        Print the objective to stderr whenever it's evaluated.
    """
    if method not in ("mu", "bmme"):
        raise ValueError(f"unknown method {method!r}")
    if init not in ("nndsvd", "random"):
        raise ValueError(f"unknown init {init!r}")
    X = _as_csr(X)
    m, n = X.shape

    if X.data.size and X.data.min() < 0:
        raise ValueError("X must be non-negative")
    total = max(float(X.data.sum()), 1.0)

    n_sub = int(warm_start_fraction * m)
    if warm_start is None:
        warm_start = fit_H and W0 is None and H0 is None and n_sub >= WARM_START_MIN_CELLS
    if warm_start and not fit_H:
        raise ValueError("warm_start requires fit_H=True")
    if warm_start and not (k <= n_sub < m):
        raise ValueError(f"warm start subsample of {n_sub} cells is too small or too large")

    rng = np.random.default_rng(seed)
    if not fit_H and H0 is None:
        raise ValueError("fit_H=False requires H0")
    if (W0 is None) != (H0 is None) and fit_H:
        raise ValueError("W0 and H0 must be given together")
    t0 = time.perf_counter()
    if warm_start:
        idx = np.sort(rng.choice(m, n_sub, replace=False))
        X_sub = X[idx]
    # Rows of W without an initial value start from a constant, and are first fit with H fixed.
    flat_W = lambda: np.full((m, k), np.sqrt(total / (m * n) / k), dtype=np.float32)  # noqa: E731
    if W0 is None and H0 is not None:
        W0 = flat_W()
    elif W0 is None and init == "random":
        # Uniform init scaled so that E[(W H)_ij] = mean(X).
        scale = np.sqrt(4.0 * total / (m * n) / k)
        W0 = scale * rng.random((m, k), dtype=np.float32)
        H0 = scale * rng.random((k, n), dtype=np.float32)
    elif W0 is None and warm_start:
        W0 = flat_W()
        W0[idx], H0 = _nndsvd(X_sub, k, rng, n_threads, eps)
    elif W0 is None:
        W0, H0 = _nndsvd(X, k, rng, n_threads, eps)
    W0, H0 = np.asarray(W0), np.asarray(H0)
    if W0.shape != (m, k) or H0.shape != (k, n):
        raise ValueError(f"expected W0 {(m, k)} and H0 {(k, n)}, got {W0.shape} and {H0.shape}")

    w = np.array(W0, dtype=np.float32, order="C")
    ht = np.ascontiguousarray(H0.T, dtype=np.float32)
    if h_pseudocount < 0 or (h_rate is not None and h_rate < 0):
        raise ValueError("h_pseudocount and h_rate must be non-negative")
    if h_rate is None:
        h_rate = h_pseudocount / max(float(ht.mean(dtype=np.float64)), eps)
    h_shape = h_rate_j = None
    if h_pseudocount > 0 or h_rate > 0:
        h_shape, h_rate_j = np.full(n, h_pseudocount), np.full(n, h_rate)
        if h_prior != "flat":
            c = np.bincount(X.indices, weights=X.data, minlength=n) + 1.0
            f = c / c.mean()
            if h_prior != "gene-rate":
                raise ValueError(f"unknown h_prior {h_prior!r}")
            h_rate_j /= f
        h_shape, h_rate_j = h_shape.astype(np.float32), h_rate_j.astype(np.float32)
    opts = dict(verbose=verbose, n_threads=n_threads, method=method, restart=restart, eps=eps,
                h_shape=h_shape, h_rate=h_rate_j)

    if warm_start:
        w_sub, ht, _, _ = _fit(
            X_sub, w[idx], ht, max_iter=warm_start_iter, tol=-np.inf, eval_every=0,
            max_time=None, fit_H=True, **opts,
        )
        w[idx] = w_sub
        w, _, _, _ = _fit(
            X, w, ht, max_iter=warm_start_w_passes, tol=-np.inf, eval_every=0,
            max_time=None, fit_H=False, **opts,
        )
    init_time = time.perf_counter() - t0
    if verbose and init_time > 0.1:
        print(f"initialization: {init_time:.2f}s", file=sys.stderr)
    if max_time is not None:
        max_time = max(max_time - init_time, 0.0)

    w, ht, loss, n_iter = _fit(
        X, w, ht, max_iter=max_iter, tol=tol, eval_every=eval_every, max_time=max_time,
        fit_H=fit_H, **opts,
    )
    loss = [(i, t + init_time, l) for i, t, l in loss]
    return NMFResult(W=w, H=ht.T, loss=loss, n_iter=n_iter, init_time=init_time)
