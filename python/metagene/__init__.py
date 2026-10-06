from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .metagene import _nmf

__all__ = ["nmf", "NMFResult"]


@dataclass
class NMFResult:
    W: np.ndarray  # [m, k]
    H: np.ndarray  # [k, n]
    loss: list = field(default_factory=list)  # (iteration, elapsed seconds, kl) tuples
    n_iter: int = 0
    warm_start_time: float = 0.0  # seconds spent on the warm start, included in loss times


# The warm start is enabled automatically when the subsample would have at least this many cells.
WARM_START_MIN_CELLS = 20_000


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


def _fit(X, w, ht, *, max_iter, tol, eval_every, verbose, n_threads, max_time, method, restart, fit_H):
    """Run the Rust solver on a canonical CSR matrix. Returns (w, ht, loss, n_iter)."""
    data = np.ascontiguousarray(X.data, dtype=np.float32)
    return _nmf(
        data, _as_u32(X.indices), _as_u32(X.indptr), w, ht, max_iter, tol, eval_every, verbose,
        n_threads, max_time, method == "bmme", restart, fit_H,
    )


def nmf(
    X,
    k: int,
    *,
    method: str = "bmme",
    restart: bool = False,
    fit_H: bool = True,
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
    restart : bool
        With method="bmme", reset the extrapolation whenever an evaluated objective increases.
    fit_H : bool
        If False, H is held fixed at H0 (which must be given) and only W is fit, e.g. to project
        new cells onto existing factors.
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
        Initial factors. Both or neither must be given. Random if omitted.
    seed : int, optional
        Seed for random initialization and the warm start's subsample.
    n_threads : int, optional
        Number of threads. Defaults to rayon's global pool (all cores, or RAYON_NUM_THREADS).
    verbose : bool
        Print the objective to stderr whenever it's evaluated.
    """
    if method not in ("mu", "bmme"):
        raise ValueError(f"unknown method {method!r}")
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
    if W0 is None and H0 is not None and not fit_H:
        W0 = np.full((m, k), np.sqrt(total / (m * n) / k), dtype=np.float32)
    if (W0 is None) != (H0 is None):
        raise ValueError("W0 and H0 must be given together")
    if W0 is None:
        # Uniform init scaled so that E[(W H)_ij] = mean(X).
        scale = np.sqrt(4.0 * total / (m * n) / k)
        W0 = scale * rng.random((m, k), dtype=np.float32)
        H0 = scale * rng.random((k, n), dtype=np.float32)
    W0 = np.asarray(W0)
    H0 = np.asarray(H0)
    if W0.shape != (m, k) or H0.shape != (k, n):
        raise ValueError(f"expected W0 {(m, k)} and H0 {(k, n)}, got {W0.shape} and {H0.shape}")

    w = np.array(W0, dtype=np.float32, order="C")
    ht = np.ascontiguousarray(H0.T, dtype=np.float32)
    opts = dict(verbose=verbose, n_threads=n_threads, method=method, restart=restart)

    warm_start_time = 0.0
    if warm_start:
        t0 = time.perf_counter()
        idx = np.sort(rng.choice(m, n_sub, replace=False))
        w_sub, ht, _, _ = _fit(
            X[idx], w[idx], ht, max_iter=warm_start_iter, tol=-np.inf, eval_every=0,
            max_time=None, fit_H=True, **opts,
        )
        w[idx] = w_sub
        w, _, _, _ = _fit(
            X, w, ht, max_iter=warm_start_w_passes, tol=-np.inf, eval_every=0,
            max_time=None, fit_H=False, **opts,
        )
        warm_start_time = time.perf_counter() - t0
        if verbose:
            print(f"warm start: {warm_start_time:.2f}s", file=sys.stderr)
        if max_time is not None:
            max_time = max(max_time - warm_start_time, 0.0)

    w, ht, loss, n_iter = _fit(
        X, w, ht, max_iter=max_iter, tol=tol, eval_every=eval_every, max_time=max_time,
        fit_H=fit_H, **opts,
    )
    loss = [(i, t + warm_start_time, l) for i, t, l in loss]
    return NMFResult(W=w, H=ht.T, loss=loss, n_iter=n_iter, warm_start_time=warm_start_time)
