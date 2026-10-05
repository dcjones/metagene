from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .metagene import _nmf

__all__ = ["nmf", "NMFResult"]


@dataclass
class NMFResult:
    W: np.ndarray  # [m, k]
    H: np.ndarray  # [k, n]
    loss: list = field(default_factory=list)  # (iteration, kl) pairs
    n_iter: int = 0


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


def nmf(
    X,
    k: int,
    *,
    max_iter: int = 200,
    tol: float = 1e-4,
    eval_every: int = 10,
    W0: np.ndarray | None = None,
    H0: np.ndarray | None = None,
    seed: int | None = None,
    n_threads: int | None = None,
    verbose: bool = False,
) -> NMFResult:
    """KL-NMF of a non-negative [m, n] matrix X ≈ W H, using multiplicative updates.

    Parameters
    ----------
    X : scipy sparse matrix or array-like, shape [m, n]
        Count matrix, typically cells × genes. Converted to CSR float32.
    k : int
        Number of factors.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Stop when the relative decrease in KL divergence between evaluations is below this.
    eval_every : int
        Compute the KL divergence every this many iterations (0 disables it, and early stopping).
    W0, H0 : arrays of shape [m, k] and [k, n], optional
        Initial factors. Both or neither must be given. Random if omitted.
    seed : int, optional
        Seed for random initialization.
    n_threads : int, optional
        Number of threads. Defaults to rayon's global pool (all cores, or RAYON_NUM_THREADS).
    verbose : bool
        Print the objective to stderr whenever it's evaluated.
    """
    X = _as_csr(X)
    m, n = X.shape

    data = np.ascontiguousarray(X.data, dtype=np.float32)
    if data.size and data.min() < 0:
        raise ValueError("X must be non-negative")
    indices = _as_u32(X.indices)
    indptr = _as_u32(X.indptr)

    if (W0 is None) != (H0 is None):
        raise ValueError("W0 and H0 must be given together")
    if W0 is None:
        # Uniform init scaled so that E[(W H)_ij] = mean(X).
        rng = np.random.default_rng(seed)
        scale = np.sqrt(4.0 * max(data.sum(), 1.0) / (m * n) / k)
        W0 = scale * rng.random((m, k), dtype=np.float32)
        H0 = scale * rng.random((k, n), dtype=np.float32)
    W0 = np.asarray(W0)
    H0 = np.asarray(H0)
    if W0.shape != (m, k) or H0.shape != (k, n):
        raise ValueError(f"expected W0 {(m, k)} and H0 {(k, n)}, got {W0.shape} and {H0.shape}")

    w = np.ascontiguousarray(W0, dtype=np.float32)
    ht = np.ascontiguousarray(H0.T, dtype=np.float32)

    w, ht, loss, n_iter = _nmf(
        data, indices, indptr, w, ht, max_iter, tol, eval_every, verbose, n_threads
    )
    return NMFResult(W=w, H=ht.T, loss=loss, n_iter=n_iter)
