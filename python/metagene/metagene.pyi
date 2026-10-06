import numpy as np

def _nmf(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    w: np.ndarray,
    ht: np.ndarray,
    max_iter: int,
    tol: float,
    eval_every: int,
    verbose: bool,
    n_threads: int | None = None,
    max_time: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, float, float]], int]: ...
