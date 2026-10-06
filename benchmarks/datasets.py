"""Loaders for benchmark datasets, returning a cells × genes scipy CSR matrix.

These avoid depending on scanpy/anndata, at the cost of only supporting the specific on-disk
layouts below.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp


def read_10x_h5(path) -> sp.csr_matrix:
    """10x Cell Ranger filtered_feature_bc_matrix.h5 (stored genes × cells CSC)."""
    import h5py

    with h5py.File(path) as f:
        g = f["matrix"]
        ngenes, ncells = g["shape"][:]
        return sp.csr_matrix(
            (g["data"][:].astype(np.float32), g["indices"][:], g["indptr"][:]),
            shape=(ncells, ngenes),
        )


def read_anndata_zarr(path, ncells=None) -> sp.csr_matrix:
    """`X` of an AnnData zarr store, which must be a csr_matrix. Optionally only the first ncells rows."""
    import zarr

    g = zarr.open_group(str(path), mode="r")["X"]
    if g.attrs.get("encoding-type") != "csr_matrix":
        raise ValueError(f"expected X to be a csr_matrix, got {g.attrs.get('encoding-type')}")
    m, n = g.attrs["shape"]
    ncells = m if ncells is None else min(ncells, m)
    indptr = g["indptr"][: ncells + 1].astype(np.int64)
    end = int(indptr[-1])
    return sp.csr_matrix((g["data"][:end], g["indices"][:end], indptr), shape=(ncells, n))


def load(path, ncells=None) -> sp.csr_matrix:
    path = Path(path)
    if path.suffix == ".h5":
        X = read_10x_h5(path)
        if ncells is not None:
            X = X[:ncells]
    elif path.suffix == ".zarr":
        X = read_anndata_zarr(path, ncells)
    else:
        raise ValueError(f"don't know how to read {path}")
    X.sum_duplicates()
    return X
