use pyo3::prelude::*;

mod nmf;

#[pymodule]
mod metagene {
    use crate::nmf::{NMFOptions, nmf};
    use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// Low-level KL-NMF entry point. Use `metagene.nmf` instead.
    ///
    /// Factorizes the [m, n] CSR matrix (data, indices, indptr) as X ≈ W Hᵀ, starting from
    /// `w` [m, k] and `ht` [n, k]. Returns (w, ht, loss, n_iter), where loss is a list of
    /// (iteration, kl) pairs.
    #[pyfunction]
    #[pyo3(signature = (data, indices, indptr, w, ht, max_iter, tol, eval_every, verbose, n_threads=None))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn _nmf<'py>(
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f32>,
        indices: PyReadonlyArray1<'py, u32>,
        indptr: PyReadonlyArray1<'py, u32>,
        w: PyReadonlyArray2<'py, f32>,
        ht: PyReadonlyArray2<'py, f32>,
        max_iter: usize,
        tol: f64,
        eval_every: usize,
        verbose: bool,
        n_threads: Option<usize>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Vec<(usize, f64)>,
        usize,
    )> {
        let data = data.as_array();
        let indices = indices.as_array();
        let indptr = indptr.as_array();
        let w = w.as_array().to_owned();
        let ht = ht.as_array().to_owned();

        let (m, k) = w.dim();
        let n = ht.nrows();
        if ht.ncols() != k {
            return Err(PyValueError::new_err(
                "w and ht must have the same number of columns",
            ));
        }
        if indptr.len() != m + 1 {
            return Err(PyValueError::new_err(
                "indptr length must be w.shape[0] + 1",
            ));
        }
        if data.len() != indices.len() || indptr[m] as usize != data.len() {
            return Err(PyValueError::new_err("inconsistent CSR arrays"));
        }
        if indices.iter().any(|&j| j as usize >= n) {
            return Err(PyValueError::new_err("column index out of bounds for ht"));
        }

        let opts = NMFOptions {
            max_iter,
            tol,
            eval_every,
            verbose,
        };

        let pool = match n_threads {
            Some(n_threads) => Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .build()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
            None => None,
        };

        let result = py.detach(|| {
            let run = || nmf(data, indices, indptr, w, ht, &opts);
            match &pool {
                Some(pool) => pool.install(run),
                None => run(),
            }
        });

        Ok((
            result.w.into_pyarray(py),
            result.ht.into_pyarray(py),
            result.loss,
            result.n_iter,
        ))
    }
}
