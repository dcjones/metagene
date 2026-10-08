use pyo3::prelude::*;

mod kernels;
mod nmf;
mod spmm;

#[pymodule]
mod metagene {
    use crate::nmf::{BlockedCSC, CSR, H_PASS_BLOCK_BYTES, Minibatch, NMFOptions, nmf};
    use crate::spmm::{csc_matmul, csr_matmul};
    use numpy::{
        IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    };
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    fn thread_pool(n_threads: Option<usize>) -> PyResult<Option<rayon::ThreadPool>> {
        n_threads
            .map(|n_threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .build()
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            })
            .transpose()
    }

    fn check_csr(
        data: &PyReadonlyArray1<f32>,
        indices: &PyReadonlyArray1<u32>,
        indptr: &PyReadonlyArray1<u32>,
        m: usize,
        n: usize,
    ) -> PyResult<()> {
        let indptr = indptr.as_array();
        if indptr.len() != m + 1 {
            return Err(PyValueError::new_err("indptr length must be m + 1"));
        }
        let nnz = data.as_array().len();
        if indices.as_array().len() != nnz || indptr[m] as usize != nnz {
            return Err(PyValueError::new_err("inconsistent CSR arrays"));
        }
        if indices.as_array().iter().any(|&j| j as usize >= n) {
            return Err(PyValueError::new_err("column index out of bounds"));
        }
        Ok(())
    }

    /// A CSR matrix X [m, n] (borrowing its arrays) for computing X B and Xᵀ B with dense B, as
    /// needed by the randomized SVD. Building it makes a cache-blocked CSC copy of X (8 bytes per
    /// nonzero) for Xᵀ B, sized for B with up to `l` columns.
    #[pyclass(frozen)]
    struct _SparseMatrix {
        data: Py<PyArray1<f32>>,
        indices: Py<PyArray1<u32>>,
        indptr: Py<PyArray1<u32>>,
        m: usize,
        n: usize,
        csc: BlockedCSC,
        pool: Option<rayon::ThreadPool>,
    }

    impl _SparseMatrix {
        // Run f (without the GIL) on the thread pool, if there is one.
        fn run<R: Send>(&self, py: Python, f: impl FnOnce() -> R + Send) -> R {
            py.detach(|| match &self.pool {
                Some(pool) => pool.install(f),
                None => f(),
            })
        }
    }

    #[pymethods]
    impl _SparseMatrix {
        #[new]
        #[pyo3(signature = (data, indices, indptr, n, l, n_threads=None))]
        fn new(
            py: Python,
            data: PyReadonlyArray1<f32>,
            indices: PyReadonlyArray1<u32>,
            indptr: PyReadonlyArray1<u32>,
            n: usize,
            l: usize,
            n_threads: Option<usize>,
        ) -> PyResult<Self> {
            let m = indptr.as_array().len().saturating_sub(1);
            check_csr(&data, &indices, &indptr, m, n)?;
            let pool = thread_pool(n_threads)?;
            let block_rows = (H_PASS_BLOCK_BYTES / (l.max(1) * size_of::<f32>())).max(1);
            let x = CSR {
                data: data.as_array(),
                indices: indices.as_array(),
                indptr: indptr.as_array(),
            };
            let build = || BlockedCSC::from_csr(&x, n, block_rows);
            let csc = py.detach(|| match &pool {
                Some(pool) => pool.install(build),
                None => build(),
            });
            Ok(Self {
                data: data.as_unbound().clone_ref(py),
                indices: indices.as_unbound().clone_ref(py),
                indptr: indptr.as_unbound().clone_ref(py),
                m,
                n,
                csc,
                pool,
            })
        }

        /// X B, for B [n, l]. Returns [m, l].
        fn matmul<'py>(
            &self,
            py: Python<'py>,
            b: PyReadonlyArray2<'py, f32>,
        ) -> PyResult<Bound<'py, PyArray2<f32>>> {
            if b.as_array().nrows() != self.n {
                return Err(PyValueError::new_err("B must have n rows"));
            }
            let (data, indices, indptr) = (
                self.data.bind(py).readonly(),
                self.indices.bind(py).readonly(),
                self.indptr.bind(py).readonly(),
            );
            let x = CSR {
                data: data.as_array(),
                indices: indices.as_array(),
                indptr: indptr.as_array(),
            };
            let b = b.as_array();
            Ok(self.run(py, || csr_matmul(&x, b)).into_pyarray(py))
        }

        /// Xᵀ B, for B [m, l]. Returns [n, l].
        fn rmatmul<'py>(
            &self,
            py: Python<'py>,
            b: PyReadonlyArray2<'py, f32>,
        ) -> PyResult<Bound<'py, PyArray2<f32>>> {
            if b.as_array().nrows() != self.m {
                return Err(PyValueError::new_err("B must have m rows"));
            }
            let b = b.as_array();
            Ok(self.run(py, || csc_matmul(&self.csc, self.n, b)).into_pyarray(py))
        }
    }

    /// Low-level KL-NMF entry point. Use `metagene.nmf` instead.
    ///
    /// Factorizes the [m, n] CSR matrix (data, indices, indptr) as X ≈ W Hᵀ, starting from
    /// `w` [m, k] and `ht` [n, k]. Returns (w, ht, loss, n_iter), where loss is a list of
    /// (iteration, elapsed seconds, objective) tuples; the objective is the KL divergence plus H's
    /// prior penalty, if any. Given `batch_perm` (a permutation of the m rows), trains with
    /// minibatches of `batch_size` consecutive rows of it instead (see `Minibatch`).
    #[pyfunction]
    #[pyo3(signature = (data, indices, indptr, w, ht, max_iter, tol, eval_every, verbose, n_threads=None, max_time=None, extrapolate=false, restart=false, fit_h=true, eps=1e-6, h_shape=None, h_rate=None, batch_perm=None, batch_size=1000, batch_step=0.1, batch_w_steps=1, batch_final_w_passes=0, batch_seed=0))]
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
        max_time: Option<f64>,
        extrapolate: bool,
        restart: bool,
        fit_h: bool,
        eps: f32,
        h_shape: Option<PyReadonlyArray1<'py, f32>>,
        h_rate: Option<PyReadonlyArray1<'py, f32>>,
        batch_perm: Option<PyReadonlyArray1<'py, u32>>,
        batch_size: usize,
        batch_step: f32,
        batch_w_steps: usize,
        batch_final_w_passes: usize,
        batch_seed: u64,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Vec<(usize, f64, f64)>,
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

        let h_prior = match (h_shape, h_rate) {
            (Some(a), Some(b)) => {
                if a.len()? != n || b.len()? != n {
                    return Err(PyValueError::new_err(
                        "h_shape and h_rate must have length n",
                    ));
                }
                Some((a.as_array().to_owned(), b.as_array().to_owned()))
            }
            (None, None) => None,
            _ => {
                return Err(PyValueError::new_err(
                    "h_shape and h_rate must be given together",
                ));
            }
        };

        let minibatch = match batch_perm {
            Some(perm) => {
                let perm = perm.as_array().to_vec();
                let mut seen = vec![false; m];
                for &i in &perm {
                    if (i as usize) >= m || std::mem::replace(&mut seen[i as usize], true) {
                        return Err(PyValueError::new_err(
                            "batch_perm must be a permutation of m",
                        ));
                    }
                }
                if perm.len() != m || batch_size == 0 || !(batch_step > 0.0 && batch_step <= 1.0) {
                    return Err(PyValueError::new_err(
                        "need batch_perm of length m, batch_size > 0, 0 < batch_step <= 1",
                    ));
                }
                if extrapolate || !fit_h {
                    return Err(PyValueError::new_err(
                        "minibatch training requires extrapolate=False and fit_h=True",
                    ));
                }
                Some(Minibatch {
                    perm,
                    batch_size,
                    step: batch_step,
                    w_steps: batch_w_steps,
                    final_w_passes: batch_final_w_passes,
                    seed: batch_seed,
                })
            }
            None => None,
        };

        let opts = NMFOptions {
            max_iter,
            tol,
            eval_every,
            max_time,
            extrapolate,
            restart,
            fit_h,
            eps,
            h_prior,
            minibatch,
            verbose,
        };

        let pool = thread_pool(n_threads)?;

        let result = py.detach(|| {
            let x = CSR {
                data,
                indices,
                indptr,
            };
            let run = || nmf(&x, w, ht, &opts);
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
