use pyo3::prelude::*;

mod nmf;
use nmf::nmf;

/// A Python module implemented in Rust.
#[pymodule]
mod metagene {
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
}
