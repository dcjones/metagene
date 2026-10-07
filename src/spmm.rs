// Sparse × dense products X B and Xᵀ B, for the randomized SVD behind NNDSVD initialization. They
// have the same structure as the NMF passes: X B is row-parallel over CSR (like the W pass), Xᵀ B is
// column-parallel over the cache-blocked CSC copy (like the H pass), so each thread owns the rows
// it writes.

use crate::kernels::{Isa, axpy, axpy_n, dispatch, prefetch};
use crate::nmf::{BlockedCSC, CSR, NNZ_BLOCK, PAR_GRAIN, PREFETCH_DISTANCE};
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use std::ops::Range;

// out += Σ_p data[p] b[indices[p]] over p in `range`, where b is row-major with rows of out.len().
#[inline(always)]
#[allow(clippy::needless_range_loop)] // indices address several parallel arrays
fn gather_axpy<const AVX2: bool>(
    out: &mut [f32],
    b: &[f32],
    data: &[f32],
    indices: &[u32],
    range: Range<usize>,
) {
    let l = out.len();
    let row = |p: usize| {
        let j = indices[p] as usize;
        &b[j * l..(j + 1) * l]
    };
    let indices = &indices[..range.end];

    // for each nonzero, NNZ_BLOCK at a time
    let mut p = range.start;
    while p + NNZ_BLOCK <= range.end {
        for d in 0..NNZ_BLOCK {
            if let Some(&j) = indices.get(p + PREFETCH_DISTANCE * NNZ_BLOCK + d) {
                let j = j as usize;
                prefetch(&b[j * l..(j + 1) * l]);
            }
        }
        let b_js: [&[f32]; NNZ_BLOCK] = std::array::from_fn(|d| row(p + d));
        let a: [f32; NNZ_BLOCK] = std::array::from_fn(|d| data[p + d]);
        axpy_n::<AVX2, NNZ_BLOCK>(a, b_js, out);
        p += NNZ_BLOCK;
    }
    for p in p..range.end {
        axpy::<AVX2>(data[p], row(p), out);
    }
}

// X B, for X [m, n] and B [n, l]. Returns [m, l].
pub fn csr_matmul(x: &CSR, b: ArrayView2<f32>) -> Array2<f32> {
    let m = x.indptr.len() - 1;
    let l = b.ncols();
    let isa = Isa::detect();
    let b = b.as_standard_layout();
    let b = b.as_slice().unwrap();
    let (data, indices, indptr) = x.slices();

    let mut out = Array2::<f32>::zeros((m, l));
    if l == 0 {
        return out;
    }
    out.as_slice_mut()
        .unwrap()
        .par_chunks_mut(l)
        .with_max_len(PAR_GRAIN)
        .enumerate()
        .for_each(|(i, out_i)| {
            let range = indptr[i] as usize..indptr[i + 1] as usize;
            dispatch!(isa, gather_axpy(out_i, b, data, indices, range));
        });
    out
}

// Xᵀ B, for X [m, n] given as its blocked CSC copy, and B [m, l]. Returns [n, l].
pub fn csc_matmul(csc: &BlockedCSC, n: usize, b: ArrayView2<f32>) -> Array2<f32> {
    let l = b.ncols();
    let isa = Isa::detect();
    let b = b.as_standard_layout();
    let b = b.as_slice().unwrap();

    let mut out = Array2::<f32>::zeros((n, l));
    if l == 0 {
        return out;
    }
    let out_slice = out.as_slice_mut().unwrap();
    for block in &csc.blocks {
        out_slice
            .par_chunks_mut(l)
            .with_max_len(PAR_GRAIN)
            .enumerate()
            .for_each(|(j, out_j)| {
                let range = block.indptr[j]..block.indptr[j + 1];
                dispatch!(
                    isa,
                    gather_axpy(out_j, b, &block.data, &block.indices, range)
                );
            });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    // Small sparse test matrix in CSR form, with its dense equivalent.
    fn test_matrix(m: usize, n: usize) -> (Array1<f32>, Array1<u32>, Array1<u32>, Array2<f32>) {
        let mut dense = Array2::<f32>::zeros((m, n));
        let (mut data, mut indices, mut indptr) = (Vec::new(), Vec::new(), vec![0_u32]);
        for i in 0..m {
            for j in 0..n {
                // a fixed, irregular sparsity pattern, including empty rows and columns
                if (i * 7 + j * 13) % 5 == 0 && i % 9 != 4 && j != 3 {
                    let v = 1.0 + ((i * 31 + j * 17) % 11) as f32;
                    dense[[i, j]] = v;
                    data.push(v);
                    indices.push(j as u32);
                }
            }
            indptr.push(data.len() as u32);
        }
        (data.into(), indices.into(), indptr.into(), dense)
    }

    fn test_dense(r: usize, c: usize) -> Array2<f32> {
        Array2::from_shape_fn((r, c), |(i, j)| ((i * 3 + j * 5) % 7) as f32 - 2.5)
    }

    #[test]
    fn products_match_dense() {
        let (m, n) = (53, 21);
        let (data, indices, indptr, dense) = test_matrix(m, n);
        let x = CSR {
            data: data.view(),
            indices: indices.view(),
            indptr: indptr.view(),
        };
        // l = 13 exercises the masked tails of the AVX2 kernels
        for l in [1, 8, 13] {
            let b = test_dense(n, l);
            let xb = csr_matmul(&x, b.view());
            assert!((&xb - &dense.dot(&b)).iter().all(|e| e.abs() < 1e-3));

            let c = test_dense(m, l);
            // blocks of 1 row, a few rows, and all rows
            for block_rows in [1, 10, m] {
                let csc = BlockedCSC::from_csr(&x, n, block_rows);
                let xtc = csc_matmul(&csc, n, c.view());
                assert!((&xtc - &dense.t().dot(&c)).iter().all(|e| e.abs() < 1e-3));
            }
        }
    }

    #[test]
    fn transposed_b_is_handled() {
        let (m, n, l) = (20, 9, 5);
        let (data, indices, indptr, dense) = test_matrix(m, n);
        let x = CSR {
            data: data.view(),
            indices: indices.view(),
            indptr: indptr.view(),
        };
        let b = test_dense(l, n); // [l, n], so b.t() is a non-standard-layout [n, l] view
        let xb = csr_matmul(&x, b.t());
        assert!((&xb - &dense.dot(&b.t())).iter().all(|e| e.abs() < 1e-3));
    }
}
