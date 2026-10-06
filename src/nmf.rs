use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip, s};
use rayon::prelude::*;
use std::ops::Range;

// Maximum number of rows (or columns) per rayon task. Per-row work varies a lot (cells differ in
// depth, genes in detection rate) and inputs are often ordered in ways that cluster heavy rows, so
// without this rayon's default ~1 chunk per thread can leave most threads idle.
const PAR_GRAIN: usize = 4;

// Target size of the slice of W touched by one cell block in the H pass. Every thread works on the
// same block at once, so this should fit comfortably in (shared) L3. Tuned on a Ryzen 9 5950X
// (2 × 32MB L3); 4–16MB all perform similarly.
const H_PASS_BLOCK_BYTES: usize = 8 << 20;

pub struct NMFOptions {
    // maximum number of iterations
    pub max_iter: usize,

    // stop when the relative decrease in the objective falls below this
    pub tol: f64,

    // evaluate the objective every this many iterations (0 to never evaluate)
    pub eval_every: usize,

    pub verbose: bool,
}

impl Default for NMFOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-4,
            eval_every: 10,
            verbose: false,
        }
    }
}

pub struct NMFResult {
    pub w: Array2<f32>,  // [m, k]
    pub ht: Array2<f32>, // [n, k]

    // (iteration, objective) pairs
    pub loss: Vec<(usize, f64)>,

    pub n_iter: usize,
}

// A CSR matrix borrowing its arrays (e.g. from numpy).
pub struct CSR<'a> {
    pub data: ArrayView1<'a, f32>,
    pub indices: ArrayView1<'a, u32>, // column (gene) indices
    pub indptr: ArrayView1<'a, u32>,
}

// An optimized KL-NMF implementation, using extrapolated multiplicative updates, and parallelized across
// rows (typically, cells).
//
//     Hien,L.T.K., Leplat,V. and Gillis,N. (2025) Block Majorization
//     Minimization with extrapolation and application to β-NMF. SIAM J.
//     Math. Data Sci., 7, 1292–1314.
//
// `x` is the [m, n] count matrix. `w` and `ht` are the initial factors, so that X ≈ W Hᵀ.
pub fn nmf(
    x: &CSR,
    mut w: Array2<f32>,  // [m, k]
    mut ht: Array2<f32>, // [n, k]
    opts: &NMFOptions,
) -> NMFResult {
    let (m, k) = w.dim();
    let n = ht.nrows();
    assert_eq!(ht.ncols(), k);
    assert_eq!(x.indptr.len(), m + 1);
    assert_eq!(x.data.len(), x.indices.len());
    assert_eq!(x.indptr[m] as usize, x.data.len());

    let block_rows = (H_PASS_BLOCK_BYTES / (k * size_of::<f32>())).max(1);
    let csc = BlockedCSC::from_csr(x, n, block_rows);
    let mut ρht = Array2::<f32>::zeros((n, k));
    let mut loss = Vec::new();
    let mut prev_loss = f64::INFINITY;
    let mut n_iter = 0;

    let mut record = |iter: usize, l: f64| {
        loss.push((iter, l));
        if opts.verbose {
            eprintln!("iter {iter}: kl = {l:.6e}");
        }
    };

    for iter in 0..opts.max_iter {
        // The objective is computed during the step, for the state before the step.
        let eval = opts.eval_every > 0 && iter % opts.eval_every == 0;
        let l = mu_step(x, &csc, &mut ρht, &mut w, &mut ht, eval);
        n_iter = iter + 1;

        if let Some(l) = l {
            record(iter, l);
            if (prev_loss - l) / l.abs().max(f64::MIN_POSITIVE) < opts.tol {
                break;
            }
            prev_loss = l;
        }
    }

    if opts.eval_every > 0 {
        record(n_iter, kl_divergence(x, w.view(), ht.view()));
    }

    NMFResult {
        w,
        ht,
        loss,
        n_iter,
    }
}

// Generalized KL divergence D(X || W Hᵀ). Only nonzero entries of X contribute log terms, and
// Σ_ij u_ij factorizes as Σ_k (Σ_i w_ik)(Σ_j h_jk).
pub fn kl_divergence(
    x: &CSR,
    w: ArrayView2<f32>,  // [m, k]
    ht: ArrayView2<f32>, // [n, k]
) -> f64 {
    let nz_part: f64 = (0..w.nrows())
        .into_par_iter()
        .with_max_len(PAR_GRAIN)
        .map(|i| {
            let idx_from = x.indptr[i] as usize;
            let idx_to = x.indptr[i + 1] as usize;
            let w_i = w.row(i);
            let mut acc = 0_f64;
            for (&j, &x_ij) in x
                .indices
                .slice(s![idx_from..idx_to])
                .iter()
                .zip(x.data.slice(s![idx_from..idx_to]))
            {
                if x_ij > 0.0 {
                    let x_ij = x_ij as f64;
                    let u_ij = w_i.dot(&ht.row(j as usize)) as f64;
                    acc += x_ij * (x_ij / u_ij).ln() - x_ij;
                }
            }
            acc
        })
        .sum();

    nz_part + col_sum_f64(w).dot(&col_sum_f64(ht))
}

// Column sums accumulated in f64, in parallel over rows.
fn col_sum_f64(a: ArrayView2<f32>) -> Array1<f64> {
    let k = a.ncols();
    a.axis_iter(Axis(0))
        .into_par_iter()
        .fold(
            || Array1::<f64>::zeros(k),
            |mut acc, row| {
                Zip::from(&mut acc)
                    .and(&row)
                    .for_each(|acc_k, &v| *acc_k += v as f64);
                acc
            },
        )
        .reduce(|| Array1::<f64>::zeros(k), |a, b| a + b)
}

// Transposed (CSC) copy of a contiguous block of rows of X. Row indices are global.
struct CSC {
    data: Vec<f32>,
    indices: Vec<u32>, // row (cell) indices
    indptr: Vec<usize>,
}

impl CSC {
    fn from_csr(x: &CSR, n: usize, rows: Range<usize>) -> Self {
        let p_from = x.indptr[rows.start] as usize;
        let p_to = x.indptr[rows.end] as usize;
        let nnz = p_to - p_from;

        let mut csc_indptr = vec![0_usize; n + 1];
        for &j in x.indices.slice(s![p_from..p_to]) {
            csc_indptr[j as usize + 1] += 1;
        }
        for j in 0..n {
            csc_indptr[j + 1] += csc_indptr[j];
        }

        let mut next = csc_indptr.clone();
        let mut csc_data = vec![0_f32; nnz];
        let mut csc_indices = vec![0_u32; nnz];
        for i in rows {
            for p in x.indptr[i] as usize..x.indptr[i + 1] as usize {
                let j = x.indices[p] as usize;
                csc_data[next[j]] = x.data[p];
                csc_indices[next[j]] = i as u32;
                next[j] += 1;
            }
        }

        Self {
            data: csc_data,
            indices: csc_indices,
            indptr: csc_indptr,
        }
    }
}

// CSC copy of X split into blocks of cells, so H can be updated in a column-parallel pass where each
// thread owns the rows of H it writes, while the rows of W being gathered stay in cache.
struct BlockedCSC {
    blocks: Vec<CSC>,
}

impl BlockedCSC {
    fn from_csr(x: &CSR, n: usize, block_rows: usize) -> Self {
        let m = x.indptr.len() - 1;
        let blocks = (0..m.div_ceil(block_rows).max(1))
            .into_par_iter()
            .map(|b| CSC::from_csr(x, n, b * block_rows..((b + 1) * block_rows).min(m)))
            .collect();
        Self { blocks }
    }
}

// One multiplicative update of W (row-parallel over CSR), followed by one of H using the updated W
// (column-parallel over CSC, one block of cells at a time). If `compute_loss`, also returns the KL divergence of the state prior
// to the update, which is nearly free since the W pass already computes u_ij at each nonzero.
fn mu_step(
    x: &CSR,
    csc: &BlockedCSC,
    ρht: &mut Array2<f32>, // [n, k] accumulator, all zeros on entry and exit
    w: &mut Array2<f32>,   // [m, k]
    ht: &mut Array2<f32>,  // [n, k]
    compute_loss: bool,
) -> Option<f64> {
    const EPS: f32 = 1e-6;
    let k = w.ncols();

    // Σ_ij u_ij term of the objective, from the state prior to the update
    let u_sum = compute_loss.then(|| col_sum_f64(w.view()).dot(&col_sum_f64(ht.view())));

    // update W
    let h_col_sum = ht.sum_axis(Axis(0));
    let ht_ro = &*ht;
    let nz_loss: f64 = w
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .with_max_len(PAR_GRAIN)
        .enumerate()
        .map_init(
            || Array1::<f32>::zeros(k),
            |ρw_i, (i, mut w_i)| {
                ρw_i.fill(0_f32);
                let mut nz_loss_i = 0_f64;

                // for each j
                for p in x.indptr[i] as usize..x.indptr[i + 1] as usize {
                    let h_j = ht_ro.row(x.indices[p] as usize);
                    let x_ij = x.data[p];
                    let u_ij = w_i.dot(&h_j);
                    if compute_loss && x_ij > 0.0 {
                        let x_ij = x_ij as f64;
                        nz_loss_i += x_ij * (x_ij / u_ij as f64).ln() - x_ij;
                    }
                    ρw_i.scaled_add(x_ij / u_ij, &h_j);
                }

                // for each k
                Zip::from(&mut w_i).and(&*ρw_i).and(&h_col_sum).for_each(
                    |w_ik, ρw_ik, h_col_sum_k| {
                        *w_ik = (*w_ik * ρw_ik / h_col_sum_k).max(EPS);
                    },
                );

                nz_loss_i
            },
        )
        .sum();

    // update H
    let w_col_sum = w.sum_axis(Axis(0));
    let w_ro = &*w;
    let ht_ro = &*ht;
    for block in &csc.blocks {
        ρht.axis_iter_mut(Axis(0))
            .into_par_iter()
            .with_max_len(PAR_GRAIN)
            .enumerate()
            .for_each(|(j, mut ρh_j)| {
                let h_j = ht_ro.row(j);

                // for each i in the block
                for q in block.indptr[j]..block.indptr[j + 1] {
                    let w_i = w_ro.row(block.indices[q] as usize);
                    let r_ij = block.data[q] / w_i.dot(&h_j);
                    ρh_j.scaled_add(r_ij, &w_i);
                }
            });
    }

    Zip::from(ht.rows_mut())
        .and(ρht.rows_mut())
        .par_for_each(|mut h_j, mut ρh_j| {
            // for each k
            Zip::from(&mut h_j)
                .and(&ρh_j)
                .and(&w_col_sum)
                .for_each(|h_jk, ρh_jk, w_col_sum_k| {
                    *h_jk = (*h_jk * ρh_jk / w_col_sum_k).max(EPS);
                });
            ρh_j.fill(0_f32);
        });

    u_sum.map(|u_sum| nz_loss + u_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Small deterministic PRNG so tests don't need an extra dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }
    }

    // Random sparse count matrix in CSR form, with varying per-gene sparsity.
    fn random_csr(rng: &mut Lcg, m: usize, n: usize) -> (Array1<f32>, Array1<u32>, Array1<u32>) {
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0_u32];
        for _ in 0..m {
            for j in 0..n {
                let p = 0.05 + 0.6 * (j as f32 / n as f32);
                if rng.next_f32() < p {
                    data.push(1.0 + (rng.next_f32() * 10.0).floor());
                    indices.push(j as u32);
                }
            }
            indptr.push(data.len() as u32);
        }
        (data.into(), indices.into(), indptr.into())
    }

    #[test]
    fn objective_is_nonincreasing() {
        let (m, n, k) = (200, 80, 5);
        let mut rng = Lcg(1234);
        let (data, indices, indptr) = random_csr(&mut rng, m, n);
        let w = Array2::from_shape_simple_fn((m, k), || 0.1 + rng.next_f32());
        let ht = Array2::from_shape_simple_fn((n, k), || 0.1 + rng.next_f32());

        let opts = NMFOptions {
            max_iter: 100,
            tol: f64::NEG_INFINITY,
            eval_every: 1,
            verbose: false,
        };
        let x = CSR {
            data: data.view(),
            indices: indices.view(),
            indptr: indptr.view(),
        };
        let result = nmf(&x, w, ht, &opts);

        assert_eq!(result.n_iter, opts.max_iter);
        assert_eq!(result.loss.len(), opts.max_iter + 1);
        for pair in result.loss.windows(2) {
            let (prev, next) = (pair[0].1, pair[1].1);
            assert!(next.is_finite());
            assert!(
                next <= prev * (1.0 + 1e-6),
                "objective increased: {prev} -> {next}"
            );
        }
        let (first, last) = (result.loss[0].1, result.loss.last().unwrap().1);
        assert!(
            last < 0.9 * first,
            "objective barely decreased: {first} -> {last}"
        );
    }

    #[test]
    fn blocking_does_not_change_result() {
        let (m, n, k) = (200, 80, 5);
        let mut rng = Lcg(5678);
        let (data, indices, indptr) = random_csr(&mut rng, m, n);
        let x = CSR {
            data: data.view(),
            indices: indices.view(),
            indptr: indptr.view(),
        };
        let w0 = Array2::from_shape_simple_fn((m, k), || 0.1 + rng.next_f32());
        let ht0 = Array2::from_shape_simple_fn((n, k), || 0.1 + rng.next_f32());

        let run = |block_rows: usize| {
            let csc = BlockedCSC::from_csr(&x, n, block_rows);
            let mut ρht = Array2::zeros((n, k));
            let (mut w, mut ht) = (w0.clone(), ht0.clone());
            let losses: Vec<_> = (0..10)
                .map(|_| mu_step(&x, &csc, &mut ρht, &mut w, &mut ht, true).unwrap())
                .collect();
            (csc.blocks.len(), w, ht, losses)
        };

        let (nb1, w1, ht1, l1) = run(m);
        let (nb2, w2, ht2, l2) = run(7);
        assert_eq!((nb1, nb2), (1, m.div_ceil(7)));
        assert_eq!(w1, w2);
        assert_eq!(ht1, ht2);
        assert_eq!(l1, l2);
    }
}
