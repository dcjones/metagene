use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip, s};
use rayon::prelude::*;
use std::{ops::AddAssign, sync::Mutex};

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

// An optimized KL-NMF implementation, using extrapolated multiplicative updates, and parallelized across
// rows (typically, cells).
//
//     Hien,L.T.K., Leplat,V. and Gillis,N. (2025) Block Majorization
//     Minimization with extrapolation and application to β-NMF. SIAM J.
//     Math. Data Sci., 7, 1292–1314.
//
// The [m, n] count matrix is given in CSR form by `data`, `indices`, `indptr`. `w` and `ht` are the
// initial factors, so that X ≈ W Hᵀ.
pub fn nmf(
    data: ArrayView1<f32>,
    indices: ArrayView1<u32>,
    indptr: ArrayView1<u32>,
    mut w: Array2<f32>,  // [m, k]
    mut ht: Array2<f32>, // [n, k]
    opts: &NMFOptions,
) -> NMFResult {
    // TODO: There are gains to be had by re-ordering the cells/genes to improve cache locality.

    let (m, k) = w.dim();
    let n = ht.nrows();
    assert_eq!(ht.ncols(), k);
    assert_eq!(indptr.len(), m + 1);
    assert_eq!(data.len(), indices.len());
    assert_eq!(indptr[m] as usize, data.len());

    let mut work = FusedMUWork::new(rayon::current_num_threads(), n, k);
    let mut loss = Vec::new();
    let mut prev_loss = f64::INFINITY;
    let mut n_iter = 0;

    for iter in 0..opts.max_iter {
        fused_mu(data, indices, indptr, &mut w, &mut ht, &mut work);
        n_iter = iter + 1;

        let last = n_iter == opts.max_iter;
        if opts.eval_every > 0 && (n_iter % opts.eval_every == 0 || last) {
            let l = kl_divergence(data, indices, indptr, w.view(), ht.view());
            loss.push((n_iter, l));
            if opts.verbose {
                eprintln!("iter {n_iter}: kl = {l:.6e}");
            }

            if (prev_loss - l) / l.abs().max(f64::MIN_POSITIVE) < opts.tol {
                break;
            }
            prev_loss = l;
        }
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
    data: ArrayView1<f32>,
    indices: ArrayView1<u32>,
    indptr: ArrayView1<u32>,
    w: ArrayView2<f32>,  // [m, k]
    ht: ArrayView2<f32>, // [n, k]
) -> f64 {
    let nz_part: f64 = (0..w.nrows())
        .into_par_iter()
        .map(|i| {
            let idx_from = indptr[i] as usize;
            let idx_to = indptr[i + 1] as usize;
            let w_i = w.row(i);
            let mut acc = 0_f64;
            for (&j, &x_ij) in indices
                .slice(s![idx_from..idx_to])
                .iter()
                .zip(data.slice(s![idx_from..idx_to]))
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

    let w_col_sum = w.mapv(|v| v as f64).sum_axis(Axis(0));
    let h_col_sum = ht.mapv(|v| v as f64).sum_axis(Axis(0));

    nz_part + w_col_sum.dot(&h_col_sum)
}

struct FusedMUThreadLocal {
    // re-used on each row/thread to compute updates factors
    ρw_i: Array1<f32>,

    // accumulated across rows/threads to compute update factors for all of H
    ρht: Array2<f32>,
}

struct FusedMUWork {
    slots: Vec<Mutex<FusedMUThreadLocal>>,
}

impl FusedMUWork {
    fn new(nthreads: usize, n: usize, k: usize) -> Self {
        Self {
            slots: (0..nthreads)
                .map(|_| {
                    Mutex::new(FusedMUThreadLocal {
                        ρw_i: Array1::zeros(k),
                        ρht: Array2::zeros((n, k)),
                    })
                })
                .collect(),
        }
    }
}

// Update both W and U in the same loop using a multiplicative update.
fn fused_mu(
    data: ArrayView1<f32>,
    indices: ArrayView1<u32>,
    indptr: ArrayView1<u32>,
    w: &mut Array2<f32>,  // [m, k]
    ht: &mut Array2<f32>, // [n, k]
    work: &mut FusedMUWork,
) {
    const EPS: f32 = 1e-6;

    // clear accumulators
    work.slots.iter().for_each(|slot| {
        slot.lock().unwrap().ρht.fill(0_f32);
    });

    let h_col_sum = ht.sum_axis(Axis(0));

    // for each i
    Zip::indexed(indptr.slice(s![..-1]))
        .and(indptr.slice(s![1..]))
        .and(w.rows_mut())
        .par_for_each(|_i, &idx_from, &idx_to, mut w_i| {
            let thread_id = rayon::current_thread_index().unwrap();
            let mut tl = work.slots[thread_id].lock().unwrap();

            tl.ρw_i.fill(0_f32);

            let idx_from = idx_from as usize;
            let idx_to = idx_to as usize;

            let data_row = data.slice(s![idx_from..idx_to]);
            let indices_row = indices.slice(s![idx_from..idx_to]);

            // for each j
            Zip::from(indices_row).and(data_row).for_each(|&j, &x_ij| {
                let j = j as usize;
                let u_ij = w_i.dot(&ht.row(j));

                // for each k
                Zip::from(&mut tl.ρw_i)
                    .and(ht.row(j))
                    .for_each(|ρw_ik, h_jk| {
                        *ρw_ik += h_jk * x_ij / u_ij;
                    });
            });

            // multiplicative update of w_i
            // for each k
            Zip::from(&mut w_i).and(&tl.ρw_i).and(&h_col_sum).for_each(
                |w_ik, ρw_ik, h_col_sum_k| {
                    *w_ik = (*w_ik * ρw_ik / h_col_sum_k).max(EPS);
                },
            );

            // for each j (second pass to accumulate updates to ρh)
            Zip::from(indices_row).and(data_row).for_each(|&j, &x_ij| {
                let j = j as usize;
                let u_ij = w_i.dot(&ht.row(j));

                // for each k
                Zip::from(tl.ρht.row_mut(j))
                    .and(&w_i)
                    .for_each(|ρh_jk, &w_ik| {
                        *ρh_jk += w_ik * x_ij / u_ij;
                    });
            });
        });

    let w_col_sum = w.sum_axis(Axis(0));

    // accumulate everything into the first thread's ρht matrix
    let mut slot0 = work.slots.first().unwrap().lock().unwrap();
    let ρht = &mut slot0.ρht;

    for slot in &work.slots[1..] {
        ρht.add_assign(&slot.lock().unwrap().ρht);
    }

    // for each j (multiplicative update of h)
    Zip::from(ht.rows_mut())
        .and(ρht.rows())
        .for_each(|ht_j, ρht_j| {
            // for each k
            Zip::from(ht_j)
                .and(ρht_j)
                .and(&w_col_sum)
                .for_each(|h_kj, ρh_kj, w_col_sum_k| {
                    *h_kj = (*h_kj * ρh_kj / w_col_sum_k).max(EPS);
                });
        })
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
        let result = nmf(data.view(), indices.view(), indptr.view(), w, ht, &opts);

        assert_eq!(result.n_iter, opts.max_iter);
        assert_eq!(result.loss.len(), opts.max_iter);
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
}
