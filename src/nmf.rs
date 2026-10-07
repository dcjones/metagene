use crate::kernels::{Isa, axpy, axpy_n, dispatch, dot, dot_n, prefetch};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip, s};
use rayon::prelude::*;
use std::ops::Range;
use std::time::{Duration, Instant};

// Maximum number of rows (or columns) per rayon task. Per-row work varies a lot (cells differ in
// depth, genes in detection rate) and inputs are often ordered in ways that cluster heavy rows, so
// without this rayon's default ~1 chunk per thread can leave most threads idle.
const PAR_GRAIN: usize = 4;

// Target size of the slice of W touched by one cell block in the H pass. Every thread works on the
// same block at once, so this should fit comfortably in (shared) L3. Tuned on a Ryzen 9 5950X
// (2 × 32MB L3); 4–16MB all perform similarly.
const H_PASS_BLOCK_BYTES: usize = 8 << 20;

// How many nonzeros ahead to prefetch the gathered factor row. Helps mainly when SMT isn't already
// hiding the latency.
const PREFETCH_DISTANCE: usize = 2;

// Nonzeros processed together in the inner loops, so their dependency chains (gather, dot product,
// division, axpy) overlap, the shared row is loaded once, and the accumulator row is updated once.
const NNZ_BLOCK: usize = 2;

pub struct NMFOptions {
    // maximum number of iterations
    pub max_iter: usize,

    // stop when the relative decrease in the objective falls below this
    pub tol: f64,

    // evaluate the objective every this many iterations (0 to never evaluate)
    pub eval_every: usize,

    // stop after this many seconds (excluding time spent evaluating the objective separately)
    pub max_time: Option<f64>,

    // Use BMMe: extrapolate each factor (Nesterov-style, positive part only) before its MU step.
    pub extrapolate: bool,

    // With `extrapolate`, reset the extrapolation sequence whenever an evaluated objective is
    // higher than the previous evaluation.
    pub restart: bool,

    // Update H. If false, H is held fixed and only W is fit.
    pub fit_h: bool,

    pub verbose: bool,
}

impl Default for NMFOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-4,
            eval_every: 10,
            max_time: None,
            extrapolate: true,
            restart: false,
            fit_h: true,
            verbose: false,
        }
    }
}

pub struct NMFResult {
    pub w: Array2<f32>,  // [m, k]
    pub ht: Array2<f32>, // [n, k]

    // (iteration, elapsed seconds, objective)
    pub loss: Vec<(usize, f64, f64)>,

    pub n_iter: usize,
}

// A CSR matrix borrowing its arrays (e.g. from numpy).
pub struct CSR<'a> {
    pub data: ArrayView1<'a, f32>,
    pub indices: ArrayView1<'a, u32>, // column (gene) indices
    pub indptr: ArrayView1<'a, u32>,
}

impl CSR<'_> {
    fn slices(&self) -> (&[f32], &[u32], &[u32]) {
        (
            self.data.as_slice().expect("data must be contiguous"),
            self.indices.as_slice().expect("indices must be contiguous"),
            self.indptr.as_slice().expect("indptr must be contiguous"),
        )
    }
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

    let isa = Isa::detect();
    let start = Instant::now();
    // time spent evaluating the objective outside of the update passes, excluded from timings
    let mut eval_time = Duration::ZERO;
    let elapsed = |eval_time: Duration| (start.elapsed() - eval_time).as_secs_f64();

    let block_rows = (H_PASS_BLOCK_BYTES / (k * size_of::<f32>())).max(1);
    // only needed to update H
    let csc = opts.fit_h.then(|| BlockedCSC::from_csr(x, n, block_rows));
    let mut ρht = Array2::<f32>::zeros((if opts.fit_h { n } else { 0 }, k));

    // previous iterates, for extrapolation
    let mut w_prev = opts.extrapolate.then(|| w.clone());
    let mut ht_prev = (opts.extrapolate && opts.fit_h).then(|| ht.clone());
    let mut t_nesterov = 1_f64;

    let mut loss = Vec::new();
    let mut prev_loss = f64::INFINITY;
    let mut best_loss = f64::INFINITY;
    let mut n_iter = 0;

    let mut record = |iter: usize, t: f64, l: f64| {
        loss.push((iter, t, l));
        if opts.verbose {
            eprintln!("iter {iter} ({t:.2}s): kl = {l:.6e}");
        }
    };

    for iter in 0..opts.max_iter {
        let t_iter = elapsed(eval_time);

        let β = if opts.extrapolate {
            let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_nesterov * t_nesterov).sqrt());
            let β = (t_nesterov - 1.0) / t_next;
            t_nesterov = t_next;
            β as f32
        } else {
            0.0
        };

        // The objective of the state before the step. This can be computed nearly for free in the
        // W pass, unless W is extrapolated first.
        let eval = opts.eval_every > 0 && iter % opts.eval_every == 0;
        let fused_eval = eval && β == 0.0;
        let mut l = None;
        if eval && !fused_eval {
            let t0 = Instant::now();
            l = Some(kl_divergence(x, w.view(), ht.view(), isa));
            eval_time += t0.elapsed();
        }

        if let Some(w_prev) = &mut w_prev {
            extrapolate(&mut w, w_prev, β);
        }
        let l_fused = update_w(x, &mut w, &ht, fused_eval, isa);
        if opts.fit_h {
            if let Some(ht_prev) = &mut ht_prev {
                extrapolate(&mut ht, ht_prev, β);
            }
            update_h(csc.as_ref().unwrap(), &mut ρht, &w, &mut ht, isa);
        }
        n_iter = iter + 1;

        if let Some(l) = l.or(l_fused) {
            record(iter, t_iter, l);

            if opts.restart && l > prev_loss {
                t_nesterov = 1.0;
            }
            prev_loss = l;

            // With extrapolation the objective isn't monotone, so an increase doesn't count as
            // having converged.
            let improvement = (best_loss - l) / l.abs().max(f64::MIN_POSITIVE);
            if (0.0..opts.tol).contains(&improvement) {
                break;
            }
            best_loss = best_loss.min(l);
        }

        if opts
            .max_time
            .is_some_and(|max_time| elapsed(eval_time) >= max_time)
        {
            break;
        }
    }

    if opts.eval_every > 0 {
        let t = elapsed(eval_time);
        let t0 = Instant::now();
        let l = kl_divergence(x, w.view(), ht.view(), isa);
        eval_time += t0.elapsed();
        record(n_iter, t, l);
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
fn kl_divergence(x: &CSR, w: ArrayView2<f32>, ht: ArrayView2<f32>, isa: Isa) -> f64 {
    let k = w.ncols();
    let (data, indices, indptr) = x.slices();
    let w_s = w.as_slice().expect("w must be contiguous");
    let ht_s = ht.as_slice().expect("ht must be contiguous");

    let nz_part: f64 = (0..w.nrows())
        .into_par_iter()
        .with_max_len(PAR_GRAIN)
        .map(|i| {
            let range = indptr[i] as usize..indptr[i + 1] as usize;
            let w_i = &w_s[i * k..(i + 1) * k];
            dispatch!(isa, kl_row(w_i, ht_s, k, data, indices, range))
        })
        .sum();

    nz_part + col_sum_f64(w).dot(&col_sum_f64(ht))
}

// Σ_j x_ij log(x_ij / u_ij) - x_ij over the nonzeros of row i.
#[inline(always)]
fn kl_row<const AVX2: bool>(
    w_i: &[f32],
    ht: &[f32],
    k: usize,
    data: &[f32],
    indices: &[u32],
    range: Range<usize>,
) -> f64 {
    let mut acc = 0_f64;
    for p in range {
        let x_ij = data[p];
        if x_ij > 0.0 {
            let j = indices[p] as usize;
            let x_ij = x_ij as f64;
            let u_ij = dot::<AVX2>(w_i, &ht[j * k..(j + 1) * k]) as f64;
            acc += x_ij * (x_ij / u_ij).ln() - x_ij;
        }
    }
    acc
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

const EPS: f32 = 1e-6;

// Add row j's contributions from one block of cells, whose nonzeros are at `range`, to ρh_j.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // indices address several parallel arrays
fn accumulate_h_row<const AVX2: bool>(
    ρh_j: &mut [f32],
    h_j: &[f32],
    w: &[f32],
    data: &[f32],
    indices: &[u32],
    range: Range<usize>,
) {
    let k = h_j.len();
    let row = |q: usize| {
        let i = indices[q] as usize;
        &w[i * k..(i + 1) * k]
    };
    let indices = &indices[..range.end];

    // for each i in the block, NNZ_BLOCK at a time
    let mut q = range.start;
    while q + NNZ_BLOCK <= range.end {
        for d in 0..NNZ_BLOCK {
            if let Some(&i) = indices.get(q + PREFETCH_DISTANCE * NNZ_BLOCK + d) {
                let i = i as usize;
                prefetch(&w[i * k..(i + 1) * k]);
            }
        }
        let w_is: [&[f32]; NNZ_BLOCK] = std::array::from_fn(|d| row(q + d));
        let u = dot_n::<AVX2, NNZ_BLOCK>(h_j, w_is);
        let r: [f32; NNZ_BLOCK] = std::array::from_fn(|d| data[q + d] / u[d]);
        axpy_n::<AVX2, NNZ_BLOCK>(r, w_is, ρh_j);
        q += NNZ_BLOCK;
    }
    for q in q..range.end {
        let w_i = row(q);
        axpy::<AVX2>(data[q] / dot::<AVX2>(w_i, h_j), w_i, ρh_j);
    }
}

// BMMe extrapolation: a += β max(a - a_prev, 0), after setting a_prev to the current a.
fn extrapolate(a: &mut Array2<f32>, a_prev: &mut Array2<f32>, β: f32) {
    Zip::from(a).and(a_prev).par_for_each(|a, a_prev| {
        let step = (*a - *a_prev).max(0.0);
        *a_prev = *a;
        *a += β * step;
    });
}

// One multiplicative update of W followed by one of H using the updated W.
#[cfg(test)]
fn mu_step(
    x: &CSR,
    csc: &BlockedCSC,
    ρht: &mut Array2<f32>,
    w: &mut Array2<f32>,
    ht: &mut Array2<f32>,
    compute_loss: bool,
) -> Option<f64> {
    let isa = Isa::detect();
    let l = update_w(x, w, ht, compute_loss, isa);
    update_h(csc, ρht, w, ht, isa);
    l
}

// Multiplicative update of W, row-parallel over CSR. If `compute_loss`, also returns the KL
// divergence of the state prior to the update, which is nearly free since this pass already
// computes u_ij at each nonzero.
fn update_w(
    x: &CSR,
    w: &mut Array2<f32>, // [m, k]
    ht: &Array2<f32>,    // [n, k]
    compute_loss: bool,
    isa: Isa,
) -> Option<f64> {
    let k = w.ncols();

    // Σ_ij u_ij term of the objective, from the state prior to the update
    let u_sum = compute_loss.then(|| col_sum_f64(w.view()).dot(&col_sum_f64(ht.view())));

    let h_col_sum = ht.sum_axis(Axis(0));
    let h_col_sum = h_col_sum.as_slice().unwrap();
    let ht = ht.as_slice().expect("ht must be contiguous");
    let (data, indices, indptr) = x.slices();

    let nz_loss: f64 = w
        .as_slice_mut()
        .expect("w must be contiguous")
        .par_chunks_mut(k)
        .with_max_len(PAR_GRAIN)
        .enumerate()
        .map_init(
            || vec![0_f32; k],
            |ρw_i, (i, w_i)| {
                let range = indptr[i] as usize..indptr[i + 1] as usize;
                dispatch!(
                    isa,
                    update_w_row(w_i, ρw_i, ht, h_col_sum, data, indices, range, compute_loss)
                )
            },
        )
        .sum();

    u_sum.map(|u_sum| nz_loss + u_sum)
}

// Multiplicative update of row i of W, whose nonzeros in X are at `range`. Returns the row's
// contribution to the objective's nonzero terms if `compute_loss`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // indices address several parallel arrays
fn update_w_row<const AVX2: bool>(
    w_i: &mut [f32],
    ρw_i: &mut [f32],
    ht: &[f32],
    h_col_sum: &[f32],
    data: &[f32],
    indices: &[u32],
    range: Range<usize>,
    compute_loss: bool,
) -> f64 {
    let k = w_i.len();
    ρw_i.fill(0_f32);
    let mut nz_loss_i = 0_f64;

    let row = |p: usize| {
        let j = indices[p] as usize;
        &ht[j * k..(j + 1) * k]
    };
    let loss = |x_ij: f32, u_ij: f32| {
        if compute_loss && x_ij > 0.0 {
            let x_ij = x_ij as f64;
            x_ij * (x_ij / u_ij as f64).ln() - x_ij
        } else {
            0.0
        }
    };
    let indices = &indices[..range.end];

    // for each j, NNZ_BLOCK at a time
    let mut p = range.start;
    while p + NNZ_BLOCK <= range.end {
        for d in 0..NNZ_BLOCK {
            if let Some(&j) = indices.get(p + PREFETCH_DISTANCE * NNZ_BLOCK + d) {
                let j = j as usize;
                prefetch(&ht[j * k..(j + 1) * k]);
            }
        }
        let h_js: [&[f32]; NNZ_BLOCK] = std::array::from_fn(|d| row(p + d));
        let u = dot_n::<AVX2, NNZ_BLOCK>(w_i, h_js);
        let r: [f32; NNZ_BLOCK] = std::array::from_fn(|d| data[p + d] / u[d]);
        for d in 0..NNZ_BLOCK {
            nz_loss_i += loss(data[p + d], u[d]);
        }
        axpy_n::<AVX2, NNZ_BLOCK>(r, h_js, ρw_i);
        p += NNZ_BLOCK;
    }
    for p in p..range.end {
        let h_j = row(p);
        let u_ij = dot::<AVX2>(w_i, h_j);
        nz_loss_i += loss(data[p], u_ij);
        axpy::<AVX2>(data[p] / u_ij, h_j, ρw_i);
    }

    // for each k
    for ((w_ik, ρw_ik), h_col_sum_k) in w_i.iter_mut().zip(&*ρw_i).zip(h_col_sum) {
        *w_ik = (*w_ik * ρw_ik / h_col_sum_k).max(EPS);
    }

    nz_loss_i
}

// Multiplicative update of H, column-parallel over CSC, one block of cells at a time.
fn update_h(
    csc: &BlockedCSC,
    ρht: &mut Array2<f32>, // [n, k] accumulator, all zeros on entry and exit
    w: &Array2<f32>,       // [m, k]
    ht: &mut Array2<f32>,  // [n, k]
    isa: Isa,
) {
    let k = w.ncols();
    let w_col_sum = w.sum_axis(Axis(0));
    let w_col_sum = w_col_sum.as_slice().unwrap();
    let w = w.as_slice().expect("w must be contiguous");
    let ht = ht.as_slice_mut().expect("ht must be contiguous");
    let ρht = ρht.as_slice_mut().expect("ρht must be contiguous");

    for block in &csc.blocks {
        let ht = &*ht;
        ρht.par_chunks_mut(k)
            .with_max_len(PAR_GRAIN)
            .enumerate()
            .for_each(|(j, ρh_j)| {
                let h_j = &ht[j * k..(j + 1) * k];
                let range = block.indptr[j]..block.indptr[j + 1];
                dispatch!(
                    isa,
                    accumulate_h_row(ρh_j, h_j, w, &block.data, &block.indices, range)
                );
            });
    }

    ht.par_chunks_mut(k)
        .zip(ρht.par_chunks_mut(k))
        .with_max_len(PAR_GRAIN)
        .for_each(|(h_j, ρh_j)| {
            // for each k
            for ((h_jk, ρh_jk), w_col_sum_k) in h_j.iter_mut().zip(&*ρh_j).zip(w_col_sum) {
                *h_jk = (*h_jk * ρh_jk / w_col_sum_k).max(EPS);
            }
            ρh_j.fill(0_f32);
        });
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
            max_time: None,
            extrapolate: false,
            restart: false,
            fit_h: true,
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
            let (prev, next) = (pair[0].2, pair[1].2);
            assert!(next.is_finite());
            assert!(
                next <= prev * (1.0 + 1e-6),
                "objective increased: {prev} -> {next}"
            );
        }
        let (first, last) = (result.loss[0].2, result.loss.last().unwrap().2);
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
