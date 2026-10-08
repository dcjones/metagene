use crate::kernels::{Isa, axpy, axpy_n, dispatch, dot, dot_n, prefetch};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use rayon::prelude::*;
use std::ops::Range;
use std::time::{Duration, Instant};

// Maximum number of rows (or columns) per rayon task. Per-row work varies a lot (cells differ in
// depth, genes in detection rate) and inputs are often ordered in ways that cluster heavy rows, so
// without this rayon's default ~1 chunk per thread can leave most threads idle.
pub(crate) const PAR_GRAIN: usize = 4;

// Target size of the slice of W touched by one cell block in the H pass. Every thread works on the
// same block at once, so this should fit comfortably in (shared) L3. Tuned on a Ryzen 9 5950X
// (2 × 32MB L3); 4–16MB all perform similarly.
pub(crate) const H_PASS_BLOCK_BYTES: usize = 8 << 20;

// How many nonzeros ahead to prefetch the gathered factor row. Helps mainly when SMT isn't already
// hiding the latency.
pub(crate) const PREFETCH_DISTANCE: usize = 2;

// Nonzeros processed together in the inner loops, so their dependency chains (gather, dot product,
// division, axpy) overlap, the shared row is loaded once, and the accumulator row is updated once.
pub(crate) const NNZ_BLOCK: usize = 2;

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

    // Floor applied to factor values after each multiplicative update.
    pub eps: f32,

    // Optional Gamma(a_j + 1, b_j) prior on each entry h_jk of H, as per-gene (a [n], b [n]): the
    // objective gains Σ_jk b_j h_jk - a_j ln h_jk, and a_j acts as a pseudocount in H's update.
    pub h_prior: Option<(Array1<f32>, Array1<f32>)>,

    // Train with minibatches of cells (stochastic MM) instead of full passes. Requires
    // `extrapolate = false` and `fit_h`.
    pub minibatch: Option<Minibatch>,

    pub verbose: bool,
}

// Minibatch training (stochastic majorization-minimization; Mairal 2013, NeurIPS 26).
// Each iteration is one epoch: the minibatches are visited in a random order, and for each, its rows
// of W get `w_steps` multiplicative updates, then H takes a step on a running average of its
// majorizer. H's majorizer at H̃ is Σ_jk b_j h_jk - (h̃_jk ρh_jk + a_j) ln h_jk + h_jk Σ_i w_ik, so
// the average only needs A [n, k] (of h̃_jk ρh_jk) and B [k] (of Σ_i w_ik), each scaled by m / |batch|
// to estimate the full-data sum. With weight λ_t = max(λ, 1 / t) on step t's batch (t counting
// from where λ last changed in its schedule),
//
//     A ← (1 - λ_t) A + λ_t m/|b| h̃ ρh_b,   B ← (1 - λ_t) B + λ_t m/|b| Σ_{i∈b} w_i,
//     h_jk ← (A_jk + a_j) / (B_k + b_j).
//
// A single batch with λ = 1 is the full-batch update. Smaller λ averages over more batches. λ = 1
// (each H step uses only the newest batch) is the noisiest, and works well early on; dropping it
// later lets H average over more cells.
pub struct Minibatch {
    // Order of the cells; consecutive runs of `batch_size` form the minibatches.
    pub perm: Vec<u32>,
    pub batch_size: usize,
    // λ, the minimum weight of the newest batch in H's running average, as a schedule of
    // (first epoch, λ), sorted by epoch and starting at epoch 0
    pub step: Vec<(usize, f32)>,
    // multiplicative updates of a batch's rows of W per visit
    pub w_steps: usize,
    // full passes updating only W after the last epoch (if any), so every row of W is fit to the
    // final H
    pub final_w_passes: usize,
    // seeds the order the batches are visited in each epoch
    pub seed: u64,
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
            eps: EPS,
            h_prior: None,
            minibatch: None,
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
    pub(crate) fn slices(&self) -> (&[f32], &[u32], &[u32]) {
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

    let mut minibatch = opts.minibatch.as_ref().map(|mb| {
        assert!(opts.fit_h && !opts.extrapolate);
        assert_eq!(mb.perm.len(), m);
        assert!(mb.step.first().is_some_and(|&(e, _)| e == 0));
        assert!(mb.step.is_sorted_by_key(|&(e, _)| e));
        (MinibatchState::new(x, n, k, mb), mb)
    });
    let full_h = opts.fit_h && minibatch.is_none();

    let block_rows = (H_PASS_BLOCK_BYTES / (k * size_of::<f32>())).max(1);
    // only needed to update H in full passes
    let csc = full_h.then(|| BlockedCSC::from_csr(x, n, block_rows));
    let mut ρht = Array2::<f32>::zeros((if full_h { n } else { 0 }, k));

    // previous iterates, for extrapolation
    let mut w_prev = opts.extrapolate.then(|| w.clone());
    let mut ht_prev = (opts.extrapolate && opts.fit_h).then(|| ht.clone());
    let mut t_nesterov = 1_f64;

    let mut loss = Vec::new();
    let mut prev_loss = f64::INFINITY;
    let mut best_loss = f64::INFINITY;
    let mut n_iter = 0;

    if let Some((a, b)) = &opts.h_prior {
        assert_eq!(a.len(), n);
        assert_eq!(b.len(), n);
    }
    let penalty = |ht: &Array2<f32>| match &opts.h_prior {
        Some((a, b)) if opts.fit_h => h_penalty(ht.view(), a.view(), b.view()),
        _ => 0.0,
    };

    let mut record = |iter: usize, t: f64, l: f64| {
        loss.push((iter, t, l));
        if opts.verbose {
            eprintln!("iter {iter} ({t:.2}s): objective = {l:.6e}");
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
        let fused_eval = eval && β == 0.0 && minibatch.is_none();
        let mut l = None;
        if eval {
            // H isn't changed until after the W pass, so its penalty can be taken here.
            let t0 = Instant::now();
            let mut l_eval = penalty(&ht);
            if !fused_eval {
                l_eval += kl_divergence(x, w.view(), ht.view(), isa);
            }
            l = Some(l_eval);
            eval_time += t0.elapsed();
        }

        let mut l_fused = None;
        if let Some((state, mb)) = &mut minibatch {
            state.epoch(iter, x, &mut w, &mut ht, mb, opts, isa);
        } else {
            if let Some(w_prev) = &mut w_prev {
                extrapolate(&mut w, w_prev, β);
            }
            l_fused = update_w(x, None, &mut w, &ht, fused_eval, opts.eps, isa);
        }
        if full_h {
            if let Some(ht_prev) = &mut ht_prev {
                extrapolate(&mut ht, ht_prev, β);
            }
            update_h(csc.as_ref().unwrap(), &mut ρht, &w, &mut ht, opts, isa);
        }
        n_iter = iter + 1;

        if let Some(l) = l.map(|l| l + l_fused.unwrap_or(0.0)) {
            record(iter, t_iter, l);

            if opts.restart && l > prev_loss {
                t_nesterov = 1.0;
            }
            prev_loss = l;

            // With extrapolation the objective isn't monotone, so an increase doesn't count as
            // having converged. Minibatch training only counts from the last change of λ, before
            // which the objective follows the schedule.
            let final_phase = minibatch
                .as_ref()
                .is_none_or(|(_, mb)| iter >= mb.step.last().unwrap().0);
            if final_phase {
                let improvement = (best_loss - l) / l.abs().max(f64::MIN_POSITIVE);
                if (0.0..opts.tol).contains(&improvement) {
                    break;
                }
                best_loss = best_loss.min(l);
            }
        }

        if opts
            .max_time
            .is_some_and(|max_time| elapsed(eval_time) >= max_time)
        {
            break;
        }
    }

    if let Some((_, mb)) = minibatch.as_ref().filter(|_| n_iter > 0) {
        for _ in 0..mb.final_w_passes {
            update_w(x, None, &mut w, &ht, false, opts.eps, isa);
        }
    }

    if opts.eval_every > 0 {
        let t = elapsed(eval_time);
        let t0 = Instant::now();
        let l = kl_divergence(x, w.view(), ht.view(), isa) + penalty(&ht);
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

// Transposed (CSC) copy of some rows of X: CSR row `i` is stored under row index `r`, for each
// (r, i) in `rows`.
pub(crate) struct CSC {
    pub(crate) data: Vec<f32>,
    pub(crate) indices: Vec<u32>, // row (cell) indices
    pub(crate) indptr: Vec<usize>,
}

impl CSC {
    fn from_csr(x: &CSR, n: usize, rows: impl Iterator<Item = (u32, usize)> + Clone) -> Self {
        let range = |i: usize| x.indptr[i] as usize..x.indptr[i + 1] as usize;

        let mut csc_indptr = vec![0_usize; n + 1];
        for (_, i) in rows.clone() {
            for p in range(i) {
                csc_indptr[x.indices[p] as usize + 1] += 1;
            }
        }
        for j in 0..n {
            csc_indptr[j + 1] += csc_indptr[j];
        }
        let nnz = csc_indptr[n];

        let mut next = csc_indptr.clone();
        let mut csc_data = vec![0_f32; nnz];
        let mut csc_indices = vec![0_u32; nnz];
        for (r, i) in rows {
            for p in range(i) {
                let j = x.indices[p] as usize;
                csc_data[next[j]] = x.data[p];
                csc_indices[next[j]] = r;
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
pub(crate) struct BlockedCSC {
    pub(crate) blocks: Vec<CSC>,
}

impl BlockedCSC {
    // All of X, with global row indices.
    pub(crate) fn from_csr(x: &CSR, n: usize, block_rows: usize) -> Self {
        let m = x.indptr.len() - 1;
        let blocks = (0..m.div_ceil(block_rows).max(1))
            .into_par_iter()
            .map(|b| {
                let rows = b * block_rows..((b + 1) * block_rows).min(m);
                CSC::from_csr(x, n, rows.map(|i| (i as u32, i)))
            })
            .collect();
        Self { blocks }
    }

    // The CSR rows `rows` of X, with row indices into `rows`.
    fn from_csr_rows(x: &CSR, n: usize, rows: &[u32], block_rows: usize) -> Self {
        let blocks = rows
            .chunks(block_rows)
            .enumerate()
            .map(|(b, chunk)| {
                let offset = b * block_rows;
                let rows = chunk.iter().enumerate();
                CSC::from_csr(x, n, rows.map(|(r, &i)| ((offset + r) as u32, i as usize)))
            })
            .collect();
        Self { blocks }
    }
}

// Default floor on factor values (NMFOptions::eps).
const EPS: f32 = 1e-6;

// Negative log density of H's Gamma prior, up to a constant: Σ_jk b_j h_jk - a_j ln h_jk.
fn h_penalty(ht: ArrayView2<f32>, a: ArrayView1<f32>, b: ArrayView1<f32>) -> f64 {
    let k = ht.ncols();
    ht.as_slice()
        .expect("ht must be contiguous")
        .par_chunks(k)
        .zip(a.as_slice().unwrap())
        .zip(b.as_slice().unwrap())
        .with_min_len(64)
        .map(|((h_j, &a_j), &b_j)| {
            let (a_j, b_j) = (a_j as f64, b_j as f64);
            h_j.iter()
                .map(|&h| b_j * h as f64 - a_j * (h as f64).ln())
                .sum::<f64>()
        })
        .sum()
}

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
    let opts = NMFOptions::default();
    let l = update_w(x, None, w, ht, compute_loss, opts.eps, isa);
    update_h(csc, ρht, w, ht, &opts, isa);
    l
}

// Multiplicative update of W, row-parallel over CSR. Row i of `w` is CSR row `rows[i]` if given,
// otherwise row i. If `compute_loss`, also returns the KL divergence of the state prior to the
// update, which is nearly free since this pass already computes u_ij at each nonzero.
fn update_w(
    x: &CSR,
    rows: Option<&[u32]>,
    w: &mut Array2<f32>, // [m, k], or [rows.len(), k]
    ht: &Array2<f32>,    // [n, k]
    compute_loss: bool,
    eps: f32,
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
                let i = rows.map_or(i, |rows| rows[i] as usize);
                let range = indptr[i] as usize..indptr[i + 1] as usize;
                dispatch!(
                    isa,
                    update_w_row(
                        w_i,
                        ρw_i,
                        ht,
                        h_col_sum,
                        data,
                        indices,
                        range,
                        compute_loss,
                        eps
                    )
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
    eps: f32,
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
        *w_ik = (*w_ik * ρw_ik / h_col_sum_k).max(eps);
    }

    nz_loss_i
}

// Multiplicative update of H, column-parallel over CSC, one block of cells at a time. With a
// Gamma(a_j + 1, b_j) prior (opts.h_prior) the majorizer's minimizer is
// h_jk ← (h_jk ρh_jk + a_j) / (Σ_i w_ik + b_j).
fn update_h(
    csc: &BlockedCSC,
    ρht: &mut Array2<f32>, // [n, k] accumulator, all zeros on entry and exit
    w: &Array2<f32>,       // [m, k]
    ht: &mut Array2<f32>,  // [n, k]
    opts: &NMFOptions,
    isa: Isa,
) {
    let eps = opts.eps;
    let prior = h_prior_slices(opts);
    let k = w.ncols();
    let w_col_sum = w.sum_axis(Axis(0));
    let w_col_sum = w_col_sum.as_slice().unwrap();
    accumulate_h(csc, ρht, w, ht, isa);

    let ht = ht.as_slice_mut().expect("ht must be contiguous");
    let ρht = ρht.as_slice_mut().expect("ρht must be contiguous");
    ht.par_chunks_mut(k)
        .zip(ρht.par_chunks_mut(k))
        .with_max_len(PAR_GRAIN)
        .enumerate()
        .for_each(|(j, (h_j, ρh_j))| {
            let (a_j, b_j) = prior.map_or((0.0, 0.0), |(a, b)| (a[j], b[j]));
            // for each k
            for ((h_jk, ρh_jk), w_col_sum_k) in h_j.iter_mut().zip(&*ρh_j).zip(w_col_sum) {
                *h_jk = ((*h_jk * ρh_jk + a_j) / (w_col_sum_k + b_j)).max(eps);
            }
            ρh_j.fill(0_f32);
        });
}

fn h_prior_slices(opts: &NMFOptions) -> Option<(&[f32], &[f32])> {
    opts.h_prior
        .as_ref()
        .map(|(a, b)| (a.as_slice().unwrap(), b.as_slice().unwrap()))
}

// ρh_jk += Σ_i w_ik x_ij / (W Hᵀ)_ij over the nonzeros in `csc`, whose row indices index `w`.
fn accumulate_h(
    csc: &BlockedCSC,
    ρht: &mut Array2<f32>, // [n, k]
    w: &Array2<f32>,       // [m, k]
    ht: &Array2<f32>,      // [n, k]
    isa: Isa,
) {
    let k = w.ncols();
    let w = w.as_slice().expect("w must be contiguous");
    let ht = ht.as_slice().expect("ht must be contiguous");
    let ρht = ρht.as_slice_mut().expect("ρht must be contiguous");

    for block in &csc.blocks {
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
}

// State of minibatch training (see `Minibatch`).
struct MinibatchState {
    batches: Vec<(Range<usize>, BlockedCSC)>, // ranges of `perm`, and those rows' CSC copies
    ρht: Array2<f32>,                         // [n, k] accumulator, all zeros between steps
    a_ht: Array2<f32>,                        // [n, k] running average A
    b: Array1<f32>,                           // [k] running average B
    t: usize,                                 // steps taken since λ last changed
    λ: f32,                                   // the current λ
    rng: u64,
}

impl MinibatchState {
    fn new(x: &CSR, n: usize, k: usize, mb: &Minibatch) -> Self {
        let m = mb.perm.len();
        let batch_size = mb.batch_size.clamp(1, m.max(1));
        let block_rows = (H_PASS_BLOCK_BYTES / (k * size_of::<f32>())).max(1);
        let batches = (0..m.div_ceil(batch_size))
            .into_par_iter()
            .map(|b| {
                let range = b * batch_size..((b + 1) * batch_size).min(m);
                let csc = BlockedCSC::from_csr_rows(x, n, &mb.perm[range.clone()], block_rows);
                (range, csc)
            })
            .collect();
        Self {
            batches,
            ρht: Array2::zeros((n, k)),
            a_ht: Array2::zeros((n, k)),
            b: Array1::zeros(k),
            t: 0,
            λ: f32::NAN,
            rng: mb.seed,
        }
    }

    // One pass over all minibatches, in a random order.
    #[allow(clippy::too_many_arguments)]
    fn epoch(
        &mut self,
        epoch: usize,
        x: &CSR,
        w: &mut Array2<f32>,  // [m, k]
        ht: &mut Array2<f32>, // [n, k]
        mb: &Minibatch,
        opts: &NMFOptions,
        isa: Isa,
    ) {
        let m = w.nrows();
        let λ_epoch = mb.step.iter().rfind(|&&(e, _)| e <= epoch).unwrap().1;
        if λ_epoch != self.λ {
            self.λ = λ_epoch;
            self.t = 0;
        }
        let mut order: Vec<usize> = (0..self.batches.len()).collect();
        // Fisher-Yates
        for i in (1..order.len()).rev() {
            order.swap(i, (splitmix64(&mut self.rng) % (i as u64 + 1)) as usize);
        }

        for b in order {
            let (range, csc) = &self.batches[b];
            let rows = &mb.perm[range.clone()];
            // the batch's rows of W, [|b|, k]
            let mut w_b = Array2::zeros((rows.len(), w.ncols()));
            // for each i in the batch
            for (mut w_bi, &i) in w_b.outer_iter_mut().zip(rows) {
                w_bi.assign(&w.row(i as usize));
            }
            for _ in 0..mb.w_steps {
                update_w(x, Some(rows), &mut w_b, ht, false, opts.eps, isa);
            }
            for (w_bi, &i) in w_b.outer_iter().zip(rows) {
                w.row_mut(i as usize).assign(&w_bi);
            }

            accumulate_h(csc, &mut self.ρht, &w_b, ht, isa);
            self.t += 1;
            let λ = self.λ.max(1.0 / self.t as f32);
            let scale = λ * m as f32 / rows.len() as f32;
            Zip::from(&mut self.b)
                .and(&w_b.sum_axis(Axis(0)))
                .for_each(|b_k, &s_k| *b_k = (1.0 - λ) * *b_k + scale * s_k);
            self.step_h(ht, λ, scale, opts);
        }
    }

    // A ← (1 - λ) A + scale h̃ ρh, then h ← (A + a) / (B + b).
    fn step_h(&mut self, ht: &mut Array2<f32>, λ: f32, scale: f32, opts: &NMFOptions) {
        let eps = opts.eps;
        let prior = h_prior_slices(opts);
        let k = ht.ncols();
        let b = self.b.as_slice().unwrap();
        ht.as_slice_mut()
            .unwrap()
            .par_chunks_mut(k)
            .zip(self.ρht.as_slice_mut().unwrap().par_chunks_mut(k))
            .zip(self.a_ht.as_slice_mut().unwrap().par_chunks_mut(k))
            .with_max_len(PAR_GRAIN)
            .enumerate()
            .for_each(|(j, ((h_j, ρh_j), a_hj))| {
                let (a_j, b_j) = prior.map_or((0.0, 0.0), |(a, b)| (a[j], b[j]));
                // for each k
                for (((h_jk, ρh_jk), a_hjk), b_k) in
                    h_j.iter_mut().zip(&*ρh_j).zip(a_hj.iter_mut()).zip(b)
                {
                    *a_hjk = (1.0 - λ) * *a_hjk + scale * *h_jk * ρh_jk;
                    *h_jk = ((*a_hjk + a_j) / (b_k + b_j)).max(eps);
                }
                ρh_j.fill(0_f32);
            });
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
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
            eps: EPS,
            h_prior: None,
            minibatch: None,
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

    #[test]
    fn minibatch_matches_full_batch() {
        let (m, n, k) = (200, 80, 5);
        let mut rng = Lcg(91011);
        let (data, indices, indptr) = random_csr(&mut rng, m, n);
        let x = CSR {
            data: data.view(),
            indices: indices.view(),
            indptr: indptr.view(),
        };
        let w0 = Array2::from_shape_simple_fn((m, k), || 0.1 + rng.next_f32());
        let ht0 = Array2::from_shape_simple_fn((n, k), || 0.1 + rng.next_f32());
        let a = Array1::from_shape_fn(n, |j| 0.5 + j as f32 / n as f32);
        let b = Array1::from_shape_fn(n, |j| 0.1 + 0.01 * j as f32);
        // a fixed non-identity permutation
        let shuffled: Vec<u32> = (0..m as u32).map(|i| (i * 37 + 11) % m as u32).collect();

        let run = |minibatch: Option<Minibatch>| {
            let opts = NMFOptions {
                max_iter: 10,
                tol: f64::NEG_INFINITY,
                eval_every: 1,
                extrapolate: false,
                h_prior: Some((a.clone(), b.clone())),
                minibatch,
                ..Default::default()
            };
            nmf(&x, w0.clone(), ht0.clone(), &opts)
        };
        let one_batch = |perm: Vec<u32>| Minibatch {
            perm,
            batch_size: m,
            step: vec![(0, 1.0)],
            w_steps: 1,
            final_w_passes: 0,
            seed: 0,
        };

        let full = run(None);
        let identity = run(Some(one_batch((0..m as u32).collect())));
        assert_eq!(full.w, identity.w);
        assert_eq!(full.ht, identity.ht);

        // the same, but with cells summed in a different order
        let permuted = run(Some(one_batch(shuffled.clone())));
        let close = |a: &Array2<f32>, b: &Array2<f32>| {
            Zip::from(a)
                .and(b)
                .all(|&a, &b| (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1e-3))
        };
        assert!(close(&full.w, &permuted.w));
        assert!(close(&full.ht, &permuted.ht));

        let batched = run(Some(Minibatch {
            perm: shuffled,
            batch_size: 30,
            step: vec![(0, 1.0), (4, 0.3)],
            w_steps: 2,
            final_w_passes: 3,
            seed: 1,
        }));
        let (first, last) = (batched.loss[0].2, batched.loss.last().unwrap().2);
        assert!(last.is_finite() && last < 0.9 * first, "{first} -> {last}");
        assert!(last < full.loss.last().unwrap().2 * 1.1);
    }
}
