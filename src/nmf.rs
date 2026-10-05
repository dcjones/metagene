use ndarray::{Array1, Array2, Axis, Zip, s};
use rayon;
use std::{
    ops::{AddAssign, MulAssign},
    sync::Mutex,
};

// An optimized KL-NMF implementation, using extrapolated multiplicative updates, and parallelized across
// rows (typically, cells).
//
//     Hien,L.T.K., Leplat,V. and Gillis,N. (2025) Block Majorization
//     Minimization with extrapolation and application to β-NMF. SIAM J.
//     Math. Data Sci., 7, 1292–1314.
//
pub fn nmf(
    data: &Array1<f32>,
    indices: &Array1<u32>,
    indptr: &Array1<u32>,
    k: usize,
) -> (Array2<f32>, Array2<f32>) {
    // TODO: There are gains to be had by re-ordering the cells/genes to improve cache locality.

    // TODO: Do a bunch of fused_mu iterations. Track loss.

    todo!();
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
    data: &Array1<f32>,
    indices: &Array1<u32>,
    indptr: &Array1<u32>,
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

            // TODO: I can just do the update here
            // for each k
            Zip::from(&mut tl.ρw_i)
                .and(&h_col_sum)
                .for_each(|ρw_ik, h_col_sum_k| {
                    *ρw_ik = (*ρw_ik / h_col_sum_k).max(EPS);
                });

            // multiplicative update of w_i
            w_i.mul_assign(&tl.ρw_i);

            // for each j (second pass to accumulate updates to ρh)
            Zip::from(indices_row)
                .and(data_row)
                .and(tl.ρht.rows_mut())
                .for_each(|&j, &x_ij, ρht_j| {
                    let j = j as usize;
                    let u_ij = w_i.dot(&ht.row(j));

                    // for each k
                    Zip::from(ρht_j).and(&w_i).for_each(|ρh_jk, &w_ik| {
                        *ρh_jk += w_ik * x_ij / u_ij;
                    });
                });
        });

    let w_row_sum = ht.sum_axis(Axis(0));

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
                .and(&w_row_sum)
                .for_each(|h_kj, ρh_kj, w_row_sum_k| {
                    *h_kj *= (ρh_kj / w_row_sum_k).max(EPS);
                });
        })
}
