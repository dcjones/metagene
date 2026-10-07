// Inner-loop kernels on factor rows: dot products and axpy, in a portable version and, on x86_64,
// an explicit AVX2+FMA version.
//
// Callers are generic over `const AVX2: bool` and are instantiated inside a function with
// `#[target_feature(enable = "avx2,fma")]` for the AVX2 case, so that the whole loop around the
// kernels is compiled for AVX2 and the kernels inline into it. See `Isa::detect` for selection.

// Which kernel implementation to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Portable,
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
}

impl Isa {
    // The best available on this CPU. Setting METAGENE_SIMD=portable forces the portable kernels.
    pub fn detect() -> Self {
        if std::env::var("METAGENE_SIMD").is_ok_and(|v| v == "portable") {
            return Isa::Portable;
        }
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Isa::Avx2Fma;
        }
        Isa::Portable
    }
}

// Dispatch a call to a function generic over `const AVX2: bool`, compiling the AVX2 instantiation
// with AVX2 and FMA enabled. `$f::<AVX2>($args)` is evaluated with `AVX2` a constant.
macro_rules! dispatch {
    ($isa:expr, $f:ident($($arg:expr),* $(,)?)) => {{
        match $isa {
            crate::kernels::Isa::Portable => $f::<false>($($arg),*),
            #[cfg(target_arch = "x86_64")]
            crate::kernels::Isa::Avx2Fma => {
                #[target_feature(enable = "avx2,fma")]
                #[inline]
                unsafe fn avx2<R>(f: impl FnOnce() -> R) -> R {
                    f()
                }
                // SAFETY: Isa::Avx2Fma is only selected when the CPU supports AVX2 and FMA.
                unsafe { avx2(|| $f::<true>($($arg),*)) }
            }
        }
    }};
}
pub(crate) use dispatch;

#[inline(always)]
pub fn dot<const AVX2: bool>(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if AVX2 {
        // SAFETY: only instantiated with AVX2 = true via `dispatch!`, after detection.
        return unsafe { avx2::dot(a, b) };
    }
    portable::dot(a, b)
}

// y += alpha x
#[inline(always)]
pub fn axpy<const AVX2: bool>(alpha: f32, x: &[f32], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if AVX2 {
        // SAFETY: as for `dot`.
        return unsafe { avx2::axpy(alpha, x, y) };
    }
    portable::axpy(alpha, x, y)
}

// Dot products of `a` with each of N rows. Computing several at once gives independent dependency
// chains to overlap, and loads `a` only once.
#[inline(always)]
pub fn dot_n<const AVX2: bool, const N: usize>(a: &[f32], bs: [&[f32]; N]) -> [f32; N] {
    #[cfg(target_arch = "x86_64")]
    if AVX2 {
        // SAFETY: as for `dot`.
        return unsafe { avx2::dot_n(a, bs) };
    }
    bs.map(|b| portable::dot(a, b))
}

// y += Σ_s alphas[s] xs[s], reading and writing y only once.
#[inline(always)]
pub fn axpy_n<const AVX2: bool, const N: usize>(alphas: [f32; N], xs: [&[f32]; N], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if AVX2 {
        // SAFETY: as for `dot`.
        return unsafe { avx2::axpy_n(alphas, xs, y) };
    }
    for (alpha, x) in alphas.into_iter().zip(xs) {
        portable::axpy(alpha, x, y);
    }
}

// Hint that a factor row will be needed soon. Only the first couple of cache lines are prefetched;
// the hardware prefetcher picks up the rest of the row, and prefetching it all costs more than it
// saves (as does prefetching into L2 instead).
#[inline(always)]
pub fn prefetch(row: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    for line in row.chunks(16).take(2) {
        // SAFETY: prefetching is only a hint and never faults, and SSE is baseline on x86_64.
        #[allow(unused_unsafe)]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            _mm_prefetch::<_MM_HINT_T0>(line.as_ptr() as *const i8);
        }
    }
}

pub mod portable {
    // Dot product with independent partial sums, which the compiler can (partly) vectorize.
    #[inline(always)]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        const L: usize = 8;
        let n = a.len().min(b.len());
        let (a_chunks, a_tail) = a[..n].as_chunks::<L>();
        let (b_chunks, b_tail) = b[..n].as_chunks::<L>();

        let mut acc = [0_f32; L];
        for (a, b) in a_chunks.iter().zip(b_chunks) {
            for l in 0..L {
                acc[l] += a[l] * b[l];
            }
        }

        // pairwise reduction, so the partial sums aren't added in one long dependent chain
        let mut width = L;
        while width > 1 {
            width /= 2;
            for l in 0..width {
                acc[l] += acc[l + width];
            }
        }

        acc[0] + a_tail.iter().zip(b_tail).map(|(a, b)| a * b).sum::<f32>()
    }

    #[inline(always)]
    pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
        for (y, x) in y.iter_mut().zip(x) {
            *y += alpha * x;
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use std::arch::x86_64::*;

    // Mask selecting the first r < 8 lanes, for the tail of a row.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    fn tail_mask(r: usize) -> __m256i {
        _mm256_cmpgt_epi32(
            _mm256_set1_epi32(r as i32),
            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
        )
    }

    // Horizontal sum of the 8 lanes.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_movehdup_ps(s));
        _mm_cvtss_f32(s)
    }

    // Four independent 8-wide accumulators, to cover FMA latency.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        // SAFETY: every full load reads 8 floats at offset i with i + 8 <= n, within both slices.
        unsafe {
            while i + 32 <= n {
                for (l, acc) in acc.iter_mut().enumerate() {
                    let o = i + 8 * l;
                    *acc = _mm256_fmadd_ps(
                        _mm256_loadu_ps(pa.add(o)),
                        _mm256_loadu_ps(pb.add(o)),
                        *acc,
                    );
                }
                i += 32;
            }
            while i + 8 <= n {
                acc[0] = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(i)),
                    _mm256_loadu_ps(pb.add(i)),
                    acc[0],
                );
                i += 8;
            }
            if i < n {
                // masked loads only touch the n - i remaining floats, and zero the other lanes
                let m = tail_mask(n - i);
                acc[1] = _mm256_fmadd_ps(
                    _mm256_maskload_ps(pa.add(i), m),
                    _mm256_maskload_ps(pb.add(i), m),
                    acc[1],
                );
            }
        }
        let acc = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        hsum(acc)
    }

    // Independent 8-wide accumulators per row in `dot_n`, shortening each row's FMA dependency chain.
    const DOT_N_ACC: usize = 2;

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    #[allow(clippy::needless_range_loop)] // indices address several parallel arrays
    pub fn dot_n<const N: usize>(a: &[f32], bs: [&[f32]; N]) -> [f32; N] {
        const U: usize = DOT_N_ACC;
        let n = bs.iter().fold(a.len(), |n, b| n.min(b.len()));
        let pa = a.as_ptr();
        let pbs = bs.map(|b| b.as_ptr());
        let mut acc = [[_mm256_setzero_ps(); U]; N];
        let mut i = 0;
        // SAFETY: every full load reads 8 floats at offset o with o + 8 <= n, within all slices.
        unsafe {
            while i + 8 * U <= n {
                for u in 0..U {
                    let o = i + 8 * u;
                    let va = _mm256_loadu_ps(pa.add(o));
                    for s in 0..N {
                        acc[s][u] = _mm256_fmadd_ps(va, _mm256_loadu_ps(pbs[s].add(o)), acc[s][u]);
                    }
                }
                i += 8 * U;
            }
            // remaining full vectors (fewer than U); accumulator indices stay compile-time constants
            // so the accumulators stay in registers
            for u in 0..U - 1 {
                if i + 8 <= n {
                    let va = _mm256_loadu_ps(pa.add(i));
                    for s in 0..N {
                        acc[s][u] = _mm256_fmadd_ps(va, _mm256_loadu_ps(pbs[s].add(i)), acc[s][u]);
                    }
                    i += 8;
                }
            }
            if i < n {
                // masked loads only touch the n - i remaining floats, and zero the other lanes
                let m = tail_mask(n - i);
                let va = _mm256_maskload_ps(pa.add(i), m);
                for s in 0..N {
                    acc[s][U - 1] =
                        _mm256_fmadd_ps(va, _mm256_maskload_ps(pbs[s].add(i), m), acc[s][U - 1]);
                }
            }
        }
        acc.map(|mut acc| {
            // pairwise reduction of the accumulators
            let mut width = U;
            while width > 1 {
                width /= 2;
                for l in 0..width {
                    acc[l] = _mm256_add_ps(acc[l], acc[l + width]);
                }
            }
            hsum(acc[0])
        })
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub fn axpy_n<const N: usize>(alphas: [f32; N], xs: [&[f32]; N], y: &mut [f32]) {
        let n = xs.iter().fold(y.len(), |n, x| n.min(x.len()));
        let pxs = xs.map(|x| x.as_ptr());
        let py = y.as_mut_ptr();
        let mut vas = [_mm256_setzero_ps(); N];
        for s in 0..N {
            vas[s] = _mm256_set1_ps(alphas[s]);
        }
        let mut i = 0;
        // SAFETY: every load/store touches 8 floats at offset i with i + 8 <= n, within all slices.
        unsafe {
            while i + 8 <= n {
                let mut v = _mm256_loadu_ps(py.add(i));
                for s in 0..N {
                    v = _mm256_fmadd_ps(vas[s], _mm256_loadu_ps(pxs[s].add(i)), v);
                }
                _mm256_storeu_ps(py.add(i), v);
                i += 8;
            }
            if i < n {
                // masked loads/stores only touch the n - i remaining floats
                let m = tail_mask(n - i);
                let mut v = _mm256_maskload_ps(py.add(i), m);
                for s in 0..N {
                    v = _mm256_fmadd_ps(vas[s], _mm256_maskload_ps(pxs[s].add(i), m), v);
                }
                _mm256_maskstore_ps(py.add(i), m, v);
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
        let n = x.len().min(y.len());
        let (px, py) = (x.as_ptr(), y.as_mut_ptr());
        let va = _mm256_set1_ps(alpha);
        let mut i = 0;
        // SAFETY: every load/store touches 8 floats at offset i with i + 8 <= n, within both slices.
        unsafe {
            while i + 8 <= n {
                let v = _mm256_fmadd_ps(va, _mm256_loadu_ps(px.add(i)), _mm256_loadu_ps(py.add(i)));
                _mm256_storeu_ps(py.add(i), v);
                i += 8;
            }
            if i < n {
                // masked loads/stores only touch the n - i remaining floats
                let m = tail_mask(n - i);
                let v = _mm256_fmadd_ps(
                    va,
                    _mm256_maskload_ps(px.add(i), m),
                    _mm256_maskload_ps(py.add(i), m),
                );
                _mm256_maskstore_ps(py.add(i), m, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vecs(n: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
        let f = |i: usize, s: u32| {
            ((i as u32).wrapping_mul(2654435761).wrapping_add(s) % 1000) as f32 / 500.0
        };
        (
            (0..n).map(|i| f(i, seed)).collect(),
            (0..n).map(|i| f(i, seed + 7)).collect(),
        )
    }

    #[inline(always)]
    fn dot2<const AVX2: bool>(a: &[f32], b: &[f32], c: &[f32]) -> [f32; 2] {
        dot_n::<AVX2, 2>(a, [b, c])
    }

    #[inline(always)]
    fn axpy2<const AVX2: bool>(a: &[f32], b: &[f32], y: &mut [f32]) {
        axpy_n::<AVX2, 2>([0.5, -0.25], [a, b], y)
    }

    #[test]
    fn kernels_agree() {
        let isas: Vec<Isa> = [Isa::Portable, Isa::detect()].into_iter().collect();
        for n in 0..80 {
            let (a, b) = vecs(n, n as u32);
            let expect: f64 = a.iter().zip(&b).map(|(a, b)| *a as f64 * *b as f64).sum();
            for &isa in &isas {
                let d = dispatch!(isa, dot(&a, &b));
                assert!(
                    (d as f64 - expect).abs() <= 1e-5 * expect.abs().max(1.0),
                    "{isa:?} n={n}"
                );

                let (c, _) = vecs(n, n as u32 + 3);
                let expect_c: f64 = a.iter().zip(&c).map(|(a, c)| *a as f64 * *c as f64).sum();
                let [d0, d1] = dispatch!(isa, dot2(&a, &b, &c));
                assert!(
                    (d0 as f64 - expect).abs() <= 1e-5 * expect.abs().max(1.0),
                    "{isa:?} n={n}"
                );
                assert!(
                    (d1 as f64 - expect_c).abs() <= 1e-5 * expect_c.abs().max(1.0),
                    "{isa:?} n={n}"
                );

                let mut y = c.clone();
                dispatch!(isa, axpy2(&a, &b, &mut y));
                for j in 0..n {
                    let e = c[j] + 0.5 * a[j] - 0.25 * b[j];
                    assert!((y[j] - e).abs() <= 1e-6 * e.abs().max(1.0));
                }

                let mut y = b.clone();
                dispatch!(isa, axpy(0.75, &a, &mut y));
                for j in 0..n {
                    assert!((y[j] - (b[j] + 0.75 * a[j])).abs() <= 1e-6 * y[j].abs().max(1.0));
                }
            }
        }
    }
}
