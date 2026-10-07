# metagene

Fast KL-divergence NMF (X ≈ W H) for large, sparse single-cell / spatial count matrices
(cells × genes, typically 1e4–1e6 cells, ~2e4 genes, 1–5% dense, k = 100–200). A Rust core
(`pyo3`/`maturin` extension) with a thin Python wrapper.

## Layout

- `src/nmf.rs` — the solver: `nmf()` main loop, W and H update passes, KL objective, BMMe
  extrapolation, CSR/CSC structures. Tests at the bottom.
- `src/kernels.rs` — inner-loop kernels (`dot`, `axpy`, fused `dot_n`/`axpy_n`, `prefetch`) in
  portable and AVX2+FMA versions, `Isa` runtime detection, and the `dispatch!` macro.
- `src/lib.rs` — the `_nmf` Python binding (GIL released, optional rayon pool via `n_threads`).
- `python/metagene/__init__.py` — `metagene.nmf()`: input conversion (any scipy sparse/dense →
  canonical CSR f32/u32 without copying where possible), initialization, warm start.
- `python/tests/test_all.py` — tests against dense numpy reference implementations of MU and BMMe.
- `benchmarks/convergence.py` — convergence harness (objective vs. time/iteration); `datasets.py`
  loads 10x `.h5` and AnnData `.zarr` without scanpy/anndata.

## Build and test

```sh
uv venv && uv pip install maturin numpy scipy pytest   # once; add h5py zarr matplotlib for benchmarks
.venv/bin/maturin develop --release
cargo test                                              # Rust unit tests
.venv/bin/pytest python/tests
METAGENE_SIMD=portable .venv/bin/pytest python/tests    # also exercise the portable kernels
```

Don't add scanpy/anndata as dependencies (too heavy); the `bench` extra is h5py, zarr, matplotlib.

## Algorithm and design decisions

Each iteration is a W pass then an H pass (alternating multiplicative updates). Defaults:
`method="bmme"`, warm start auto-enabled for large m. All of the below were measured; see git log.

- **W pass is row-parallel over CSR; H pass is column-parallel over a CSC copy.** Each thread owns
  the rows it writes. The original design fused both updates in one row-parallel pass, scattering
  H contributions into per-thread n×k buffers — that was memory-bound and 3–8× slower. Rule:
  parallelize over the dimension you write. Cost: the CSC copy (8 bytes/nonzero).
- **H pass is cache-blocked by cells** (`BlockedCSC`, `H_PASS_BLOCK_BYTES` = 8 MB so a block's W rows
  stay in L3). All threads work on one block at a time, accumulating into a persistent n×k `ρht`.
  3–4× faster on 250k cells; blocked and unblocked results are bit-identical (tested).
- **Load balancing:** rayon loops use `with_max_len(PAR_GRAIN = 4)`. Without it, inputs ordered by
  depth or gene density ran ~2.5× slower.
- **BMMe** (Hien, Leplat & Gillis 2025; MATLAB reference `MUe_KLNMF.m`): extrapolate each factor
  (positive part only, Nesterov β) before its MU step. Same cost per iteration as MU, 2–4× fewer
  iterations. Implemented as an elementwise pre-pass (`extrapolate`) before each factor's update.
  The β cap in the paper is omitted (its default disables it). Objective isn't guaranteed monotone,
  but no increases were ever observed; `restart` exists but never triggered.
- **Objective:** computed for free in the W pass when no extrapolation is applied; otherwise a
  separate `kl_divergence` pass, excluded from reported timings. Loss records are
  `(iteration, seconds, kl)` describing the state *before* that iteration's step.
- **Warm start** (Python side): fit 10% of cells for 200 iterations, then 5 W-only passes
  (`fit_H=False`) over all cells, then the full fit. Auto-enabled when no init is given and the
  subsample would have ≥ 20k cells. 2–4× faster to a given objective on 250k and 660k cells; useless
  on ~10k cells. Tuned on 250k cells and checked at 660k: smaller subsamples (or a fixed cap) were
  worse.
- **Kernels:** explicit AVX2+FMA intrinsics, chosen at runtime (`Isa::detect`, override with
  `METAGENE_SIMD=portable`). Per-row loop bodies are generic over `const AVX2: bool` and run via
  `dispatch!` inside a `#[target_feature]` wrapper so the whole loop is compiled for AVX2. LLVM did
  not vectorize the dot product's reduction well on its own (2-wide), hence intrinsics.
  Nonzeros are processed in pairs (`NNZ_BLOCK = 2`) with fused `dot_n`/`axpy_n` to overlap
  dependency chains; tails use masked loads/stores; prefetch only the first 2 lines of the row
  2 blocks ahead. Only x86_64 AVX2 has explicit kernels so far; other ISAs (NEON, AVX-512) would be
  added as further `Isa` variants.
- **Precision:** f32 storage and kernels, f64 accumulation for the objective. Values are clamped at
  `EPS = 1e-6` after each multiplicative update.

### Tried and rejected (don't redo without new evidence)

Reordering cells/genes for locality (≤ 10% at best); splitting genes into dense + sparse (GEMM) parts
(dense genes are too sparse here); extra W updates per row per iteration (each costs ~a full W pass);
L2-sized H blocks; more `dot_n` accumulators; software pipelining; prefetching whole rows or into
L2; prefetch off. Remaining cost is mostly L3 latency of gathered factor rows — roughly 2.5× the
arithmetic throughput floor, which we consider near the practical floor for this design.

## Benchmarking

- Convergence: `benchmarks/convergence.py run DATA --k 100 --seeds 0 1 2 --methods mu bmme bmme-warm
  --max-time 30 --out benchmarks/results/<name> --plot`, plus a long reference per seed
  (`--methods --ref-iters 15000 --ref-method bmme-warm --ref-name ref-bmme-warm`). Gaps are relative
  to the best reference; without one, runs measured against their own end plunge to 0 (an artifact).
  Runs can land in different local minima (~1e-3 apart on scRNA), so keep several references
  (`ref-*`) and use several seeds. `benchmarks/results/` is gitignored.
- Per-iteration speed: time an N-iteration run minus a 1-iteration run (excludes setup like the CSC
  build); `perf stat` instructions/cycles per nonzero per pass are steadier than wall time, but
  summed cycles include rayon threads spin-waiting.
- Set `OPENBLAS_NUM_THREADS=1` (the harness does) — numpy's OpenBLAS threads otherwise spin and steal
  CPU. Same-build repeats vary ~10%; run variants alternately and repeat before trusting small gains.

Reference numbers (Ryzen 9 5950X, 32 threads, k=100): ~10 ms/iteration on 9.5k cells × 18k genes
(5.5M nonzeros); ~0.6 s on 250k cells (259M nonzeros); ~1.5 s on 660k cells (632M nonzeros, peak
memory ~11.6 GB: X + CSC copy dominate).

Datasets used during development (local paths):
- scRNA, 9.5k cells: `/mnt/extra2/against-normalization/datasets/scrnaseq-cervical-cancer/filtered_feature_bc_matrix.h5`
- spatial, 660k cells: `/mnt/extra2/against-normalization/results/atera-cervical-cancer/seg-proseg/counts.zarr`
  (benchmarks mostly used `--ncells 250000`)

## Working conventions

- The user reviews and makes git commits themselves — don't commit. Keep each change a cleanly
  scoped unit so it can be committed on its own.
- Measure before and after any performance change; keep only what helps beyond noise, and report
  what didn't.
- Match the existing code style: Greek identifiers for update factors (`ρw_i`, `ρht`, `β`),
  `// for each j`-style loop comments, `[m, k]` / `[n, k]` shape comments.
