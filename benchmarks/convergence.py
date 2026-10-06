"""Convergence benchmark: objective vs. time and iteration for different optimization methods.

Every method is started from the same initialization for a given seed. Each (method, seed) run
writes its loss trace to its own CSV in the output directory, so methods can be added to an
existing comparison later:

    python benchmarks/convergence.py run DATA --k 100 --seeds 0 1 2 --methods mu --out results/cc
    python benchmarks/convergence.py run DATA --k 100 --seeds 0 1 2 --methods bmme --out results/cc
    python benchmarks/convergence.py report results/cc --plot

Each trace is measured against the best objective reached for that seed. Measured against its own
final value, a run's gap always plunges to zero at its last iteration whether or not it has
converged, so a much longer reference run per seed should be included (`--ref-iters`, saved as
ref_seed*.csv). Reference runs only set the best objective; they are not reported or plotted.
Gaps much smaller than the reference's own remaining gap are still unreliable.

Requires the `bench` extra (h5py, zarr, matplotlib).
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import metagene

sys.path.insert(0, str(Path(__file__).parent))
from datasets import load  # noqa: E402

# Each method is called as f(X, k, W0, H0, max_iter, max_time, n_threads, eval_every) and returns
# a metagene.NMFResult whose loss records are (iteration, elapsed seconds, kl). Any work a method
# does before calling metagene.nmf (e.g. fitting a subsample) must be included in those times.
METHODS = {
    "mu": lambda X, k, W0, H0, max_iter, max_time, n_threads, eval_every: metagene.nmf(
        X, k, W0=W0, H0=H0, max_iter=max_iter, max_time=max_time,
        tol=-np.inf, eval_every=eval_every, n_threads=n_threads,
    ),
}

REF = "ref"

THRESHOLDS = (1e-2, 1e-3, 1e-4)

# Categorical palette, assigned to methods in a fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def init_factors(X, k, seed):
    """Same scheme as metagene.nmf's default initialization."""
    m, n = X.shape
    rng = np.random.default_rng(seed)
    scale = np.sqrt(4.0 * max(X.sum(), 1.0) / (m * n) / k)
    W0 = scale * rng.random((m, k), dtype=np.float32)
    H0 = scale * rng.random((k, n), dtype=np.float32)
    return W0, H0


def cmd_run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    X = load(args.data, args.ncells)
    print(f"loaded {X.shape[0]} × {X.shape[1]}, nnz={X.nnz} in {time.perf_counter() - t0:.1f}s")

    meta = {"data": str(args.data), "ncells": args.ncells, "shape": list(X.shape), "nnz": int(X.nnz), "k": args.k}
    meta_path = out / "meta.json"
    if meta_path.exists():
        prev = json.loads(meta_path.read_text())
        if prev != meta:
            sys.exit(f"{meta_path} describes a different setup:\n  {prev}\nvs\n  {meta}")
    meta_path.write_text(json.dumps(meta, indent=2))

    def save(name, seed, res):
        with open(out / f"{name}_seed{seed}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "time", "kl"])
            w.writerows(res.loss)
        it, t, kl = res.loss[-1]
        print(f"seed {seed} {name:>12}: {it} iters, {t:.1f}s, kl={kl:.8e}", flush=True)

    for seed in args.seeds:
        W0, H0 = init_factors(X, args.k, seed)
        if args.ref_iters:
            res = METHODS[args.ref_method](
                X, args.k, W0, H0, args.ref_iters, None, args.threads, max(args.ref_iters // 1000, 1)
            )
            save(REF, seed, res)
        for name in args.methods:
            res = METHODS[name](X, args.k, W0, H0, args.max_iter, args.max_time, args.threads, 1)
            save(name, seed, res)

    report(out, args.plot)


def read_traces(out):
    """{seed: {method: (iters, times, kls)}}"""
    traces = defaultdict(dict)
    for path in sorted(out.glob("*_seed*.csv")):
        method, seed = path.stem.rsplit("_seed", 1)
        a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        traces[int(seed)][method] = (a[:, 0].astype(int), a[:, 1], a[:, 2])
    return traces


def first_reaching(gap, xs, rtol):
    hit = np.nonzero(gap <= rtol)[0]
    return xs[hit[0]] if len(hit) else np.nan


def report(out, plot=False):
    out = Path(out)
    traces = read_traces(out)
    best = {seed: min(kl.min() for _, _, kl in runs.values()) for seed, runs in traces.items()}

    missing_ref = sorted(s for s, runs in traces.items() if REF not in runs)
    if missing_ref:
        print(f"warning: no reference run for seeds {missing_ref}; gaps near the end of runs are meaningless")
    ref_tail = []
    for seed, runs in traces.items():
        if REF in runs:
            its, _, kls = runs[REF]
            ref_kl = kls.min()
            # how much the reference still improved over its last 10% of iterations
            tail = kls[its >= 0.9 * its[-1]]
            ref_tail.append((tail[0] - tail[-1]) / tail[-1])
            for m, (_, _, kl) in runs.items():
                if m != REF and kl.min() < ref_kl:
                    print(f"warning: seed {seed} {m} beats the reference by "
                          f"{(ref_kl - kl.min()) / ref_kl:.1e}; use a longer --ref-iters")
        traces[seed] = {m: tr for m, tr in runs.items() if m != REF}
    methods = sorted({m for runs in traces.values() for m in runs}, key=list(METHODS).index)

    # median over seeds of time / iterations to reach each relative gap to the best objective
    header = "".join(f"  {'≤' + format(r, '.0e'):>17}" for r in THRESHOLDS)
    print(f"\n{'method':>12}  {'final gap':>9}{header}")
    print(f"{'':>12}  {'':>9}" + "".join(f"  {'time (s)':>8} {'iters':>8}" for _ in THRESHOLDS))
    for m in methods:
        runs = [(traces[s][m], best[s]) for s in traces if m in traces[s]]
        gaps = [(kl - b) / b for (_, _, kl), b in runs]
        row = f"{m:>12}  {np.median([g[-1] for g in gaps]):9.1e}"
        for r in THRESHOLDS:
            t = np.median([first_reaching(g, ts, r) for g, ((_, ts, _), _) in zip(gaps, runs)])
            i = np.median([first_reaching(g, its, r) for g, ((its, _, _), _) in zip(gaps, runs)])
            row += f"  {t:8.2f} {i:8.0f}"
        print(row)
    print(f"({len(traces)} seeds; medians are nan if any seed never reached the threshold)")
    if ref_tail:
        print(f"reference still improved by up to {max(ref_tail):.1e} over its last 10% of iterations, a lower "
              f"bound on its own gap to optimal; treat thresholds within ~10x of that with suspicion")

    if plot:
        plot_traces(out, traces, methods, best)


def plot_traces(out, traces, methods, best):
    import matplotlib.pyplot as plt

    meta = json.loads((out / "meta.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    floor = 1e-7  # the best run reaches a gap of exactly 0
    for ax, xi, xlabel in ((axes[0], 1, "time (s)"), (axes[1], 0, "iteration")):
        for mi, m in enumerate(methods):
            for seed, runs in traces.items():
                if m in runs:
                    x, kl = runs[m][xi], runs[m][2]
                    gap = np.maximum((kl - best[seed]) / best[seed], floor)
                    ax.plot(x, gap, color=COLORS[mi], lw=1.5, alpha=0.85,
                            label=m if seed == min(traces) else None)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="major", color="#e4e3dc", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("relative gap to best objective")
    axes[0].legend(frameon=False)
    shape = "×".join(map(str, meta["shape"]))
    fig.suptitle(f"{Path(meta['data']).parent.name}  ({shape}, nnz={meta['nnz']:,}, k={meta['k']})", fontsize=11)
    path = out / "convergence.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run methods and write loss traces, then report")
    r.add_argument("data", help="10x .h5 or AnnData .zarr")
    r.add_argument("--ncells", type=int, help="use only the first NCELLS cells")
    r.add_argument("--k", type=int, default=100)
    r.add_argument("--seeds", type=int, nargs="+", default=[0])
    r.add_argument("--methods", nargs="*", default=list(METHODS), choices=list(METHODS))
    r.add_argument("--max-iter", type=int, default=1000)
    r.add_argument("--max-time", type=float, help="per-run time limit in seconds")
    r.add_argument("--threads", type=int)
    r.add_argument("--ref-iters", type=int, help="also run a reference of this many iterations per seed")
    r.add_argument("--ref-method", default="mu", choices=list(METHODS))
    r.add_argument("--out", required=True)
    r.add_argument("--plot", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("report", help="summarize (and optionally plot) existing traces")
    s.add_argument("out")
    s.add_argument("--plot", action="store_true")
    s.set_defaults(func=lambda a: report(a.out, a.plot))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
