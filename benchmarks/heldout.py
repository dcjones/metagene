"""Held-out deviance benchmark: does the fit generalize, or overfit the counts' noise?

Each count x_ij is split by binomial thinning into x_train ~ Binomial(x_ij, p) and
x_test = x_ij - x_train. For Poisson counts the two halves are independent Poissons with means
p μ_ij and (1 - p) μ_ij, so a model fit to the training half predicts the test half as
(1 - p) / p · W H, and its test deviance measures how well it recovered μ rather than the noise.
Both are scored against the rank-1 null model μ_ij = r_i c_j / T (cell depth × gene frequency).

Fits are rerun from the same (deterministic, NNDSVD) initialization for each iteration count in
--iters, so the test deviance can be followed along the optimization path. metagene.nmf's default
warm start applies (it's only used from 200k cells). Variants of the solver are given as
comma-separated settings on top of plain full-batch KL-NMF (`base`: BMMe with no prior on H, so
results stay comparable with runs from before minibatch training became metagene's default):

    method=minibatch    solver method (metagene's default; "base" is method=bmme)
    eps=1e-3        floor on factor values
    a=0.1           flat Gamma prior on H with this pseudocount (h_pseudocount), default rate
    a=0.1,b=2       ... and rate (h_rate)
    gr=0.1          gene-rate prior with this pseudocount (gr=1 is a good strength)

    python benchmarks/heldout.py run DATA --ncells 500 2500 9503 --k 25 100 \\
        --variants base eps=1e-2 a=0.1 --out results/ho
    python benchmarks/heldout.py report results/ho --plot

Per-gene deviances are also saved, so the report can break the gain over the null model down by
how many counts each gene has.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import metagene  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from datasets import load  # noqa: E402

FIELDS = ["variant", "ncells", "k", "seed", "iters", "seconds", "train_dev", "test_dev", "train_null",
          "test_null", "train_counts", "test_counts"]
INT_FIELDS = {"ncells", "k", "seed", "iters"}

# Variant setting names → metagene.nmf keyword arguments (and fixed extra arguments).
VARIANT_KEYS = {
    "eps": ("eps", {}),
    "a": ("h_pseudocount", {"h_prior": "flat"}),
    "b": ("h_rate", {}),
    "gr": ("h_pseudocount", {"h_prior": "gene-rate"}),
    "method": ("method", {}),
}

# Gene total count bins (in the full, unsplit matrix) for the per-gene breakdown.
GENE_BINS = [0, 10, 100, 1_000, 10_000, np.inf]

# Categorical palette (as in convergence.py), assigned to variants in the order given.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def parse_variant(spec):
    kw = {"h_pseudocount": 0.0, "method": "bmme"}
    if spec == "base":
        return kw
    for setting in spec.split(","):
        key, _, value = setting.partition("=")
        if key not in VARIANT_KEYS:
            raise ValueError(f"unknown variant setting {key!r} in {spec!r}")
        name, extra = VARIANT_KEYS[key]
        kw[name] = value if name == "method" else float(value)
        kw.update(extra)
    return kw


def split_counts(X, p, rng):
    """Binomial thinning of an integer count matrix into (train, test)."""
    if not np.all(X.data == np.round(X.data)):
        raise ValueError("binomial splitting needs integer counts")
    train_data = rng.binomial(X.data.astype(np.int64), p).astype(np.float32)
    # Separate index arrays: eliminate_zeros works in place.
    train = sp.csr_matrix((train_data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    test = sp.csr_matrix((X.data - train_data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    train.eliminate_zeros()
    test.eliminate_zeros()
    return train, test


def gene_deviance(X, W, Ht, scale=1.0, chunk=1 << 18):
    """Per-gene Poisson deviance of X [m, n] under μ = scale · W Ht.T. Returns [n] float64.

    D_j = 2 Σ_i [x_ij log(x_ij / μ_ij) - x_ij + μ_ij], where the μ sum runs over all cells
    (colsum(W) · Ht_j) and the log term only over nonzeros.
    """
    rows = np.repeat(np.arange(X.shape[0], dtype=np.int64), np.diff(X.indptr))
    dev = np.zeros(X.shape[1])
    for s in range(0, X.nnz, chunk):
        r, c, x = rows[s:s + chunk], X.indices[s:s + chunk], X.data[s:s + chunk].astype(np.float64)
        μ = scale * np.einsum("ij,ij->i", W[r], Ht[c], dtype=np.float64)
        dev += np.bincount(c, weights=x * np.log(x / μ) - x, minlength=X.shape[1])
    dev += scale * (Ht.astype(np.float64) @ W.sum(axis=0, dtype=np.float64))
    return 2.0 * dev


def null_factors(X):
    """Rank-1 Poisson MLE μ_ij = r_i c_j / T as (W [m, 1], Ht [n, 1])."""
    r = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
    c = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
    return (r / r.sum())[:, None], c[:, None]


def gene_file(out, variant, ncells, k, seed, iters):
    tag = "" if variant == "base" else f"_v{variant.replace('=', '').replace(',', '_')}"
    return Path(out) / f"genes_n{ncells}_k{k}_s{seed}_i{iters}{tag}.npz"


def read_results(out):
    csv_path = Path(out) / "results.csv"
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("variant", "base")  # results from before variants existed
        for key in FIELDS[1:]:
            r[key] = int(r[key]) if key in INT_FIELDS else float(r[key])
    return rows


def cmd_run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = {v: parse_variant(v) for v in args.variants}
    X_full = load(args.data)
    print(f"loaded {X_full.shape[0]} × {X_full.shape[1]}, nnz={X_full.nnz}")
    meta = {"data": str(args.data), "shape": list(X_full.shape), "p": args.p, "split_seed": args.split_seed}
    # Optionally make the dataset resemble a smaller or shallower one: a random subset of cells, and
    # binomial thinning of every count (as if sequenced less deeply).
    pre_rng = np.random.default_rng([args.split_seed, 1])
    if args.subsample is not None:
        X_full = X_full[np.sort(pre_rng.choice(X_full.shape[0], args.subsample, replace=False))]
        meta["subsample"] = args.subsample
    if args.thin is not None:
        X_full, _ = split_counts(X_full, args.thin, pre_rng)
        meta["thin"] = args.thin
    if args.subsample is not None or args.thin is not None:
        depth = np.asarray(X_full.sum(axis=1)).ravel()
        print(f"resampled to {X_full.shape[0]} × {X_full.shape[1]}, nnz={X_full.nnz}, "
              f"depth mean {depth.mean():.0f} median {np.median(depth):.0f}")
    meta_path = out / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        sys.exit(f"{meta_path} describes a different setup")
    meta_path.write_text(json.dumps(meta, indent=2))

    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(X_full.shape[0])  # nested cell subsets: the first ncells of one order
    train_all, test_all = split_counts(X_full, args.p, rng)

    # (Re)write the CSV with the current columns, then append.
    rows = read_results(out)
    done = {(r["variant"], r["ncells"], r["k"], r["seed"], r["iters"]) for r in rows}
    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        for ncells in args.ncells:
            idx = np.sort(perm[:min(ncells, X_full.shape[0])])
            train, test = train_all[idx], test_all[idx]
            # Drop genes with no training counts: no model can predict them, and the
            # null model would predict exactly zero for them.
            genes = np.flatnonzero(np.asarray(train.sum(axis=0)).ravel() > 0)
            train, test = sp.csr_matrix(train[:, genes]), sp.csr_matrix(test[:, genes])
            gene_counts = np.asarray(X_full[idx][:, genes].sum(axis=0)).ravel()
            scale = (1 - args.p) / args.p
            Wn, Htn = null_factors(train)
            train_null = gene_deviance(train, Wn, Htn)
            test_null = gene_deviance(test, Wn, Htn, scale)
            np.savez(out / f"null_n{ncells}.npz", genes=genes, gene_counts=gene_counts,
                     train=train_null, test=test_null)
            print(f"ncells={len(idx)} genes={len(genes)} train nnz={train.nnz} "
                  f"null deviance/count: train {train_null.sum() / train.sum():.4f} "
                  f"test {test_null.sum() / test.sum():.4f}")
            for variant, kw in variants.items():
                for k in args.k:
                    for seed in args.seeds:
                        for iters in args.iters:
                            if (variant, ncells, k, seed, iters) in done:
                                continue
                            t0 = time.perf_counter()
                            res = metagene.nmf(train, k, max_iter=iters, tol=-np.inf, eval_every=0, seed=seed,
                                               n_threads=args.n_threads, **kw)
                            secs = time.perf_counter() - t0
                            Ht = np.ascontiguousarray(res.H.T)
                            tr = gene_deviance(train, res.W, Ht)
                            te = gene_deviance(test, res.W, Ht, scale)
                            np.savez(gene_file(out, variant, ncells, k, seed, iters), train=tr, test=te)
                            writer.writerow(dict(
                                variant=variant, ncells=ncells, k=k, seed=seed, iters=iters, seconds=secs,
                                train_dev=tr.sum(), test_dev=te.sum(), train_null=train_null.sum(),
                                test_null=test_null.sum(), train_counts=train.sum(), test_counts=test.sum(),
                            ))
                            f.flush()
                            print(f"  {variant:>12s} k={k:4d} seed={seed} iters={iters:5d} {secs:6.1f}s  "
                                  f"explained train {1 - tr.sum() / train_null.sum():.4f} "
                                  f"test {1 - te.sum() / test_null.sum():.4f}")


def explained(sel, part):
    """Explained deviance 1 - D / D_null of the train or test half, averaged over runs (seeds)."""
    return np.mean([1 - r[f"{part}_dev"] / r[f"{part}_null"] for r in sel])


def variant_order(rows, variants):
    seen = list(dict.fromkeys(r["variant"] for r in rows))
    return [v for v in variants if v in seen] if variants else seen


def cmd_report(args):
    out = Path(args.out)
    rows = read_results(out)
    if args.variants:
        rows = [r for r in rows if r["variant"] in args.variants]
    if args.ncells:
        rows = [r for r in rows if r["ncells"] in args.ncells]
    if args.k:
        rows = [r for r in rows if r["k"] in args.k]
    variants = variant_order(rows, args.variants)
    ncells_all = sorted({r["ncells"] for r in rows})
    ks = sorted({r["k"] for r in rows})
    w = max(len(v) for v in variants) + 2

    def select(**key):
        return [r for r in rows if all(r[f] == v for f, v in key.items())]

    # Test explained deviance at the last iteration count, and at the best one (the iteration count
    # an oracle would early-stop at).
    for title, best in (("at the last iteration count", False), ("best over iteration counts (iters)", True)):
        print(f"\ntest explained deviance (1 - D / D_null) {title}, mean over seeds")
        print(" " * w + "".join(f"{f'n={nc} k={k}':>17s}" for nc in ncells_all for k in ks))
        for v in variants:
            cells = []
            for nc in ncells_all:
                for k in ks:
                    its = sorted({r["iters"] for r in select(variant=v, ncells=nc, k=k)})
                    if not its:
                        cells.append(f"{'':>17s}")
                        continue
                    te = {i: explained(select(variant=v, ncells=nc, k=k, iters=i), "test") for i in its}
                    if best:
                        i = max(te, key=te.get)
                        cells.append(f"{te[i]:>10.4f} ({i:>4d})")
                    else:
                        cells.append(f"{te[its[-1]]:>17.4f}")
            print(f"{v:>{w}s}" + "".join(cells))

    def last(sel):
        it = max((r["iters"] for r in sel), default=None)
        return [r for r in sel if r["iters"] == it]

    print("\ntrain explained deviance at the last iteration count")
    for v in variants:
        cells = [last(select(variant=v, ncells=nc, k=k)) for nc in ncells_all for k in ks]
        print(f"{v:>{w}s}" + "".join(f"{explained(c, 'train'):>17.4f}" if c else f"{'':>17s}" for c in cells))

    # Per-gene breakdown at the last iteration count: test gain over null per gene-count bin.
    print("\ntest deviance gain over null by gene total count, last iteration count, seed-mean "
          "(fraction of the bin's null deviance; negative = worse than rank 1)")
    for nc in ncells_all:
        null = np.load(out / f"null_n{nc}.npz")
        bins = np.digitize(null["gene_counts"], GENE_BINS[1:-1])
        labels = [f"[{GENE_BINS[b]:g},{GENE_BINS[b + 1]:g})" for b in range(len(GENE_BINS) - 1)]
        print(f"ncells={nc}  genes per bin: " + ", ".join(
            f"{lab}: {(bins == b).sum()}" for b, lab in enumerate(labels)))
        for k in ks:
            for v in variants:
                sel = select(variant=v, ncells=nc, k=k)
                if not sel:
                    continue
                it = max(r["iters"] for r in sel)
                seeds = sorted({r["seed"] for r in sel if r["iters"] == it})
                te = np.mean([np.load(gene_file(out, v, nc, k, s, it))["test"] for s in seeds], axis=0)
                gains = []
                for b in range(len(labels)):
                    d0 = null["test"][bins == b].sum()
                    gains.append(f"{(d0 - te[bins == b].sum()) / d0:+8.3f}" if d0 > 0 else f"{'':>8s}")
                print(f"  k={k:<4d}{v:>{w}s} " + "".join(gains))

    if args.plot:
        plot(out, rows, variants, ncells_all, ks)


def plot(out, rows, variants, ncells_all, ks):
    import matplotlib.pyplot as plt

    if len(variants) > len(COLORS):
        sys.exit(f"at most {len(COLORS)} variants can be plotted; select some with --variants")
    fig, axes = plt.subplots(1, len(ncells_all), figsize=(4.2 * len(ncells_all), 3.6), squeeze=False)
    for ax, nc in zip(axes[0], ncells_all):
        for v, color in zip(variants, COLORS):
            xs, ys = [], []
            for k in ks:
                sel = [r for r in rows if (r["variant"], r["ncells"], r["k"]) == (v, nc, k)]
                if sel:
                    it = max(r["iters"] for r in sel)
                    xs.append(k)
                    ys.append(explained([r for r in sel if r["iters"] == it], "test"))
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=4, label=v)
        ax.axhline(0, color="#888", lw=0.75)
        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set_title(f"{nc} cells", fontsize=10)
        ax.set_xlabel("k")
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("test explained deviance vs. rank-1 null")
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out / "heldout_variants.png", dpi=150)
    print(f"wrote {out / 'heldout_variants.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("data")
    r.add_argument("--ncells", type=int, nargs="+", required=True)
    r.add_argument("--k", type=int, nargs="+", default=[100])
    r.add_argument("--seeds", type=int, nargs="+", default=[0])
    r.add_argument("--iters", type=int, nargs="+", default=[50, 200, 800, 3200])
    r.add_argument("--variants", nargs="+", default=["base"])
    r.add_argument("--p", type=float, default=0.5, help="fraction of counts kept for training")
    r.add_argument("--split-seed", type=int, default=0)
    r.add_argument("--subsample", type=int, help="first take this many random cells")
    r.add_argument("--thin", type=float, help="first keep each count with this probability")
    r.add_argument("--n-threads", type=int)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)
    rep = sub.add_parser("report")
    rep.add_argument("out")
    rep.add_argument("--variants", nargs="+", help="only these variants, in this order")
    rep.add_argument("--ncells", type=int, nargs="+", help="only these cell counts")
    rep.add_argument("--k", type=int, nargs="+", help="only these k")
    rep.add_argument("--plot", action="store_true")
    rep.set_defaults(func=cmd_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
