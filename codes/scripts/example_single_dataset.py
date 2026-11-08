"""End-to-end example on a single dataset (MNIST by default).

Walks through every method in the AKSS paper for a *single seed* — meant as a
smoke-test / API tour, not as a benchmark:

    1) UMAP + KMeans baseline
    2) AKSS (KMeans init)
    3) UK-AKSS (UMAP+KMeans init)               <-- chain starts here
    4) SR-UMAP V1 (subspace-reweighted graph, fixed kNN)
    5) SR-UMAP V2 (subspace-residual top-m reselection)
    6) ISR-UMAP V2 (iterative SR-UMAP, T=2)
    7) (optional) Joint UMAP-driven AKSS (Algorithm 2)

The `--bic-weight` knob lets you fix one `w` for all methods. By default each
method uses the per-dataset best `w` from the paper, looked up in the table
that matches the method's init: KM-init / Algorithm 2 use `BEST_BIC_WEIGHTS_KM`;
UK-init / SR / ISR use `BEST_BIC_WEIGHTS_UK`.

Usage:
    python scripts/example_single_dataset.py --cache-dir /path/to/clip_cache
    python scripts/example_single_dataset.py --cache-dir ... --dataset CIFAR10 --joint
    python scripts/example_single_dataset.py --cache-dir ... --bic-weight 0.16
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from akss import (
    BEST_BIC_WEIGHTS_KM,
    BEST_BIC_WEIGHTS_UK,
    eval_clustering,
    isr_umap,
    k_subspaces_adaptive,
    l2_normalize_rows,
    load_dataset,
    sr_umap_from_external_akss,
    umap_kmeans_baseline,
    umap_kmeans_init,
)


def parse_args():
    p = argparse.ArgumentParser(description="AKSS single-dataset example.")
    p.add_argument("--cache-dir", type=str, required=True,
                   help="Directory holding cached .pt feature files.")
    p.add_argument("--dataset", type=str, default="MNIST",
                   help="Dataset name (MNIST, CIFAR10, STL10, ...).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--graph-nn", type=int, default=15,
                   help="kNN size for the SR / baseline UMAP graph.")
    p.add_argument("--q", type=int, default=2,
                   help="UMAP embedding dimension.")
    p.add_argument("--bic-weight", type=float, default=None,
                   help="If given, use this fixed w for every method; otherwise "
                        "each method uses its own per-dataset best from the paper.")
    p.add_argument("--joint", action="store_true",
                   help="Also run the joint UMAP-driven AKSS (Algorithm 2).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def header(name: str, X: np.ndarray, K: int, w_km: float, w_uk: float, args):
    print("\n" + "=" * 78)
    print(f"  {name}  |  n={X.shape[0]}  d={X.shape[1]}  K={K}")
    print(f"  seed={args.seed}  graph_nn={args.graph_nn}  q={args.q}")
    print(f"  bic_weight: w_KM={w_km}  w_UK={w_uk}")
    print("=" * 78)


def print_row(method: str, metrics: dict, elapsed: float):
    print(f"  {method:<30s} ARI={metrics['ari']:.4f}  "
          f"NMI={metrics['nmi']:.4f}  ACC={metrics['acc']:.4f}  "
          f"({elapsed:.1f}s)")


def main():
    args = parse_args()
    X, y, K = load_dataset(args.dataset, args.cache_dir)

    if args.bic_weight is not None:
        w_km = w_uk = float(args.bic_weight)
    else:
        w_km = BEST_BIC_WEIGHTS_KM[args.dataset]
        w_uk = BEST_BIC_WEIGHTS_UK[args.dataset]

    header(args.dataset, X, K, w_km, w_uk, args)

    Xn = l2_normalize_rows(X.astype(np.float32))
    del X
    gc.collect()

    # 1) UMAP + KMeans baseline
    t0 = time.time()
    base = umap_kmeans_baseline(
        args.dataset, Xn, y, K,
        n_neighbors=args.graph_nn, n_components=args.q,
        random_state=args.seed, verbose=args.verbose,
    )
    print_row("UMAP + KMeans (baseline)", base["metrics"], time.time() - t0)
    del base
    gc.collect()

    # 2) AKSS (KMeans init)  --  KM-table w
    t0 = time.time()
    pred_km, _, _ = k_subspaces_adaptive(
        Xn, K, all_labels=y,
        init="kmeans", random_state=args.seed,
        l2_normalize=False,
        rank_mode="bic", r_min=4, r_max=128, bic_weight=w_km,
        max_iter=20, verbose=args.verbose,
    )
    print_row(f"AKSS (KMeans init, w={w_km})",
              eval_clustering(y, pred_km),
              time.time() - t0)
    del pred_km
    gc.collect()

    # Story chain starts here: UK-AKSS -> SR-UMAP V1/V2 -> ISR-UMAP V2.
    # All four share the same UK-table w (since they share AKSS structure).

    # 3) UK-AKSS (UMAP+KMeans init)
    t0 = time.time()
    pred_init = umap_kmeans_init(Xn, K, random_state=args.seed,
                                 n_neighbors=args.graph_nn, n_components=args.q)
    pred_uk, model_uk, _ = k_subspaces_adaptive(
        Xn, K, all_labels=y,
        init="external", pred_init=pred_init, random_state=args.seed,
        l2_normalize=False,
        rank_mode="bic", r_min=4, r_max=128, bic_weight=w_uk,
        max_iter=20, verbose=args.verbose,
    )
    print_row(f"UK-AKSS (UMAP+KM init, w={w_uk})",
              eval_clustering(y, pred_uk),
              time.time() - t0)
    del pred_init
    gc.collect()

    # 4 + 5) SR-UMAP V1 / V2 from UK-AKSS
    t0 = time.time()
    sr = sr_umap_from_external_akss(
        args.dataset, Xn, y, K,
        pred_akss=pred_uk, model=model_uk,
        random_state=args.seed,
        graph_n_neighbors=args.graph_nn, q=args.q,
        compare_reselect=True, verbose=args.verbose,
    )
    elapsed = time.time() - t0
    print_row("SR-UMAP V1 (reweight-fixed)", sr["V1"]["metrics"], elapsed)
    if "V2" in sr:
        print_row("SR-UMAP V2 (reselect-topm)", sr["V2"]["metrics"], elapsed)
    del sr
    gc.collect()

    # 6) ISR-UMAP V2 (outer T=2)
    t0 = time.time()
    pred_isr, _ = isr_umap(
        args.dataset, Xn, y, K,
        outer_T=2, update_variant="V2",
        pred_init=pred_uk, model_init=model_uk,
        graph_n_neighbors=args.graph_nn, q=args.q,
        bic_weight=w_uk, random_state=args.seed,
        verbose=args.verbose,
    )
    print_row("ISR-UMAP V2 (T=2)",
              eval_clustering(y, pred_isr),
              time.time() - t0)
    del pred_uk, model_uk, pred_isr
    gc.collect()

    # 7) Joint UMAP-driven AKSS (Algorithm 2) — KM-table w (kmeans init on Y)
    if args.joint:
        from akss import joint_umap_ksubspaces
        t0 = time.time()
        pred_joint, _, _ = joint_umap_ksubspaces(
            Xn, K, all_labels=y,
            n_neighbors=args.graph_nn, q=128,
            outer_iter=10, inner_steps=200,
            rank_mode="bic", r_min=4, r_max=64, bic_weight=w_km,
            beta=1.0, gamma=1.0, seed=args.seed,
            verbose=args.verbose,
        )
        print_row(f"Joint AKSS Alg 2 (w={w_km})",
                  eval_clustering(y, pred_joint),
                  time.time() - t0)

    print("-" * 78 + "\n")


if __name__ == "__main__":
    main()
