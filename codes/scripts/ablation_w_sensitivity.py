"""Ablation: BIC weight `w` sensitivity for the UK-init -> SR / ISR chain.

For each (dataset, seed, w in W_GRID) this script runs

    UMAP+KMeans init -> AKSS (w) -> SR-UMAP V1/V2 (+ ISR-V2, T=2)

on the paper's 10 datasets. Output is one row per (dataset, seed, w) so the
resulting CSV can be melted into the w-sensitivity figure / table.

Per (dataset, seed) the UMAP+KMeans pred_init and the brute-force kNN over
the features are cached once and reused across every `w` value (those parts
of the pipeline do not depend on `w`).

Usage:
  python scripts/ablation_w_sensitivity.py --cache-dir /path/to/clip_cache
  python scripts/ablation_w_sensitivity.py --cache-dir ... --datasets MNIST CIFAR10
  python scripts/ablation_w_sensitivity.py --cache-dir ... --w-grid 0.05 0.16 0.3
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from akss import (
    DATASET_REGISTRY,
    l2_normalize_rows,
    load_dataset,
    run_uk_chain,
    umap_kmeans_init,
)
from akss.sr_umap import exact_knn


DEFAULT_W_GRID = [0.05, 0.1, 0.15, 0.2, 0.3]

METRIC_COLUMNS = [
    "uk_akss_ari", "uk_akss_nmi", "uk_akss_acc",
    "sr_v1_ari", "sr_v1_nmi", "sr_v1_acc",
    "sr_v2_ari", "sr_v2_nmi", "sr_v2_acc",
    "isr_v2_ari", "isr_v2_nmi", "isr_v2_acc",
]

FIELDNAMES = [
    "dataset", "K", "seed", "bic_weight", "graph_nn", "q",
    *METRIC_COLUMNS,
    "time_sec", "error",
]


def parse_args():
    p = argparse.ArgumentParser(description="w-sensitivity ablation (UK-init chain).")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--out", type=str, default="ablation_w_sensitivity.csv")
    p.add_argument("--partial", type=str, default="ablation_w_sensitivity.partial.csv")
    p.add_argument("--datasets", nargs="*", default=list(DATASET_REGISTRY.keys()))
    p.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    p.add_argument("--w-grid", nargs="*", type=float, default=DEFAULT_W_GRID,
                   help="BIC weights to sweep (default: 0.05 0.1 0.15 0.2 0.3).")
    p.add_argument("--graph-nn", type=int, default=15)
    p.add_argument("--q", type=int, default=2)
    p.add_argument("--reselect-cand-factor", type=int, default=2)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    partial_path = Path(args.partial)

    write_header = not partial_path.exists()
    fpart = open(partial_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fpart, fieldnames=FIELDNAMES, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        fpart.flush()

    all_rows: list[dict] = []
    total = len(args.datasets) * len(args.seeds) * len(args.w_grid)
    done = 0

    print(f"[w-sweep] {len(args.datasets)} datasets x {len(args.seeds)} seeds "
          f"x {len(args.w_grid)} weights = {total} runs")
    print(f"[w-sweep] w grid: {args.w_grid}")
    print()

    for dataset in args.datasets:
        try:
            X, y, K = load_dataset(dataset, cache_dir)
        except FileNotFoundError as e:
            print(f"[SKIP] {dataset}: {e}")
            done += len(args.seeds) * len(args.w_grid)
            continue
        Xn = l2_normalize_rows(X.astype(np.float32))
        del X
        gc.collect()
        print(f"[Load] {dataset:<14s} Xn={Xn.shape} K={K}")

        for seed in args.seeds:
            # pred_init, knn_inds_X, knn_inds_cand do NOT depend on w — compute
            # once per (dataset, seed) and reuse across the whole w grid.
            pred_init = umap_kmeans_init(
                Xn, K, random_state=seed,
                n_neighbors=args.graph_nn, n_components=args.q,
            )
            knn_inds_X, _ = exact_knn(Xn, n_neighbors=args.graph_nn)
            knn_inds_cand, _ = exact_knn(
                Xn, n_neighbors=args.reselect_cand_factor * args.graph_nn,
            )

            for w in args.w_grid:
                t0 = time.time()
                row = {
                    "dataset": dataset, "K": K, "seed": int(seed),
                    "bic_weight": float(w), "graph_nn": int(args.graph_nn),
                    "q": int(args.q), "error": "",
                }
                try:
                    row.update(run_uk_chain(
                        dataset, Xn, y, K,
                        seed=seed, bic_weight=float(w),
                        bic_weight_sr=float(w),
                        graph_nn=args.graph_nn, q=args.q,
                        reselect_cand_factor=args.reselect_cand_factor,
                        with_baseline=False,
                        pred_init=pred_init,
                        knn_inds_X=knn_inds_X,
                        knn_inds_cand=knn_inds_cand,
                        verbose=args.verbose,
                    ))
                except Exception as e:  # noqa: BLE001
                    row["error"] = repr(e)
                    for col in METRIC_COLUMNS:
                        row[col] = float("nan")
                    row["time_sec"] = time.time() - t0
                    print(f"[ERR] {dataset} seed={seed} w={w}: {e}")

                writer.writerow(row)
                fpart.flush()
                all_rows.append(row)
                done += 1

                print(f"  [{done:>4d}/{total}] {dataset:<14s} seed={seed} w={w:<5g}  "
                      f"uk={row['uk_akss_ari']:.3f}  sr_v2={row['sr_v2_ari']:.3f}  "
                      f"isr={row['isr_v2_ari']:.3f}  t={row['time_sec']:.1f}s")

            del pred_init, knn_inds_X, knn_inds_cand
            gc.collect()

        del Xn
        gc.collect()

    fpart.close()
    pd.DataFrame(all_rows).to_csv(args.out, index=False)
    print(f"\n[Done] Final CSV: {args.out}")


if __name__ == "__main__":
    main()
