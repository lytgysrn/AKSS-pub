"""Ablation: UMAP embedding dimensionality `q`.

The headline pipeline uses q=2 (UMAP n_components=2). This ablation re-runs
the UK-init -> SR / ISR chain at multiple q values to show the choice is not
masking a dimensionality advantage. Default grid: q in {2, 128}.

For each (dataset, seed, q) it writes one row with the metrics for
  - UMAP+KMeans baseline (at that q)
  - UK-AKSS
  - SR-UMAP V1 / V2
  - ISR-UMAP V2 (T=2)

The brute-force kNN over feature space is cached once per (dataset, seed)
since it does not depend on q (only `umap_kmeans_init` and the UMAP layout do).

Usage:
  python scripts/ablation_q_dim.py --cache-dir /path/to/clip_cache
  python scripts/ablation_q_dim.py --cache-dir ... --q-grid 2 8 32 128
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
    FIXED_BIC_WEIGHT,
    l2_normalize_rows,
    load_dataset,
    run_uk_chain,
)
from akss.sr_umap import exact_knn


DEFAULT_Q_GRID = [2, 128]

METRIC_COLUMNS = [
    "baseline_ari", "baseline_nmi", "baseline_acc",
    "uk_akss_ari", "uk_akss_nmi", "uk_akss_acc",
    "sr_v1_ari", "sr_v1_nmi", "sr_v1_acc",
    "sr_v2_ari", "sr_v2_nmi", "sr_v2_acc",
    "isr_v2_ari", "isr_v2_nmi", "isr_v2_acc",
]

FIELDNAMES = [
    "dataset", "K", "seed", "q", "bic_weight", "graph_nn",
    *METRIC_COLUMNS,
    "time_sec", "error",
]


def parse_args():
    p = argparse.ArgumentParser(description="q-dim ablation (UK-init chain).")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--out", type=str, default="ablation_q_dim.csv")
    p.add_argument("--partial", type=str, default="ablation_q_dim.partial.csv")
    p.add_argument("--datasets", nargs="*", default=list(DATASET_REGISTRY.keys()))
    p.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    p.add_argument("--q-grid", nargs="*", type=int, default=DEFAULT_Q_GRID,
                   help=f"UMAP n_components values to sweep (default: {DEFAULT_Q_GRID}).")
    p.add_argument("--graph-nn", type=int, default=15)
    p.add_argument("--reselect-cand-factor", type=int, default=2)
    p.add_argument("--bic-weight", type=float, default=FIXED_BIC_WEIGHT,
                   help=f"BIC weight w (default {FIXED_BIC_WEIGHT}).")
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
    total = len(args.datasets) * len(args.seeds) * len(args.q_grid)
    done = 0

    print(f"[q-sweep] {len(args.datasets)} datasets x {len(args.seeds)} seeds "
          f"x {len(args.q_grid)} q values = {total} runs")
    print(f"[q-sweep] q grid: {args.q_grid}, w={args.bic_weight}")
    print()

    for dataset in args.datasets:
        try:
            X, y, K = load_dataset(dataset, cache_dir)
        except FileNotFoundError as e:
            print(f"[SKIP] {dataset}: {e}")
            done += len(args.seeds) * len(args.q_grid)
            continue
        Xn = l2_normalize_rows(X.astype(np.float32))
        del X
        gc.collect()
        print(f"[Load] {dataset:<14s} Xn={Xn.shape} K={K}")

        for seed in args.seeds:
            # kNN over Xn is q-independent; cache it once per (dataset, seed).
            knn_inds_X, _ = exact_knn(Xn, n_neighbors=args.graph_nn)
            knn_inds_cand, _ = exact_knn(
                Xn, n_neighbors=args.reselect_cand_factor * args.graph_nn,
            )

            for q in args.q_grid:
                t0 = time.time()
                row = {
                    "dataset": dataset, "K": K, "seed": int(seed),
                    "q": int(q), "bic_weight": float(args.bic_weight),
                    "graph_nn": int(args.graph_nn), "error": "",
                }
                try:
                    row.update(run_uk_chain(
                        dataset, Xn, y, K,
                        seed=seed, bic_weight=float(args.bic_weight),
                        graph_nn=args.graph_nn, q=int(q),
                        reselect_cand_factor=args.reselect_cand_factor,
                        with_baseline=True,
                        knn_inds_X=knn_inds_X,
                        knn_inds_cand=knn_inds_cand,
                        verbose=args.verbose,
                    ))
                except Exception as e:  # noqa: BLE001
                    row["error"] = repr(e)
                    for col in METRIC_COLUMNS:
                        row[col] = float("nan")
                    row["time_sec"] = time.time() - t0
                    print(f"[ERR] {dataset} seed={seed} q={q}: {e}")

                writer.writerow(row)
                fpart.flush()
                all_rows.append(row)
                done += 1

                print(f"  [{done:>4d}/{total}] {dataset:<14s} seed={seed} q={q:<3d}  "
                      f"base={row['baseline_ari']:.3f}  uk={row['uk_akss_ari']:.3f}  "
                      f"sr_v2={row['sr_v2_ari']:.3f}  isr={row['isr_v2_ari']:.3f}  "
                      f"t={row['time_sec']:.1f}s")

            del knn_inds_X, knn_inds_cand
            gc.collect()

        del Xn
        gc.collect()

    fpart.close()
    pd.DataFrame(all_rows).to_csv(args.out, index=False)
    print(f"\n[Done] Final CSV: {args.out}")


if __name__ == "__main__":
    main()
