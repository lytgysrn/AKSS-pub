"""Reproduce the main 10-dataset x 10-seed table from the paper.

Story-line chain (UK-init only): for every (dataset, seed):
  1. UMAP + KMeans baseline (umap-learn, n_components=2)
  2. UK-AKSS (UMAP+KMeans init -> AKSS in CLIP feature space)
  3. SR-UMAP V1 / V2 (single-pass subspace-refined UMAP + KMeans)
  4. ISR-UMAP V2 (outer-iter T=2)

KMeans-init AKSS and joint UMAP-driven AKSS (Algorithm 2) are *not* part of
the headline chain and are excluded here (see scripts/example_single_dataset.py
for an end-to-end demo that runs them).

`w` mode (BIC weight):
  default          : fixed w = akss.FIXED_BIC_WEIGHT (= 0.16)
  --best-w         : per-dataset best from akss.BEST_BIC_WEIGHTS_UK
  --bic-weight W   : force this single value for every dataset

Outputs:
  results.csv : long-format CSV with one row per (dataset, seed)

Usage:
  python scripts/reproduce_10datasets_10seeds.py --cache-dir /path/to/clip_cache
  python scripts/reproduce_10datasets_10seeds.py --cache-dir ... --best-w
  python scripts/reproduce_10datasets_10seeds.py --cache-dir ... --datasets MNIST CIFAR10
  python scripts/reproduce_10datasets_10seeds.py --cache-dir ... --seeds 0 1 2

The full sweep (10 x 10) takes hours on a single laptop GPU. Rows are appended
to results.partial.csv after each (dataset, seed) so a run can be resumed.
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
    BEST_BIC_WEIGHTS_UK,
    DATASET_REGISTRY,
    FIXED_BIC_WEIGHT,
    l2_normalize_rows,
    load_dataset,
    run_uk_chain,
)


METHOD_COLUMNS = [
    "baseline_ari", "baseline_nmi", "baseline_acc",
    "uk_akss_ari", "uk_akss_nmi", "uk_akss_acc",
    "sr_v1_ari", "sr_v1_nmi", "sr_v1_acc",
    "sr_v2_ari", "sr_v2_nmi", "sr_v2_acc",
    "isr_v2_ari", "isr_v2_nmi", "isr_v2_acc",
]

FIELDNAMES = [
    "dataset", "K", "seed", "bic_weight", "graph_nn", "q",
    *METHOD_COLUMNS,
    "time_sec", "error",
]


def parse_args():
    p = argparse.ArgumentParser(description="AKSS 10-dataset x 10-seed reproduction.")
    p.add_argument("--cache-dir", type=str, required=True,
                   help="Directory holding cached .pt feature files.")
    p.add_argument("--out", type=str, default="results.csv",
                   help="Final summary CSV path.")
    p.add_argument("--partial", type=str, default="results.partial.csv",
                   help="Per-row crash-safe CSV (appended).")
    p.add_argument("--datasets", nargs="*", default=list(DATASET_REGISTRY.keys()),
                   help="Subset of datasets to run.")
    p.add_argument("--seeds", nargs="*", type=int, default=list(range(10)),
                   help="Seeds (defaults to 0..9).")
    p.add_argument("--graph-nn", type=int, default=15)
    p.add_argument("--q", type=int, default=2)
    p.add_argument("--bic-weight", type=float, default=None,
                   help="Force this fixed w for every dataset and every method "
                        "(overrides per-dataset best-w).")
    p.add_argument("--fixed-w", action="store_true",
                   help=f"Use fixed w={FIXED_BIC_WEIGHT} for UK-AKSS instead of "
                        "the per-dataset best. SR/ISR always use fixed w.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _resolve_w(dataset: str, args) -> float:
    """BIC weight for UK-AKSS. Default: per-dataset best from paper."""
    if args.bic_weight is not None:
        return float(args.bic_weight)
    if args.fixed_w:
        return float(FIXED_BIC_WEIGHT)
    return float(BEST_BIC_WEIGHTS_UK[dataset])


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    partial_path = Path(args.partial)
    out_path = Path(args.out)

    write_header = not partial_path.exists()
    fpart = open(partial_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fpart, fieldnames=FIELDNAMES, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        fpart.flush()

    all_rows: list[dict] = []
    total = len(args.datasets) * len(args.seeds)
    done = 0

    if args.bic_weight is not None:
        w_mode = f"fixed (forced)={args.bic_weight}"
    elif args.fixed_w:
        w_mode = f"fixed={FIXED_BIC_WEIGHT}"
    else:
        w_mode = "per-dataset best (UK) [default]"

    print(f"[Reproduce] {len(args.datasets)} datasets x {len(args.seeds)} seeds "
          f"= {total} runs")
    print(f"[Reproduce] cache_dir={cache_dir}")
    print(f"[Reproduce] partial={partial_path} final={out_path}")
    print(f"[Reproduce] w mode : {w_mode}")
    print()

    for dataset in args.datasets:
        try:
            X, y, K = load_dataset(dataset, cache_dir)
        except FileNotFoundError as e:
            print(f"[SKIP] {dataset}: {e}")
            continue
        print(f"[Load] {dataset:<14s} X={X.shape} K={K}")
        Xn = l2_normalize_rows(X.astype(np.float32))
        del X
        gc.collect()

        w = _resolve_w(dataset, args)

        for seed in args.seeds:
            t0 = time.time()
            row = {
                "dataset": dataset, "K": K, "seed": int(seed),
                "bic_weight": float(w), "graph_nn": int(args.graph_nn),
                "q": int(args.q), "error": "",
            }
            try:
                row.update(run_uk_chain(
                    dataset, Xn, y, K,
                    seed=seed, bic_weight=w,
                    graph_nn=args.graph_nn, q=args.q,
                    with_baseline=True, verbose=args.verbose,
                ))
            except Exception as e:  # noqa: BLE001 — crash-safe sweep
                row["error"] = repr(e)
                for col in METHOD_COLUMNS:
                    row[col] = float("nan")
                row["time_sec"] = time.time() - t0
                print(f"[ERR] {dataset} seed={seed}: {e}")

            writer.writerow(row)
            fpart.flush()
            all_rows.append(row)
            done += 1

            print(f"  [{done:>3d}/{total}] {dataset:<14s} seed={seed}  "
                  f"base={row['baseline_ari']:.3f}  uk={row['uk_akss_ari']:.3f}  "
                  f"sr_v1={row['sr_v1_ari']:.3f}  sr_v2={row['sr_v2_ari']:.3f}  "
                  f"isr={row['isr_v2_ari']:.3f}  t={row['time_sec']:.1f}s")

        del Xn
        gc.collect()

    fpart.close()
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\n[Done] Final CSV: {out_path}")


if __name__ == "__main__":
    main()
