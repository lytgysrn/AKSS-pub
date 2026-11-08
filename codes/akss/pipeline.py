"""High-level wrapper around the UK-init -> AKSS -> SR-UMAP -> ISR-UMAP chain.

`run_uk_chain` is the single entry point used by all driver scripts
(reproduction, w-sensitivity, q-dim ablation). It computes the brute-force
kNN over `Xn` exactly once and threads it through every SR/ISR step,
which is the dominant cost when sweeping `w` (where the kNN does not
depend on `w` at all) or running ISR's outer iterations.
"""
from __future__ import annotations

import gc
import time

import numpy as np

from .akss import k_subspaces_adaptive, l2_normalize_rows, umap_kmeans_init
from .datasets import FIXED_BIC_WEIGHT
from .eval import eval_clustering
from .sr_umap import (
    exact_knn,
    isr_umap,
    sr_umap_from_external_akss,
    umap_kmeans_baseline,
)


__all__ = ["run_uk_chain"]


def _empty_metrics() -> dict[str, float]:
    return {"ari": float("nan"), "nmi": float("nan"), "acc": float("nan")}


def run_uk_chain(
    dataset: str,
    Xn: np.ndarray,
    y: np.ndarray,
    K: int,
    *,
    seed: int,
    bic_weight: float,
    bic_weight_sr: float | None = None,
    graph_nn: int = 15,
    q: int = 2,
    isr_outer_T: int = 2,
    isr_update_variant: str = "V2",
    r_min: int = 4,
    r_max: int = 128,
    kss_max_iter: int = 20,
    reselect_cand_factor: int = 2,
    with_baseline: bool = True,
    pred_init: np.ndarray | None = None,
    knn_inds_X: np.ndarray | None = None,
    knn_inds_cand: np.ndarray | None = None,
    verbose: bool = False,
) -> dict:
    """Run UMAP+KMeans baseline (optional) -> UK-AKSS -> SR-V1/V2 -> ISR-V2.

    `Xn` must already be L2-row-normalized. `y` is the integer label vector.

    Inputs that can be cached across calls (sweeping `w` over the same
    dataset/seed): `pred_init` (UK-init labels — does not depend on `w`),
    `knn_inds_X` (size `graph_nn`), `knn_inds_cand` (size
    `reselect_cand_factor * graph_nn`). If `None`, they are computed.

    Returns a flat dict with keys
        baseline_{ari,nmi,acc}, uk_akss_{ari,nmi,acc},
        sr_v1_{ari,nmi,acc},   sr_v2_{ari,nmi,acc},
        isr_v2_{ari,nmi,acc},  time_sec
    Methods that did not run get NaN. `time_sec` is the wall time for this call.
    """
    t0 = time.time()
    y_int = np.asarray(y, dtype=int)
    out: dict = {}

    if with_baseline:
        base = umap_kmeans_baseline(
            dataset, Xn, y_int, K,
            n_neighbors=graph_nn, n_components=q,
            random_state=seed, verbose=verbose,
        )
        out["baseline_ari"] = base["metrics"]["ari"]
        out["baseline_nmi"] = base["metrics"]["nmi"]
        out["baseline_acc"] = base["metrics"]["acc"]
        del base
        gc.collect()
    else:
        for tag in ("baseline_ari", "baseline_nmi", "baseline_acc"):
            out[tag] = float("nan")

    if pred_init is None:
        pred_init = umap_kmeans_init(
            Xn, K, random_state=seed,
            n_neighbors=graph_nn, n_components=q,
        )

    pred_uk, model_uk, _ = k_subspaces_adaptive(
        Xn, K, all_labels=y_int,
        init="external", pred_init=pred_init, random_state=seed,
        l2_normalize=False,
        rank_mode="bic", r_min=r_min, r_max=r_max, bic_weight=bic_weight,
        max_iter=kss_max_iter, verbose=verbose,
    )
    met_uk = eval_clustering(y_int, pred_uk)
    out["uk_akss_ari"] = met_uk["ari"]
    out["uk_akss_nmi"] = met_uk["nmi"]
    out["uk_akss_acc"] = met_uk["acc"]

    _w_sr = bic_weight_sr if bic_weight_sr is not None else FIXED_BIC_WEIGHT
    if _w_sr == bic_weight:
        pred_sr, model_sr = pred_uk, model_uk
    else:
        pred_sr, model_sr, _ = k_subspaces_adaptive(
            Xn, K, all_labels=y_int,
            init="external", pred_init=pred_init, random_state=seed,
            l2_normalize=False,
            rank_mode="bic", r_min=r_min, r_max=r_max, bic_weight=_w_sr,
            max_iter=kss_max_iter, verbose=verbose,
        )

    if knn_inds_X is None:
        knn_inds_X, _ = exact_knn(Xn, n_neighbors=graph_nn)
    if knn_inds_cand is None:
        knn_inds_cand, _ = exact_knn(Xn, n_neighbors=reselect_cand_factor * graph_nn)

    sr = sr_umap_from_external_akss(
        dataset, Xn, y_int, K,
        pred_akss=pred_sr, model=model_sr,
        random_state=seed,
        graph_n_neighbors=graph_nn, q=q,
        compare_reselect=True, reselect_cand_factor=reselect_cand_factor,
        knn_inds_X=knn_inds_X, knn_inds_cand=knn_inds_cand,
        verbose=verbose,
    )
    out["sr_v1_ari"] = sr["V1"]["metrics"]["ari"]
    out["sr_v1_nmi"] = sr["V1"]["metrics"]["nmi"]
    out["sr_v1_acc"] = sr["V1"]["metrics"]["acc"]
    sr_v2_met = sr["V2"]["metrics"] if sr["V2"] is not None else _empty_metrics()
    out["sr_v2_ari"] = sr_v2_met["ari"]
    out["sr_v2_nmi"] = sr_v2_met["nmi"]
    out["sr_v2_acc"] = sr_v2_met["acc"]
    del sr
    gc.collect()

    pred_isr, _ = isr_umap(
        dataset, Xn, y_int, K,
        outer_T=isr_outer_T, update_variant=isr_update_variant,
        pred_init=pred_sr, model_init=model_sr,
        graph_n_neighbors=graph_nn, q=q,
        knn_inds_X=knn_inds_X, knn_inds_cand=knn_inds_cand,
        bic_weight=_w_sr, random_state=seed,
        verbose=verbose,
    )
    met_isr = eval_clustering(y_int, pred_isr)
    out["isr_v2_ari"] = met_isr["ari"]
    out["isr_v2_nmi"] = met_isr["nmi"]
    out["isr_v2_acc"] = met_isr["acc"]

    out["time_sec"] = time.time() - t0
    return out
