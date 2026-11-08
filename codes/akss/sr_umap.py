"""Subspace-Refined UMAP (SR-UMAP) and Iterative SR-UMAP (ISR-UMAP).

Implements Algorithm 3 of the paper: AKSS provides per-cluster bases that are
fed back into the UMAP graph either as edge re-weights (V1) or as kNN
re-selection (V2). The ISR variant iterates the SR-UMAP -> KMeans -> refit
loop for T rounds.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import randomized_svd

from .akss import k_subspaces_adaptive, l2_normalize_rows
from .eval import eval_clustering
# Use the manual UMAP graph builder so that fuzzy-graph construction matches
# the numbers reported in the paper (umap-learn's fuzzy_simplicial_set has
# slightly different sigma/rho rounding and would shift results).
from .umap_manual import fuzzy_simplicial_set


def exact_knn(
    X: np.ndarray,
    n_neighbors: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force kNN with self as the first neighbor."""
    nn = NearestNeighbors(
        n_neighbors=n_neighbors, metric=metric, algorithm="auto", n_jobs=-1
    )
    nn.fit(X)
    dists, inds = nn.kneighbors(X, return_distance=True)
    return inds.astype(np.int64), dists.astype(np.float32)


# ----------------------------------------------------------------------------
# Subspace-aware kNN distances (V1 = reweight, V2 = reselect)
# ----------------------------------------------------------------------------

def subspace_reweighted_knn_symmetric(
    X: np.ndarray,
    pred: np.ndarray,
    model: list[dict],
    knn_inds_X: np.ndarray,
    n_neighbors: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """SR-UMAP V1: keep the original-space kNN topology, replace edge weights
    with the symmetric subspace-residual distance

        d_sym(i, j) = 0.5 * (||(I - U_{k_i} U_{k_i}^T)(x_i - x_j)||
                            + ||(I - U_{k_j} U_{k_j}^T)(x_i - x_j)||).
    """
    X = X.astype(np.float32, copy=False)
    pred = pred.astype(np.int64, copy=False)
    n = X.shape[0]
    assert knn_inds_X.shape == (n, n_neighbors)

    K = len(model)
    U_list = [m["B"][:, : int(m["r"])].astype(np.float32, copy=False) for m in model]

    if verbose:
        ranks = np.array([int(m["r"]) for m in model])
        print(f"[SR-V1] ranks(min/med/max)={ranks.min()}/{int(np.median(ranks))}/{ranks.max()} "
              f"neighbors={n_neighbors}")

    knn_dists = np.empty((n, n_neighbors), dtype=np.float32)
    for i in range(n):
        nbrs = knn_inds_X[i]
        diff = X[nbrs] - X[i][None, :]
        diff2 = np.einsum("ij,ij->i", diff, diff).astype(np.float32)

        Ui = U_list[int(pred[i])]
        proj_i = diff @ Ui
        res_i2 = np.maximum(diff2 - np.einsum("ij,ij->i", proj_i, proj_i), 0.0)

        res_j2 = np.empty_like(res_i2)
        kj = pred[nbrs]
        for k in np.unique(kj):
            sel = np.where(kj == k)[0]
            pj = diff[sel] @ U_list[int(k)]
            res_j2[sel] = np.maximum(diff2[sel] - np.einsum("ij,ij->i", pj, pj), 0.0)

        knn_dists[i] = 0.5 * (np.sqrt(res_i2) + np.sqrt(res_j2)).astype(np.float32)

    return knn_inds_X.astype(np.int64), knn_dists


def reselect_knn_by_symmetric_residual(
    X: np.ndarray,
    pred: np.ndarray,
    model: list[dict],
    m: int,
    cand_factor: int = 2,
    knn_inds_cand: np.ndarray | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """SR-UMAP V2: take a candidate pool of size `cand_factor * m` from
    original-space kNN, then keep the top-`m` by subspace-residual distance.

    Pass a precomputed `knn_inds_cand` (shape `(n, >=cand_factor*m)`) to skip
    the brute-force kNN — useful when sweeping `w` or running ISR's outer
    iterations, where the candidate set depends only on `X`."""
    X = X.astype(np.float32, copy=False)
    pred = pred.astype(np.int64, copy=False)
    n, _ = X.shape
    cand = max(int(cand_factor * m), m)

    if knn_inds_cand is None:
        knn_inds_cand, _ = exact_knn(X, n_neighbors=cand)
    elif knn_inds_cand.shape[1] < cand:
        raise ValueError(
            f"knn_inds_cand has {knn_inds_cand.shape[1]} neighbours; need >= {cand}"
        )
    else:
        knn_inds_cand = knn_inds_cand[:, :cand]

    U_list = [m_["B"][:, : int(m_["r"])].astype(np.float32, copy=False) for m_ in model]

    if verbose:
        ranks = np.array([int(m_["r"]) for m_ in model])
        print(f"[SR-V2] cand={cand} -> m={m}  ranks(min/med/max)="
              f"{ranks.min()}/{int(np.median(ranks))}/{ranks.max()}")

    knn_inds_out = np.empty((n, m), dtype=np.int64)
    knn_dists_out = np.empty((n, m), dtype=np.float32)

    for i in range(n):
        nbrs = knn_inds_cand[i]
        diff = X[nbrs] - X[i][None, :]
        diff2 = np.einsum("ij,ij->i", diff, diff).astype(np.float32)

        Ui = U_list[int(pred[i])]
        proj_i = diff @ Ui
        res_i2 = np.maximum(diff2 - np.einsum("ij,ij->i", proj_i, proj_i), 0.0)

        res_j2 = np.empty_like(res_i2)
        kj = pred[nbrs]
        for k in np.unique(kj):
            sel = np.where(kj == k)[0]
            pj = diff[sel] @ U_list[int(k)]
            res_j2[sel] = np.maximum(diff2[sel] - np.einsum("ij,ij->i", pj, pj), 0.0)

        d_sym = 0.5 * (np.sqrt(res_i2) + np.sqrt(res_j2))

        sel = np.argpartition(d_sym, m - 1)[:m]
        sel = sel[np.argsort(d_sym[sel])]
        knn_inds_out[i] = nbrs[sel]
        knn_dists_out[i] = d_sym[sel].astype(np.float32)

    return knn_inds_out, knn_dists_out


# ----------------------------------------------------------------------------
# Graph -> UMAP layout -> KMeans
# ----------------------------------------------------------------------------

def umap_from_graph(
    Xn: np.ndarray,
    graph: sp.spmatrix,
    n_components: int = 2,
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_epochs: int = 200,
    init: str = "spectral",
    random_state: int = 0,
    negative_sample_rate: int = 5,
    gamma: float = 1.0,
    verbose: bool = False,
) -> np.ndarray:
    """Run UMAP layout optimization on a custom fuzzy graph."""
    from umap.umap_ import find_ab_params, simplicial_set_embedding

    graph = graph.tocsr().astype(np.float32)
    graph.eliminate_zeros()

    a, b = find_ab_params(spread=spread, min_dist=min_dist)
    rs = np.random.RandomState(random_state)

    out = simplicial_set_embedding(
        data=Xn.astype(np.float32, copy=False),
        graph=graph,
        n_components=int(n_components),
        initial_alpha=1.0,
        a=a, b=b,
        gamma=float(gamma),
        negative_sample_rate=int(negative_sample_rate),
        n_epochs=int(n_epochs),
        init=init,
        random_state=rs,
        metric="euclidean",
        metric_kwds={},
        densmap=False,
        densmap_kwds={},
        output_dens=False,
        verbose=verbose,
    )
    return out[0] if isinstance(out, tuple) else out


def umap_kmeans_on_graph(
    Xn: np.ndarray,
    y: np.ndarray,
    K: int,
    graph: sp.spmatrix,
    q: int = 2,
    umap_n_epochs: int = 200,
    umap_min_dist: float = 0.1,
    umap_spread: float = 1.0,
    negative_sample_rate: int = 5,
    random_state: int = 0,
    verbose_tag: str = "",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """UMAP-on-graph + KMeans, returning (embedding, pred, metrics)."""
    t0 = time.time()
    Y = umap_from_graph(
        Xn, graph,
        n_components=q,
        min_dist=umap_min_dist,
        spread=umap_spread,
        n_epochs=umap_n_epochs,
        init="spectral",
        random_state=random_state,
        negative_sample_rate=negative_sample_rate,
        gamma=1.0,
    )
    if verbose and verbose_tag:
        print(f"[UMAP {verbose_tag}] q={q} epochs={umap_n_epochs} "
              f"time={(time.time()-t0)/60:.2f} min")
    pred = KMeans(n_clusters=K, n_init=10, random_state=random_state).fit_predict(Y)
    met = eval_clustering(y, pred)
    if verbose and verbose_tag:
        print(f"[KMeans {verbose_tag}] ARI={met['ari']:.4f} NMI={met['nmi']:.4f} ACC={met['acc']:.4f}")
    return Y, pred, met


# ----------------------------------------------------------------------------
# Graph builders (V1 / V2) bundled
# ----------------------------------------------------------------------------

def build_graphs_from_model(
    Xn: np.ndarray,
    pred: np.ndarray,
    model: list[dict],
    graph_n_neighbors: int,
    compare_reselect: bool = True,
    reselect_cand_factor: int = 2,
    knn_inds_X: np.ndarray | None = None,
    knn_inds_cand: np.ndarray | None = None,
    random_state: int = 0,
    verbose: bool = False,
) -> tuple[sp.spmatrix, Optional[sp.spmatrix]]:
    """Pass precomputed `knn_inds_X` / `knn_inds_cand` to skip the inner
    `exact_knn` calls; both depend only on `Xn` and are reused across ISR
    iterations and parameter sweeps."""
    if knn_inds_X is None:
        knn_inds_X, _ = exact_knn(Xn, n_neighbors=graph_n_neighbors)
    inds_v1, dists_v1 = subspace_reweighted_knn_symmetric(
        Xn, pred, model, knn_inds_X=knn_inds_X,
        n_neighbors=graph_n_neighbors, verbose=verbose,
    )
    graph_v1, _, _ = fuzzy_simplicial_set(
        inds_v1, dists_v1, n_neighbors=graph_n_neighbors,
        local_connectivity=1.0, set_op_mix_ratio=1.0,
        apply_set_operations=True,
    )

    graph_v2 = None
    if compare_reselect:
        inds_v2, dists_v2 = reselect_knn_by_symmetric_residual(
            Xn, pred, model, m=graph_n_neighbors,
            cand_factor=reselect_cand_factor,
            knn_inds_cand=knn_inds_cand, verbose=verbose,
        )
        graph_v2, _, _ = fuzzy_simplicial_set(
            inds_v2, dists_v2, n_neighbors=graph_n_neighbors,
            local_connectivity=1.0, set_op_mix_ratio=1.0,
            apply_set_operations=True,
        )
    return graph_v1, graph_v2


def fit_subspaces_from_labels(
    X: np.ndarray,
    labels: np.ndarray,
    K: int,
    prev_model: list[dict] | None = None,
    random_state: int = 0,
) -> list[dict]:
    """Fast PCA-only M-step (no rank reselection). Reuses ranks from prev_model
    when available; otherwise defaults to r=16."""
    X = X.astype(np.float32, copy=False)
    labels = labels.astype(np.int64, copy=False)
    n, d = X.shape
    model = []
    rs = int(random_state)

    for k in range(K):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            if prev_model is not None and k < len(prev_model):
                model.append(prev_model[k])
            else:
                mu = X[np.random.default_rng(rs + k).integers(0, n)].astype(np.float32)
                r = min(8, d)
                A = np.random.default_rng(rs + 1337 + k).normal(size=(d, r)).astype(np.float32)
                Q, _ = np.linalg.qr(A)
                model.append({"mu": mu, "B": Q[:, :r].astype(np.float32), "r": int(r)})
            continue

        mu = X[idx].mean(axis=0).astype(np.float32)
        Xc = X[idx] - mu[None, :]

        r = int(prev_model[k].get("r", 16)) if (prev_model is not None and k < len(prev_model)) else 16
        r = max(1, min(r, d, idx.size - 1))

        try:
            _, _, Vt = randomized_svd(Xc, n_components=r, random_state=rs + k)
            B = Vt.T.astype(np.float32, copy=False)
        except Exception:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            B = Vt[:r].T.astype(np.float32, copy=False)

        model.append({"mu": mu, "B": B, "r": int(r)})

    return model


def ensure_contiguous_cluster_labels(
    pred: np.ndarray, K_hint: int, verbose: bool = False,
) -> tuple[np.ndarray, int]:
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    if pred.size == 0:
        return pred, K_hint
    if pred.min() >= 0 and pred.max() < K_hint:
        return pred, K_hint
    if verbose:
        print(f"[labels] remap cluster ids -> 0..K'-1 "
              f"(pred min/max={pred.min()}/{pred.max()}, K_hint={K_hint})")
    le = LabelEncoder()
    pred_new = le.fit_transform(pred).astype(np.int64)
    return pred_new, int(len(le.classes_))


# ----------------------------------------------------------------------------
# Iterative SR-UMAP (Algorithm 3, outer loop)
# ----------------------------------------------------------------------------

def isr_umap(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    outer_T: int = 2,
    update_variant: str = "V2",
    pred_init: np.ndarray | None = None,
    model_init: list | None = None,
    kss_max_iter: int = 25,
    rank_mode: str = "bic",
    r_min: int = 4,
    r_max: int = 128,
    bic_weight: float = 0.16,
    delta: float = 0.95,
    graph_n_neighbors: int = 15,
    compare_reselect: bool = True,
    reselect_cand_factor: int = 2,
    knn_inds_X: np.ndarray | None = None,
    knn_inds_cand: np.ndarray | None = None,
    q: int = 2,
    umap_n_epochs: int = 200,
    umap_min_dist: float = 0.1,
    umap_spread: float = 1.0,
    negative_sample_rate: int = 5,
    random_state: int = 0,
    verbose: bool = False,
):
    """Outer iteration: build SR graphs -> UMAP -> KMeans -> refit subspaces.

    Pass precomputed `knn_inds_X` / `knn_inds_cand` to skip the inner
    `exact_knn` calls inside `build_graphs_from_model` — these depend only on
    `X` and are reused across every outer iteration."""
    if verbose:
        print(f"[ISR] {name} n={X.shape[0]} d={X.shape[1]} K={K} T={outer_T} update={update_variant}")

    Xn = l2_normalize_rows(X.astype(np.float32))
    y = np.asarray(y, dtype=np.int64)

    if pred_init is not None and model_init is not None:
        pred, model = np.asarray(pred_init, dtype=np.int64, order="C"), model_init
    else:
        pred, model, _ = k_subspaces_adaptive(
            Xn, K, all_labels=y, r_min=r_min, r_max=r_max,
            max_iter=kss_max_iter, rank_mode=rank_mode,
            bic_weight=bic_weight, delta=delta,
            random_state=random_state, init="kmeans",
            l2_normalize=False, verbose=verbose,
        )

    if verbose:
        met0 = eval_clustering(y, pred)
        print(f"[ISR init] ARI={met0['ari']:.4f} NMI={met0['nmi']:.4f} ACC={met0['acc']:.4f}")

    for t in range(1, outer_T + 1):
        g1, g2 = build_graphs_from_model(
            Xn, pred, model,
            graph_n_neighbors=graph_n_neighbors,
            compare_reselect=compare_reselect,
            reselect_cand_factor=reselect_cand_factor,
            knn_inds_X=knn_inds_X, knn_inds_cand=knn_inds_cand,
            random_state=random_state, verbose=verbose,
        )

        _, pred_v1, met_v1 = umap_kmeans_on_graph(
            Xn, y, K, g1, q=q,
            umap_n_epochs=umap_n_epochs,
            umap_min_dist=umap_min_dist, umap_spread=umap_spread,
            negative_sample_rate=negative_sample_rate,
            random_state=random_state,
            verbose_tag=f"outer{t}/V1" if verbose else "",
            verbose=verbose,
        )

        pred_v2, met_v2 = None, None
        if g2 is not None:
            _, pred_v2, met_v2 = umap_kmeans_on_graph(
                Xn, y, K, g2, q=q,
                umap_n_epochs=umap_n_epochs,
                umap_min_dist=umap_min_dist, umap_spread=umap_spread,
                negative_sample_rate=negative_sample_rate,
                random_state=random_state,
                verbose_tag=f"outer{t}/V2" if verbose else "",
                verbose=verbose,
            )

        if update_variant.upper() == "V1":
            pred_new, chosen = pred_v1, met_v1
        else:
            pred_new = pred_v2 if pred_v2 is not None else pred_v1
            chosen = met_v2 if met_v2 is not None else met_v1

        model = fit_subspaces_from_labels(Xn, pred_new, K, prev_model=model, random_state=random_state)
        pred = pred_new
        if verbose:
            print(f"[ISR t={t}] -> chosen={update_variant} ARI={chosen['ari']:.4f} "
                  f"NMI={chosen['nmi']:.4f} ACC={chosen['acc']:.4f}")

    return pred, model


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------

def umap_kmeans_baseline(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    n_neighbors: int = 15,
    n_components: int = 2,
    n_epochs: int = 200,
    random_state: int = 0,
    metric: str = "euclidean",
    verbose: bool = False,
) -> dict:
    """Standard UMAP + KMeans baseline (umap-learn end-to-end)."""
    import umap
    Xn = l2_normalize_rows(X.astype(np.float32))
    y = np.asarray(y, dtype=int)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, n_components=n_components,
        n_epochs=n_epochs, random_state=random_state, metric=metric,
        verbose=verbose,
    )
    Y = reducer.fit_transform(Xn)
    pred = KMeans(n_clusters=K, n_init=10, random_state=random_state).fit_predict(Y)
    met = eval_clustering(y, pred)
    if verbose:
        print(f"[Baseline UMAP+KMeans {name}] "
              f"ARI={met['ari']:.4f} NMI={met['nmi']:.4f} ACC={met['acc']:.4f}")
    return {"Y": Y, "pred": pred, "metrics": met}


# ----------------------------------------------------------------------------
# One-pass SR-UMAP (single round, V1 + V2)
# ----------------------------------------------------------------------------

def sr_umap_from_external_akss(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    pred_akss: np.ndarray,
    model: list,
    *,
    random_state: int = 0,
    graph_n_neighbors: int = 15,
    q: int = 2,
    umap_n_epochs: int = 200,
    umap_min_dist: float = 0.1,
    umap_spread: float = 1.0,
    negative_sample_rate: int = 5,
    compare_reselect: bool = True,
    reselect_cand_factor: int = 2,
    knn_inds_X: np.ndarray | None = None,
    knn_inds_cand: np.ndarray | None = None,
    verbose: bool = False,
) -> dict:
    """SR-UMAP V1 (+ optional V2) starting from precomputed AKSS (pred, model).

    Always returns a dict with keys ``"V1"`` and ``"V2"``. Each value is either
    ``{"Y","pred","metrics"}`` or ``None`` (``"V2"`` is ``None`` when
    ``compare_reselect=False``).
    """
    Xn = l2_normalize_rows(X.astype(np.float32))
    y = np.asarray(y, dtype=int)
    pred_akss = np.asarray(pred_akss, dtype=np.int64)

    pred_akss, K = ensure_contiguous_cluster_labels(pred_akss, K, verbose=verbose)
    if verbose:
        print(f"[SR-UMAP] {name} n={X.shape[0]} d={X.shape[1]} K={K} "
              f"graph_nn={graph_n_neighbors} q={q}")

    g1, g2 = build_graphs_from_model(
        Xn, pred_akss, model,
        graph_n_neighbors=graph_n_neighbors,
        compare_reselect=compare_reselect,
        reselect_cand_factor=reselect_cand_factor,
        knn_inds_X=knn_inds_X, knn_inds_cand=knn_inds_cand,
        random_state=random_state, verbose=verbose,
    )

    out: dict = {}
    Y1, pred1, met1 = umap_kmeans_on_graph(
        Xn, y, K, g1, q=q,
        umap_n_epochs=umap_n_epochs,
        umap_min_dist=umap_min_dist, umap_spread=umap_spread,
        negative_sample_rate=negative_sample_rate,
        random_state=random_state,
        verbose_tag="V1" if verbose else "",
        verbose=verbose,
    )
    out["V1"] = {"Y": Y1, "pred": pred1, "metrics": met1}

    if g2 is None:
        out["V2"] = None
    else:
        Y2, pred2, met2 = umap_kmeans_on_graph(
            Xn, y, K, g2, q=q,
            umap_n_epochs=umap_n_epochs,
            umap_min_dist=umap_min_dist, umap_spread=umap_spread,
            negative_sample_rate=negative_sample_rate,
            random_state=random_state,
            verbose_tag="V2" if verbose else "",
            verbose=verbose,
        )
        out["V2"] = {"Y": Y2, "pred": pred2, "metrics": met2}

    return out
