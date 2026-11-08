"""Adaptive K-Subspaces (AKSS) — Algorithm 1 of the paper.

Hard-assignment K-subspaces with per-cluster PCA where the rank of each
subspace is chosen adaptively, either by a spectral-energy heuristic or by
a BIC-style criterion under a signal-plus-isotropic-noise covariance (where latter is the one in paper).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd

from .eval import eval_clustering


def _to_numpy(X):
    if hasattr(X, "detach"):
        return X.detach().cpu().numpy()
    return np.asarray(X)


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (nrm + eps)


# ----------------------------------------------------------------------------
# Initialization helpers
# ----------------------------------------------------------------------------

def _pred_init_from_external(pred_init: np.ndarray, n: int, K: int) -> np.ndarray:
    pred_ext = np.asarray(pred_init, dtype=int).reshape(-1)
    if pred_ext.shape[0] != n:
        raise ValueError(f"pred_init must have length n={n}, got {pred_ext.shape[0]}")
    u = np.unique(pred_ext)
    if u.size != K:
        raise ValueError(
            f"pred_init must contain exactly K={K} distinct labels, got {u.size}"
        )
    if u.min() >= 0 and u.max() == K - 1:
        return pred_ext.copy()
    idx_map = {int(v): i for i, v in enumerate(np.sort(u))}
    return np.array([idx_map[int(v)] for v in pred_ext], dtype=int)


def umap_kmeans_init(
    X: np.ndarray,
    K: int,
    random_state: int = 0,
    n_neighbors: int = 15,
    n_components: int = 2,
    n_epochs: int = 200,
) -> np.ndarray:
    """UMAP + KMeans on the embedding — produces the UK-init labels."""
    import umap  # local import keeps the eval-only path light

    emb = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        n_epochs=n_epochs,
        random_state=random_state,
    ).fit_transform(X)
    km = KMeans(n_clusters=K, n_init=10, random_state=random_state)
    return km.fit_predict(emb)


# ----------------------------------------------------------------------------
# Rank selection
# ----------------------------------------------------------------------------

def select_rank_trace_stop(
    eigvals: np.ndarray,
    m_full: int,
    r_min: int = 2,
    r_max: int = 64,
    delta: float = 0.95,
    eps: float = 1e-12,
) -> int:
    """Spectral-gap heuristic: pick the smallest r whose log-eigenvalue gap
    exceeds the `delta`-quantile of all gaps."""
    eigvals = np.asarray(eigvals, dtype=np.float64)
    m_obs = eigvals.size
    if m_obs == 0:
        return int(r_min)

    r_max_eff = min(r_max, m_obs)
    r_min_eff = max(r_min, 1)

    lam = eigvals[:r_max_eff]
    gaps = np.log(np.maximum(lam[:-1], eps)) - np.log(np.maximum(lam[1:], eps))

    missing = max((m_full - 1) - gaps.size, 0)
    gaps_pad = np.concatenate([gaps, np.zeros(missing)]) if missing > 0 else gaps
    tau = float(np.quantile(gaps_pad, delta))
    if tau <= eps:
        tau = float(np.quantile(gaps, delta))
        if tau <= eps:
            return int(r_min_eff)

    idx = np.where(gaps >= tau)[0]
    if idx.size == 0:
        return int(r_min_eff)

    r = int(idx[-1]) + 1
    return int(min(r_max_eff, max(r_min_eff, r)))


def select_rank_bic(
    eigvals: np.ndarray,
    n_k: int,
    p_dim: int,
    r_min: int = 2,
    r_max: int = 64,
    bic_weight: float = 0.16,
    eps: float = 1e-12,
) -> int:
    """BIC-style rank selection under a (subspace + isotropic noise) covariance
    model. `bic_weight` is the `w` knob in the paper (default matches the
    paper's fixed-w setting)."""
    eigvals = np.asarray(eigvals, dtype=np.float64)
    m = eigvals.size
    if m == 0:
        return int(r_min)

    r_max_eff = min(r_max, m - 1)
    r_min_eff = min(max(r_min, 1), r_max_eff)

    log_eigs = np.log(np.maximum(eigvals, eps))
    prefix_log = np.cumsum(log_eigs)
    tail_sum = np.cumsum(eigvals[::-1])[::-1]

    best_r, best_val = r_min_eff, np.inf
    for r in range(r_min_eff, r_max_eff + 1):
        tail_cnt = m - r
        b_r = max(tail_sum[r] / max(tail_cnt, 1), eps)
        term_logdet = prefix_log[r - 1] + tail_cnt * np.log(b_r)
        df = p_dim * r - (r * (r + 1)) / 2.0 + r + 1.0
        bic_pen = bic_weight * df * np.log(max(n_k, 2))
        crit = n_k * term_logdet + bic_pen
        if crit < best_val:
            best_val, best_r = crit, r
    return int(best_r)


# ----------------------------------------------------------------------------
# Per-cluster PCA
# ----------------------------------------------------------------------------

def fit_cluster_subspace(
    Xk: np.ndarray,
    rank_mode: str = "bic",
    r_min: int = 2,
    r_max: int = 128,
    delta: float = 0.95,
    bic_weight: float = 0.16,
    r_max_global: int | None = None,
    svd_method: str = "randomized",
    random_state: int = 0,
):
    """Fit (mu, B, r, eigvals) for a single cluster via PCA + chosen rank rule."""
    n_k, p = Xk.shape
    mu = Xk.mean(axis=0)
    Xc = Xk - mu

    if n_k <= 2:
        r = min(r_min, p)
        B = np.zeros((p, r), dtype=np.float32)
        B[:r, :] = np.eye(r, dtype=np.float32)
        return mu.astype(np.float32), B, int(r), np.zeros(r, dtype=np.float32)

    m = min(n_k - 1, p)
    need = max(r_max, r_max_global or r_max)
    n_comp = min(need, m)

    if svd_method == "full" or n_comp == m:
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    else:
        _, S, Vt = randomized_svd(Xc, n_components=n_comp, random_state=random_state, n_iter=3)

    eigvals = np.sort((S ** 2) / max(n_k - 1, 1))[::-1]

    if rank_mode == "heuristic":
        r = select_rank_trace_stop(eigvals, m_full=m, r_min=r_min, r_max=r_max, delta=delta)
    elif rank_mode == "bic":
        r = select_rank_bic(
            eigvals, n_k=n_k, p_dim=p, r_min=r_min,
            r_max=min(r_max, eigvals.size), bic_weight=bic_weight,
        )
    else:
        raise ValueError(f"Unknown rank_mode: {rank_mode}")

    r = int(max(r_min, r))
    B = Vt[:r].T.astype(np.float32)
    return mu.astype(np.float32), B, r, eigvals.astype(np.float32)


# ----------------------------------------------------------------------------
# Point-to-subspace distance
# ----------------------------------------------------------------------------

def subspace_residual_distance(
    X: np.ndarray,
    mu: np.ndarray,
    B: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    """Squared distance from each row of X to the affine subspace (mu, span(B))."""
    out = np.empty(X.shape[0], dtype=np.float32)
    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        Xc = X[start:end] - mu
        tot = np.sum(Xc * Xc, axis=1)
        if B is None or B.size == 0:
            out[start:end] = tot
            continue
        proj = Xc @ B
        out[start:end] = tot - np.sum(proj * proj, axis=1)
    return out


# ----------------------------------------------------------------------------
# AKSS main loop
# ----------------------------------------------------------------------------

def k_subspaces_adaptive(
    all_features,
    K: int,
    all_labels=None,
    max_iter: int = 20,
    init: str = "umap_kmeans",  # 'umap_kmeans' | 'kmeans' | 'random' | 'external'
    pred_init: np.ndarray | None = None,
    random_state: int = 0,
    l2_normalize: bool = True,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 2,
    umap_n_epochs: int = 200,
    rank_mode: str = "bic",
    r_min: int = 4,
    r_max: int = 128,
    delta: float = 0.95,
    bic_weight: float = 0.16,
    svd_method: str = "randomized",
    verbose: bool = False,
):
    """Adaptive K-Subspaces.

    Init modes:
      umap_kmeans : UMAP(2D) + KMeans (the "UK-init" variant; default — this is
                    the headline configuration used in the paper)
      kmeans      : sklearn.KMeans on the features (classical KSS-style init)
      random      : uniform random labels
      external    : caller supplies `pred_init` of length n with exactly K labels
    """
    X = _to_numpy(all_features).astype(np.float32)
    n, p = X.shape

    if l2_normalize:
        X = l2_normalize_rows(X)

    y_true = _to_numpy(all_labels).astype(int) if all_labels is not None else None

    if init == "kmeans":
        pred = KMeans(n_clusters=K, n_init=10, random_state=random_state).fit_predict(X)
    elif init == "random":
        rng = np.random.default_rng(random_state)
        pred = rng.integers(0, K, size=n)
    elif init == "external":
        if pred_init is None:
            raise ValueError('init="external" requires pred_init.')
        pred = _pred_init_from_external(pred_init, n, K)
    elif init == "umap_kmeans":
        pred = umap_kmeans_init(
            X, K, random_state=random_state,
            n_neighbors=umap_n_neighbors,
            n_components=umap_n_components,
            n_epochs=umap_n_epochs,
        )
    else:
        raise ValueError(f"Unknown init: {init!r}")

    history = {"obj": [], "ranks": [], "ari": [], "nmi": [], "acc": []}

    for it in range(max_iter):
        model = []
        ranks = []
        for k in range(K):
            idx = np.where(pred == k)[0]
            if idx.size == 0:
                idx = np.array([np.random.randint(0, n)], dtype=int)
            mu_k, B_k, r_k, eig_k = fit_cluster_subspace(
                X[idx], rank_mode=rank_mode, r_min=r_min, r_max=r_max,
                delta=delta, bic_weight=bic_weight, r_max_global=r_max,
                svd_method=svd_method,
                random_state=random_state + 37 * (it + 1) + k,
            )
            model.append({"mu": mu_k, "B": B_k, "r": r_k, "eigvals": eig_k})
            ranks.append(r_k)

        # E-step
        D = np.zeros((n, K), dtype=np.float32)
        for k in range(K):
            D[:, k] = subspace_residual_distance(X, model[k]["mu"], model[k]["B"])

        pred_new = D.argmin(axis=1)

        # Repair empty clusters (donate one farthest point from the largest cluster)
        nk = np.bincount(pred_new, minlength=K)
        for k_empty in np.where(nk == 0)[0]:
            nk_sorted = np.argsort(-nk)
            try:
                k_donor = next(int(kk) for kk in nk_sorted if nk[kk] > 1)
            except StopIteration:
                k_donor = int(nk_sorted[0])
            idx_donor = np.where(pred_new == k_donor)[0]
            i_move = idx_donor[int(np.argmax(D[idx_donor, k_donor]))]
            pred_new[i_move] = k_empty
            nk[k_donor] -= 1
            nk[k_empty] = 1

        obj = float(np.sum(np.min(D, axis=1)))
        history["obj"].append(obj)
        history["ranks"].append(ranks)
        if y_true is not None:
            met = eval_clustering(y_true, pred_new)
            history["ari"].append(met["ari"])
            history["nmi"].append(met["nmi"])
            history["acc"].append(met["acc"])

        if verbose:
            msg = (f"[AKSS iter {it+1:02d}] obj={obj:.4e} "
                   f"ranks(min/med/max)={int(np.min(ranks))}/{int(np.median(ranks))}/{int(np.max(ranks))}")
            if y_true is not None:
                msg += (f"  ARI={history['ari'][-1]:.4f} "
                        f"NMI={history['nmi'][-1]:.4f} "
                        f"ACC={history['acc'][-1]:.4f}")
            print(msg)

        changed = float(np.mean(pred_new != pred))
        pred = pred_new
        if changed < 1e-4:
            if verbose:
                print(f"AKSS converged at iter {it+1} (changed={changed:.6f}).")
            break

    return pred, model, history
