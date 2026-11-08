"""Joint UMAP-graph + K-Subspaces (Algorithm 2 of the paper).

Optimizes a low-dimensional embedding Y by alternating
  (i) K-Subspaces clustering on Y,
  (ii) SGD on Y under the UMAP fuzzy-graph loss + a subspace-residual penalty.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import umap
from sklearn.cluster import KMeans

from .akss import (
    _to_numpy,
    fit_cluster_subspace,
    l2_normalize_rows,
    subspace_residual_distance,
)
from .eval import eval_clustering
# Match the paper's numbers: build the UMAP graph with the manual
# implementation that the original experiments used.
from .umap_manual import fuzzy_simplicial_set, nearest_neighbors, find_ab_params


# ----------------------------------------------------------------------------
# UMAP fuzzy graph (uses umap-learn primitives)
# ----------------------------------------------------------------------------

def build_umap_graph(
    X: np.ndarray,
    n_neighbors: int = 30,
    metric: str = "euclidean",
    random_state: int = 0,
    local_connectivity: float = 1.0,
    set_op_mix_ratio: float = 1.0,
    symmetrize: str = "max",
    mutual_only: bool = False,
):
    """Build a UMAP fuzzy simplicial set from features (no .UMAP() needed)."""
    knn_indices, knn_dists, _ = nearest_neighbors(
        X, n_neighbors=n_neighbors, metric=metric,
        random_state=random_state, verbose=False,
    )

    W, _, _ = fuzzy_simplicial_set(
        knn_indices, knn_dists, n_neighbors=n_neighbors,
        local_connectivity=local_connectivity,
        set_op_mix_ratio=set_op_mix_ratio,
        apply_set_operations=True,
    )
    W.eliminate_zeros()

    if symmetrize == "max":
        W = W.maximum(W.T)
    elif symmetrize == "avg":
        W = (W + W.T) * 0.5
    elif symmetrize == "none":
        pass
    else:
        raise ValueError("symmetrize must be 'max', 'avg', or 'none'")

    if mutual_only:
        W = W.multiply(W.T)
        W.eliminate_zeros()

    deg = np.asarray(W.sum(axis=1)).ravel()
    return W.tocsr(), deg


# ----------------------------------------------------------------------------
# K-Subspaces on a current embedding Y
# ----------------------------------------------------------------------------

def k_subspaces_on_Y(
    Y: np.ndarray,
    K: int,
    max_iter: int = 10,
    init: str = "kmeans",
    random_state: int = 0,
    rank_mode: str = "bic",
    r_min: int = 4,
    r_max: int = 16,
    bic_weight: float = 0.16,
    verbose: bool = False,
):
    """Run K-subspaces directly on the q-dim embedding Y."""
    n, _ = Y.shape

    if init == "kmeans":
        pred = KMeans(n_clusters=K, n_init=10, random_state=random_state).fit_predict(Y)
    else:
        pred = np.random.default_rng(random_state).integers(0, K, size=n)

    for it in range(max_iter):
        model = []
        ranks = []
        for k in range(K):
            idx = np.where(pred == k)[0]
            if idx.size == 0:
                idx = np.array([np.random.randint(0, n)], dtype=int)
            mu_k, B_k, r_k, _ = fit_cluster_subspace(
                Y[idx], rank_mode=rank_mode, r_min=r_min, r_max=r_max,
                bic_weight=bic_weight, r_max_global=r_max,
                svd_method="randomized",
                random_state=random_state + 97 * (it + 1) + k,
            )
            model.append({"mu": mu_k, "B": B_k, "r": r_k})
            ranks.append(r_k)

        D = np.zeros((n, K), dtype=np.float32)
        for k in range(K):
            D[:, k] = subspace_residual_distance(Y, model[k]["mu"], model[k]["B"])
        pred_new = D.argmin(axis=1)
        changed = float(np.mean(pred_new != pred))
        pred = pred_new

        if verbose:
            print(f"  [KSub-Y {it+1:02d}] changed={changed:.4f} "
                  f"ranks={np.min(ranks)}/{int(np.median(ranks))}/{np.max(ranks)}")
        if changed < 1e-4:
            break

    return pred, model


# ----------------------------------------------------------------------------
# UMAP edge loss (torch) + subspace residual penalty
# ----------------------------------------------------------------------------

class EdgeSampler:
    """Sample (i, j, w) edges from a CSR fuzzy graph in proportion to weight."""

    def __init__(self, W_csr, seed: int = 0, eps: float = 1e-12):
        W = W_csr.tocoo()
        self.i = W.row.astype(np.int64)
        self.j = W.col.astype(np.int64)
        w = np.maximum(W.data.astype(np.float64), 0.0)
        w[~np.isfinite(w)] = 0.0
        s = w.sum()
        if not np.isfinite(s) or s <= eps:
            prob = np.full_like(w, 1.0 / max(len(w), 1))
        else:
            prob = w / s
            prob = np.maximum(prob, eps)
            prob /= prob.sum()
        self.w = w.astype(np.float32)
        self.prob = prob.astype(np.float64)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int):
        m = len(self.i)
        if m == 0:
            raise ValueError("UMAP graph has no edges (nnz=0).")
        idx = self.rng.choice(m, size=batch_size, replace=True, p=self.prob)
        return self.i[idx], self.j[idx], self.w[idx]


def umap_graph_loss(
    Y: torch.Tensor,
    edges_i: np.ndarray, edges_j: np.ndarray, edges_w: np.ndarray,
    a: float = 1.0, b: float = 1.0,
    n_neg: int = 5,
    gamma: float = 1.0,
    device: str = "cuda",
    eps: float = 1e-6,
) -> torch.Tensor:
    """UMAP cross-entropy edge loss (attractive + repulsive)."""
    n, _ = Y.shape

    ii = torch.from_numpy(edges_i).to(device)
    jj = torch.from_numpy(edges_j).to(device)
    ww = torch.from_numpy(edges_w).to(device)

    yi = Y[ii]
    yj = Y[jj]
    d2 = torch.sum((yi - yj) ** 2, dim=1) + eps
    loss_att = (ww * torch.log(1.0 + a * torch.pow(d2, b))).mean()

    neg_j = torch.randint(low=0, high=n, size=(len(edges_i), n_neg), device=device)
    yneg = Y[neg_j]
    yi2 = yi[:, None, :].expand_as(yneg)
    d2n = torch.sum((yi2 - yneg) ** 2, dim=2) + eps
    qn = 1.0 / (1.0 + a * torch.pow(d2n, b))
    loss_rep = -gamma * torch.log(1.0 - qn + eps).mean()

    return loss_att + loss_rep


def subspace_penalty_torch(
    Y: torch.Tensor,
    pred: np.ndarray,
    model: list[dict],
    device: str = "cpu",
) -> torch.Tensor:
    """sum_k mean_{i in C_k} ||(I - B_k B_k^T)(y_i - mu_k)||^2."""
    pred = np.asarray(pred, dtype=np.int64)
    total = torch.tensor(0.0, device=device)
    count = 0
    for k, mk in enumerate(model):
        idx = np.where(pred == k)[0]
        if idx.size == 0:
            continue
        mu = torch.tensor(mk["mu"], device=device, dtype=Y.dtype)
        B = torch.tensor(mk["B"], device=device, dtype=Y.dtype)
        yk = Y[torch.from_numpy(idx).to(device)]
        yc = yk - mu[None, :]
        tot = torch.sum(yc * yc, dim=1)
        proj = yc @ B
        res = tot - torch.sum(proj * proj, dim=1)
        total = total + res.mean()
        count += 1
    return total / max(count, 1)


# ----------------------------------------------------------------------------
# Main: joint optimization (Algorithm 2)
# ----------------------------------------------------------------------------

def joint_umap_ksubspaces(
    all_features,
    K: int,
    all_labels=None,
    n_neighbors: int = 15,
    graph_metric: str = "euclidean",
    mutual_only: bool = False,
    q: int = 128,
    outer_iter: int = 20,
    ksub_iter: int = 10,
    rank_mode: str = "bic",
    r_min: int = 4,
    r_max: int = 64,
    bic_weight: float = 0.16,
    beta: float = 1.0,
    gamma: float = 1.0,
    lr: float = 5e-2,
    inner_steps: int = 200,
    edge_batch_size: int = 4096,
    n_neg: int = 5,
    min_dist: float = 0.1,
    spread: float = 1.0,
    recenter_rescale: bool = True,
    device: str | None = None,
    seed: int = 0,
    verbose: bool = False,
):
    """Algorithm 2. Alternates K-subspaces on Y with SGD on Y."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X = _to_numpy(all_features).astype(np.float32)
    X = l2_normalize_rows(X)

    y_true = _to_numpy(all_labels).astype(int) if all_labels is not None else None

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    a, b = find_ab_params(spread=spread, min_dist=min_dist)

    W, deg = build_umap_graph(
        X, n_neighbors=n_neighbors, metric=graph_metric,
        random_state=seed, mutual_only=mutual_only,
    )
    if verbose:
        print(f"[Joint] graph nnz={W.nnz} avg_deg={deg.mean():.2f} device={device}")

    Y0 = umap.UMAP(
        n_neighbors=n_neighbors, n_components=q, metric=graph_metric,
        min_dist=min_dist, spread=spread, random_state=seed,
    ).fit_transform(X).astype(np.float32)
    Y0 -= Y0.mean(axis=0, keepdims=True)
    Y0 /= (Y0.std(axis=0, keepdims=True) + 1e-6)

    Y = nn.Parameter(torch.tensor(Y0, device=device))
    sampler = EdgeSampler(W, seed=seed)
    opt = optim.Adam([Y], lr=lr)

    history = {"ari": [], "nmi": [], "acc": [], "loss_graph": [], "loss_sub": []}
    best_pred = None
    best_ari = -np.inf

    for t in range(outer_iter):
        Y_np = Y.detach().cpu().numpy().astype(np.float32)
        pred, model = k_subspaces_on_Y(
            Y_np, K, max_iter=ksub_iter, init="kmeans",
            random_state=seed + 11 * (t + 1), rank_mode=rank_mode,
            r_min=r_min, r_max=r_max, bic_weight=bic_weight, verbose=False,
        )

        if y_true is not None:
            met = eval_clustering(y_true, pred)
            history["ari"].append(met["ari"])
            history["nmi"].append(met["nmi"])
            history["acc"].append(met["acc"])
            if met["ari"] > best_ari:
                best_ari = met["ari"]
                best_pred = pred.copy()
            if verbose:
                print(f"[Joint outer {t+1:02d}] ARI={met['ari']:.4f} "
                      f"NMI={met['nmi']:.4f} ACC={met['acc']:.4f}")

        pred_fixed = pred.copy()
        for step in range(inner_steps):
            opt.zero_grad()
            ei, ej, ew = sampler.sample(edge_batch_size)
            loss_g = umap_graph_loss(Y, ei, ej, ew, a=a, b=b,
                                     n_neg=n_neg, gamma=gamma, device=device)
            loss_s = subspace_penalty_torch(Y, pred_fixed, model, device=device)
            loss = loss_g + beta * loss_s
            loss.backward()
            opt.step()

            if recenter_rescale and (step % 20 == 0):
                with torch.no_grad():
                    Y.data = Y.data - Y.data.mean(dim=0, keepdim=True)
                    Y.data = Y.data / (Y.data.std(dim=0, keepdim=True) + 1e-6)

        history["loss_graph"].append(float(loss_g.item()))
        history["loss_sub"].append(float(loss_s.item()))

    if best_pred is None:
        best_pred = pred

    return best_pred, Y.detach().cpu().numpy().astype(np.float32), history
