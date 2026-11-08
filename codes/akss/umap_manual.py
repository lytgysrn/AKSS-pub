"""UMAP fuzzy-graph helpers used by SR-UMAP.

This module intentionally stops at graph construction. The embedding/layout
optimization is delegated to umap-learn in `sr_umap.py`, while SR-UMAP uses the
helpers here to build fuzzy simplicial sets from custom kNN distances.
"""

from __future__ import annotations

import math
import numpy as np

try:
    import scipy
    import scipy.sparse
    from scipy.optimize import curve_fit
except Exception as e:
    raise ImportError("This module requires scipy. Please `pip install scipy`.") from e

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3


def nearest_neighbors(
    X: np.ndarray,
    n_neighbors: int,
    metric: str = "euclidean",
    metric_kwds: dict | None = None,
    angular: bool = False,
    random_state: int | np.random.RandomState = 42,
    low_memory: bool = True,
    n_jobs: int = -1,
    verbose: bool = False,
):
    """
    Return (knn_indices, knn_dists, knn_search_index) matching umap-learn behavior.

    Notes:
    - umap-learn uses pynndescent.NNDescent for most metrics; the heuristics for
      n_trees and n_iters are:
        n_trees = min(64, 5 + round(sqrt(n)/20))
        n_iters = max(5, round(log2(n)))
    """
    if metric_kwds is None:
        metric_kwds = {}

    n = X.shape[0]
    n_trees = min(64, 5 + int(round((n ** 0.5) / 20.0)))
    n_iters = max(5, int(round(math.log2(n))))

    try:
        from pynndescent import NNDescent
    except Exception as e:
        raise ImportError(
            "pynndescent is required to match umap-learn's neighbor graph. "
        ) from e

    knn_search_index = NNDescent(
        X,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_kwds=metric_kwds,
        random_state=random_state,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        low_memory=low_memory,
        n_jobs=n_jobs,
        verbose=verbose,
        compressed=False,
    )
    knn_indices, knn_dists = knn_search_index.neighbor_graph
    return knn_indices, knn_dists, knn_search_index


def smooth_knn_dist(
    distances: np.ndarray,
    k: int,
    n_iter: int = 64,
    local_connectivity: float = 1.0,
    bandwidth: float = 1.0,
):
    """Compute sigmas and rhos as in umap-learn."""
    target = math.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = float(np.inf)
        mid = 1.0

        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE and non_zero_dists.shape[0] > index:
                    rho[i] += interpolation * (non_zero_dists[index] - non_zero_dists[index - 1])
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = float(np.max(non_zero_dists))

        for _ in range(n_iter):
            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = ith_distances[j] - rho[i]
                if d > 0:
                    psum += math.exp(-(d / mid))
                else:
                    psum += 1.0

            if abs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0 if hi < float(np.inf) else mid / 2.0
            else:
                lo = mid
                mid = 2.0 * mid if hi == float(np.inf) else (lo + hi) / 2.0

        result[i] = mid

        if rho[i] > 0.0:
            mean_ith_distances = float(np.mean(ith_distances))
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


def compute_membership_strengths(
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    sigmas: np.ndarray,
    rhos: np.ndarray,
):
    n_samples, n_neighbors = knn_indices.shape
    rows = np.zeros(n_samples * n_neighbors, dtype=np.int64)
    cols = np.zeros(n_samples * n_neighbors, dtype=np.int64)
    vals = np.zeros(n_samples * n_neighbors, dtype=np.float32)

    idx = 0
    for i in range(n_samples):
        for j in range(n_neighbors):
            neighbor = knn_indices[i, j]
            if neighbor == -1:
                continue
            rows[idx] = i
            cols[idx] = neighbor

            if neighbor == i:
                vals[idx] = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                vals[idx] = 1.0
            else:
                vals[idx] = math.exp(-((knn_dists[i, j] - rhos[i]) / sigmas[i]))
            idx += 1

    return rows[:idx], cols[:idx], vals[:idx]


def fuzzy_simplicial_set(
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    n_neighbors: int,
    local_connectivity: float = 1.0,
    set_op_mix_ratio: float = 1.0,
    apply_set_operations: bool = True,
):
    sigmas, rhos = smooth_knn_dist(
        knn_dists, k=n_neighbors, local_connectivity=local_connectivity
    )

    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)

    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(knn_indices.shape[0], knn_indices.shape[0])
    )

    if apply_set_operations:
        transpose = result.transpose()
        prod_matrix = result.multiply(transpose)
        result = (
            set_op_mix_ratio * (result + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )

    result.eliminate_zeros()
    return result.tocsr(), sigmas, rhos


def find_ab_params(spread: float, min_dist: float):
    """
    Fit a,b such that 1/(1+a*x^(2b)) approximates:
        1 for x <= min_dist
        exp(-(x-min_dist)/spread) for x > min_dist
    """
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2.0 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.ones_like(xv)
    mask = xv > min_dist
    yv[mask] = np.exp(-(xv[mask] - min_dist) / spread)

    params, _ = curve_fit(curve, xv, yv)
    return float(params[0]), float(params[1])