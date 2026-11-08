"""Clustering evaluation (ARI, NMI, ACC).

ACC uses Hungarian matching between predicted and true labels.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clustering accuracy via Hungarian matching."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    D = max(int(y_pred.max()), int(y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_true.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return float(w[row_ind, col_ind].sum()) / float(y_true.size)


def eval_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return {'ari', 'nmi', 'acc'} for a single prediction."""
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "acc": clustering_accuracy(y_true, y_pred),
    }
