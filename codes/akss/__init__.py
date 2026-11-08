"""AKSS: Adaptive K-Subspaces with Subspace-Refined UMAP.

----------
Algorithm 1 (AKSS, adaptive K-subspaces):
    akss.k_subspaces_adaptive
    akss.select_rank_bic, akss.select_rank_trace_stop
    akss.fit_cluster_subspace, akss.subspace_residual_distance

Algorithm 2 (joint UMAP-graph + K-subspaces):
    akss.joint_umap_ksubspaces

Algorithm 3 (SR-UMAP V1/V2 and iterative SR-UMAP):
    akss.sr_umap_from_external_akss
    akss.isr_umap
    akss.umap_kmeans_baseline

Datasets and evaluation:
    akss.load_dataset, akss.DATASET_REGISTRY
    akss.BEST_BIC_WEIGHTS_UK, akss.BEST_BIC_WEIGHTS_KM, akss.FIXED_BIC_WEIGHT
    akss.best_bic_weight, akss.eval_clustering, akss.clustering_accuracy
"""
from .akss import (
    fit_cluster_subspace,
    k_subspaces_adaptive,
    l2_normalize_rows,
    select_rank_bic,
    select_rank_trace_stop,
    subspace_residual_distance,
    umap_kmeans_init,
)
from .datasets import (
    BEST_BIC_WEIGHTS_KM,
    BEST_BIC_WEIGHTS_UK,
    DATASET_REGISTRY,
    FIXED_BIC_WEIGHT,
    best_bic_weight,
    load_dataset,
)
from .eval import clustering_accuracy, eval_clustering
from .pipeline import run_uk_chain
from .sr_umap import (
    build_graphs_from_model,
    fit_subspaces_from_labels,
    isr_umap,
    reselect_knn_by_symmetric_residual,
    sr_umap_from_external_akss,
    subspace_reweighted_knn_symmetric,
    umap_from_graph,
    umap_kmeans_baseline,
    umap_kmeans_on_graph,
)

__all__ = [
    # Algorithm 1
    "k_subspaces_adaptive",
    "select_rank_bic",
    "select_rank_trace_stop",
    "fit_cluster_subspace",
    "subspace_residual_distance",
    "umap_kmeans_init",
    "l2_normalize_rows",
    # Algorithm 3
    "sr_umap_from_external_akss",
    "isr_umap",
    "umap_kmeans_baseline",
    # High-level pipeline
    "run_uk_chain",
    "build_graphs_from_model",
    "subspace_reweighted_knn_symmetric",
    "reselect_knn_by_symmetric_residual",
    "fit_subspaces_from_labels",
    "umap_from_graph",
    "umap_kmeans_on_graph",
    # Datasets
    "load_dataset",
    "DATASET_REGISTRY",
    "BEST_BIC_WEIGHTS_UK",
    "BEST_BIC_WEIGHTS_KM",
    "FIXED_BIC_WEIGHT",
    "best_bic_weight",
    # Evaluation
    "eval_clustering",
    "clustering_accuracy",
]


def __getattr__(name):
    """Lazy-import `joint_umap_ksubspaces` so users without torch can still
    use the rest of the package."""
    if name == "joint_umap_ksubspaces":
        from .joint_umap import joint_umap_ksubspaces as _f
        return _f
    raise AttributeError(name)
