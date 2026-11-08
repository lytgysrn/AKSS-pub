"""Cached-feature dataset registry and loader.

The paper uses pre-extracted CLIP ViT-L/14 features for ten benchmark datasets.
Each dataset is loaded from `test_features.pt` in a user-supplied cache directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATASET_REGISTRY: dict[str, dict] = {
    "MNIST":        {"K": 10},
    "FashionMNIST": {"K": 10},
    "CIFAR10":      {"K": 10},
    "CIFAR20":      {"K": 20},
    "CIFAR100":     {"K": 100},
    "STL10":        {"K": 10},
    "ImageNet10":   {"K": 10},
    "ImagenetDogs": {"K": 15},
    "COCO":         {"K": 75},
    "DTD":          {"K": 47},
}

# Fixed BIC weight w used as the single cross-dataset default in the paper.
FIXED_BIC_WEIGHT: float = 0.16

# Per-dataset best w under UK-init AKSS (UMAP+KMeans pred -> AKSS).
# Selected by maximising mean ARI across 10 seeds.
BEST_BIC_WEIGHTS_UK: dict[str, float] = {
    "MNIST":        0.10,
    "FashionMNIST": 0.05,
    "CIFAR10":      0.15,
    "CIFAR20":      0.05,
    "CIFAR100":     0.30,
    "STL10":        0.30,
    "ImageNet10":   0.05,
    "ImagenetDogs": 0.10,
    "COCO":         0.05,
    "DTD":          0.10,
}

# Per-dataset best w under KMeans-init AKSS (classical KSS init).
# Selected by the same protocol; differs from UK on a few datasets.
BEST_BIC_WEIGHTS_KM: dict[str, float] = {
    "MNIST":        0.30,
    "FashionMNIST": 0.05,
    "CIFAR10":      0.05,
    "CIFAR20":      0.05,
    "CIFAR100":     0.30,
    "STL10":        0.16,
    "ImageNet10":   0.05,
    "ImagenetDogs": 0.10,
    "COCO":         0.10,
    "DTD":          0.10,
}

def best_bic_weight(dataset: str, init: str = "umap_kmeans") -> float:
    """Look up the per-dataset best `w` for the given init mode.

    init: 'umap_kmeans' / 'external' -> UK-init table;
          'kmeans' / 'random'        -> KM-init table.
    """
    table = BEST_BIC_WEIGHTS_KM if init in ("kmeans", "random") else BEST_BIC_WEIGHTS_UK
    return table[dataset]

_CACHE_FILENAME = "test_features.pt"
_FEATURE_KEY = "test_features"
_LABEL_KEY = "test_labels"


def _find_cache_file(cache_dir: Path, dataset_name: str) -> Path:
    candidates = [
        cache_dir / dataset_name / _CACHE_FILENAME,
        cache_dir / _CACHE_FILENAME,
        cache_dir / f"{dataset_name}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path

    for dataset_dir in cache_dir.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name.lower() == dataset_name.lower():
            path = dataset_dir / _CACHE_FILENAME
            if path.exists():
                return path

    raise FileNotFoundError(
        f"No cached {_CACHE_FILENAME} for '{dataset_name}' under {cache_dir}."
    )


def _load_feature_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() != ".pt":
        raise ValueError(f"Unsupported file type: {path}")

    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported .pt content: {type(obj)}")
    if _FEATURE_KEY not in obj or _LABEL_KEY not in obj:
        raise ValueError(f"Expected '{_FEATURE_KEY}' and '{_LABEL_KEY}' in {path}")

    feat = obj[_FEATURE_KEY]
    lab = obj[_LABEL_KEY]
    X = feat.detach().cpu().numpy() if hasattr(feat, "detach") else np.asarray(feat)
    y = lab.detach().cpu().numpy() if hasattr(lab, "detach") else np.asarray(lab)
    return X, y


def load_dataset(
    dataset_name: str,
    cache_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load (X, y, K) for a registered dataset from a `.pt` cache directory."""
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{dataset_name}'. "
                       f"Choose from: {list(DATASET_REGISTRY)}")
    cache_dir = Path(cache_dir)
    path = _find_cache_file(cache_dir, dataset_name)
    X, y = _load_feature_file(path)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    K = DATASET_REGISTRY[dataset_name]["K"]
    return X, y, K
