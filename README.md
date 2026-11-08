# AKSS — Adaptive K-Subspaces with Subspace-Refined UMAP

This repository contains the algorithms introduced in the paper:

- **AKSS** (Algorithm 1) — K-subspaces with per-cluster adaptive rank, chosen
  by a BIC-style criterion under a signal-plus-isotropic-noise covariance.
- **Joint UMAP-driven AKSS** (Algorithm 2) — joint optimization of a
  low-dimensional embedding and the AKSS subspace structure.
- **SR-UMAP** / **ISR-UMAP** (Algorithm 3) — subspace-refined UMAP where the
  AKSS subspaces re-weight (V1) or re-select (V2) the UMAP kNN graph, plus an
  iterative variant (T outer rounds).

## Layout

```
AKSS-pub/
├── codes/
│   ├── akss/                          # Python package
│   │   ├── __init__.py                # Public API
│   │   ├── eval.py                    # ARI / NMI / Hungarian ACC
│   │   ├── akss.py                    # Algorithm 1 + rank selection
│   │   ├── sr_umap.py                 # Algorithm 3 (SR-UMAP V1/V2 + ISR)
│   │   ├── joint_umap.py              # Algorithm 2
│   │   ├── umap_manual.py             # UMAP graph builder
│   │   ├── pipeline.py
│   │   └── datasets.py                # Dataset registry + w tables
│   ├── scripts/
│   │   ├── example_single_dataset.py  # All methods on one dataset (1 seed)
│   │   ├── reproduce_10datasets_10seeds.py
│   │   ├── ablation_w_sensitivity.py  # w sweep for the UK -> SR -> ISR chain
│   │   └── ablation_q_dim.py          # UMAP n_components ablation
│   └── requirements.txt
```

## Installation

```bash
pip install -r codes/requirements.txt
```

Python 3.10+ is recommended.

## Quick start

```python
import numpy as np
from akss import (
    k_subspaces_adaptive, sr_umap_from_external_akss, isr_umap,
    eval_clustering, load_dataset,
    BEST_BIC_WEIGHTS_UK, FIXED_BIC_WEIGHT, l2_normalize_rows,
)

X, y, K = load_dataset("MNIST", cache_dir="data/clip_cache")
Xn = l2_normalize_rows(X.astype(np.float32))

# Defaults: UK-init (UMAP+KMeans) + bic_weight=0.16 — paper fixed-w setting.
pred, model, _ = k_subspaces_adaptive(Xn, K, all_labels=y,
                                      l2_normalize=False, random_state=0)

# SR-UMAP V1 + V2 on top of AKSS
sr = sr_umap_from_external_akss("MNIST", Xn, y, K, pred, model,
                                graph_n_neighbors=15, q=2, random_state=0)
print(sr["V1"]["metrics"], sr["V2"]["metrics"])

# ISR-UMAP V2 (T=2 outer rounds)
pred_isr, _ = isr_umap("MNIST", Xn, y, K, outer_T=2, update_variant="V2",
                       pred_init=pred, model_init=model,
                       graph_n_neighbors=15, q=2, random_state=0)
print(eval_clustering(y, pred_isr))
```

Run the quick start from `codes/`, or add `codes/` to `PYTHONPATH`.

## Running the example pipeline

```bash
python codes/scripts/example_single_dataset.py --cache-dir data/clip_cache --dataset MNIST
```

Single seed. Each method runs with the per-dataset best `w` for its init
mode (`BEST_BIC_WEIGHTS_KM` for the KMeans-init AKSS row and the Algorithm 2
joint row; `BEST_BIC_WEIGHTS_UK` for the UK-AKSS / SR / ISR chain). Pass
`--bic-weight 0.16` to force a single `w` everywhere. Add `--joint` to also
evaluate Algorithm 2 (slow; needs GPU for reasonable runtimes).

## Reproducing the 10-dataset x 10-seed headline table

Only the **UK-init → SR-UMAP → ISR-UMAP** chain.

```bash
# fixed w=0.16
python codes/scripts/reproduce_10datasets_10seeds.py \
    --cache-dir data/clip_cache --out results.csv

python codes/scripts/reproduce_10datasets_10seeds.py \
    --cache-dir data/clip_cache --best-w --out results_best.csv
```

## Data

The CLIP ViT-L/14 features are provided by Yicen Li:

https://drive.google.com/drive/folders/1v4RhQiDenMCM16Ym-tJVwwlV3HXHLbd0?usp=sharing
