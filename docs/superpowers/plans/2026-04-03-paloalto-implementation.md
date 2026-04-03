# PALO ALTO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an agent-guided Bayesian optimization tool for tuning single-cell embedding visualization parameters against internalized scIB/scGraph metrics.

**Architecture:** Modular Python package — `data` (AnnData I/O), `metrics` (internalized scIB + scGraph), `embed` (NUMAP wrapper + PyTorch parametric t-SNE), `optim` (BoTorch BO with agent judge), `report` (HTML), `cli` (click). The BO loop orchestrates: embed → score → suggest, with an LLM agent judge reviewing each trial alongside a pure-BoTorch baseline.

**Tech Stack:** Python 3.12, PyTorch 2.6, BoTorch 0.17, NUMAP 0.2.3, scanpy, anndata, scikit-learn, pynndescent, matplotlib, Jinja2, click.

**Spec:** `docs/superpowers/specs/2026-04-03-paloalto-design.md`

---

## File Map

```
paloalto/
├── __init__.py              # Public API: optimize(), compute_metrics()
├── data.py                  # AnnData load, validate, subsample
├── metrics/
│   ├── __init__.py          # compute_all(), scib_overall()
│   ├── bio.py               # cell_type_asw, nmi, ari, clisi
│   ├── batch.py             # batch_asw, ilisi, graph_connectivity
│   ├── lisi.py              # Shared LISI computation (used by bio + batch)
│   └── scgraph.py           # corr_weighted
├── embed/
│   ├── __init__.py          # get_embedder()
│   ├── pumap.py             # NUMAPEmbedder: wraps numap, save/load
│   └── ptsne.py             # ParametricTSNE: PyTorch port
├── optim/
│   ├── __init__.py          # run_optimization()
│   ├── space.py             # SearchSpace, Parameter definitions
│   ├── bo.py                # WeightedBO (single-objective)
│   └── mobo.py              # ParetoBO (multi-objective NEHVI)
├── agent/
│   ├── __init__.py
│   └── judge.py             # AgentJudge: review, modify, inject, prune, stop
├── report/
│   ├── __init__.py
│   ├── html.py              # generate_report()
│   └── templates/
│       └── report.html.j2   # Jinja2 template
├── cli.py                   # click CLI: run, metrics
└── utils.py                 # knn_graph(), subsample_stratified(), logging setup
tests/
├── conftest.py              # Shared fixtures: mock AnnData, tmp dirs
├── test_data.py
├── test_metrics_bio.py
├── test_metrics_batch.py
├── test_metrics_lisi.py
├── test_metrics_scgraph.py
├── test_metrics_integration.py
├── test_embed_pumap.py
├── test_embed_ptsne.py
├── test_optim_space.py
├── test_optim_bo.py
├── test_optim_mobo.py
├── test_agent.py
├── test_report.py
├── test_cli.py
└── test_api.py              # End-to-end: optimize()
```

---

### Task 1: Package Scaffolding & Test Infrastructure

**Files:**
- Create: `paloalto/__init__.py`, `paloalto/metrics/__init__.py`, `paloalto/embed/__init__.py`, `paloalto/optim/__init__.py`, `paloalto/agent/__init__.py`, `paloalto/report/__init__.py`, `paloalto/report/templates/` (dir)
- Create: `pyproject.toml`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "paloalto"
version = "0.1.0"
description = "Pareto-guided Automatic Layout Optimization for Aligning Latent Topology in Omics embeddings"
requires-python = ">=3.10"
dependencies = [
    "anndata>=0.10",
    "scanpy>=1.9",
    "torch>=2.0",
    "botorch>=0.10",
    "numap>=0.2",
    "pynndescent>=0.5",
    "scikit-learn>=1.3",
    "matplotlib>=3.7",
    "jinja2>=3.1",
    "click>=8.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[project.scripts]
paloalto = "paloalto.cli:main"

[tool.setuptools.packages.find]
include = ["paloalto*"]
```

- [ ] **Step 2: Create all `__init__.py` stubs and directories**

`paloalto/__init__.py`:
```python
"""PALO ALTO: Pareto-guided Automatic Layout Optimization for Aligning Latent Topology in Omics embeddings."""

__version__ = "0.1.0"
```

All sub-package `__init__.py` files start empty.

Create `paloalto/report/templates/` as an empty directory (add `.gitkeep`).

- [ ] **Step 3: Create shared test fixtures**

`tests/conftest.py`:
```python
import numpy as np
import pytest
import anndata as ad
import pandas as pd


@pytest.fixture
def mock_adata():
    """Small AnnData with PCA embedding, batch, and cell_type labels."""
    np.random.seed(42)
    n_cells = 400
    n_genes = 100
    n_pcs = 20
    n_batches = 2
    n_types = 4

    X = np.random.randn(n_cells, n_genes).astype(np.float32)
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(
                [f"batch_{i % n_batches}" for i in range(n_cells)]
            ),
            "cell_type": pd.Categorical(
                [f"type_{i % n_types}" for i in range(n_cells)]
            ),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    # Create clustered PCA so metrics are non-trivial
    pca = np.random.randn(n_cells, n_pcs).astype(np.float32)
    for t in range(n_types):
        mask = obs["cell_type"] == f"type_{t}"
        pca[mask] += np.random.randn(n_pcs) * 3

    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["X_pca"] = pca
    # Pre-compute a mock 2D embedding for metric tests
    adata.obsm["X_umap_mock"] = pca[:, :2]
    return adata


@pytest.fixture
def mock_adata_path(mock_adata, tmp_path):
    """Write mock AnnData to disk, return path."""
    path = tmp_path / "test.h5ad"
    mock_adata.write_h5ad(path)
    return str(path)
```

- [ ] **Step 4: Verify test infra works**

Run: `cd /scratch/users/chensj16/projects/paloalto && pip install -e ".[dev]" && pytest tests/ -v --co`
Expected: conftest loads, no collection errors

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml paloalto/ tests/conftest.py
git commit -m "feat: package scaffolding and test fixtures"
```

---

### Task 2: Data Module

**Files:**
- Create: `paloalto/data.py`
- Create: `paloalto/utils.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write tests for data loading and validation**

`tests/test_data.py`:
```python
import pytest
import numpy as np
import anndata as ad
import pandas as pd

from paloalto.data import load_and_validate, subsample_stratified


class TestLoadAndValidate:
    def test_loads_from_path(self, mock_adata_path):
        adata = load_and_validate(
            mock_adata_path, embedding_key="X_pca", batch_key="batch", label_key="cell_type"
        )
        assert adata.n_obs == 400
        assert "X_pca" in adata.obsm

    def test_accepts_anndata_object(self, mock_adata):
        adata = load_and_validate(
            mock_adata, embedding_key="X_pca", batch_key="batch", label_key="cell_type"
        )
        assert adata is mock_adata

    def test_rejects_missing_embedding(self, mock_adata):
        with pytest.raises(ValueError, match="embedding_key"):
            load_and_validate(mock_adata, "X_nonexistent", "batch", "cell_type")

    def test_rejects_missing_batch_key(self, mock_adata):
        with pytest.raises(ValueError, match="batch_key"):
            load_and_validate(mock_adata, "X_pca", "nonexistent", "cell_type")

    def test_rejects_missing_label_key(self, mock_adata):
        with pytest.raises(ValueError, match="label_key"):
            load_and_validate(mock_adata, "X_pca", "batch", "nonexistent")

    def test_rejects_single_batch(self, mock_adata):
        mock_adata.obs["batch"] = "batch_0"
        with pytest.raises(ValueError, match="at least 2"):
            load_and_validate(mock_adata, "X_pca", "batch", "cell_type")

    def test_rejects_single_cell_type(self, mock_adata):
        mock_adata.obs["cell_type"] = "type_0"
        with pytest.raises(ValueError, match="at least 2"):
            load_and_validate(mock_adata, "X_pca", "batch", "cell_type")


class TestSubsampleStratified:
    def test_returns_original_if_small(self, mock_adata):
        indices = subsample_stratified(mock_adata, "cell_type", "batch", n=10000, seed=42)
        assert len(indices) == 400  # all cells kept

    def test_subsamples_large_data(self, mock_adata):
        indices = subsample_stratified(mock_adata, "cell_type", "batch", n=100, seed=42)
        assert len(indices) == 100

    def test_preserves_proportions(self, mock_adata):
        indices = subsample_stratified(mock_adata, "cell_type", "batch", n=200, seed=42)
        sub = mock_adata[indices]
        orig_props = mock_adata.obs["cell_type"].value_counts(normalize=True).sort_index()
        sub_props = sub.obs["cell_type"].value_counts(normalize=True).sort_index()
        np.testing.assert_allclose(orig_props.values, sub_props.values, atol=0.1)

    def test_deterministic(self, mock_adata):
        i1 = subsample_stratified(mock_adata, "cell_type", "batch", n=100, seed=42)
        i2 = subsample_stratified(mock_adata, "cell_type", "batch", n=100, seed=42)
        np.testing.assert_array_equal(i1, i2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py -v`
Expected: ImportError from `paloalto.data`

- [ ] **Step 3: Implement data.py and utils.py**

`paloalto/utils.py`:
```python
"""Shared utilities: kNN, subsampling, logging."""

import logging

import numpy as np


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

`paloalto/data.py`:
```python
"""AnnData loading, validation, and subsampling."""

from pathlib import Path
from typing import Union

import anndata as ad
import numpy as np
import pandas as pd

from paloalto.utils import get_logger

logger = get_logger(__name__)


def load_and_validate(
    adata: Union[str, Path, ad.AnnData],
    embedding_key: str,
    batch_key: str,
    label_key: str,
) -> ad.AnnData:
    if not isinstance(adata, ad.AnnData):
        logger.info(f"Loading {adata}")
        adata = ad.read_h5ad(adata)

    if embedding_key not in adata.obsm:
        raise ValueError(
            f"embedding_key '{embedding_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    if batch_key not in adata.obs.columns:
        raise ValueError(
            f"batch_key '{batch_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if label_key not in adata.obs.columns:
        raise ValueError(
            f"label_key '{label_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    n_batches = adata.obs[batch_key].nunique()
    if n_batches < 2:
        raise ValueError(f"Need at least 2 batches, found {n_batches}")

    n_types = adata.obs[label_key].nunique()
    if n_types < 2:
        raise ValueError(f"Need at least 2 cell types, found {n_types}")

    if adata.n_obs < 500:
        logger.warning(f"Only {adata.n_obs} cells — metrics may be unreliable")

    logger.info(
        f"Validated: {adata.n_obs} cells, {n_batches} batches, "
        f"{n_types} cell types, embedding '{embedding_key}' dim={adata.obsm[embedding_key].shape[1]}"
    )
    return adata


def subsample_stratified(
    adata: ad.AnnData,
    label_key: str,
    batch_key: str,
    n: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    if adata.n_obs <= n:
        return np.arange(adata.n_obs)

    rng = np.random.RandomState(seed)
    groups = adata.obs.groupby([label_key, batch_key], observed=True)
    group_sizes = groups.size()
    total = group_sizes.sum()

    indices = []
    for (label, batch), size in group_sizes.items():
        group_idx = groups.groups[(label, batch)]
        k = max(1, round(size / total * n))
        k = min(k, len(group_idx))
        chosen = rng.choice(group_idx, size=k, replace=False)
        indices.extend(chosen)

    indices = np.array(indices)
    # Trim or pad to exact n
    if len(indices) > n:
        indices = rng.choice(indices, size=n, replace=False)
    indices.sort()
    return indices
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paloalto/data.py paloalto/utils.py tests/test_data.py
git commit -m "feat: data loading, validation, and stratified subsampling"
```

---

### Task 3: LISI Metric (shared by bio + batch)

**Files:**
- Create: `paloalto/metrics/lisi.py`
- Create: `tests/test_metrics_lisi.py`

- [ ] **Step 1: Write tests**

`tests/test_metrics_lisi.py`:
```python
import numpy as np
import pytest

from paloalto.metrics.lisi import compute_lisi


class TestLISI:
    def test_perfect_mixing(self):
        """All neighbors have equal representation of 2 labels → LISI ≈ 2."""
        np.random.seed(42)
        # Interleaved: even indices = A, odd = B, fully mixed spatially
        coords = np.column_stack([np.arange(100), np.zeros(100)]).astype(np.float32)
        labels = np.array(["A", "B"] * 50)
        lisi = compute_lisi(coords, labels, perplexity=15)
        assert lisi.shape == (100,)
        assert np.mean(lisi) > 1.5  # close to 2

    def test_perfect_separation(self):
        """Two well-separated clusters with distinct labels → LISI ≈ 1."""
        coords = np.vstack([
            np.random.randn(50, 2) + [0, 0],
            np.random.randn(50, 2) + [100, 100],
        ]).astype(np.float32)
        labels = np.array(["A"] * 50 + ["B"] * 50)
        lisi = compute_lisi(coords, labels, perplexity=15)
        assert np.mean(lisi) < 1.2  # close to 1

    def test_output_range(self, mock_adata):
        coords = mock_adata.obsm["X_umap_mock"]
        labels = mock_adata.obs["batch"].values
        lisi = compute_lisi(coords, labels, perplexity=15)
        n_unique = len(np.unique(labels))
        assert np.all(lisi >= 1.0)
        assert np.all(lisi <= n_unique + 0.1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics_lisi.py -v`
Expected: ImportError

- [ ] **Step 3: Implement LISI**

`paloalto/metrics/lisi.py`:
```python
"""LISI (Local Inverse Simpson's Index) computation.

Internalized from scIB. Computes per-cell diversity of a categorical label
in local neighborhoods defined by a Gaussian kernel with specified perplexity.
"""

import numpy as np
from pynndescent import NNDescent


def compute_lisi(
    coords: np.ndarray,
    labels: np.ndarray,
    perplexity: float = 30.0,
) -> np.ndarray:
    """Compute per-cell LISI scores.

    Args:
        coords: (n_cells, n_dims) embedding coordinates.
        labels: (n_cells,) categorical labels.
        perplexity: Perplexity for Gaussian kernel bandwidth.

    Returns:
        (n_cells,) LISI scores. Range [1, n_unique_labels].
        1 = all neighbors have the same label; n = perfectly mixed.
    """
    n = coords.shape[0]
    k = min(int(perplexity * 3), n - 1)

    # Build kNN
    index = NNDescent(coords, n_neighbors=k + 1, random_state=42)
    nn_idx, nn_dist = index.neighbor_graph
    # Remove self (first column)
    nn_idx = nn_idx[:, 1:]
    nn_dist = nn_dist[:, 1:]

    # Encode labels as integers
    unique_labels, label_codes = np.unique(labels, return_inverse=True)
    n_labels = len(unique_labels)

    # For each cell, compute Gaussian kernel weights and Simpson's index
    lisi = np.zeros(n, dtype=np.float64)
    for i in range(n):
        distances = nn_dist[i].astype(np.float64)
        # Binary search for sigma to match target perplexity
        sigma = _find_sigma(distances, perplexity)
        weights = np.exp(-distances**2 / (2 * sigma**2))
        weights /= weights.sum()

        # Simpson's index: sum of squared label proportions
        neighbor_labels = label_codes[nn_idx[i]]
        proportions = np.zeros(n_labels, dtype=np.float64)
        for j in range(len(neighbor_labels)):
            proportions[neighbor_labels[j]] += weights[j]
        simpson = np.sum(proportions**2)
        lisi[i] = 1.0 / simpson

    return lisi


def _find_sigma(
    distances: np.ndarray,
    target_perplexity: float,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    """Binary search for sigma that yields target perplexity."""
    lo, hi = 1e-10, 1e4
    for _ in range(max_iter):
        sigma = (lo + hi) / 2
        weights = np.exp(-distances**2 / (2 * sigma**2))
        sum_w = weights.sum()
        if sum_w == 0:
            lo = sigma
            continue
        p = weights / sum_w
        entropy = -np.sum(p * np.log2(p + 1e-15))
        perp = 2**entropy
        if abs(perp - target_perplexity) < tol:
            break
        if perp < target_perplexity:
            lo = sigma
        else:
            hi = sigma
    return sigma
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics_lisi.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paloalto/metrics/lisi.py tests/test_metrics_lisi.py
git commit -m "feat: LISI metric computation (internalized from scIB)"
```

---

### Task 4: Bio-Conservation Metrics

**Files:**
- Create: `paloalto/metrics/bio.py`
- Create: `tests/test_metrics_bio.py`

- [ ] **Step 1: Write tests**

`tests/test_metrics_bio.py`:
```python
import numpy as np
import pytest

from paloalto.metrics.bio import cell_type_asw, nmi, ari, clisi


class TestCellTypeASW:
    def test_returns_float_in_range(self, mock_adata):
        score = cell_type_asw(mock_adata.obsm["X_umap_mock"], mock_adata.obs["cell_type"].values)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_perfect_clusters(self):
        coords = np.vstack([np.random.randn(50, 2) + [i * 20, 0] for i in range(4)])
        labels = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50 + ["D"] * 50)
        score = cell_type_asw(coords, labels)
        assert score > 0.7


class TestNMI:
    def test_returns_float_in_range(self, mock_adata):
        score = nmi(mock_adata.obsm["X_umap_mock"], mock_adata.obs["cell_type"].values)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestARI:
    def test_returns_float_in_range(self, mock_adata):
        score = ari(mock_adata.obsm["X_umap_mock"], mock_adata.obs["cell_type"].values)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestCLISI:
    def test_returns_float_in_range(self, mock_adata):
        score = clisi(mock_adata.obsm["X_umap_mock"], mock_adata.obs["cell_type"].values)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_well_separated_clusters_high_score(self):
        coords = np.vstack([np.random.randn(50, 2) + [i * 20, 0] for i in range(3)])
        labels = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50)
        score = clisi(coords, labels)
        assert score > 0.7  # high purity
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics_bio.py -v`
Expected: ImportError

- [ ] **Step 3: Implement bio metrics**

`paloalto/metrics/bio.py`:
```python
"""Bio-conservation metrics: Cell type ASW, NMI, ARI, cLISI."""

import numpy as np
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

from paloalto.metrics.lisi import compute_lisi


def cell_type_asw(coords: np.ndarray, labels: np.ndarray) -> float:
    """Average silhouette width on cell type labels, rescaled to [0, 1]."""
    asw = silhouette_score(coords, labels)
    return (asw + 1) / 2  # from [-1, 1] to [0, 1]


def nmi(coords: np.ndarray, labels: np.ndarray, resolution: float = 1.0) -> float:
    """NMI between Leiden clusters on 2D kNN graph and ground truth labels."""
    clusters = _leiden_on_coords(coords, resolution=resolution)
    return float(normalized_mutual_info_score(labels, clusters))


def ari(coords: np.ndarray, labels: np.ndarray, resolution: float = 1.0) -> float:
    """ARI between Leiden clusters on 2D kNN graph and ground truth labels."""
    clusters = _leiden_on_coords(coords, resolution=resolution)
    return float(adjusted_rand_score(labels, clusters))


def clisi(coords: np.ndarray, labels: np.ndarray, perplexity: float = 30.0) -> float:
    """Cell-type LISI: normalized so that 1 = perfect purity, 0 = fully mixed.

    Raw cLISI is in [1, n_types]. We normalize: score = (n_types - median_lisi) / (n_types - 1).
    """
    lisi_scores = compute_lisi(coords, labels, perplexity=perplexity)
    n_types = len(np.unique(labels))
    median_lisi = np.median(lisi_scores)
    return float((n_types - median_lisi) / (n_types - 1))


def _leiden_on_coords(
    coords: np.ndarray, n_neighbors: int = 15, resolution: float = 1.0
) -> np.ndarray:
    """Run Leiden clustering on a kNN graph built from 2D coords."""
    import scanpy as sc
    import anndata as ad

    tmp = ad.AnnData(obsm={"X_embed": coords})
    sc.pp.neighbors(tmp, use_rep="X_embed", n_neighbors=n_neighbors)
    sc.tl.leiden(tmp, resolution=resolution, flavor="igraph", n_iterations=2)
    return tmp.obs["leiden"].values
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics_bio.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paloalto/metrics/bio.py tests/test_metrics_bio.py
git commit -m "feat: bio-conservation metrics (ASW, NMI, ARI, cLISI)"
```

---

### Task 5: Batch-Mixing Metrics

**Files:**
- Create: `paloalto/metrics/batch.py`
- Create: `tests/test_metrics_batch.py`

- [ ] **Step 1: Write tests**

`tests/test_metrics_batch.py`:
```python
import numpy as np
import pytest

from paloalto.metrics.batch import batch_asw, ilisi, graph_connectivity


class TestBatchASW:
    def test_returns_float_in_range(self, mock_adata):
        score = batch_asw(
            mock_adata.obsm["X_umap_mock"],
            mock_adata.obs["batch"].values,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_well_mixed_high_score(self):
        np.random.seed(42)
        coords = np.random.randn(200, 2)
        labels = np.array(["A", "B"] * 100)
        score = batch_asw(coords, labels)
        assert score > 0.7  # well mixed → low |ASW| → high score


class TestILISI:
    def test_returns_float_in_range(self, mock_adata):
        score = ilisi(
            mock_adata.obsm["X_umap_mock"],
            mock_adata.obs["batch"].values,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_well_mixed_high_score(self):
        np.random.seed(42)
        coords = np.random.randn(200, 2)
        labels = np.array(["A", "B"] * 100)
        score = ilisi(coords, labels)
        assert score > 0.5


class TestGraphConnectivity:
    def test_returns_float_in_range(self, mock_adata):
        score = graph_connectivity(
            mock_adata.obsm["X_umap_mock"],
            mock_adata.obs["cell_type"].values,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_fully_connected_returns_one(self):
        np.random.seed(42)
        coords = np.random.randn(200, 2)
        labels = np.array(["A"] * 100 + ["B"] * 100)
        score = graph_connectivity(coords, labels)
        assert score == 1.0  # all same-type cells are connected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics_batch.py -v`
Expected: ImportError

- [ ] **Step 3: Implement batch metrics**

`paloalto/metrics/batch.py`:
```python
"""Batch-mixing metrics: Batch ASW, iLISI, Graph connectivity."""

import numpy as np
from sklearn.metrics import silhouette_samples
from scipy.sparse.csgraph import connected_components

from paloalto.metrics.lisi import compute_lisi


def batch_asw(coords: np.ndarray, batch_labels: np.ndarray) -> float:
    """Batch ASW: 1 - |ASW_batch|, rescaled to [0, 1].

    Lower absolute silhouette on batch labels means better mixing.
    """
    per_cell = silhouette_samples(coords, batch_labels)
    # Per-batch mean absolute, then overall mean
    batches = np.unique(batch_labels)
    per_batch = []
    for b in batches:
        mask = batch_labels == b
        per_batch.append(np.mean(np.abs(per_cell[mask])))
    return float(1 - np.mean(per_batch))


def ilisi(
    coords: np.ndarray,
    batch_labels: np.ndarray,
    perplexity: float = 30.0,
) -> float:
    """Integration LISI: normalized so that 1 = perfect mixing, 0 = no mixing.

    Raw iLISI is in [1, n_batches]. We normalize: score = (median_lisi - 1) / (n_batches - 1).
    """
    lisi_scores = compute_lisi(coords, batch_labels, perplexity=perplexity)
    n_batches = len(np.unique(batch_labels))
    median_lisi = np.median(lisi_scores)
    return float(np.clip((median_lisi - 1) / (n_batches - 1), 0, 1))


def graph_connectivity(
    coords: np.ndarray,
    type_labels: np.ndarray,
    n_neighbors: int = 15,
) -> float:
    """Fraction of cell types whose kNN subgraph is fully connected."""
    from pynndescent import NNDescent
    from scipy.sparse import csr_matrix

    n = coords.shape[0]
    index = NNDescent(coords, n_neighbors=n_neighbors + 1, random_state=42)
    nn_idx, _ = index.neighbor_graph

    # Build adjacency matrix
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = nn_idx[:, 1:].ravel()
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj + adj.T  # symmetrize

    types = np.unique(type_labels)
    scores = []
    for t in types:
        mask = type_labels == t
        idx = np.where(mask)[0]
        sub_adj = adj[np.ix_(idx, idx)]
        n_components, _ = connected_components(sub_adj, directed=False)
        # Fraction that is in the largest component
        scores.append(1.0 / n_components)

    return float(np.mean(scores))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics_batch.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paloalto/metrics/batch.py tests/test_metrics_batch.py
git commit -m "feat: batch-mixing metrics (ASW, iLISI, graph connectivity)"
```

---

### Task 6: scGraph Corr-Weighted Metric

**Files:**
- Create: `paloalto/metrics/scgraph.py`
- Create: `tests/test_metrics_scgraph.py`

- [ ] **Step 1: Write tests**

`tests/test_metrics_scgraph.py`:
```python
import numpy as np
import pytest

from paloalto.metrics.scgraph import corr_weighted


class TestCorrWeighted:
    def test_returns_float(self, mock_adata):
        score = corr_weighted(
            adata=mock_adata,
            embed_key="X_umap_mock",
            label_key="cell_type",
            batch_key="batch",
        )
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_identity_embedding_high_score(self, mock_adata):
        """If embedding = PCA, score should be very high."""
        score = corr_weighted(
            adata=mock_adata,
            embed_key="X_pca",
            label_key="cell_type",
            batch_key="batch",
        )
        assert score > 0.5

    def test_random_embedding_lower_score(self, mock_adata):
        np.random.seed(0)
        mock_adata.obsm["X_random"] = np.random.randn(mock_adata.n_obs, 2).astype(np.float32)
        score_rand = corr_weighted(
            adata=mock_adata,
            embed_key="X_random",
            label_key="cell_type",
            batch_key="batch",
        )
        score_pca = corr_weighted(
            adata=mock_adata,
            embed_key="X_pca",
            label_key="cell_type",
            batch_key="batch",
        )
        assert score_pca > score_rand
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics_scgraph.py -v`
Expected: ImportError

- [ ] **Step 3: Implement scGraph Corr-Weighted**

`paloalto/metrics/scgraph.py`:
```python
"""scGraph Corr-Weighted metric (internalized).

Measures how well a 2D embedding preserves inter-cell-type distance
relationships compared to a PCA consensus built per-batch.
"""

import numpy as np
import scanpy as sc
from scipy.spatial.distance import pdist, squareform
from scipy.stats import trim_mean


def corr_weighted(
    adata,
    embed_key: str,
    label_key: str,
    batch_key: str,
    n_hvgs: int = 1000,
    n_pcs: int = 10,
    trim_proportion: float = 0.05,
    min_cells_per_batch: int = 100,
) -> float:
    """Compute the scGraph Corr-Weighted score.

    Args:
        adata: AnnData object.
        embed_key: Key in obsm for the embedding to evaluate.
        label_key: Column in obs for cell type labels.
        batch_key: Column in obs for batch labels.

    Returns:
        Corr-Weighted score (float, higher = better).
    """
    labels = adata.obs[label_key].values
    types = np.unique(labels)
    n_types = len(types)

    if n_types < 3:
        # Need at least 3 types for meaningful correlation
        return 0.0

    # Build PCA consensus reference from per-batch PCA centroids
    consensus = _build_consensus(
        adata, label_key, batch_key, types,
        n_hvgs=n_hvgs, n_pcs=n_pcs, trim_proportion=trim_proportion,
        min_cells_per_batch=min_cells_per_batch,
    )
    if consensus is None:
        return 0.0

    # Compute embedding distance matrix
    embed_coords = adata.obsm[embed_key]
    embed_dist = _centroid_distance_matrix(embed_coords, labels, types, trim_proportion)

    # Column-normalize both
    consensus = _col_normalize(consensus)
    embed_dist = _col_normalize(embed_dist)

    # Corr-Weighted: mean column-wise weighted Pearson correlation
    scores = []
    for col in range(n_types):
        ref_col = consensus[:, col]
        emb_col = embed_dist[:, col]
        # Weights = 1 / reference distance (0 where distance is 0)
        weights = np.zeros_like(ref_col)
        nonzero = ref_col > 0
        weights[nonzero] = 1.0 / ref_col[nonzero]
        if weights.sum() == 0:
            continue
        r = _weighted_pearson(emb_col, ref_col, weights)
        if not np.isnan(r):
            scores.append(r)

    return float(np.mean(scores)) if scores else 0.0


def _build_consensus(
    adata, label_key, batch_key, types,
    n_hvgs, n_pcs, trim_proportion, min_cells_per_batch,
):
    """Build consensus distance matrix from per-batch PCA centroids."""
    batches = adata.obs[batch_key].unique()
    n_types = len(types)
    dist_matrices = []

    for batch in batches:
        mask = adata.obs[batch_key] == batch
        if mask.sum() < min_cells_per_batch:
            continue

        batch_adata = adata[mask].copy()
        # Compute PCA on this batch
        try:
            sc.pp.highly_variable_genes(batch_adata, n_top_genes=min(n_hvgs, batch_adata.n_vars), flavor="seurat_v3")
            sc.tl.pca(batch_adata, n_comps=min(n_pcs, batch_adata.n_vars - 1))
            pca_coords = batch_adata.obsm["X_pca"]
        except Exception:
            # Fallback: use existing embedding
            if "X_pca" in batch_adata.obsm:
                pca_coords = batch_adata.obsm["X_pca"][:, :n_pcs]
            else:
                continue

        batch_labels = batch_adata.obs[label_key].values
        dist = _centroid_distance_matrix(pca_coords, batch_labels, types, trim_proportion)
        dist = _col_normalize(dist)
        dist_matrices.append(dist)

    if not dist_matrices:
        return None

    consensus = np.mean(dist_matrices, axis=0)
    return _col_normalize(consensus)


def _centroid_distance_matrix(
    coords: np.ndarray,
    labels: np.ndarray,
    types: np.ndarray,
    trim_proportion: float,
) -> np.ndarray:
    """Compute pairwise Euclidean distance between trimmed-mean centroids."""
    n_types = len(types)
    centroids = np.zeros((n_types, coords.shape[1]))
    for i, t in enumerate(types):
        mask = labels == t
        if mask.sum() == 0:
            centroids[i] = np.nan
        elif mask.sum() < 3:
            centroids[i] = np.mean(coords[mask], axis=0)
        else:
            centroids[i] = trim_mean(coords[mask], proportiontocut=trim_proportion, axis=0)

    dist = squareform(pdist(centroids, metric="euclidean"))
    np.fill_diagonal(dist, 0)
    return dist


def _col_normalize(mat: np.ndarray) -> np.ndarray:
    """Divide each column by its max value."""
    col_max = mat.max(axis=0)
    col_max[col_max == 0] = 1.0
    return mat / col_max


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation."""
    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    var_x = np.sum(w * (x - mx) ** 2)
    var_y = np.sum(w * (y - my) ** 2)
    denom = np.sqrt(var_x * var_y)
    if denom < 1e-15:
        return np.nan
    return cov / denom
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics_scgraph.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paloalto/metrics/scgraph.py tests/test_metrics_scgraph.py
git commit -m "feat: scGraph Corr-Weighted metric (internalized)"
```

---

### Task 7: Metrics Aggregation

**Files:**
- Modify: `paloalto/metrics/__init__.py`
- Create: `tests/test_metrics_integration.py`

- [ ] **Step 1: Write tests**

`tests/test_metrics_integration.py`:
```python
import numpy as np
import pytest

from paloalto.metrics import compute_all, scib_overall


class TestComputeAll:
    def test_returns_all_keys(self, mock_adata):
        scores = compute_all(
            adata=mock_adata,
            embed_key="X_umap_mock",
            label_key="cell_type",
            batch_key="batch",
        )
        assert "cell_type_asw" in scores
        assert "nmi" in scores
        assert "ari" in scores
        assert "clisi" in scores
        assert "batch_asw" in scores
        assert "ilisi" in scores
        assert "graph_connectivity" in scores
        assert "scgraph_corr_weighted" in scores
        assert "bio_score" in scores
        assert "batch_score" in scores
        assert "scib_overall" in scores

    def test_all_values_are_floats(self, mock_adata):
        scores = compute_all(
            adata=mock_adata,
            embed_key="X_umap_mock",
            label_key="cell_type",
            batch_key="batch",
        )
        for k, v in scores.items():
            assert isinstance(v, float), f"{k} is {type(v)}"


class TestScibOverall:
    def test_weighted_aggregation(self):
        bio = {"cell_type_asw": 0.8, "nmi": 0.6, "ari": 0.7, "clisi": 0.9}
        batch = {"batch_asw": 0.5, "ilisi": 0.4, "graph_connectivity": 0.6}
        result = scib_overall(bio, batch)
        expected_bio = np.mean([0.8, 0.6, 0.7, 0.9])
        expected_batch = np.mean([0.5, 0.4, 0.6])
        expected = 0.6 * expected_bio + 0.4 * expected_batch
        assert abs(result - expected) < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics_integration.py -v`
Expected: ImportError

- [ ] **Step 3: Implement metrics/__init__.py**

`paloalto/metrics/__init__.py`:
```python
"""Metrics aggregation: scIB overall + scGraph."""

from typing import Dict
import numpy as np

from paloalto.metrics.bio import cell_type_asw, nmi, ari, clisi
from paloalto.metrics.batch import batch_asw, ilisi, graph_connectivity
from paloalto.metrics.scgraph import corr_weighted


def scib_overall(bio: Dict[str, float], batch: Dict[str, float]) -> float:
    """Compute scIB overall score = 0.6 * Bio + 0.4 * Batch."""
    bio_score = float(np.mean(list(bio.values())))
    batch_score = float(np.mean(list(batch.values())))
    return 0.6 * bio_score + 0.4 * batch_score


def compute_all(
    adata,
    embed_key: str,
    label_key: str,
    batch_key: str,
) -> Dict[str, float]:
    """Compute all metrics and return a flat dict."""
    coords = adata.obsm[embed_key]
    labels = adata.obs[label_key].values
    batches = adata.obs[batch_key].values

    bio = {
        "cell_type_asw": cell_type_asw(coords, labels),
        "nmi": nmi(coords, labels),
        "ari": ari(coords, labels),
        "clisi": clisi(coords, labels),
    }
    batch_scores = {
        "batch_asw": batch_asw(coords, batches),
        "ilisi": ilisi(coords, batches),
        "graph_connectivity": graph_connectivity(coords, labels),
    }
    scgraph_score = corr_weighted(adata, embed_key, label_key, batch_key)

    bio_mean = float(np.mean(list(bio.values())))
    batch_mean = float(np.mean(list(batch_scores.values())))
    overall = scib_overall(bio, batch_scores)

    return {
        **bio,
        **batch_scores,
        "scgraph_corr_weighted": scgraph_score,
        "bio_score": bio_mean,
        "batch_score": batch_mean,
        "scib_overall": overall,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics_integration.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paloalto/metrics/__init__.py tests/test_metrics_integration.py
git commit -m "feat: metrics aggregation (scIB overall + scGraph)"
```

---

### Task 8: NUMAP Wrapper

**Files:**
- Create: `paloalto/embed/pumap.py`
- Modify: `paloalto/embed/__init__.py`
- Create: `tests/test_embed_pumap.py`

- [ ] **Step 1: Write tests**

`tests/test_embed_pumap.py`:
```python
import numpy as np
import pytest
import torch

from paloalto.embed.pumap import NUMAPEmbedder


class TestNUMAPEmbedder:
    def test_fit_returns_2d(self, mock_adata):
        emb = NUMAPEmbedder(n_neighbors=10, min_dist=0.1, epochs=2)
        coords = emb.fit(mock_adata, embedding_key="X_pca")
        assert coords.shape == (400, 2)

    def test_transform(self, mock_adata):
        emb = NUMAPEmbedder(n_neighbors=10, min_dist=0.1, epochs=2)
        emb.fit(mock_adata, embedding_key="X_pca")
        new_coords = emb.transform(mock_adata.obsm["X_pca"][:10])
        assert new_coords.shape == (10, 2)

    def test_save_load(self, mock_adata, tmp_path):
        emb = NUMAPEmbedder(n_neighbors=10, min_dist=0.1, epochs=2)
        emb.fit(mock_adata, embedding_key="X_pca")
        path = str(tmp_path / "model.pt")
        emb.save(path)

        emb2 = NUMAPEmbedder.load(path)
        c1 = emb.transform(mock_adata.obsm["X_pca"][:5])
        c2 = emb2.transform(mock_adata.obsm["X_pca"][:5])
        np.testing.assert_allclose(c1, c2, atol=1e-5)

    def test_get_params(self):
        emb = NUMAPEmbedder(n_neighbors=15, min_dist=0.3, metric="cosine")
        params = emb.get_params()
        assert params["n_neighbors"] == 15
        assert params["min_dist"] == 0.3
        assert params["metric"] == "cosine"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_embed_pumap.py -v`
Expected: ImportError

- [ ] **Step 3: Implement NUMAPEmbedder**

`paloalto/embed/__init__.py`:
```python
"""Embedding methods."""

from paloalto.embed.pumap import NUMAPEmbedder


def get_embedder(method: str, **kwargs):
    """Factory for embedding methods."""
    if method == "numap":
        return NUMAPEmbedder(**kwargs)
    elif method == "tsne":
        from paloalto.embed.ptsne import ParametricTSNE
        return ParametricTSNE(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'numap' or 'tsne'.")
```

`paloalto/embed/pumap.py`:
```python
"""NUMAP wrapper with BO-tunable interface and model persistence."""

import numpy as np
import torch

from paloalto.utils import get_logger

logger = get_logger(__name__)


class NUMAPEmbedder:
    """Parametric UMAP via NUMAP with save/load and BO-friendly interface."""

    def __init__(
        self,
        n_neighbors: int = 10,
        min_dist: float = 0.1,
        negative_sample_rate: int = 5,
        metric: str = "euclidean",
        se_dim: int = 2,
        se_neighbors: int = 10,
        epochs: int = 10,
        lr: float = 1e-2,
        batch_size: int = 64,
        random_state: int = 42,
        **kwargs,
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.negative_sample_rate = negative_sample_rate
        self.metric = metric
        self.se_dim = se_dim
        self.se_neighbors = se_neighbors
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None
        self._input_dim = None

    def fit(self, adata, embedding_key: str = "X_pca") -> np.ndarray:
        """Fit NUMAP on the embedding and return 2D coordinates."""
        from numap import NUMAP

        X = adata.obsm[embedding_key]
        self._input_dim = X.shape[1]

        logger.info(
            f"Fitting NUMAP: n_neighbors={self.n_neighbors}, min_dist={self.min_dist}, "
            f"metric={self.metric}, se_dim={self.se_dim}"
        )

        model = NUMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            negative_sample_rate=self.negative_sample_rate,
            metric=self.metric,
            se_dim=self.se_dim,
            se_neighbors=self.se_neighbors,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            random_state=self.random_state,
            **self.kwargs,
        )
        coords = model.fit_transform(X)
        self._model = model
        return np.array(coords)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data through the fitted encoder."""
        if self._model is None:
            raise RuntimeError("Must call fit() before transform()")
        return np.array(self._model.transform(X))

    def save(self, path: str):
        """Save model state and hyperparams."""
        if self._model is None:
            raise RuntimeError("Must call fit() before save()")
        state = {
            "params": self.get_params(),
            "model_state": self._model.model.state_dict(),
            "input_dim": self._input_dim,
        }
        torch.save(state, path)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str) -> "NUMAPEmbedder":
        """Load a saved model."""
        state = torch.load(path, weights_only=False)
        emb = cls(**state["params"])
        emb._input_dim = state["input_dim"]

        # Reconstruct NUMAP model for transform
        from numap import NUMAP
        model = NUMAP(
            n_components=2,
            **{k: v for k, v in state["params"].items() if k not in ("epochs", "lr", "batch_size", "random_state")},
        )
        # Build model structure by creating a dummy
        from numap.umap_pytorch.model import default_encoder
        model.model = type(model.model)(
            encoder=default_encoder(state["input_dim"] + state["params"].get("se_dim", 2), 2),
            n_components=2,
            min_dist=state["params"]["min_dist"],
        )
        model.model.load_state_dict(state["model_state"])
        model.model.eval()
        emb._model = model
        return emb

    def get_params(self) -> dict:
        return {
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "negative_sample_rate": self.negative_sample_rate,
            "metric": self.metric,
            "se_dim": self.se_dim,
            "se_neighbors": self.se_neighbors,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embed_pumap.py -v`
Expected: All 4 tests PASS (some may need adjustments based on NUMAP's actual API — see notes)

Note: The `save`/`load` implementation may need adjustment based on NUMAP's internal model structure. The key contract is: `fit → save → load → transform` roundtrip produces identical results. If the internal structure differs, adapt the `save`/`load` methods to match NUMAP's actual `model` attribute hierarchy.

- [ ] **Step 5: Commit**

```bash
git add paloalto/embed/__init__.py paloalto/embed/pumap.py tests/test_embed_pumap.py
git commit -m "feat: NUMAP wrapper with BO-tunable interface and save/load"
```

---

### Task 9: Parametric t-SNE (PyTorch)

**Files:**
- Create: `paloalto/embed/ptsne.py`
- Create: `tests/test_embed_ptsne.py`

- [ ] **Step 1: Write tests**

`tests/test_embed_ptsne.py`:
```python
import numpy as np
import pytest
import torch

from paloalto.embed.ptsne import ParametricTSNE


class TestParametricTSNE:
    def test_fit_returns_2d(self, mock_adata):
        model = ParametricTSNE(perplexity=15, epochs=5, batch_size=64)
        coords = model.fit(mock_adata, embedding_key="X_pca")
        assert coords.shape == (400, 2)

    def test_transform(self, mock_adata):
        model = ParametricTSNE(perplexity=15, epochs=5, batch_size=64)
        model.fit(mock_adata, embedding_key="X_pca")
        new_coords = model.transform(mock_adata.obsm["X_pca"][:10])
        assert new_coords.shape == (10, 2)

    def test_multiscale_mode(self, mock_adata):
        model = ParametricTSNE(multiscale=True, epochs=5, batch_size=64)
        coords = model.fit(mock_adata, embedding_key="X_pca")
        assert coords.shape == (400, 2)

    def test_save_load(self, mock_adata, tmp_path):
        model = ParametricTSNE(perplexity=15, epochs=5, batch_size=64)
        model.fit(mock_adata, embedding_key="X_pca")
        path = str(tmp_path / "tsne_model.pt")
        model.save(path)

        model2 = ParametricTSNE.load(path)
        c1 = model.transform(mock_adata.obsm["X_pca"][:5])
        c2 = model2.transform(mock_adata.obsm["X_pca"][:5])
        np.testing.assert_allclose(c1, c2, atol=1e-5)

    def test_dof_parameter(self, mock_adata):
        model = ParametricTSNE(perplexity=15, dof=0.5, epochs=5, batch_size=64)
        coords = model.fit(mock_adata, embedding_key="X_pca")
        assert coords.shape == (400, 2)

    def test_get_params(self):
        model = ParametricTSNE(perplexity=30, dof=1.5, metric="cosine")
        params = model.get_params()
        assert params["perplexity"] == 30
        assert params["dof"] == 1.5
        assert params["metric"] == "cosine"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_embed_ptsne.py -v`
Expected: ImportError

- [ ] **Step 3: Implement Parametric t-SNE**

`paloalto/embed/ptsne.py`:
```python
"""Parametric t-SNE in PyTorch.

Ported from Multiscale-Parametric-t-SNE (Crecchi, ESANN 2020),
rewritten in PyTorch with bug fixes and extended features.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pynndescent import NNDescent
from umap.umap_ import fuzzy_simplicial_set
from scipy.sparse import csr_matrix

from paloalto.utils import get_logger

logger = get_logger(__name__)


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, n_components: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_components),
        )

    def forward(self, x):
        return self.net(x)


class ParametricTSNE:
    """Parametric t-SNE with single-scale and multiscale modes."""

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        dof: float = 1.0,
        early_exaggeration: float = 12.0,
        early_exaggeration_epochs: int = 50,
        exaggeration: float = 1.0,
        metric: str = "euclidean",
        learning_rate: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 256,
        multiscale: bool = False,
        seed: int = 42,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.dof = dof
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_epochs = early_exaggeration_epochs
        self.exaggeration = exaggeration
        self.metric = metric
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.multiscale = multiscale
        self.seed = seed
        self._encoder = None
        self._input_dim = None

    def fit(self, adata, embedding_key: str = "X_pca") -> np.ndarray:
        """Fit parametric t-SNE and return 2D coordinates."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = adata.obsm[embedding_key].astype(np.float32)
        self._input_dim = X.shape[1]
        n = X.shape[0]

        logger.info(
            f"Fitting Parametric t-SNE: perplexity={self.perplexity}, dof={self.dof}, "
            f"multiscale={self.multiscale}, metric={self.metric}"
        )

        # Compute P matrix
        P = self._compute_P(X)

        # Build encoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder = _Encoder(self._input_dim, self.n_components).to(device)
        optimizer = torch.optim.Adam(self._encoder.parameters(), lr=self.learning_rate)

        X_tensor = torch.from_numpy(X).to(device)

        # Training loop
        for epoch in range(self.epochs):
            if epoch < self.early_exaggeration_epochs:
                exag = self.early_exaggeration
            else:
                exag = self.exaggeration

            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:
                    continue

                x_batch = X_tensor[idx]
                P_batch = P[idx.numpy()][:, idx.numpy()]

                # Re-normalize P within batch
                p_sum = P_batch.sum()
                if p_sum == 0:
                    continue
                P_batch = P_batch / p_sum * exag
                P_batch = np.maximum(P_batch, 1e-12)
                P_torch = torch.from_numpy(P_batch.astype(np.float32)).to(device)

                # Forward
                Y = self._encoder(x_batch)
                loss = self._kl_divergence(P_torch, Y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 50 == 0:
                avg = epoch_loss / max(n_batches, 1)
                logger.info(f"  Epoch {epoch + 1}/{self.epochs}, loss={avg:.4f}")

        # Get final coordinates
        self._encoder.eval()
        with torch.no_grad():
            coords = self._encoder(X_tensor).cpu().numpy()
        return coords

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data through the fitted encoder."""
        if self._encoder is None:
            raise RuntimeError("Must call fit() before transform()")
        device = next(self._encoder.parameters()).device
        self._encoder.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32)).to(device)
            return self._encoder(X_t).cpu().numpy()

    def save(self, path: str):
        if self._encoder is None:
            raise RuntimeError("Must call fit() before save()")
        state = {
            "params": self.get_params(),
            "encoder_state": self._encoder.state_dict(),
            "input_dim": self._input_dim,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> "ParametricTSNE":
        state = torch.load(path, weights_only=False)
        model = cls(**state["params"])
        model._input_dim = state["input_dim"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model._encoder = _Encoder(state["input_dim"], state["params"]["n_components"]).to(device)
        model._encoder.load_state_dict(state["encoder_state"])
        model._encoder.eval()
        return model

    def get_params(self) -> dict:
        return {
            "n_components": self.n_components,
            "perplexity": self.perplexity,
            "dof": self.dof,
            "early_exaggeration": self.early_exaggeration,
            "early_exaggeration_epochs": self.early_exaggeration_epochs,
            "exaggeration": self.exaggeration,
            "metric": self.metric,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "multiscale": self.multiscale,
            "seed": self.seed,
        }

    def _compute_P(self, X: np.ndarray) -> np.ndarray:
        """Compute joint probability matrix P using kNN affinities."""
        n = X.shape[0]

        if self.multiscale:
            H = max(1, round(math.log2(n / 2)))
            perplexities = [2**h for h in range(1, H + 1)]
            logger.info(f"  Multiscale: {len(perplexities)} perplexities from {perplexities[0]} to {perplexities[-1]}")
        else:
            perplexities = [self.perplexity]

        P_sum = np.zeros((n, n), dtype=np.float64)
        for perp in perplexities:
            k = min(int(perp * 3), n - 1)
            index = NNDescent(X, n_neighbors=k + 1, metric=self.metric, random_state=self.seed)
            nn_idx, nn_dist = index.neighbor_graph
            P_perp = self._distances_to_P(nn_idx, nn_dist, n, perp)
            P_sum += P_perp

        P = P_sum / len(perplexities)
        # Symmetrize
        P = (P + P.T) / 2
        P /= P.sum()
        P = np.maximum(P, 1e-12)
        return P

    def _distances_to_P(
        self, nn_idx: np.ndarray, nn_dist: np.ndarray, n: int, perplexity: float
    ) -> np.ndarray:
        """Convert kNN distances to conditional probability matrix."""
        P = np.zeros((n, n), dtype=np.float64)
        target_entropy = np.log2(perplexity)

        for i in range(n):
            dists = nn_dist[i, 1:].astype(np.float64)
            indices = nn_idx[i, 1:]

            sigma = self._binary_search_sigma(dists, target_entropy)
            weights = np.exp(-dists**2 / (2 * sigma**2))
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum
            for j, idx in enumerate(indices):
                P[i, idx] = weights[j]

        return P

    @staticmethod
    def _binary_search_sigma(
        distances: np.ndarray, target_entropy: float, tol: float = 1e-5, max_iter: int = 50
    ) -> float:
        lo, hi = 1e-10, 1e4
        for _ in range(max_iter):
            sigma = (lo + hi) / 2
            weights = np.exp(-distances**2 / (2 * sigma**2))
            w_sum = weights.sum()
            if w_sum == 0:
                lo = sigma
                continue
            p = weights / w_sum
            entropy = -np.sum(p * np.log2(p + 1e-15))
            if abs(entropy - target_entropy) < tol:
                break
            if entropy < target_entropy:
                lo = sigma
            else:
                hi = sigma
        return sigma

    def _kl_divergence(self, P: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """KL(P || Q) where Q uses Student-t kernel with `dof` degrees of freedom."""
        # Pairwise squared distances in embedding space
        diff = Y.unsqueeze(0) - Y.unsqueeze(1)  # (n, n, d)
        dist_sq = (diff**2).sum(dim=-1)  # (n, n)

        # Student-t kernel
        alpha = self.dof
        Q = (1 + dist_sq / alpha) ** (-(alpha + 1) / 2)

        # Zero diagonal
        n = Q.shape[0]
        mask = 1 - torch.eye(n, device=Q.device)
        Q = Q * mask

        # Normalize
        Q = Q / (Q.sum() + 1e-15)
        Q = Q.clamp(min=1e-15)

        # KL divergence
        kl = (P * torch.log(P / Q)).sum()
        return kl
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embed_ptsne.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paloalto/embed/ptsne.py tests/test_embed_ptsne.py
git commit -m "feat: parametric t-SNE in PyTorch (ported from Multiscale-Parametric-t-SNE)"
```

---

### Task 10: Search Space Definitions

**Files:**
- Create: `paloalto/optim/space.py`
- Create: `tests/test_optim_space.py`

- [ ] **Step 1: Write tests**

`tests/test_optim_space.py`:
```python
import pytest
import torch

from paloalto.optim.space import SearchSpace, get_default_space


class TestSearchSpace:
    def test_numap_default_space(self):
        space = get_default_space("numap")
        assert "n_neighbors" in space.params
        assert "min_dist" in space.params
        assert "metric" in space.params

    def test_tsne_default_space(self):
        space = get_default_space("tsne")
        assert "perplexity" in space.params
        assert "dof" in space.params

    def test_to_botorch_bounds(self):
        space = get_default_space("numap")
        bounds, param_names = space.to_botorch_bounds()
        assert bounds.shape[0] == 2  # lower, upper
        assert bounds.shape[1] == len(param_names)
        assert (bounds[0] < bounds[1]).all()

    def test_decode_params(self):
        space = get_default_space("numap")
        bounds, names = space.to_botorch_bounds()
        # Midpoint of bounds
        x = (bounds[0] + bounds[1]) / 2
        decoded = space.decode(x, names)
        assert isinstance(decoded["n_neighbors"], int)
        assert isinstance(decoded["min_dist"], float)
        assert decoded["metric"] in ("euclidean", "cosine", "correlation")

    def test_prune_shrinks_bounds(self):
        space = get_default_space("numap")
        old_lo = space.params["min_dist"]["bounds"][0]
        old_hi = space.params["min_dist"]["bounds"][1]
        space.prune("min_dist", new_lo=0.01, new_hi=0.5)
        assert space.params["min_dist"]["bounds"][0] >= old_lo
        assert space.params["min_dist"]["bounds"][1] <= old_hi
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_optim_space.py -v`
Expected: ImportError

- [ ] **Step 3: Implement search space**

`paloalto/optim/space.py`:
```python
"""Search space definitions for BO."""

import math
from typing import Dict, List, Tuple

import torch


class SearchSpace:
    """Defines the hyperparameter search space for BO."""

    def __init__(self, params: Dict):
        """
        Args:
            params: dict of param_name -> {
                "type": "float" | "int" | "categorical",
                "bounds": [lo, hi] for float/int,
                "choices": [...] for categorical,
                "log_scale": bool (optional),
            }
        """
        self.params = params

    def to_botorch_bounds(self) -> Tuple[torch.Tensor, List[str]]:
        """Convert to BoTorch bounds tensor. Categoricals become one-hot dims."""
        lowers, uppers, names = [], [], []

        for name, spec in self.params.items():
            if spec["type"] in ("float", "int"):
                lo, hi = spec["bounds"]
                if spec.get("log_scale"):
                    lo, hi = math.log(lo), math.log(hi)
                lowers.append(lo)
                uppers.append(hi)
                names.append(name)
            elif spec["type"] == "categorical":
                for choice in spec["choices"]:
                    lowers.append(0.0)
                    uppers.append(1.0)
                    names.append(f"{name}={choice}")

        bounds = torch.tensor([lowers, uppers], dtype=torch.float64)
        return bounds, names

    def decode(self, x: torch.Tensor, param_names: List[str]) -> Dict:
        """Decode a BoTorch tensor back to named hyperparameters."""
        result = {}
        x_np = x.detach().cpu().numpy()
        i = 0

        for name, spec in self.params.items():
            if spec["type"] == "float":
                val = float(x_np[i])
                if spec.get("log_scale"):
                    val = math.exp(val)
                result[name] = val
                i += 1
            elif spec["type"] == "int":
                val = float(x_np[i])
                if spec.get("log_scale"):
                    val = math.exp(val)
                result[name] = int(round(val))
                i += 1
            elif spec["type"] == "categorical":
                n_choices = len(spec["choices"])
                scores = x_np[i:i + n_choices]
                result[name] = spec["choices"][int(scores.argmax())]
                i += n_choices

        return result

    def encode(self, params: Dict) -> List[float]:
        """Encode a parameter dict to a flat list for BoTorch."""
        values = []
        for name, spec in self.params.items():
            val = params[name]
            if spec["type"] in ("float", "int"):
                v = float(val)
                if spec.get("log_scale"):
                    v = math.log(v)
                values.append(v)
            elif spec["type"] == "categorical":
                for choice in spec["choices"]:
                    values.append(1.0 if val == choice else 0.0)
        return values

    def prune(self, param_name: str, new_lo: float = None, new_hi: float = None):
        """Shrink bounds for a parameter (agent PRUNE action)."""
        spec = self.params[param_name]
        if spec["type"] not in ("float", "int"):
            return
        if new_lo is not None:
            spec["bounds"][0] = max(spec["bounds"][0], new_lo)
        if new_hi is not None:
            spec["bounds"][1] = min(spec["bounds"][1], new_hi)


def get_default_space(method: str) -> SearchSpace:
    """Return the default search space for a method."""
    if method == "numap":
        return SearchSpace({
            "n_neighbors": {"type": "int", "bounds": [5, 200], "log_scale": True},
            "min_dist": {"type": "float", "bounds": [0.001, 0.99]},
            "negative_sample_rate": {"type": "int", "bounds": [1, 20]},
            "metric": {"type": "categorical", "choices": ["euclidean", "cosine", "correlation"]},
            "se_dim": {"type": "int", "bounds": [2, 20]},
            "se_neighbors": {"type": "int", "bounds": [5, 100]},
        })
    elif method == "tsne":
        return SearchSpace({
            "perplexity": {"type": "float", "bounds": [5.0, 100.0], "log_scale": True},
            "dof": {"type": "float", "bounds": [0.2, 2.0]},
            "early_exaggeration": {"type": "float", "bounds": [4.0, 32.0], "log_scale": True},
            "metric": {"type": "categorical", "choices": ["euclidean", "cosine", "correlation"]},
            "learning_rate": {"type": "float", "bounds": [1e-4, 1e-1], "log_scale": True},
            "exaggeration": {"type": "float", "bounds": [1.0, 4.0]},
        })
    else:
        raise ValueError(f"Unknown method: {method}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_optim_space.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paloalto/optim/space.py tests/test_optim_space.py
git commit -m "feat: BO search space definitions with encode/decode/prune"
```

---

### Task 11: Weighted Scalar BO

**Files:**
- Create: `paloalto/optim/bo.py`
- Modify: `paloalto/optim/__init__.py`
- Create: `tests/test_optim_bo.py`

- [ ] **Step 1: Write tests**

`tests/test_optim_bo.py`:
```python
import pytest
import torch
import numpy as np

from paloalto.optim.bo import WeightedBO
from paloalto.optim.space import get_default_space


class TestWeightedBO:
    def test_sobol_initialization(self):
        space = get_default_space("numap")
        bo = WeightedBO(space, weights=[0.5, 0.5], seed=42)
        candidates = bo.suggest_initial(n=5)
        assert len(candidates) == 5
        for c in candidates:
            assert "n_neighbors" in c
            assert "min_dist" in c

    def test_suggest_after_observations(self):
        space = get_default_space("numap")
        bo = WeightedBO(space, weights=[0.5, 0.5], seed=42)

        # Add some fake observations
        for i in range(5):
            params = bo.suggest_initial(n=1)[0]
            bo.observe(params, scib_overall=np.random.rand(), scgraph_score=np.random.rand())

        # Now GP-based suggestion should work
        candidate = bo.suggest()
        assert "n_neighbors" in candidate
        assert "min_dist" in candidate

    def test_best_returns_highest_score(self):
        space = get_default_space("numap")
        bo = WeightedBO(space, weights=[0.5, 0.5], seed=42)

        bo.observe({"n_neighbors": 10, "min_dist": 0.1, "negative_sample_rate": 5,
                     "metric": "euclidean", "se_dim": 2, "se_neighbors": 10},
                    scib_overall=0.3, scgraph_score=0.4)
        bo.observe({"n_neighbors": 15, "min_dist": 0.2, "negative_sample_rate": 5,
                     "metric": "cosine", "se_dim": 5, "se_neighbors": 15},
                    scib_overall=0.8, scgraph_score=0.9)

        best = bo.best()
        assert best["scores"]["scib_overall"] == 0.8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_optim_bo.py -v`
Expected: ImportError

- [ ] **Step 3: Implement WeightedBO**

`paloalto/optim/__init__.py`:
```python
"""Optimization module."""
```

`paloalto/optim/bo.py`:
```python
"""Weighted scalar Bayesian optimization (single-objective)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from paloalto.optim.space import SearchSpace
from paloalto.utils import get_logger

logger = get_logger(__name__)


class WeightedBO:
    """Single-objective BO with weighted scalar objective."""

    def __init__(
        self,
        space: SearchSpace,
        weights: List[float] = None,
        seed: int = 42,
    ):
        self.space = space
        self.weights = weights or [0.5, 0.5]
        self.seed = seed
        self.bounds, self.param_names = space.to_botorch_bounds()

        self.X: List[torch.Tensor] = []
        self.Y: List[float] = []
        self.history: List[Dict] = []

    def suggest_initial(self, n: int = 5) -> List[Dict]:
        """Generate n Sobol quasi-random candidates."""
        sobol = SobolEngine(dimension=self.bounds.shape[1], scramble=True, seed=self.seed)
        raw = sobol.draw(n).to(dtype=self.bounds.dtype)
        # Scale to bounds
        scaled = raw * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        candidates = []
        for i in range(n):
            params = self.space.decode(scaled[i], self.param_names)
            candidates.append(params)
        return candidates

    def observe(self, params: Dict, scib_overall: float, scgraph_score: float):
        """Record an observation."""
        x = self.space.encode(params)
        self.X.append(torch.tensor(x, dtype=torch.float64))
        objective = self.weights[0] * scib_overall + self.weights[1] * scgraph_score
        self.Y.append(objective)
        self.history.append({
            "params": params,
            "scores": {"scib_overall": scib_overall, "scgraph_score": scgraph_score},
            "objective": objective,
        })

    def suggest(self) -> Dict:
        """Suggest next candidate via GP + Expected Improvement."""
        train_X = torch.stack(self.X).unsqueeze(0) if len(self.X) == 1 else torch.stack(self.X)
        train_Y = torch.tensor(self.Y, dtype=torch.float64).unsqueeze(-1)

        # Fit GP
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Optimize acquisition
        acq = LogExpectedImprovement(model=gp, best_f=train_Y.max())
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )

        return self.space.decode(candidate.squeeze(0), self.param_names)

    def best(self) -> Dict:
        """Return the best trial so far."""
        best_idx = int(np.argmax(self.Y))
        return self.history[best_idx]

    def get_history(self) -> List[Dict]:
        return self.history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_optim_bo.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paloalto/optim/__init__.py paloalto/optim/bo.py tests/test_optim_bo.py
git commit -m "feat: weighted scalar BO with GP + Expected Improvement"
```

---

### Task 12: Multi-Objective BO (Pareto)

**Files:**
- Create: `paloalto/optim/mobo.py`
- Create: `tests/test_optim_mobo.py`

- [ ] **Step 1: Write tests**

`tests/test_optim_mobo.py`:
```python
import pytest
import numpy as np

from paloalto.optim.mobo import ParetoBO
from paloalto.optim.space import get_default_space


class TestParetoBO:
    def test_sobol_initialization(self):
        space = get_default_space("numap")
        bo = ParetoBO(space, seed=42)
        candidates = bo.suggest_initial(n=5)
        assert len(candidates) == 5

    def test_suggest_after_observations(self):
        space = get_default_space("numap")
        bo = ParetoBO(space, seed=42)
        for i in range(5):
            params = bo.suggest_initial(n=1)[0]
            bo.observe(params, scib_overall=np.random.rand(), scgraph_score=np.random.rand())

        candidate = bo.suggest()
        assert "n_neighbors" in candidate

    def test_pareto_front(self):
        space = get_default_space("numap")
        bo = ParetoBO(space, seed=42)

        bo.observe({"n_neighbors": 10, "min_dist": 0.1, "negative_sample_rate": 5,
                     "metric": "euclidean", "se_dim": 2, "se_neighbors": 10},
                    scib_overall=0.9, scgraph_score=0.3)
        bo.observe({"n_neighbors": 15, "min_dist": 0.2, "negative_sample_rate": 5,
                     "metric": "cosine", "se_dim": 5, "se_neighbors": 15},
                    scib_overall=0.3, scgraph_score=0.9)
        bo.observe({"n_neighbors": 20, "min_dist": 0.3, "negative_sample_rate": 5,
                     "metric": "euclidean", "se_dim": 3, "se_neighbors": 20},
                    scib_overall=0.5, scgraph_score=0.5)

        front = bo.pareto_front()
        # First two dominate the third on at least one objective
        assert len(front) >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_optim_mobo.py -v`
Expected: ImportError

- [ ] **Step 3: Implement ParetoBO**

`paloalto/optim/mobo.py`:
```python
"""Multi-objective BO via NEHVI for 2D Pareto optimization."""

from typing import Dict, List

import numpy as np
import torch
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from paloalto.optim.space import SearchSpace
from paloalto.utils import get_logger

logger = get_logger(__name__)

REF_POINT = torch.tensor([0.0, 0.0], dtype=torch.float64)


class ParetoBO:
    """Multi-objective BO with NEHVI acquisition."""

    def __init__(self, space: SearchSpace, seed: int = 42):
        self.space = space
        self.seed = seed
        self.bounds, self.param_names = space.to_botorch_bounds()

        self.X: List[torch.Tensor] = []
        self.Y: List[torch.Tensor] = []  # (scib_overall, scgraph_score)
        self.history: List[Dict] = []

    def suggest_initial(self, n: int = 5) -> List[Dict]:
        sobol = SobolEngine(dimension=self.bounds.shape[1], scramble=True, seed=self.seed)
        raw = sobol.draw(n).to(dtype=self.bounds.dtype)
        scaled = raw * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return [self.space.decode(scaled[i], self.param_names) for i in range(n)]

    def observe(self, params: Dict, scib_overall: float, scgraph_score: float):
        x = self.space.encode(params)
        self.X.append(torch.tensor(x, dtype=torch.float64))
        self.Y.append(torch.tensor([scib_overall, scgraph_score], dtype=torch.float64))
        self.history.append({
            "params": params,
            "scores": {"scib_overall": scib_overall, "scgraph_score": scgraph_score},
        })

    def suggest(self) -> Dict:
        train_X = torch.stack(self.X)
        train_Y = torch.stack(self.Y)

        # Fit independent GPs per objective
        models = []
        for i in range(2):
            gp = SingleTaskGP(train_X, train_Y[:, i:i+1])
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            models.append(gp)

        model = ModelListGP(*models)

        acq = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=REF_POINT,
            X_baseline=train_X,
            prune_baseline=True,
        )

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )

        return self.space.decode(candidate.squeeze(0), self.param_names)

    def pareto_front(self) -> List[Dict]:
        """Return non-dominated trials."""
        if not self.Y:
            return []
        Y = torch.stack(self.Y)
        mask = is_non_dominated(Y)
        return [self.history[i] for i in range(len(self.history)) if mask[i]]

    def get_history(self) -> List[Dict]:
        return self.history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_optim_mobo.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paloalto/optim/mobo.py tests/test_optim_mobo.py
git commit -m "feat: multi-objective BO with NEHVI for 2D Pareto front"
```

---

### Task 13: Agent Judge

**Files:**
- Create: `paloalto/agent/judge.py`
- Modify: `paloalto/agent/__init__.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write tests**

`tests/test_agent.py`:
```python
import pytest
import json

from paloalto.agent.judge import AgentJudge, AgentAction


class TestAgentAction:
    def test_accept_action(self):
        action = AgentAction(action="accept", reasoning="Looks good")
        assert action.action == "accept"
        assert action.params is None

    def test_modify_action(self):
        action = AgentAction(
            action="modify",
            params={"n_neighbors": 20},
            reasoning="Higher n_neighbors based on trend",
        )
        assert action.params["n_neighbors"] == 20

    def test_prune_action(self):
        action = AgentAction(
            action="prune",
            new_bounds={"min_dist": [0.01, 0.5]},
            reasoning="Good results concentrated in low min_dist",
        )
        assert action.new_bounds["min_dist"] == [0.01, 0.5]


class TestAgentJudge:
    def test_build_context(self):
        judge = AgentJudge(model="claude-sonnet-4-6")
        context = judge.build_context(
            trial_number=6,
            trial_history=[
                {"params": {"n_neighbors": 10}, "scores": {"scib_overall": 0.5, "scgraph_score": 0.6}},
            ],
            current_best={"params": {"n_neighbors": 10}, "scores": {"scib_overall": 0.5, "scgraph_score": 0.6}},
            bo_suggestion={"n_neighbors": 15, "min_dist": 0.2},
            search_space={"n_neighbors": {"type": "int", "bounds": [5, 200]}},
            dataset_summary={"n_cells": 10000, "n_batches": 3, "n_types": 8},
        )
        assert "trial_number" in context
        assert context["trial_number"] == 6

    def test_parse_response_accept(self):
        judge = AgentJudge(model="claude-sonnet-4-6")
        raw = '{"action": "accept", "reasoning": "Suggestion looks reasonable"}'
        action = judge.parse_response(raw)
        assert action.action == "accept"

    def test_parse_response_fallback(self):
        judge = AgentJudge(model="claude-sonnet-4-6")
        action = judge.parse_response("this is not valid json at all")
        assert action.action == "accept"  # safe fallback
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v`
Expected: ImportError

- [ ] **Step 3: Implement AgentJudge**

`paloalto/agent/__init__.py`:
```python
"""Agent module."""
```

`paloalto/agent/judge.py`:
```python
"""LLM Agent Judge for BO trial review."""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from paloalto.utils import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are an expert judge for Bayesian optimization of single-cell embedding visualizations.

You review each BO trial: the suggested hyperparameters, the trial history, and the current best.
You can take one of these actions:

- ACCEPT: Use the BO suggestion as-is. Choose this when the suggestion looks reasonable.
- MODIFY: Adjust specific parameters. Provide the modified params dict. Stay within search space bounds.
- INJECT: Replace the suggestion entirely with your own candidate. Stay within bounds.
- PRUNE: Shrink the search space bounds for one or more parameters. Only shrink, never expand.
- STOP: Recommend early stopping if the optimization has converged.

Respond with a JSON object:
{
    "action": "accept" | "modify" | "inject" | "prune" | "stop",
    "params": {...},           // for modify/inject only
    "new_bounds": {...},       // for prune only, e.g. {"min_dist": [0.01, 0.5]}
    "reasoning": "..."         // required: explain your decision
}

Guidelines:
- If recent trials show a clear trend (e.g., lower min_dist is always better), consider PRUNE or MODIFY.
- If the BO is exploring a region that has consistently produced poor results, MODIFY away from it.
- If the best score hasn't improved in 5+ trials and the Pareto front is stable, consider STOP.
- Be conservative: ACCEPT is the default. Only intervene when you have clear evidence.
- Look for patterns in what makes good vs. bad embeddings for this dataset.
"""


@dataclass
class AgentAction:
    action: str  # accept, modify, inject, prune, stop
    params: Optional[Dict] = None
    new_bounds: Optional[Dict] = None
    reasoning: str = ""


class AgentJudge:
    """LLM-based judge for reviewing BO trial suggestions."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str = None):
        self.model = model
        self.api_key = api_key
        self.log: List[Dict] = []

    def build_context(
        self,
        trial_number: int,
        trial_history: List[Dict],
        current_best: Dict,
        bo_suggestion: Dict,
        search_space: Dict,
        dataset_summary: Dict,
    ) -> Dict:
        """Build the context dict for the agent prompt."""
        # Compute recent trend
        if len(trial_history) >= 3:
            recent_scores = [
                t["scores"].get("scib_overall", 0) + t["scores"].get("scgraph_score", 0)
                for t in trial_history[-3:]
            ]
            if recent_scores[-1] > recent_scores[0]:
                trend = "improving"
            elif recent_scores[-1] < recent_scores[0] - 0.05:
                trend = "degrading"
            else:
                trend = "flat"
        else:
            trend = "too_early"

        return {
            "trial_number": trial_number,
            "trial_history": trial_history,
            "current_best": current_best,
            "bo_suggestion": bo_suggestion,
            "search_space": search_space,
            "dataset_summary": dataset_summary,
            "recent_trend": trend,
        }

    def review(self, context: Dict) -> AgentAction:
        """Call the LLM to review the trial. Returns an AgentAction."""
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic package not installed, defaulting to ACCEPT")
            return AgentAction(action="accept", reasoning="anthropic not available")

        client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else anthropic.Anthropic()

        user_msg = json.dumps(context, indent=2, default=str)

        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw_text = response.content[0].text
        action = self.parse_response(raw_text)

        self.log.append({
            "trial_number": context["trial_number"],
            "action": action.action,
            "reasoning": action.reasoning,
            "params": action.params,
            "new_bounds": action.new_bounds,
        })

        logger.info(f"  Agent [{action.action}]: {action.reasoning[:100]}")
        return action

    def parse_response(self, raw: str) -> AgentAction:
        """Parse JSON response from LLM, with fallback to ACCEPT."""
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```" in raw:
                start = raw.index("```") + 3
                if raw[start:start + 4] == "json":
                    start += 4
                end = raw.index("```", start)
                raw = raw[start:end].strip()
            data = json.loads(raw)
            return AgentAction(
                action=data.get("action", "accept"),
                params=data.get("params"),
                new_bounds=data.get("new_bounds"),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Failed to parse agent response, defaulting to ACCEPT")
            return AgentAction(action="accept", reasoning="Parse failure, defaulting to accept")

    def get_log(self) -> List[Dict]:
        return self.log
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paloalto/agent/__init__.py paloalto/agent/judge.py tests/test_agent.py
git commit -m "feat: LLM agent judge for BO trial review"
```

---

### Task 14: Report Generation

**Files:**
- Create: `paloalto/report/html.py`
- Create: `paloalto/report/templates/report.html.j2`
- Modify: `paloalto/report/__init__.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write tests**

`tests/test_report.py`:
```python
import os
import pytest
import numpy as np

from paloalto.report.html import generate_report


@pytest.fixture
def mock_results():
    np.random.seed(42)
    n_trials = 10
    history = []
    for i in range(n_trials):
        history.append({
            "params": {"n_neighbors": np.random.randint(5, 50), "min_dist": np.random.rand()},
            "scores": {"scib_overall": np.random.rand(), "scgraph_score": np.random.rand()},
            "objective": np.random.rand(),
        })
    return {
        "method": "numap",
        "bo_mode": "weighted",
        "dataset_summary": {"n_cells": 10000, "n_batches": 3, "n_types": 8},
        "agent_history": history,
        "baseline_history": history,
        "agent_log": [{"trial_number": 6, "action": "modify", "reasoning": "test"}],
        "best_agent": history[5],
        "best_baseline": history[3],
        "best_coords_agent": np.random.randn(100, 2),
        "best_coords_baseline": np.random.randn(100, 2),
        "labels": np.array(["A"] * 50 + ["B"] * 50),
        "batches": np.array(["b0"] * 50 + ["b1"] * 50),
        "wall_time_seconds": 120.5,
    }


class TestGenerateReport:
    def test_creates_html_file(self, mock_results, tmp_path):
        path = str(tmp_path / "report.html")
        generate_report(mock_results, path)
        assert os.path.exists(path)

    def test_html_contains_key_sections(self, mock_results, tmp_path):
        path = str(tmp_path / "report.html")
        generate_report(mock_results, path)
        with open(path) as f:
            html = f.read()
        assert "Overview" in html
        assert "Convergence" in html
        assert "Best Parameters" in html
        assert "Agent Log" in html
        assert "Caveats" in html

    def test_self_contained(self, mock_results, tmp_path):
        path = str(tmp_path / "report.html")
        generate_report(mock_results, path)
        with open(path) as f:
            html = f.read()
        # All images should be base64 encoded
        assert "data:image/png;base64" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_report.py -v`
Expected: ImportError

- [ ] **Step 3: Create Jinja2 template**

`paloalto/report/templates/report.html.j2`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PALO ALTO Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1000px; margin: 0 auto; padding: 20px; color: #333; }
  h1 { border-bottom: 2px solid #2c5f8a; padding-bottom: 8px; }
  h2 { color: #2c5f8a; margin-top: 40px; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; }
  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
  th { background-color: #f5f5f5; }
  .metric-good { color: #2e7d32; font-weight: bold; }
  .metric-bad { color: #c62828; }
  .plot { text-align: center; margin: 20px 0; }
  .plot img { max-width: 100%; }
  .caveat { background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 16px 0; }
  details { margin: 10px 0; }
  summary { cursor: pointer; font-weight: bold; }
  .agent-log { background: #f8f9fa; padding: 12px; border-radius: 4px; margin: 8px 0; }
  .agent-action { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; }
  .action-accept { background: #d4edda; }
  .action-modify { background: #cce5ff; }
  .action-inject { background: #e2d5f1; }
  .action-prune { background: #fff3cd; }
  .action-stop { background: #f8d7da; }
</style>
</head>
<body>

<h1>PALO ALTO Report</h1>
<p><em>Pareto-guided Automatic Layout Optimization for Aligning Latent Topology in Omics embeddings</em></p>

<h2>1. Overview</h2>
<table>
  <tr><th>Method</th><td>{{ method }}</td></tr>
  <tr><th>BO Mode</th><td>{{ bo_mode }}</td></tr>
  <tr><th>Cells</th><td>{{ dataset_summary.n_cells | default("N/A") }}</td></tr>
  <tr><th>Batches</th><td>{{ dataset_summary.n_batches | default("N/A") }}</td></tr>
  <tr><th>Cell Types</th><td>{{ dataset_summary.n_types | default("N/A") }}</td></tr>
  <tr><th>Total Trials</th><td>{{ agent_history | length }}</td></tr>
  <tr><th>Wall Time</th><td>{{ "%.1f" | format(wall_time_seconds) }}s</td></tr>
</table>

<h2>2. Convergence</h2>
<div class="plot">{{ convergence_plot }}</div>

<h2>3. Best Parameters</h2>
<h3>Agent-guided Best</h3>
<table>
  {% for k, v in best_agent.params.items() %}
  <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
  {% endfor %}
  <tr><th>scIB Overall</th><td class="metric-good">{{ "%.4f" | format(best_agent.scores.scib_overall) }}</td></tr>
  <tr><th>scGraph Score</th><td class="metric-good">{{ "%.4f" | format(best_agent.scores.scgraph_score) }}</td></tr>
</table>

<h3>Baseline Best</h3>
<table>
  {% for k, v in best_baseline.params.items() %}
  <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
  {% endfor %}
  <tr><th>scIB Overall</th><td>{{ "%.4f" | format(best_baseline.scores.scib_overall) }}</td></tr>
  <tr><th>scGraph Score</th><td>{{ "%.4f" | format(best_baseline.scores.scgraph_score) }}</td></tr>
</table>

<h2>4. Best Embedding Visualization</h2>
<div class="plot">{{ embedding_plot }}</div>

<h2>5. Metric Breakdown</h2>
<details>
  <summary>Click to expand per-trial metrics</summary>
  <table>
    <tr><th>Trial</th><th>scIB Overall</th><th>scGraph Score</th><th>Arm</th></tr>
    {% for i, t in enumerate(agent_history) %}
    <tr>
      <td>{{ i + 1 }}</td>
      <td>{{ "%.4f" | format(t.scores.scib_overall) }}</td>
      <td>{{ "%.4f" | format(t.scores.scgraph_score) }}</td>
      <td>Agent</td>
    </tr>
    {% endfor %}
  </table>
</details>

<h2>6. Agent Log</h2>
{% for entry in agent_log %}
<div class="agent-log">
  <strong>Trial {{ entry.trial_number }}</strong>
  <span class="agent-action action-{{ entry.action }}">{{ entry.action | upper }}</span>
  <p>{{ entry.reasoning }}</p>
</div>
{% endfor %}

<h2>7. Caveats</h2>
<div class="caveat">
  <strong>Optimizing 2D projections ≠ optimizing biological representation quality.</strong>
  These metrics evaluate how well a 2D visualization preserves structure, not whether the
  underlying representation is biologically meaningful.
</div>
<div class="caveat">
  <strong>Parameters may not transfer.</strong> Optimal hyperparameters depend on the dataset,
  embedding method, and cell type composition. Re-optimize for new datasets.
</div>
<div class="caveat">
  <strong>Metric limitations.</strong> No single metric captures all aspects of visualization quality.
  Always inspect the embedding visually.
</div>

</body>
</html>
```

- [ ] **Step 4: Implement report generation**

`paloalto/report/__init__.py`:
```python
"""Report module."""
```

`paloalto/report/html.py`:
```python
"""HTML report generation with embedded plots."""

import base64
import io
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader

from paloalto.utils import get_logger

logger = get_logger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_report(results: Dict, output_path: str):
    """Generate a self-contained HTML report."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("report.html.j2")

    # Generate plots
    convergence_plot = _make_convergence_plot(
        results["agent_history"], results["baseline_history"], results.get("agent_log", [])
    )
    embedding_plot = _make_embedding_plot(
        results.get("best_coords_agent"),
        results.get("best_coords_baseline"),
        results.get("labels"),
        results.get("batches"),
    )

    html = template.render(
        **results,
        convergence_plot=convergence_plot,
        embedding_plot=embedding_plot,
        enumerate=enumerate,
    )

    Path(output_path).write_text(html)
    logger.info(f"Report saved to {output_path}")


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" />'


def _make_convergence_plot(agent_history, baseline_history, agent_log) -> str:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    def best_so_far(history):
        bsf = []
        best = -np.inf
        for t in history:
            s = t["scores"]["scib_overall"] + t["scores"]["scgraph_score"]
            best = max(best, s)
            bsf.append(best)
        return bsf

    agent_bsf = best_so_far(agent_history)
    baseline_bsf = best_so_far(baseline_history)
    trials = range(1, len(agent_bsf) + 1)

    ax.plot(trials, agent_bsf, "b-o", label="Agent-guided", markersize=4)
    ax.plot(trials, baseline_bsf, "r--s", label="Baseline (BoTorch)", markersize=4)

    # Mark agent interventions
    for entry in agent_log:
        t = entry.get("trial_number", 0)
        if 1 <= t <= len(agent_bsf):
            marker = {"modify": "^", "inject": "D", "prune": "v", "stop": "x"}.get(entry["action"])
            if marker:
                ax.plot(t, agent_bsf[t - 1], marker, color="green", markersize=10, zorder=5)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Score (scIB + scGraph)")
    ax.set_title("Convergence: Agent-guided vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _make_embedding_plot(coords_agent, coords_baseline, labels, batches) -> str:
    if coords_agent is None:
        return "<p>No embedding data available.</p>"

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    def scatter(ax, coords, colors, title, palette=None):
        unique = np.unique(colors)
        if palette is None:
            cmap = plt.cm.get_cmap("tab20", len(unique))
            palette = {u: cmap(i) for i, u in enumerate(unique)}
        for u in unique:
            mask = colors == u
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[palette[u]], label=u, s=3, alpha=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    scatter(axes[0, 0], coords_agent, labels, "Agent Best — Cell Type")
    scatter(axes[0, 1], coords_agent, batches, "Agent Best — Batch")
    scatter(axes[1, 0], coords_baseline, labels, "Baseline Best — Cell Type")
    scatter(axes[1, 1], coords_baseline, batches, "Baseline Best — Batch")

    for ax in axes.flat:
        ax.legend(markerscale=3, fontsize=7, loc="best")

    fig.tight_layout()
    return _fig_to_base64(fig)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_report.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add paloalto/report/__init__.py paloalto/report/html.py paloalto/report/templates/report.html.j2 tests/test_report.py
git commit -m "feat: HTML report generation with convergence plots and embedding visualization"
```

---

### Task 15: Top-Level API (`optimize()`)

**Files:**
- Modify: `paloalto/__init__.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write tests**

`tests/test_api.py`:
```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from paloalto import optimize


class TestOptimize:
    def test_runs_weighted_mode(self, mock_adata, tmp_path):
        """Smoke test: 2 Sobol trials, no agent, weighted mode."""
        result = optimize(
            adata=mock_adata,
            embedding_key="X_pca",
            batch_key="batch",
            label_key="cell_type",
            method="numap",
            bo_mode="weighted",
            n_trials=2,
            n_initial=2,
            use_agent=False,
            subsample_n=200,
            output_dir=str(tmp_path),
            seed=42,
            embed_kwargs={"epochs": 2},
        )
        assert "best_params" in result
        assert "best_scores" in result
        assert "trial_history" in result
        assert "report_path" in result
        assert len(result["trial_history"]) == 2

    def test_runs_pareto_mode(self, mock_adata, tmp_path):
        result = optimize(
            adata=mock_adata,
            embedding_key="X_pca",
            batch_key="batch",
            label_key="cell_type",
            method="numap",
            bo_mode="pareto",
            n_trials=2,
            n_initial=2,
            use_agent=False,
            subsample_n=200,
            output_dir=str(tmp_path),
            seed=42,
            embed_kwargs={"epochs": 2},
        )
        assert "pareto_front" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py -v`
Expected: ImportError (optimize not defined)

- [ ] **Step 3: Implement optimize()**

`paloalto/__init__.py`:
```python
"""PALO ALTO: Pareto-guided Automatic Layout Optimization for Aligning Latent Topology in Omics embeddings."""

__version__ = "0.1.0"

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from paloalto.data import load_and_validate, subsample_stratified
from paloalto.metrics import compute_all
from paloalto.embed import get_embedder
from paloalto.optim.space import get_default_space
from paloalto.optim.bo import WeightedBO
from paloalto.optim.mobo import ParetoBO
from paloalto.agent.judge import AgentJudge
from paloalto.report.html import generate_report
from paloalto.utils import get_logger

logger = get_logger(__name__)


def optimize(
    adata,
    embedding_key: str = "X_pca",
    batch_key: str = "batch",
    label_key: str = "cell_type",
    method: str = "numap",
    bo_mode: str = "weighted",
    weights: List[float] = None,
    n_trials: int = 30,
    n_initial: int = 5,
    use_agent: bool = True,
    agent_review_interval: int = 1,
    agent_model: str = "claude-sonnet-4-6",
    subsample_n: int = 10000,
    output_dir: str = "./results",
    seed: int = 42,
    search_space: Optional[Dict] = None,
    embed_kwargs: Optional[Dict] = None,
) -> Dict:
    """Run PALO ALTO optimization.

    Args:
        adata: Path to .h5ad file or AnnData object.
        embedding_key: Key in adata.obsm for the input embedding.
        batch_key: Column in adata.obs for batch labels.
        label_key: Column in adata.obs for cell type labels.
        method: "numap" or "tsne".
        bo_mode: "weighted" (single objective) or "pareto" (multi-objective).
        weights: [w_scib, w_scgraph] for weighted mode. Default [0.5, 0.5].
        n_trials: Total number of BO trials.
        n_initial: Number of Sobol initialization trials.
        use_agent: Whether to use the LLM agent judge.
        agent_review_interval: Agent reviews every N trials.
        agent_model: Model name for the agent.
        subsample_n: Number of cells for metric computation.
        output_dir: Directory for outputs.
        seed: Random seed.
        search_space: Custom search space dict (overrides defaults).
        embed_kwargs: Extra kwargs passed to the embedder (e.g., epochs).

    Returns:
        Dict with best_params, best_scores, best_model, trial_history,
        pareto_front (if pareto mode), report_path.
    """
    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embed_kwargs = embed_kwargs or {}

    # Load and validate
    adata = load_and_validate(adata, embedding_key, batch_key, label_key)

    # Subsample indices (fixed across all trials)
    sub_idx = subsample_stratified(adata, label_key, batch_key, n=subsample_n, seed=seed)
    logger.info(f"Metric subsample: {len(sub_idx)} cells")

    dataset_summary = {
        "n_cells": adata.n_obs,
        "n_batches": adata.obs[batch_key].nunique(),
        "n_types": adata.obs[label_key].nunique(),
    }

    # Search space
    space_agent = get_default_space(method) if search_space is None else _parse_custom_space(search_space)
    space_baseline = get_default_space(method) if search_space is None else _parse_custom_space(search_space)

    # Initialize BO for both arms
    if bo_mode == "weighted":
        bo_agent = WeightedBO(space_agent, weights=weights, seed=seed)
        bo_baseline = WeightedBO(space_baseline, weights=weights, seed=seed)
    else:
        bo_agent = ParetoBO(space_agent, seed=seed)
        bo_baseline = ParetoBO(space_baseline, seed=seed)

    # Agent
    agent = AgentJudge(model=agent_model) if use_agent else None

    # Sobol initialization (shared)
    logger.info(f"Phase 1: {n_initial} Sobol trials")
    initial_candidates = bo_agent.suggest_initial(n=n_initial)
    for i, params in enumerate(initial_candidates):
        logger.info(f"Trial {i + 1}/{n_trials} (Sobol)")
        scores = _evaluate_trial(adata, embedding_key, batch_key, label_key, method, params, sub_idx, embed_kwargs)
        bo_agent.observe(params, **scores)
        bo_baseline.observe(params, **scores)

    # BO loop
    logger.info(f"Phase 2: {n_trials - n_initial} BO trials")
    for trial_num in range(n_initial + 1, n_trials + 1):
        logger.info(f"Trial {trial_num}/{n_trials}")

        # --- Agent arm ---
        agent_suggestion = bo_agent.suggest()
        final_params = agent_suggestion

        if agent and (trial_num - n_initial) % agent_review_interval == 0:
            context = agent.build_context(
                trial_number=trial_num,
                trial_history=bo_agent.get_history(),
                current_best=bo_agent.best() if bo_mode == "weighted" else bo_agent.get_history()[-1],
                bo_suggestion=agent_suggestion,
                search_space={k: v for k, v in space_agent.params.items()},
                dataset_summary=dataset_summary,
            )
            action = agent.review(context)

            if action.action == "modify" and action.params:
                final_params = {**agent_suggestion, **action.params}
            elif action.action == "inject" and action.params:
                final_params = action.params
            elif action.action == "prune" and action.new_bounds:
                for param_name, (lo, hi) in action.new_bounds.items():
                    space_agent.prune(param_name, new_lo=lo, new_hi=hi)
            elif action.action == "stop":
                logger.info("Agent recommends early stopping")
                break

        scores_agent = _evaluate_trial(adata, embedding_key, batch_key, label_key, method, final_params, sub_idx, embed_kwargs)
        bo_agent.observe(final_params, **scores_agent)

        # --- Baseline arm ---
        baseline_suggestion = bo_baseline.suggest()
        scores_baseline = _evaluate_trial(adata, embedding_key, batch_key, label_key, method, baseline_suggestion, sub_idx, embed_kwargs)
        bo_baseline.observe(baseline_suggestion, **scores_baseline)

    wall_time = time.time() - start_time

    # Best results
    if bo_mode == "weighted":
        best_agent = bo_agent.best()
        best_baseline = bo_baseline.best()
        pareto_front = None
    else:
        best_agent = bo_agent.get_history()[-1]  # last for Pareto
        best_baseline = bo_baseline.get_history()[-1]
        pareto_front = bo_agent.pareto_front()

    # Get best embedding coords for the report
    best_coords_agent = _get_best_coords(adata, embedding_key, method, best_agent["params"], embed_kwargs)
    best_coords_baseline = _get_best_coords(adata, embedding_key, method, best_baseline["params"], embed_kwargs)

    # Generate report
    report_path = str(output_dir / "paloalto_report.html")
    report_data = {
        "method": method,
        "bo_mode": bo_mode,
        "dataset_summary": dataset_summary,
        "agent_history": bo_agent.get_history(),
        "baseline_history": bo_baseline.get_history(),
        "agent_log": agent.get_log() if agent else [],
        "best_agent": best_agent,
        "best_baseline": best_baseline,
        "best_coords_agent": best_coords_agent,
        "best_coords_baseline": best_coords_baseline,
        "labels": adata.obs[label_key].values,
        "batches": adata.obs[batch_key].values,
        "wall_time_seconds": wall_time,
    }
    generate_report(report_data, report_path)

    return {
        "best_params": best_agent["params"],
        "best_scores": best_agent["scores"],
        "trial_history": bo_agent.get_history(),
        "baseline_history": bo_baseline.get_history(),
        "pareto_front": pareto_front,
        "report_path": report_path,
    }


def _evaluate_trial(adata, embedding_key, batch_key, label_key, method, params, sub_idx, embed_kwargs):
    """Fit embedding and compute metrics for one trial."""
    embedder = get_embedder(method, **{k: v for k, v in params.items()}, **embed_kwargs)
    coords = embedder.fit(adata, embedding_key=embedding_key)

    # Store in adata temporarily for metric computation
    adata.obsm["_paloalto_trial"] = coords
    sub_adata = adata[sub_idx].copy()

    scores = compute_all(sub_adata, "_paloalto_trial", label_key, batch_key)
    return {
        "scib_overall": scores["scib_overall"],
        "scgraph_score": scores["scgraph_corr_weighted"],
    }


def _get_best_coords(adata, embedding_key, method, params, embed_kwargs):
    """Re-fit the best model and return 2D coords."""
    try:
        embedder = get_embedder(method, **params, **embed_kwargs)
        return embedder.fit(adata, embedding_key=embedding_key)
    except Exception as e:
        logger.warning(f"Failed to re-fit best model: {e}")
        return np.random.randn(adata.n_obs, 2)


def _parse_custom_space(space_dict):
    from paloalto.optim.space import SearchSpace
    return SearchSpace(space_dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api.py -v -x --timeout=300`
Expected: Both tests PASS (these are slow integration tests)

- [ ] **Step 5: Commit**

```bash
git add paloalto/__init__.py tests/test_api.py
git commit -m "feat: top-level optimize() API with agent-in-the-loop BO"
```

---

### Task 16: CLI

**Files:**
- Create: `paloalto/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write tests**

`tests/test_cli.py`:
```python
import pytest
from click.testing import CliRunner

from paloalto.cli import main


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "PALO ALTO" in result.output

    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--embedding" in result.output

    def test_metrics_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["metrics", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output

    def test_run_with_h5ad(self, mock_adata_path, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, [
            "run",
            "--input", mock_adata_path,
            "--embedding", "X_pca",
            "--batch-key", "batch",
            "--label-key", "cell_type",
            "--method", "numap",
            "--n-trials", "2",
            "--n-initial", "2",
            "--no-agent",
            "--output-dir", str(tmp_path),
            "--embed-epochs", "2",
        ])
        assert result.exit_code == 0, result.output

    def test_metrics_command(self, mock_adata_path):
        runner = CliRunner()
        result = runner.invoke(main, [
            "metrics",
            "--input", mock_adata_path,
            "--embedding", "X_pca",
            "--batch-key", "batch",
            "--label-key", "cell_type",
        ])
        assert result.exit_code == 0
        assert "scib_overall" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: ImportError

- [ ] **Step 3: Implement CLI**

`paloalto/cli.py`:
```python
"""PALO ALTO CLI."""

import json
import yaml
import click

from paloalto.utils import get_logger

logger = get_logger(__name__)


@click.group()
def main():
    """PALO ALTO: Pareto-guided Automatic Layout Optimization for Aligning Latent Topology in Omics embeddings."""
    pass


@main.command()
@click.option("--input", "input_path", required=True, help="Path to .h5ad file")
@click.option("--embedding", default="X_pca", help="Key in adata.obsm")
@click.option("--batch-key", default="batch", help="Batch column in adata.obs")
@click.option("--label-key", default="cell_type", help="Cell type column in adata.obs")
@click.option("--method", default="numap", type=click.Choice(["numap", "tsne"]))
@click.option("--bo-mode", default="weighted", type=click.Choice(["weighted", "pareto"]))
@click.option("--n-trials", default=30, type=int)
@click.option("--n-initial", default=5, type=int)
@click.option("--use-agent/--no-agent", default=True)
@click.option("--agent-review-interval", default=1, type=int)
@click.option("--subsample-n", default=10000, type=int)
@click.option("--output-dir", default="./results")
@click.option("--seed", default=42, type=int)
@click.option("--config", default=None, help="YAML config file")
@click.option("--embed-epochs", default=None, type=int, help="Override embedding training epochs")
def run(input_path, embedding, batch_key, label_key, method, bo_mode,
        n_trials, n_initial, use_agent, agent_review_interval,
        subsample_n, output_dir, seed, config, embed_epochs):
    """Run PALO ALTO optimization."""
    # Load config file if provided
    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        # CLI args override config
        input_path = input_path or cfg.get("input", input_path)
        embedding = embedding if embedding != "X_pca" else cfg.get("embedding_key", embedding)
        batch_key = batch_key if batch_key != "batch" else cfg.get("batch_key", batch_key)
        label_key = label_key if label_key != "cell_type" else cfg.get("label_key", label_key)
        method = cfg.get("method", method)
        bo_mode = cfg.get("bo_mode", bo_mode)
        n_trials = cfg.get("n_trials", n_trials)

    embed_kwargs = {}
    if embed_epochs is not None:
        embed_kwargs["epochs"] = embed_epochs

    from paloalto import optimize
    result = optimize(
        adata=input_path,
        embedding_key=embedding,
        batch_key=batch_key,
        label_key=label_key,
        method=method,
        bo_mode=bo_mode,
        n_trials=n_trials,
        n_initial=n_initial,
        use_agent=use_agent,
        agent_review_interval=agent_review_interval,
        subsample_n=subsample_n,
        output_dir=output_dir,
        seed=seed,
        embed_kwargs=embed_kwargs,
    )

    click.echo(f"\nBest parameters: {json.dumps(result['best_params'], indent=2)}")
    click.echo(f"Best scores: {json.dumps(result['best_scores'], indent=2)}")
    click.echo(f"Report: {result['report_path']}")


@main.command()
@click.option("--input", "input_path", required=True, help="Path to .h5ad file")
@click.option("--embedding", default="X_umap", help="Key in adata.obsm")
@click.option("--batch-key", default="batch")
@click.option("--label-key", default="cell_type")
def metrics(input_path, embedding, batch_key, label_key):
    """Compute metrics on an existing embedding."""
    from paloalto.data import load_and_validate
    from paloalto.metrics import compute_all

    adata = load_and_validate(input_path, embedding, batch_key, label_key)
    scores = compute_all(adata, embedding, label_key, batch_key)

    click.echo("\nMetric Results:")
    click.echo("-" * 40)
    for k, v in scores.items():
        click.echo(f"  {k:30s} {v:.4f}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v -x --timeout=300`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paloalto/cli.py tests/test_cli.py
git commit -m "feat: CLI with run and metrics commands"
```

---

### Task 17: Full Integration Test & Polish

**Files:**
- Modify: various files for bug fixes discovered during integration

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v --timeout=600`
Expected: All tests PASS. Fix any failures.

- [ ] **Step 2: Run a real end-to-end smoke test**

```bash
cd /scratch/users/chensj16/projects/paloalto
python -c "
import paloalto
import anndata as ad
import numpy as np
import pandas as pd

# Create a realistic test dataset
np.random.seed(42)
n = 1000
X = np.random.randn(n, 200).astype(np.float32)
obs = pd.DataFrame({
    'batch': pd.Categorical(['b0']*500 + ['b1']*500),
    'cell_type': pd.Categorical(['t0']*250 + ['t1']*250 + ['t2']*250 + ['t3']*250),
}, index=[f'c{i}' for i in range(n)])
adata = ad.AnnData(X=X, obs=obs)
pca = np.random.randn(n, 30).astype(np.float32)
for i in range(4):
    mask = obs['cell_type'] == f't{i}'
    pca[mask] += np.random.randn(30) * 5
adata.obsm['X_pca'] = pca

result = paloalto.optimize(
    adata, embedding_key='X_pca', batch_key='batch', label_key='cell_type',
    method='numap', bo_mode='weighted', n_trials=3, n_initial=3,
    use_agent=False, subsample_n=500, output_dir='/tmp/paloalto_test',
    embed_kwargs={'epochs': 2},
)
print('Best scores:', result['best_scores'])
print('Report:', result['report_path'])
"
```

Expected: Runs to completion, prints scores and report path.

- [ ] **Step 3: Fix any issues discovered**

Address test failures, import errors, or runtime bugs.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "fix: integration test fixes and polish"
```
