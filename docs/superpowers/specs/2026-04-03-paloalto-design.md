# PALO ALTO Design Spec

**Pareto-guided Automatic Layout Optimization for Aligning Latent Topology in Omics embeddings**

Date: 2026-04-03

---

## 1. Overview

PALO ALTO is an agent-guided Bayesian optimization tool for tuning single-cell embedding visualization parameters. It optimizes parametric UMAP (via NUMAP) and parametric t-SNE hyperparameters against internalized scIB and scGraph metrics, without requiring users to install those packages.

The key contribution is an **Agent-in-the-Loop BO** architecture: a BoTorch-driven optimization loop where an LLM agent acts as a judge — reviewing suggestions, injecting candidates, pruning the search space, and diagnosing anomalies. A built-in A/B comparison against pure BoTorch baseline demonstrates the agent's value.

### Key Decisions

- **Parametric embeddings** (NUMAP, PyTorch parametric t-SNE) so models can be saved and reused for reproducibility.
- **Two aggregate metrics**: scIB overall (0.4 × Batch + 0.6 × Bio) and scGraph Corr-Weighted score.
- **Two BO modes**: weighted scalar (single objective) and multi-objective Pareto (NEHVI).
- **Agent review every trial** by default (`agent_review_interval=1`), user-configurable.
- **Python API + CLI** with optional YAML config.
- **HTML report** with embedded plots, agent logs, and baseline comparison.

---

## 2. Package Structure

```
paloalto/
├── __init__.py          # public API: optimize(), compute_metrics()
├── data.py              # AnnData loading, validation, subsampling
├── metrics/
│   ├── __init__.py      # compute_all(), metric registry
│   ├── bio.py           # Cell type ASW, NMI, ARI, cLISI
│   ├── batch.py         # Batch ASW, iLISI, Graph connectivity
│   └── scgraph.py       # Corr-Weighted (internalized from scGraph)
├── embed/
│   ├── __init__.py      # get_embedder(method)
│   ├── pumap.py         # NUMAP wrapper, BO-tunable params, model save/load
│   └── ptsne.py         # PyTorch parametric t-SNE (ported from Multiscale-Parametric-t-SNE)
├── optim/
│   ├── __init__.py
│   ├── space.py         # search space definitions, parameter constraints
│   ├── bo.py            # weighted scalar BO (GP + EI)
│   └── mobo.py          # multi-objective BO (NEHVI)
├── agent/
│   ├── __init__.py
│   └── judge.py         # Agent judge: review, modify, inject, prune, stop
├── report/
│   ├── __init__.py
│   ├── html.py          # HTML report generation
│   └── templates/       # Jinja2 template(s)
├── cli.py               # click-based CLI
└── utils.py             # kNN helpers, subsampling, logging
```

---

## 3. Data Module (`data.py`)

### Input

- `.h5ad` file or `AnnData` object
- Required fields:
  - `adata.obsm[embedding_key]` — pre-computed embedding (e.g., `X_pca`, `X_scVI`)
  - `adata.obs[batch_key]` — batch labels
  - `adata.obs[label_key]` — cell type labels

### Validation

- Check that `embedding_key` exists in `obsm`
- Check that `batch_key` and `label_key` exist in `obs` and are categorical
- Check that there are at least 2 batches and 2 cell types
- Warn if n_cells < 500 (too few for meaningful metrics)

### Subsampling

- For metric computation: stratified subsample (by cell_type × batch) to `subsample_n` cells (default 10,000)
- Fixed across all trials for fair comparison
- Embedding is trained on the full dataset (parametric models handle scale)
- 2D coordinates are extracted for the subsample indices after training

---

## 4. Metrics

### scIB Overall Score

**Batch correction** (mean of 3 metrics):

| Metric | Description | Implementation |
|--------|-------------|----------------|
| Batch ASW | Average silhouette width on batch labels (1 - \|ASW\|, so higher = better mixing) | `sklearn.metrics.silhouette_samples` on 2D coords, grouped by batch |
| iLISI | Integration LISI — batch diversity in k-neighborhoods | Internalized from scIB's LISI computation (kNN + Simpson's index) |
| Graph connectivity | Fraction of cells whose same-type subgraph is connected across batches | kNN graph on 2D → per-type connected components |

**Bio conservation** (mean of 4 metrics):

| Metric | Description | Implementation |
|--------|-------------|----------------|
| Cell type ASW | Average silhouette width on cell type labels | `sklearn.metrics.silhouette_score` on 2D coords |
| NMI | Normalized mutual info: Leiden clusters vs ground truth | Leiden clustering on 2D kNN graph → `sklearn.metrics.normalized_mutual_info_score` |
| ARI | Adjusted Rand index: Leiden clusters vs ground truth | Same clustering → `sklearn.metrics.adjusted_rand_score` |
| cLISI | Cell-type LISI — cell type purity in k-neighborhoods | Same LISI code, on cell type labels |

**Aggregation**: `scIB_overall = 0.4 × Batch + 0.6 × Bio`

All individual metrics are normalized to [0, 1], higher is better.

### scGraph Score (Corr-Weighted)

Internalized from [scGraph](https://github.com/chansigit/scGraph):

1. For each batch (≥100 cells): select top HVGs, compute PCA, compute trimmed-mean centroids per cell type, build column-normalized pairwise distance matrix.
2. Average across batches → consensus reference matrix. Column-normalize.
3. On the 2D embedding: compute same centroids and distance matrix.
4. **Corr-Weighted**: mean column-wise distance-weighted Pearson correlation between embedding distances and consensus distances.

Output: single float in [-1, 1], where higher = better topology preservation.

### For BO

- **Weighted scalar mode**: `objective = w1 × scIB_overall + w2 × scGraph_score` (default [0.5, 0.5])
- **Multi-objective mode**: `[scIB_overall, scGraph_score]` → 2D Pareto front

---

## 5. Embedding Methods

### 5.1 Parametric UMAP (NUMAP wrapper — `embed/pumap.py`)

Wraps the installed `numap` package (v0.2.3, PyTorch Lightning).

**BO search space:**

| Param | Type | Range | Tier | Notes |
|-------|------|-------|------|-------|
| `n_neighbors` | int | [5, 200] log | 1 | Local vs global structure |
| `min_dist` | float | [0.001, 0.99] | 1 | Cluster tightness |
| `negative_sample_rate` | int | [1, 20] | 1 | Repulsion strength |
| `metric` | categorical | {euclidean, cosine, correlation} | 2 | High-D distance |
| `se_dim` | int | [2, 20] | 2 | Spectral embedding dims |
| `se_neighbors` | int | [5, 100] | 2 | Spectral graph neighborhood |

NUMAP-specific booleans (`use_residual_connections`, `learn_from_se`, `use_concat`, `init_method`) frozen to defaults initially.

**Model persistence**: Save `encoder.state_dict()` + hyperparams dict per trial. Best model(s) returned with `.transform(X_new)` method.

### 5.2 Parametric t-SNE (PyTorch port — `embed/ptsne.py`)

Ported from [Multiscale-Parametric-t-SNE](https://github.com/FrancescoCrecchi/Multiscale-Parametric-t-SNE), rewritten in PyTorch.

**Key changes from original:**
- TF1/Keras → PyTorch
- Fix batch-only P computation: use pynndescent for global kNN-based affinities
- Expose `dof` (alpha) as tunable
- Support cosine/correlation metrics (not just Euclidean)
- Fix data truncation bug
- Two sub-modes: single-scale (explicit perplexity) and multiscale (perplexity-free)

**BO search space:**

| Param | Type | Range | Tier | Notes |
|-------|------|-------|------|-------|
| `perplexity` | float | [5, 100] log | 1 | Single-scale only |
| `dof` | float | [0.2, 2.0] | 1 | Student-t tail weight |
| `early_exaggeration` | float | [4, 32] log | 1 | Initial cluster separation |
| `metric` | categorical | {euclidean, cosine, correlation} | 2 | High-D distance |
| `learning_rate` | float | [1e-4, 1e-1] log | 2 | Encoder LR |
| `exaggeration` | float | [1.0, 4.0] | 2 | Normal-phase exaggeration |

**Shared encoder architecture**: `input → 256 → 256 → 128 → 2` (MLP with ReLU). Fixed, not searched. Users can override.

---

## 6. Bayesian Optimization

### 6.1 Weighted Scalar Mode (default)

- Single GP surrogate (Matérn 5/2 kernel) over the combined objective
- Acquisition: `LogExpectedImprovement` (BoTorch)
- Continuous relaxation for integer params (rounded at evaluation)
- One-hot encoding for categorical params

### 6.2 Multi-Objective Mode

- Two independent GPs, one per objective (scIB_overall, scGraph_score)
- Acquisition: `qNoisyExpectedHypervolumeImprovement` (BoTorch)
- Reference point: [0, 0]
- Returns Pareto front of non-dominated solutions

### 6.3 Trial Flow

```
Trials 1..n_initial (default 5):
    Sobol quasi-random initialization — shared by both arms

Trials n_initial+1..n_trials (default 30):
  [Agent arm]
    1. GP_agent fit on agent history → acquisition → candidate x_bo
    2. Agent Judge reviews x_bo → returns x_agent (may differ from x_bo)
    3. Evaluate x_agent: train embedding → 2D coords → metrics
    4. Agent reviews result → logs reasoning
    5. Update GP_agent with (x_agent, scores)

  [Baseline arm]
    1. GP_baseline fit on baseline history → acquisition → candidate x_base
    2. Evaluate x_base directly (no agent): train embedding → 2D coords → metrics
    3. Update GP_baseline with (x_base, scores)

  The two arms maintain independent GP models and trial histories.
  They diverge after the shared Sobol phase. This doubles compute
  but ensures a fair comparison (each GP conditions on its own data).
```

### 6.4 Parameter Handling

- Continuous: native BoTorch bounds
- Integer (`n_neighbors`, `negative_sample_rate`, `se_dim`, `se_neighbors`): continuous relaxation → round
- Categorical (`metric`): one-hot in GP feature space
- Search space bounds can be dynamically modified by the Agent (PRUNE action)

### 6.5 Defaults

| Setting | Default |
|---------|---------|
| `n_trials` | 30 |
| `n_initial` | 5 (Sobol) |
| `method` | `"numap"` |
| `bo_mode` | `"weighted"` |
| `weights` | `[0.5, 0.5]` |
| `subsample_n` | 10000 |
| `seed` | 42 |

---

## 7. Agent Judge (`agent/judge.py`)

### Role

An LLM agent that reviews each BO trial (configurable via `agent_review_interval`, default 1). It acts as a domain-aware judge that can intervene in specific, constrained ways.

### Actions

| Action | Description | Constraints |
|--------|-------------|-------------|
| **ACCEPT** | Use BoTorch's suggestion as-is | — |
| **MODIFY** | Adjust specific params in the suggestion | Must stay within search space bounds |
| **INJECT** | Replace suggestion with agent's own candidate | Must be within search space bounds |
| **PRUNE** | Tighten search space bounds | Can only shrink, not expand. Minimum range enforced. |
| **STOP** | Recommend early stopping | Requires reasoning. User can override. |

### Input Context (per trial)

```python
{
    "trial_number": int,
    "trial_history": [{"params": {...}, "scores": {...}}, ...],
    "current_best": {"params": {...}, "scores": {...}},
    "bo_suggestion": {"params": {...}},
    "search_space": {"param_name": {"type": ..., "bounds": ...}, ...},
    "dataset_summary": {"n_cells": int, "n_batches": int, "n_types": int},
    "recent_trend": "improving" | "flat" | "degrading",
}
```

### Output (structured JSON)

```python
{
    "action": "accept" | "modify" | "inject" | "prune" | "stop",
    "params": {...},           # for modify/inject
    "new_bounds": {...},       # for prune
    "reasoning": "...",        # always required, logged to report
}
```

### Experimental Comparison

Both arms (Agent-guided and Baseline) share the same Sobol initialization. After that:

- **Baseline arm**: pure BoTorch, no agent intervention
- **Agent arm**: BoTorch + agent judge

Both arms are evaluated and reported side-by-side. Comparison metrics:
- Convergence speed (trials to reach X% of final best)
- Final best score after N trials
- Pareto hypervolume (multi-objective mode)
- Agent intervention effectiveness (did modified/injected trials outperform the original suggestion?)

---

## 8. Report (`report/html.py`)

Self-contained HTML file with embedded Matplotlib plots (base64).

### Sections

1. **Overview** — dataset summary, method, BO mode, total trials, wall time
2. **Convergence curves** — Baseline vs Agent-guided best-so-far, agent intervention markers
3. **Pareto front** (multi-objective) — scIB vs scGraph scatter, highlighted Pareto optimal, both arms overlaid
4. **Best parameters** — top-K trials table, Baseline best vs Agent best side-by-side
5. **Metric breakdown** — per-trial scIB sub-metrics and scGraph score (collapsible)
6. **Agent log** — each intervention: action, reasoning, score delta, effectiveness stats
7. **Best embedding visualization** — 2D scatter plots colored by cell type and batch (both arms)
8. **Caveats** — standard disclaimers about 2D optimization ≠ representation quality, metric limitations, non-transferability of parameters

### Implementation

- Jinja2 template + Matplotlib figures → base64 PNG
- Single `.html` file, no external dependencies
- Default path: `{output_dir}/paloalto_report.html`

---

## 9. CLI (`cli.py`)

Built with `click`.

```bash
# Full optimization
paloalto run \
    --input data.h5ad \
    --embedding X_pca \
    --batch-key batch \
    --label-key cell_type \
    --method numap \
    --bo-mode weighted \
    --n-trials 30 \
    --use-agent \
    --agent-review-interval 1 \
    --output-dir ./results

# Metrics only
paloalto metrics \
    --input data.h5ad \
    --embedding X_umap \
    --batch-key batch \
    --label-key cell_type

# From config file
paloalto run --config paloalto.yaml
```

### YAML Config

```yaml
input: data.h5ad
embedding_key: X_pca
batch_key: batch
label_key: cell_type
method: numap
bo_mode: pareto
n_trials: 50
use_agent: true
agent_review_interval: 1
weights: [0.5, 0.5]
subsample_n: 10000
seed: 42
search_space:
  n_neighbors: [5, 200]
  min_dist: [0.001, 0.99]
  negative_sample_rate: [1, 20]
  metric: [euclidean, cosine, correlation]
```

---

## 10. Dependencies

| Package | Purpose | Already installed |
|---------|---------|-------------------|
| `anndata` | Data I/O | Yes (0.12.10) |
| `scanpy` | Leiden clustering, neighbors | Yes (1.11.5) |
| `torch` | Parametric t-SNE, NUMAP backend | Yes (2.6.0) |
| `botorch` | Bayesian optimization | Yes (0.17.2) |
| `numap` | Parametric UMAP | Yes (0.2.3) |
| `pynndescent` | Approximate kNN | Yes (via umap-learn) |
| `scikit-learn` | Silhouette, NMI, ARI | Yes |
| `matplotlib` | Report plots | Yes |
| `jinja2` | HTML templating | Yes |
| `click` | CLI | Yes |
| `openTSNE` | t-SNE affinity computation (optional, for P matrix) | **Needs install** |

---

## 11. Out of Scope (for now)

- Raw count processing (users provide pre-computed embeddings)
- Non-h5ad input formats
- Searching over encoder architectures
- GPU multi-node distributed training
- Interactive dashboard (report is static HTML)
- Extending to non-visualization embeddings (e.g., optimizing scVI latent space)
