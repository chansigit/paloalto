"""scGraph Corr-Weighted metric (internalized).

Measures how well a 2D embedding preserves inter-cell-type distance
relationships compared to a PCA reference consensus.

Attribution: scGraph (https://github.com/scgraph/scgraph). Internalized to
avoid requiring end users to install scgraph. Core algorithm retained; changes:
- HVG selection falls back to using existing X_pca when count data is absent
  or when seurat_v3 HVG selection fails (e.g., on log-normalized input).
- n_top_genes is clamped to n_vars to handle small gene panels.
- Per-batch distance matrices are averaged with np.nanmean so that batches
  lacking a cell type do not invalidate rows/columns for types they do contain.
- When per-batch coverage leaves fewer valid type pairs than n_types*(n_types-1)/2,
  the implementation falls back to a global PCA centroid reference (using the
  precomputed X_pca in adata.obsm if available, otherwise recomputing PCA on
  the full dataset).  This handles confounded batch-by-celltype designs.
- _col_normalize and the correlation loop skip fully-NaN columns.
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
        n_hvgs: Number of highly variable genes for per-batch PCA reference.
        n_pcs: Number of PCs to use when building the consensus.
        trim_proportion: Trimming fraction for robust centroid estimation.
        min_cells_per_batch: Batches with fewer cells are skipped.

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

    # Compute embedding distance matrix over all cells
    embed_coords = adata.obsm[embed_key]
    embed_dist = _centroid_distance_matrix(embed_coords, labels, types, trim_proportion)

    # Column-normalize both (NaN-aware)
    consensus_norm = _col_normalize(consensus)
    embed_norm = _col_normalize(embed_dist)

    # Corr-Weighted: mean column-wise weighted Pearson correlation
    scores = []
    for col in range(n_types):
        ref_col = consensus_norm[:, col]
        emb_col = embed_norm[:, col]

        # Skip columns where either reference or embedding is all NaN
        valid = ~(np.isnan(ref_col) | np.isnan(emb_col))
        if valid.sum() < 3:
            continue

        ref_col_v = ref_col[valid]
        emb_col_v = emb_col[valid]

        # Weights = 1 / reference distance; zero weight where distance is 0
        weights = np.zeros_like(ref_col_v)
        nonzero = ref_col_v > 0
        weights[nonzero] = 1.0 / ref_col_v[nonzero]
        if weights.sum() == 0:
            continue

        r = _weighted_pearson(emb_col_v, ref_col_v, weights)
        if not np.isnan(r):
            scores.append(r)

    return float(np.mean(scores)) if scores else 0.0


def _build_consensus(
    adata, label_key, batch_key, types,
    n_hvgs, n_pcs, trim_proportion, min_cells_per_batch,
):
    """Build consensus distance matrix from per-batch PCA centroids.

    Uses np.nanmean so batches that lack a cell type contribute NaN for that
    type's row/column without invalidating entries for types they do contain.

    Falls back to a global PCA reference when the per-batch consensus covers
    fewer than half the possible type-pair entries (i.e., the batch design is
    too confounded with cell type to yield a useful per-batch signal).
    """
    batches = adata.obs[batch_key].unique()
    n_types = len(types)
    dist_stack = []

    for batch in batches:
        mask = adata.obs[batch_key] == batch
        if mask.sum() < min_cells_per_batch:
            continue

        batch_adata = adata[mask].copy()
        # Compute PCA on this batch; fall back to precomputed X_pca on failure
        try:
            sc.pp.highly_variable_genes(
                batch_adata,
                n_top_genes=min(n_hvgs, batch_adata.n_vars),
                flavor="seurat_v3",
            )
            sc.tl.pca(batch_adata, n_comps=min(n_pcs, batch_adata.n_vars - 1))
            pca_coords = batch_adata.obsm["X_pca"]
            # Abort if PCA produced NaN (e.g., from degenerate HVG selection)
            if np.isnan(pca_coords).any():
                raise ValueError("PCA produced NaN values")
        except Exception:
            # Fallback: use existing X_pca embedding from the subset
            if "X_pca" in batch_adata.obsm:
                pca_coords = batch_adata.obsm["X_pca"][:, :n_pcs]
            else:
                continue

        batch_labels = batch_adata.obs[label_key].values
        dist = _centroid_distance_matrix(pca_coords, batch_labels, types, trim_proportion)
        dist_norm = _col_normalize(dist)
        dist_stack.append(dist_norm)

    if not dist_stack:
        return _global_pca_reference(adata, label_key, types, n_pcs, n_hvgs, trim_proportion)

    # nanmean: entries NaN in all batches stay NaN; otherwise average over
    # the batches that have data for those type pairs
    with np.errstate(all="ignore"):
        consensus = np.nanmean(dist_stack, axis=0)

    # Check coverage: count non-NaN upper-triangle pairs
    upper = np.triu(~np.isnan(consensus), k=1)
    n_possible = n_types * (n_types - 1) // 2
    coverage = upper.sum() / max(n_possible, 1)

    if coverage < 0.5:
        # Too many missing pairs — fall back to global PCA reference
        return _global_pca_reference(adata, label_key, types, n_pcs, n_hvgs, trim_proportion)

    return _col_normalize(consensus)


def _global_pca_reference(adata, label_key, types, n_pcs, n_hvgs, trim_proportion):
    """Build a distance reference from global PCA (full dataset).

    Uses precomputed X_pca if present; otherwise recomputes PCA.
    """
    labels = adata.obs[label_key].values

    if "X_pca" in adata.obsm:
        pca_coords = adata.obsm["X_pca"][:, :n_pcs]
    else:
        try:
            tmp = adata.copy()
            sc.pp.highly_variable_genes(
                tmp,
                n_top_genes=min(n_hvgs, tmp.n_vars),
                flavor="seurat_v3",
            )
            sc.tl.pca(tmp, n_comps=min(n_pcs, tmp.n_vars - 1))
            pca_coords = tmp.obsm["X_pca"]
        except Exception:
            return None

    dist = _centroid_distance_matrix(pca_coords, labels, types, trim_proportion)
    return _col_normalize(dist)


def _centroid_distance_matrix(
    coords: np.ndarray,
    labels: np.ndarray,
    types: np.ndarray,
    trim_proportion: float,
) -> np.ndarray:
    """Compute pairwise Euclidean distance between trimmed-mean centroids.

    Cell types absent from ``labels`` get a NaN centroid; their rows and
    columns in the resulting distance matrix will also be NaN.
    """
    n_types = len(types)
    centroids = np.full((n_types, coords.shape[1]), np.nan)
    for i, t in enumerate(types):
        mask = labels == t
        if mask.sum() == 0:
            # Type not present — leave as NaN
            continue
        elif mask.sum() < 3:
            centroids[i] = np.mean(coords[mask], axis=0)
        else:
            centroids[i] = trim_mean(coords[mask], proportiontocut=trim_proportion, axis=0)

    # pdist on rows with NaN propagates NaN for those pairs, which is correct
    dist = squareform(pdist(centroids, metric="euclidean"))
    np.fill_diagonal(dist, 0.0)
    return dist


def _col_normalize(mat: np.ndarray) -> np.ndarray:
    """Divide each column by its nanmax value; all-NaN columns stay NaN."""
    with np.errstate(all="ignore"):
        col_max = np.nanmax(mat, axis=0)
    # Where col_max is 0 or NaN, avoid division errors
    safe_max = np.where((col_max == 0) | np.isnan(col_max), 1.0, col_max)
    result = mat / safe_max
    # Restore NaN for columns that were fully NaN (nanmax returns nan for them)
    fully_nan = np.isnan(col_max)
    if fully_nan.any():
        result[:, fully_nan] = np.nan
    return result


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
