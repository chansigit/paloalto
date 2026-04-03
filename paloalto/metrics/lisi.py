"""LISI (Local Inverse Simpson's Index) computation.

Internalized from scIB. Computes per-cell diversity of a categorical label
in local neighborhoods defined by a Gaussian kernel with specified perplexity.

Attribution: Luecken et al. (2022) "Benchmarking atlas-level data integration
in single-cell genomics", Nature Methods. Original scIB code at
https://github.com/theislab/scib under BSD-3-Clause.

Behavior changes vs. scIB reference:
- Uses pynndescent for kNN (instead of sklearn or FAISS) for speed and
  reduced dependency footprint.
- Binary search uses sigma (std dev) rather than beta (precision) to keep
  the formula closer to the UMAP paper convention.
- Self-neighbor removal is done via column-indexing rather than in-graph
  masking.
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

    # Build kNN — request k+1 neighbors so we can drop the self-neighbor
    index = NNDescent(coords, n_neighbors=k + 1, random_state=42)
    nn_idx, nn_dist = index.neighbor_graph
    # Remove self (first column is always the query point itself)
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

        # Inverse Simpson's index: 1 / sum(p_c^2) where p_c is weighted
        # proportion of cells from category c among the k neighbors
        neighbor_labels = label_codes[nn_idx[i]]
        proportions = np.zeros(n_labels, dtype=np.float64)
        for j in range(len(neighbor_labels)):
            proportions[neighbor_labels[j]] += weights[j]
        simpson = np.sum(proportions**2)
        lisi[i] = 1.0 / simpson

    # Clip to theoretical bounds [1, n_labels] to guard against floating-point
    # rounding errors (e.g. simpson slightly > 1 when all neighbors share a
    # label, producing lisi slightly < 1).
    lisi = np.clip(lisi, 1.0, float(n_labels))
    return lisi


def _find_sigma(
    distances: np.ndarray,
    target_perplexity: float,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    """Binary search for sigma that yields target perplexity.

    Args:
        distances: Sorted non-negative distances to k neighbors.
        target_perplexity: Desired effective number of neighbors.
        tol: Convergence tolerance on perplexity.
        max_iter: Maximum number of bisection iterations.

    Returns:
        Sigma value (Gaussian std dev) achieving target perplexity.
    """
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
