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
    nn_idx: np.ndarray = None,
    nn_dist: np.ndarray = None,
) -> float:
    """Integration LISI: normalized so that 1 = perfect mixing, 0 = no mixing.

    Raw iLISI is in [1, n_batches]. We normalize: score = (median_lisi - 1) / (n_batches - 1).
    """
    lisi_scores = compute_lisi(coords, batch_labels, perplexity=perplexity,
                               nn_idx=nn_idx, nn_dist=nn_dist)
    n_batches = len(np.unique(batch_labels))
    median_lisi = np.median(lisi_scores)
    return float(np.clip((median_lisi - 1) / (n_batches - 1), 0, 1))


def graph_connectivity(
    coords: np.ndarray,
    type_labels: np.ndarray,
    n_neighbors: int = 15,
    precomputed_nn_idx: np.ndarray = None,
) -> float:
    """Fraction of cell types whose kNN subgraph is fully connected."""
    from scipy.sparse import csr_matrix

    n = coords.shape[0]

    if precomputed_nn_idx is not None:
        nn_idx = precomputed_nn_idx[:, :n_neighbors]
    else:
        from pynndescent import NNDescent
        index = NNDescent(coords, n_neighbors=n_neighbors + 1, random_state=42)
        nn_idx, _ = index.neighbor_graph
        nn_idx = nn_idx[:, 1:]  # drop self

    # Build adjacency matrix
    k = nn_idx.shape[1]
    rows = np.repeat(np.arange(n), k)
    cols = nn_idx.ravel()
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
        scores.append(1.0 / n_components)

    return float(np.mean(scores))
