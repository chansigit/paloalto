"""Metrics aggregation: scIB overall + scGraph.

Optimized to share expensive computation across metrics:
- One Leiden clustering shared by NMI and ARI (saves ~24s)
- One kNN graph shared by cLISI, iLISI, and graph_connectivity (saves ~4s)
"""

from typing import Dict
import numpy as np
import torch

from paloalto.metrics.bio import cell_type_asw, nmi, ari, clisi, _leiden_on_coords
from paloalto.metrics.batch import batch_asw, ilisi, graph_connectivity
from paloalto.metrics.scgraph import corr_weighted


def scib_overall(bio: Dict[str, float], batch: Dict[str, float]) -> float:
    """Compute scIB overall score = 0.6 * Bio + 0.4 * Batch."""
    bio_score = float(np.mean(list(bio.values())))
    batch_score = float(np.mean(list(batch.values())))
    return 0.6 * bio_score + 0.4 * batch_score


def _build_shared_knn(coords: np.ndarray, perplexity: float = 30.0):
    """Build one kNN graph reused by LISI and graph_connectivity.

    Uses GPU brute-force for small/low-dim data (faster than pynndescent),
    falls back to pynndescent for large/high-dim data.
    """
    n = coords.shape[0]
    k = min(int(perplexity * 3), n - 1)
    k = max(k, 15)

    # GPU brute-force is faster for small n and low dims (typical: 10k × 2D)
    if n <= 50000 and coords.shape[1] <= 10 and torch.cuda.is_available():
        device = torch.device("cuda")
        X = torch.from_numpy(coords.astype(np.float32)).to(device)
        dists = torch.cdist(X, X)
        dists.fill_diagonal_(float("inf"))
        nn_dist, nn_idx = dists.topk(k, largest=False)
        return nn_idx.cpu().numpy(), nn_dist.cpu().numpy()

    # Fallback: pynndescent
    from pynndescent import NNDescent
    index = NNDescent(coords, n_neighbors=k + 1, random_state=42)
    nn_idx, nn_dist = index.neighbor_graph
    return nn_idx[:, 1:], nn_dist[:, 1:]


def compute_all(
    adata,
    embed_key: str,
    label_key: str,
    batch_key: str,
) -> Dict[str, float]:
    """Compute all metrics and return a flat dict.

    Optimizations vs. calling each metric independently:
    - Leiden clustering computed once, shared by NMI and ARI
    - kNN graph computed once, shared by cLISI, iLISI, and graph_connectivity
    """
    coords = adata.obsm[embed_key]
    labels = adata.obs[label_key].values
    batches = adata.obs[batch_key].values

    # Shared: one Leiden clustering for NMI + ARI
    clusters = _leiden_on_coords(coords)

    # Shared: one kNN graph for cLISI + iLISI + graph_connectivity
    nn_idx, nn_dist = _build_shared_knn(coords)

    bio = {
        "cell_type_asw": cell_type_asw(coords, labels),
        "nmi": nmi(coords, labels, clusters=clusters),
        "ari": ari(coords, labels, clusters=clusters),
        "clisi": clisi(coords, labels, nn_idx=nn_idx, nn_dist=nn_dist),
    }
    batch_scores = {
        "batch_asw": batch_asw(coords, batches),
        "ilisi": ilisi(coords, batches, nn_idx=nn_idx, nn_dist=nn_dist),
        "graph_connectivity": graph_connectivity(coords, labels, precomputed_nn_idx=nn_idx),
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
