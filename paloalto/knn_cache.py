"""Pre-computed kNN cache to avoid redundant graph builds across BO trials.

In a typical 20-trial optimization, the input embedding (X_pca) is fixed and
the kNN metric has only 3 possible values. Pre-building kNN with max k for
each metric eliminates ~10s per trial (×40 evaluations = ~7 minutes saved).
"""

from typing import Dict, Optional, Tuple

import numpy as np
from pynndescent import NNDescent
from umap.umap_ import fuzzy_simplicial_set
from sklearn.utils import check_random_state

from paloalto.utils import get_logger

logger = get_logger(__name__)


class KNNCache:
    """Caches kNN graphs for different metrics on a fixed dataset."""

    def __init__(self, X: np.ndarray, max_k: int = 200, metrics: Tuple[str, ...] = ("euclidean", "cosine", "correlation"), seed: int = 42):
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.max_k = max_k

        for metric in metrics:
            logger.info(f"Pre-building kNN: metric={metric}, k={max_k}, n={X.shape[0]}")
            n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
            n_iters = max(5, int(round(np.log2(X.shape[0]))))
            nnd = NNDescent(
                X.reshape((len(X), np.prod(np.shape(X)[1:]))),
                n_neighbors=max_k,
                metric=metric,
                n_trees=n_trees,
                n_iters=n_iters,
                max_candidates=60,
                random_state=seed,
                verbose=False,
            )
            knn_idx, knn_dist = nnd.neighbor_graph
            self._cache[metric] = (knn_idx, knn_dist)
            logger.info(f"  Done: {metric}")

    def get(self, metric: str, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get kNN indices and distances, truncated to n_neighbors."""
        if metric not in self._cache:
            raise KeyError(f"Metric '{metric}' not in cache. Available: {list(self._cache.keys())}")
        knn_idx, knn_dist = self._cache[metric]
        return knn_idx[:, :n_neighbors], knn_dist[:, :n_neighbors]

    def get_umap_graph(self, X: np.ndarray, n_neighbors: int, metric: str, random_state=None):
        """Drop-in replacement for numap.umap_pytorch.modules.get_umap_graph."""
        knn_idx, knn_dist = self.get(metric, n_neighbors)
        rs = check_random_state(random_state)
        umap_graph, sigmas, rhos = fuzzy_simplicial_set(
            X=X,
            n_neighbors=n_neighbors,
            metric=metric,
            random_state=rs,
            knn_indices=knn_idx,
            knn_dists=knn_dist,
        )
        return umap_graph
