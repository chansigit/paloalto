"""Standard (non-parametric) UMAP wrapper.

Unlike NUMAP/parametric t-SNE, standard UMAP does not learn an encoder — there
is no model to save. Instead we save the 2D coordinates directly.
"""

import numpy as np

from paloalto.utils import get_logger

logger = get_logger(__name__)


class UMAPEmbedder:
    """Standard UMAP with save/load for coordinates (not a model)."""

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
        **kwargs,
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.kwargs = kwargs
        self._coords = None

    def fit(self, adata, embedding_key: str = "X_pca", **_ignored) -> np.ndarray:
        """Run UMAP and return 2D coordinates."""
        import umap

        X = adata.obsm[embedding_key]
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        X = X.astype(np.float32)

        logger.info(
            "Fitting UMAP: n_neighbors=%d, min_dist=%.3f, metric=%s",
            self.n_neighbors, self.min_dist, self.metric,
        )

        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            n_components=2,
            random_state=self.random_state,
        )
        self._coords = reducer.fit_transform(X)
        return self._coords

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Not supported for non-parametric UMAP — raises error."""
        raise NotImplementedError(
            "Standard UMAP is non-parametric and cannot transform new data. "
            "Use 'numap' method for parametric embedding with .transform() support."
        )

    def save(self, path: str) -> None:
        """Save coordinates and hyperparams (no model to save)."""
        if self._coords is None:
            raise RuntimeError("Must call fit() before save()")
        np.savez(
            path,
            coords=self._coords,
            params=self.get_params(),
        )
        logger.info("Saved UMAP coords to %s", path)

    @classmethod
    def load(cls, path: str) -> "UMAPEmbedder":
        data = np.load(path, allow_pickle=True)
        emb = cls(**data["params"].item())
        emb._coords = data["coords"]
        return emb

    def get_params(self) -> dict:
        return {
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "random_state": self.random_state,
        }
