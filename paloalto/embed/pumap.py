"""NUMAP wrapper with BO-tunable interface and model persistence."""

from __future__ import annotations

import numpy as np
import torch

from paloalto.utils import get_logger

logger = get_logger(__name__)


class NUMAPEmbedder:
    """Parametric UMAP via NUMAP with save/load and BO-friendly interface.

    Wraps the ``numap.NUMAP`` class, converting between numpy arrays
    (used by AnnData) and tensors (required by NUMAP internals), and
    providing ``save``/``load`` for model persistence.
    """

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
        self._model: "numap.NUMAP | None" = None  # noqa: F821
        self._input_dim: int | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, adata, embedding_key: str = "X_pca", knn_cache=None) -> np.ndarray:
        """Fit NUMAP on the embedding stored in *adata* and return 2-D coords.

        Args:
            adata: AnnData object.
            embedding_key: Key in adata.obsm for the input embedding.
            knn_cache: Optional KNNCache. If provided, monkey-patches
                get_umap_graph to skip redundant kNN construction.
        """
        from numap import NUMAP

        X = adata.obsm[embedding_key]
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        X = X.astype(np.float32)
        self._input_dim = X.shape[1]

        logger.info(
            "Fitting NUMAP: n_neighbors=%d, min_dist=%.3f, metric=%s, se_dim=%d",
            self.n_neighbors,
            self.min_dist,
            self.metric,
            self.se_dim,
        )

        # Monkey-patch get_umap_graph if cache is available.
        # Must patch main.py's namespace (not modules.py) because main.py
        # binds the name at import time via `from .modules import get_umap_graph`.
        import numap.umap_pytorch.main as _main
        _original_get_umap_graph = _main.get_umap_graph
        if knn_cache is not None:
            _main.get_umap_graph = knn_cache.get_umap_graph

        try:
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
                use_residual_connections=True,
                **self.kwargs,
            )

            X_tensor = torch.tensor(X, dtype=torch.float32)
            model.fit(X_tensor)

            coords = model.transform(X_tensor, is_train=True)
            self._model = model
            return np.asarray(coords)
        finally:
            # Always restore original function
            _main.get_umap_graph = _original_get_umap_graph

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data through the fitted encoder."""
        if self._model is None:
            raise RuntimeError("Must call fit() before transform()")
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        X = X.astype(np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return np.asarray(self._model.transform(X_tensor))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model state and hyperparams to *path*."""
        if self._model is None:
            raise RuntimeError("Must call fit() before save()")

        # The NUMAP object wraps:
        #   .pumap        -- PUMAP instance
        #   .pumap.model  -- pytorch-lightning Model with .encoder
        #   .knn          -- KNeighborsRegressor for OOS spectral embedding
        #   .se           -- training spectral embedding (tensor)
        import pickle

        state = {
            "params": self.get_params(),
            "input_dim": self._input_dim,
            "encoder_state": self._model.pumap.model.encoder.state_dict(),
            "knn": pickle.dumps(self._model.knn),
            "se": self._model.se,
            # Encoder config needed for reconstruction.
            "encoder_config": {
                "input_dim": self._model.pumap.model.encoder.input_dim,
                "n_components": self._model.pumap.model.encoder.n_components,
                "use_residual_connections": self._model.pumap.model.encoder.use_residual_connections,
                "learn_from_se": self._model.pumap.model.encoder.learn_from_se,
                "use_concat": self._model.pumap.model.encoder.use_concat,
                "use_alpha": self._model.pumap.model.encoder.use_alpha,
                "alpha": self._model.pumap.model.encoder.alpha,
            },
        }
        torch.save(state, path)
        logger.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: str) -> "NUMAPEmbedder":
        """Load a previously saved model from *path*."""
        state = torch.load(path, weights_only=False, map_location="cpu")
        params = state["params"]
        emb = cls(**params)
        emb._input_dim = state["input_dim"]

        # Reconstruct the NUMAP object with enough internals for transform.
        import pickle
        from numap import NUMAP
        from numap.umap_pytorch import PUMAP
        from numap.umap_pytorch.main import Model
        from numap.umap_pytorch.model import default_encoder

        enc_cfg = state["encoder_config"]

        # Rebuild encoder on CPU first, then load saved weights.  We keep
        # the encoder on CPU so that NUMAP.transform (which feeds CPU
        # tensors) works without device-mismatch errors.
        encoder = default_encoder(
            dims=enc_cfg["input_dim"],
            n_components=enc_cfg["n_components"],
            use_residual_connections=enc_cfg["use_residual_connections"],
            learn_from_se=enc_cfg["learn_from_se"],
            use_concat=enc_cfg["use_concat"],
            use_alpha=enc_cfg["use_alpha"],
            alpha=enc_cfg["alpha"],
            se=None,
            init_method="identity",
            device="cpu",
        )
        encoder.load_state_dict(state["encoder_state"])
        encoder.eval()

        # Wrap in the Lightning Model so PUMAP.transform works.
        lightning_model = Model(
            lr=params["lr"],
            encoder=encoder,
            decoder=None,
            min_dist=params["min_dist"],
            negative_sample_rate=params["negative_sample_rate"],
        )
        lightning_model.eval()

        # Attach to a minimal PUMAP shell.
        pumap = PUMAP(
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            metric=params["metric"],
            n_components=2,
            lr=params["lr"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            negative_sample_rate=params["negative_sample_rate"],
        )
        pumap.model = lightning_model

        # Reconstruct the NUMAP wrapper.
        numap_obj = NUMAP(
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            metric=params["metric"],
            se_dim=params["se_dim"],
            se_neighbors=params["se_neighbors"],
            epochs=params["epochs"],
            lr=params["lr"],
            batch_size=params["batch_size"],
            random_state=params["random_state"],
            negative_sample_rate=params["negative_sample_rate"],
        )
        numap_obj.pumap = pumap
        numap_obj.knn = pickle.loads(state["knn"])
        numap_obj.se = state["se"]

        emb._model = numap_obj
        return emb

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return the hyper-parameters as a plain dict (BO-friendly)."""
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
