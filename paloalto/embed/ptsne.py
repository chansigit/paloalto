"""Parametric t-SNE in PyTorch.

Ported from Multiscale-Parametric-t-SNE (Crecchi, ESANN 2020),
rewritten in PyTorch with bug fixes and extended features.

Attribution:
    Francesco Crecchi, "Multiscale Parametric t-SNE", ESANN 2020.
    https://github.com/FrancescoCrecchi/Multiscale-Parametric-t-SNE

Behavior changes vs. original:
    - TF1/Keras encoder replaced with a pure PyTorch MLP.
    - Uses pynndescent for global kNN-based affinities (not batch-only P).
    - Exposes ``dof`` (alpha) for Student-t tail weight tuning.
    - Multiscale mode averages P across exponentially-spaced perplexities.
    - Early exaggeration applied only during the first N epochs.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from pynndescent import NNDescent

from paloalto.utils import get_logger

logger = get_logger(__name__)


class _Encoder(nn.Module):
    """Simple MLP encoder: input → 256 → 256 → 128 → n_components."""

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
    """Parametric t-SNE with configurable dof, multiscale mode, and save/load.

    Parameters
    ----------
    n_components:
        Output dimensionality (default 2).
    perplexity:
        t-SNE perplexity. Ignored when ``multiscale=True``.
    dof:
        Degrees of freedom for the Student-t kernel (alpha). ``dof=1`` gives
        the standard Cauchy distribution used in vanilla t-SNE. Smaller values
        produce heavier tails; larger values approach a Gaussian.
    early_exaggeration:
        Exaggeration factor applied to P during the first
        ``early_exaggeration_epochs`` epochs.
    early_exaggeration_epochs:
        Number of epochs for which early exaggeration is active.
    exaggeration:
        Exaggeration factor applied after the early phase (usually 1.0).
    metric:
        Distance metric passed to pynndescent.
    learning_rate:
        Adam learning rate.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size for the KL-divergence step.
    multiscale:
        When ``True``, averages P matrices computed at exponentially-spaced
        perplexities [2, 4, 8, ..., n/2].
    seed:
        Random seed for reproducibility.
    """

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
        self._encoder: _Encoder | None = None
        self._input_dim: int | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, adata, embedding_key: str = "X_pca") -> np.ndarray:
        """Fit parametric t-SNE on *adata* and return 2-D coordinates."""
        torch.manual_seed(self.seed)

        X = adata.obsm[embedding_key].astype(np.float32)
        self._input_dim = X.shape[1]
        n = X.shape[0]

        logger.info(
            "Fitting Parametric t-SNE: perplexity=%s, dof=%s, multiscale=%s",
            self.perplexity,
            self.dof,
            self.multiscale,
        )

        # Compute global P matrix (dense, n×n) once before training.
        P = self._compute_P(X)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder = _Encoder(self._input_dim, self.n_components).to(device)
        optimizer = torch.optim.Adam(self._encoder.parameters(), lr=self.learning_rate)
        X_tensor = torch.from_numpy(X).to(device)

        for epoch in range(self.epochs):
            exag = (
                self.early_exaggeration
                if epoch < self.early_exaggeration_epochs
                else self.exaggeration
            )
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                if len(idx) < 2:
                    continue

                idx_np = idx.cpu().numpy()
                x_batch = X_tensor[idx]
                P_batch = P[np.ix_(idx_np, idx_np)]

                p_sum = P_batch.sum()
                if p_sum == 0:
                    continue
                P_batch = P_batch / p_sum * exag
                P_batch = np.maximum(P_batch, 1e-12)
                P_torch = torch.from_numpy(P_batch.astype(np.float32)).to(device)

                Y = self._encoder(x_batch)
                loss = self._kl_divergence(P_torch, Y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 50 == 0:
                avg = epoch_loss / max(n_batches, 1)
                logger.info("  Epoch %d/%d, loss=%.4f", epoch + 1, self.epochs, avg)

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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and hyperparameters to *path*."""
        if self._encoder is None:
            raise RuntimeError("Must call fit() before save()")
        state = {
            "params": self.get_params(),
            "encoder_state": self._encoder.state_dict(),
            "input_dim": self._input_dim,
        }
        torch.save(state, path)
        logger.info("Saved ParametricTSNE model to %s", path)

    @classmethod
    def load(cls, path: str) -> "ParametricTSNE":
        """Load a previously saved model from *path*."""
        state = torch.load(path, weights_only=False)
        model = cls(**state["params"])
        model._input_dim = state["input_dim"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model._encoder = _Encoder(
            state["input_dim"], state["params"]["n_components"]
        ).to(device)
        model._encoder.load_state_dict(state["encoder_state"])
        model._encoder.eval()
        return model

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return hyperparameters as a plain dict (BO-friendly)."""
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

    # ------------------------------------------------------------------
    # P-matrix computation
    # ------------------------------------------------------------------

    def _compute_P(self, X: np.ndarray) -> np.ndarray:
        """Compute the joint probability matrix P (dense, n×n).

        In multiscale mode, P is the average over exponentially-spaced
        perplexities.  The matrix is symmetrised and globally normalised.
        """
        n = X.shape[0]
        if self.multiscale:
            H = max(1, round(math.log2(n / 2)))
            perplexities = [2**h for h in range(1, H + 1)]
            logger.info(
                "  Multiscale: %d perplexities from %d to %d",
                len(perplexities),
                perplexities[0],
                perplexities[-1],
            )
        else:
            perplexities = [self.perplexity]

        P_sum = np.zeros((n, n), dtype=np.float64)
        for perp in perplexities:
            k = min(int(perp * 3), n - 1)
            index = NNDescent(
                X, n_neighbors=k + 1, metric=self.metric, random_state=self.seed
            )
            nn_idx, nn_dist = index.neighbor_graph
            P_perp = self._distances_to_P(nn_idx, nn_dist, n, perp)
            P_sum += P_perp

        P = P_sum / len(perplexities)
        P = (P + P.T) / 2
        P /= P.sum()
        P = np.maximum(P, 1e-12)
        return P

    def _distances_to_P(
        self,
        nn_idx: np.ndarray,
        nn_dist: np.ndarray,
        n: int,
        perplexity: float,
    ) -> np.ndarray:
        """Convert kNN distances to a conditional probability matrix P."""
        P = np.zeros((n, n), dtype=np.float64)
        target_entropy = np.log2(perplexity)
        for i in range(n):
            dists = nn_dist[i, 1:].astype(np.float64)  # skip self (index 0)
            indices = nn_idx[i, 1:]
            sigma = self._binary_search_sigma(dists, target_entropy)
            weights = np.exp(-(dists**2) / (2 * sigma**2))
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum
            for j, idx in enumerate(indices):
                P[i, idx] = weights[j]
        return P

    @staticmethod
    def _binary_search_sigma(
        distances: np.ndarray,
        target_entropy: float,
        tol: float = 1e-5,
        max_iter: int = 50,
    ) -> float:
        """Binary search for sigma such that the Gaussian kernel entropy matches target."""
        lo, hi = 1e-10, 1e4
        sigma = (lo + hi) / 2
        for _ in range(max_iter):
            sigma = (lo + hi) / 2
            weights = np.exp(-(distances**2) / (2 * sigma**2))
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

    # ------------------------------------------------------------------
    # KL divergence loss
    # ------------------------------------------------------------------

    def _kl_divergence(self, P: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute KL(P || Q) where Q is the Student-t kernel in output space."""
        diff = Y.unsqueeze(0) - Y.unsqueeze(1)         # (n, n, d)
        dist_sq = (diff**2).sum(dim=-1)                # (n, n)
        alpha = self.dof
        Q = (1 + dist_sq / alpha) ** (-(alpha + 1) / 2)
        n = Q.shape[0]
        mask = 1 - torch.eye(n, device=Q.device)
        Q = Q * mask
        Q = Q / (Q.sum() + 1e-15)
        Q = Q.clamp(min=1e-15)
        kl = (P * torch.log(P / Q)).sum()
        return kl
