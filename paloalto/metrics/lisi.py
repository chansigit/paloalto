"""LISI (Local Inverse Simpson's Index) computation.

Internalized from scIB. Computes per-cell diversity of a categorical label
in local neighborhoods defined by a Gaussian kernel with specified perplexity.

Torch-vectorized: binary search for sigma and Simpson's index are computed
for all cells simultaneously on GPU (if available), eliminating the per-cell
Python loop.

Attribution: Luecken et al. (2022) "Benchmarking atlas-level data integration
in single-cell genomics", Nature Methods. Original scIB code at
https://github.com/theislab/scib under BSD-3-Clause.
"""

import numpy as np
import torch
from pynndescent import NNDescent


def compute_lisi(
    coords: np.ndarray,
    labels: np.ndarray,
    perplexity: float = 30.0,
    nn_idx: np.ndarray = None,
    nn_dist: np.ndarray = None,
) -> np.ndarray:
    """Compute per-cell LISI scores (torch-vectorized).

    Args:
        coords: (n_cells, n_dims) embedding coordinates.
        labels: (n_cells,) categorical labels.
        perplexity: Perplexity for Gaussian kernel bandwidth.
        nn_idx: Pre-computed kNN indices (n_cells, k). If None, built internally.
        nn_dist: Pre-computed kNN distances (n_cells, k). If None, built internally.

    Returns:
        (n_cells,) LISI scores. Range [1, n_unique_labels].
        1 = all neighbors have the same label; n = perfectly mixed.
    """
    n = coords.shape[0]
    k = min(int(perplexity * 3), n - 1)

    if nn_idx is None or nn_dist is None:
        index = NNDescent(coords, n_neighbors=k + 1, random_state=42)
        nn_idx, nn_dist = index.neighbor_graph
        nn_idx = nn_idx[:, 1:]
        nn_dist = nn_dist[:, 1:]
    else:
        if nn_idx.shape[1] > k:
            nn_idx = nn_idx[:, :k]
            nn_dist = nn_dist[:, :k]

    # Encode labels as integers
    unique_labels, label_codes = np.unique(labels, return_inverse=True)
    n_labels = len(unique_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (n, k) tensors on device
    dist_t = torch.from_numpy(nn_dist.astype(np.float64)).to(device)
    dist_sq = dist_t ** 2

    # --- Vectorized binary search for sigma ---
    # All cells searched in parallel
    sigma = _find_sigma_batched(dist_sq, perplexity, device=device)  # (n,)

    # Compute weights: exp(-d^2 / (2*sigma^2))
    weights = torch.exp(-dist_sq / (2 * sigma.unsqueeze(1) ** 2))  # (n, k)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-30)  # normalize

    # --- Vectorized Simpson's index via scatter_add ---
    # neighbor label codes: (n, k)
    neighbor_labels = torch.from_numpy(label_codes[nn_idx]).to(device).long()

    # Accumulate weighted proportions per label: (n, n_labels)
    proportions = torch.zeros(n, n_labels, dtype=torch.float64, device=device)
    proportions.scatter_add_(1, neighbor_labels, weights)

    # Simpson = sum(p^2), LISI = 1/Simpson
    simpson = (proportions ** 2).sum(dim=1)  # (n,)
    lisi = 1.0 / simpson

    lisi = lisi.clamp(1.0, float(n_labels)).cpu().numpy()
    return lisi


def _find_sigma_batched(
    dist_sq: torch.Tensor,
    target_perplexity: float,
    tol: float = 1e-5,
    max_iter: int = 100,
    device: torch.device = None,
) -> torch.Tensor:
    """Batched binary search for sigma across all cells simultaneously.

    Args:
        dist_sq: (n, k) squared distances to k neighbors.
        target_perplexity: Target perplexity value.

    Returns:
        (n,) sigma values.
    """
    n = dist_sq.shape[0]
    lo = torch.full((n,), 1e-10, dtype=torch.float64, device=device)
    hi = torch.full((n,), 1e4, dtype=torch.float64, device=device)
    target_entropy = np.log2(target_perplexity)

    for _ in range(max_iter):
        sigma = (lo + hi) / 2  # (n,)
        # weights: (n, k)
        weights = torch.exp(-dist_sq / (2 * sigma.unsqueeze(1) ** 2))
        sum_w = weights.sum(dim=1)  # (n,)

        # Normalize to probabilities
        safe_sum = sum_w.clamp(min=1e-30)
        p = weights / safe_sum.unsqueeze(1)  # (n, k)

        # Entropy per cell
        entropy = -(p * torch.log2(p + 1e-15)).sum(dim=1)  # (n,)
        perp = 2 ** entropy

        # Update bounds
        too_low = perp < target_perplexity
        too_high = perp >= target_perplexity
        zero_sum = sum_w == 0
        lo = torch.where(too_low | zero_sum, sigma, lo)
        hi = torch.where(too_high & ~zero_sum, sigma, hi)

        # Check convergence
        if (torch.abs(perp - target_perplexity) < tol).all():
            break

    return sigma
