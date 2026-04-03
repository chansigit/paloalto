import numpy as np
import pytest

from paloalto.metrics.lisi import compute_lisi


class TestLISI:
    def test_perfect_mixing(self):
        """All neighbors have equal representation of 2 labels → LISI ≈ 2."""
        np.random.seed(42)
        # Interleaved: even indices = A, odd = B, fully mixed spatially
        coords = np.column_stack([np.arange(100), np.zeros(100)]).astype(np.float32)
        labels = np.array(["A", "B"] * 50)
        lisi = compute_lisi(coords, labels, perplexity=15)
        assert lisi.shape == (100,)
        assert np.mean(lisi) > 1.5  # close to 2

    def test_perfect_separation(self):
        """Two well-separated clusters with distinct labels → LISI ≈ 1."""
        coords = np.vstack([
            np.random.randn(50, 2) + [0, 0],
            np.random.randn(50, 2) + [100, 100],
        ]).astype(np.float32)
        labels = np.array(["A"] * 50 + ["B"] * 50)
        lisi = compute_lisi(coords, labels, perplexity=15)
        assert np.mean(lisi) < 1.2  # close to 1

    def test_output_range(self, mock_adata):
        coords = mock_adata.obsm["X_umap_mock"]
        labels = mock_adata.obs["batch"].values
        lisi = compute_lisi(coords, labels, perplexity=15)
        n_unique = len(np.unique(labels))
        assert np.all(lisi >= 1.0)
        assert np.all(lisi <= n_unique + 0.1)
