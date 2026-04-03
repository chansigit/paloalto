import numpy as np
import pytest

from paloalto.metrics.scgraph import corr_weighted


class TestCorrWeighted:
    def test_returns_float(self, mock_adata):
        score = corr_weighted(
            adata=mock_adata,
            embed_key="X_umap_mock",
            label_key="cell_type",
            batch_key="batch",
        )
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_identity_embedding_high_score(self, mock_adata):
        """If embedding = PCA, score should be very high."""
        score = corr_weighted(
            adata=mock_adata,
            embed_key="X_pca",
            label_key="cell_type",
            batch_key="batch",
        )
        assert score > 0.5

    def test_random_embedding_lower_score(self, mock_adata):
        np.random.seed(0)
        mock_adata.obsm["X_random"] = np.random.randn(mock_adata.n_obs, 2).astype(np.float32)
        score_rand = corr_weighted(
            adata=mock_adata,
            embed_key="X_random",
            label_key="cell_type",
            batch_key="batch",
        )
        score_pca = corr_weighted(
            adata=mock_adata,
            embed_key="X_pca",
            label_key="cell_type",
            batch_key="batch",
        )
        assert score_pca > score_rand
