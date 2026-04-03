import numpy as np
import pytest

from paloalto.metrics.batch import batch_asw, ilisi, graph_connectivity


class TestBatchASW:
    def test_returns_float_in_range(self, mock_adata):
        score = batch_asw(
            mock_adata.obsm["X_umap_mock"],
            mock_adata.obs["batch"].values,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_well_mixed_high_score(self):
        np.random.seed(42)
        coords = np.random.randn(200, 2)
        labels = np.array(["A", "B"] * 100)
        score = batch_asw(coords, labels)
        assert score > 0.7  # well mixed → low |ASW| → high score


class TestILISI:
    def test_returns_float_in_range(self, mock_adata):
        score = ilisi(
            mock_adata.obsm["X_umap_mock"],
            mock_adata.obs["batch"].values,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_well_mixed_high_score(self):
        np.random.seed(42)
        coords = np.random.randn(200, 2)
        labels = np.array(["A", "B"] * 100)
        score = ilisi(coords, labels)
        assert score > 0.5


class TestGraphConnectivity:
    def test_returns_float_in_range(self, mock_adata):
        score = graph_connectivity(
            mock_adata.obsm["X_umap_mock"],
            mock_adata.obs["cell_type"].values,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_fully_connected_returns_one(self):
        np.random.seed(42)
        coords = np.random.randn(200, 2)
        labels = np.array(["A"] * 100 + ["B"] * 100)
        score = graph_connectivity(coords, labels)
        assert score == 1.0  # all same-type cells are connected
