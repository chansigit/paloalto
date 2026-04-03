import numpy as np
import pytest

from paloalto.metrics.bio import cell_type_asw, nmi, ari, clisi


class TestCellTypeASW:
    def test_returns_float_in_range(self, mock_adata):
        score = cell_type_asw(mock_adata.obsm["X_umap_mock"], mock_adata.obs["cell_type"].values)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_perfect_clusters(self):
        coords = np.vstack([np.random.randn(50, 2) + [i * 20, 0] for i in range(4)])
        labels = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50 + ["D"] * 50)
        score = cell_type_asw(coords, labels)
        assert score > 0.7


class TestNMI:
    def test_returns_float_in_range(self, mock_adata):
        score = nmi(mock_adata.obsm["X_umap_mock"], mock_adata.obs["cell_type"].values)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestARI:
    def test_returns_float_in_range(self, mock_adata):
        score = ari(mock_adata.obsm["X_umap_mock"], mock_adata.obs["cell_type"].values)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestCLISI:
    def test_returns_float_in_range(self, mock_adata):
        score = clisi(mock_adata.obsm["X_umap_mock"], mock_adata.obs["cell_type"].values)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_well_separated_clusters_high_score(self):
        coords = np.vstack([np.random.randn(50, 2) + [i * 20, 0] for i in range(3)])
        labels = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50)
        score = clisi(coords, labels)
        assert score > 0.7  # high purity
