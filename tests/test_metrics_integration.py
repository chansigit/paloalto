import numpy as np
import pytest

from paloalto.metrics import compute_all, scib_overall


class TestComputeAll:
    def test_returns_all_keys(self, mock_adata):
        scores = compute_all(
            adata=mock_adata,
            embed_key="X_umap_mock",
            label_key="cell_type",
            batch_key="batch",
        )
        assert "cell_type_asw" in scores
        assert "nmi" in scores
        assert "ari" in scores
        assert "clisi" in scores
        assert "batch_asw" in scores
        assert "ilisi" in scores
        assert "graph_connectivity" in scores
        assert "scgraph_corr_weighted" in scores
        assert "bio_score" in scores
        assert "batch_score" in scores
        assert "scib_overall" in scores

    def test_all_values_are_floats(self, mock_adata):
        scores = compute_all(
            adata=mock_adata,
            embed_key="X_umap_mock",
            label_key="cell_type",
            batch_key="batch",
        )
        for k, v in scores.items():
            assert isinstance(v, float), f"{k} is {type(v)}"


class TestScibOverall:
    def test_weighted_aggregation(self):
        bio = {"cell_type_asw": 0.8, "nmi": 0.6, "ari": 0.7, "clisi": 0.9}
        batch = {"batch_asw": 0.5, "ilisi": 0.4, "graph_connectivity": 0.6}
        result = scib_overall(bio, batch)
        expected_bio = np.mean([0.8, 0.6, 0.7, 0.9])
        expected_batch = np.mean([0.5, 0.4, 0.6])
        expected = 0.6 * expected_bio + 0.4 * expected_batch
        assert abs(result - expected) < 1e-10
