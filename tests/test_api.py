import pytest
import numpy as np

from paloalto import optimize


class TestOptimize:
    def test_runs_weighted_mode(self, mock_adata, tmp_path):
        """Smoke test: 2 Sobol trials, no agent, weighted mode."""
        result = optimize(
            adata=mock_adata,
            embedding_key="X_pca",
            batch_key="batch",
            label_key="cell_type",
            method="numap",
            bo_mode="weighted",
            n_trials=2,
            n_initial=2,
            use_agent=False,
            subsample_n=200,
            output_dir=str(tmp_path),
            seed=42,
            embed_kwargs={"epochs": 2},
        )
        assert "best_params" in result
        assert "best_scores" in result
        assert "trial_history" in result
        assert "report_path" in result
        assert len(result["trial_history"]) == 2

    def test_runs_pareto_mode(self, mock_adata, tmp_path):
        result = optimize(
            adata=mock_adata,
            embedding_key="X_pca",
            batch_key="batch",
            label_key="cell_type",
            method="numap",
            bo_mode="pareto",
            n_trials=2,
            n_initial=2,
            use_agent=False,
            subsample_n=200,
            output_dir=str(tmp_path),
            seed=42,
            embed_kwargs={"epochs": 2},
        )
        assert "pareto_front" in result
