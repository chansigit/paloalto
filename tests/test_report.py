import os
import pytest
import numpy as np

from paloalto.report.html import generate_report


@pytest.fixture
def mock_results():
    np.random.seed(42)
    n_trials = 10
    history = []
    for i in range(n_trials):
        history.append({
            "params": {"n_neighbors": np.random.randint(5, 50), "min_dist": np.random.rand()},
            "scores": {"scib_overall": np.random.rand(), "scgraph_score": np.random.rand()},
            "objective": np.random.rand(),
        })
    return {
        "method": "numap",
        "bo_mode": "weighted",
        "dataset_summary": {"n_cells": 10000, "n_batches": 3, "n_types": 8},
        "agent_history": history,
        "baseline_history": history,
        "agent_log": [{"trial_number": 6, "action": "modify", "reasoning": "test"}],
        "best_agent": history[5],
        "best_baseline": history[3],
        "best_coords_agent": np.random.randn(100, 2),
        "best_coords_baseline": np.random.randn(100, 2),
        "labels": np.array(["A"] * 50 + ["B"] * 50),
        "batches": np.array(["b0"] * 50 + ["b1"] * 50),
        "wall_time_seconds": 120.5,
    }


class TestGenerateReport:
    def test_creates_html_file(self, mock_results, tmp_path):
        path = str(tmp_path / "report.html")
        generate_report(mock_results, path)
        assert os.path.exists(path)

    def test_html_contains_key_sections(self, mock_results, tmp_path):
        path = str(tmp_path / "report.html")
        generate_report(mock_results, path)
        with open(path) as f:
            html = f.read()
        assert "Overview" in html
        assert "Convergence" in html
        assert "Best Parameters" in html
        assert "Agent Log" in html
        assert "Caveats" in html

    def test_self_contained(self, mock_results, tmp_path):
        path = str(tmp_path / "report.html")
        generate_report(mock_results, path)
        with open(path) as f:
            html = f.read()
        # All images should be base64 encoded
        assert "data:image/png;base64" in html
