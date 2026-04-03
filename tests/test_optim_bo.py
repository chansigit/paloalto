import pytest
import torch
import numpy as np

from paloalto.optim.bo import WeightedBO
from paloalto.optim.space import get_default_space


class TestWeightedBO:
    def test_sobol_initialization(self):
        space = get_default_space("numap")
        bo = WeightedBO(space, weights=[0.5, 0.5], seed=42)
        candidates = bo.suggest_initial(n=5)
        assert len(candidates) == 5
        for c in candidates:
            assert "n_neighbors" in c
            assert "min_dist" in c

    def test_suggest_after_observations(self):
        space = get_default_space("numap")
        bo = WeightedBO(space, weights=[0.5, 0.5], seed=42)

        # Add some fake observations
        for i in range(5):
            params = bo.suggest_initial(n=1)[0]
            bo.observe(params, scib_overall=np.random.rand(), scgraph_score=np.random.rand())

        # Now GP-based suggestion should work
        candidate = bo.suggest()
        assert "n_neighbors" in candidate
        assert "min_dist" in candidate

    def test_best_returns_highest_score(self):
        space = get_default_space("numap")
        bo = WeightedBO(space, weights=[0.5, 0.5], seed=42)

        bo.observe({"n_neighbors": 10, "min_dist": 0.1, "negative_sample_rate": 5,
                     "metric": "euclidean", "se_dim": 2, "se_neighbors": 10},
                    scib_overall=0.3, scgraph_score=0.4)
        bo.observe({"n_neighbors": 15, "min_dist": 0.2, "negative_sample_rate": 5,
                     "metric": "cosine", "se_dim": 5, "se_neighbors": 15},
                    scib_overall=0.8, scgraph_score=0.9)

        best = bo.best()
        assert best["scores"]["scib_overall"] == 0.8
