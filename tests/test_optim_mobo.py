import pytest
import numpy as np

from paloalto.optim.mobo import ParetoBO
from paloalto.optim.space import get_default_space


class TestParetoBO:
    def test_sobol_initialization(self):
        space = get_default_space("numap")
        bo = ParetoBO(space, seed=42)
        candidates = bo.suggest_initial(n=5)
        assert len(candidates) == 5

    def test_suggest_after_observations(self):
        space = get_default_space("numap")
        bo = ParetoBO(space, seed=42)
        for i in range(5):
            params = bo.suggest_initial(n=1)[0]
            bo.observe(params, scib_overall=np.random.rand(), scgraph_score=np.random.rand())

        candidate = bo.suggest()
        assert "n_neighbors" in candidate

    def test_pareto_front(self):
        space = get_default_space("numap")
        bo = ParetoBO(space, seed=42)

        bo.observe({"n_neighbors": 10, "min_dist": 0.1, "negative_sample_rate": 5,
                     "metric": "euclidean", "se_dim": 2, "se_neighbors": 10},
                    scib_overall=0.9, scgraph_score=0.3)
        bo.observe({"n_neighbors": 15, "min_dist": 0.2, "negative_sample_rate": 5,
                     "metric": "cosine", "se_dim": 5, "se_neighbors": 15},
                    scib_overall=0.3, scgraph_score=0.9)
        bo.observe({"n_neighbors": 20, "min_dist": 0.3, "negative_sample_rate": 5,
                     "metric": "euclidean", "se_dim": 3, "se_neighbors": 20},
                    scib_overall=0.5, scgraph_score=0.5)

        front = bo.pareto_front()
        # First two dominate the third on at least one objective
        assert len(front) >= 2
