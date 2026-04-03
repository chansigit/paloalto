import pytest
import torch

from paloalto.optim.space import SearchSpace, get_default_space


class TestSearchSpace:
    def test_numap_default_space(self):
        space = get_default_space("numap")
        assert "n_neighbors" in space.params
        assert "min_dist" in space.params
        assert "metric" in space.params

    def test_tsne_default_space(self):
        space = get_default_space("tsne")
        assert "perplexity" in space.params
        assert "dof" in space.params

    def test_to_botorch_bounds(self):
        space = get_default_space("numap")
        bounds, param_names = space.to_botorch_bounds()
        assert bounds.shape[0] == 2  # lower, upper
        assert bounds.shape[1] == len(param_names)
        assert (bounds[0] < bounds[1]).all()

    def test_decode_params(self):
        space = get_default_space("numap")
        bounds, names = space.to_botorch_bounds()
        # Midpoint of bounds
        x = (bounds[0] + bounds[1]) / 2
        decoded = space.decode(x, names)
        assert isinstance(decoded["n_neighbors"], int)
        assert isinstance(decoded["min_dist"], float)
        assert decoded["metric"] in ("euclidean", "cosine", "correlation")

    def test_prune_shrinks_bounds(self):
        space = get_default_space("numap")
        old_lo = space.params["min_dist"]["bounds"][0]
        old_hi = space.params["min_dist"]["bounds"][1]
        space.prune("min_dist", new_lo=0.01, new_hi=0.5)
        assert space.params["min_dist"]["bounds"][0] >= old_lo
        assert space.params["min_dist"]["bounds"][1] <= old_hi
