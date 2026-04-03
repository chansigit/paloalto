"""Weighted scalar Bayesian optimization (single-objective)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from paloalto.optim.space import SearchSpace
from paloalto.utils import get_logger

logger = get_logger(__name__)


class WeightedBO:
    """Single-objective BO with weighted scalar objective."""

    def __init__(
        self,
        space: SearchSpace,
        weights: List[float] = None,
        seed: int = 42,
    ):
        self.space = space
        self.weights = weights or [0.5, 0.5]
        self.seed = seed
        self.bounds, self.param_names = space.to_botorch_bounds()

        self.X: List[torch.Tensor] = []
        self.Y: List[float] = []
        self.history: List[Dict] = []

    def suggest_initial(self, n: int = 5) -> List[Dict]:
        """Generate n Sobol quasi-random candidates."""
        sobol = SobolEngine(dimension=self.bounds.shape[1], scramble=True, seed=self.seed)
        raw = sobol.draw(n).to(dtype=self.bounds.dtype)
        # Scale to bounds
        scaled = raw * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        candidates = []
        for i in range(n):
            params = self.space.decode(scaled[i], self.param_names)
            candidates.append(params)
        return candidates

    def observe(self, params: Dict, scib_overall: float, scgraph_score: float):
        """Record an observation."""
        x = self.space.encode(params)
        self.X.append(torch.tensor(x, dtype=torch.float64))
        objective = self.weights[0] * scib_overall + self.weights[1] * scgraph_score
        self.Y.append(objective)
        self.history.append({
            "params": params,
            "scores": {"scib_overall": scib_overall, "scgraph_score": scgraph_score},
            "objective": objective,
        })

    def suggest(self) -> Dict:
        """Suggest next candidate via GP + Expected Improvement."""
        train_X = torch.stack(self.X)
        train_Y = torch.tensor(self.Y, dtype=torch.float64).unsqueeze(-1)

        # Fit GP
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Optimize acquisition
        acq = LogExpectedImprovement(model=gp, best_f=train_Y.max())
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )

        return self.space.decode(candidate.squeeze(0), self.param_names)

    def best(self) -> Dict:
        """Return the best trial so far."""
        best_idx = int(np.argmax(self.Y))
        return self.history[best_idx]

    def get_history(self) -> List[Dict]:
        return self.history
