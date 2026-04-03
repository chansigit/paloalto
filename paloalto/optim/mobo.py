"""Multi-objective BO via NEHVI for 2D Pareto optimization."""

from typing import Dict, List

import numpy as np
import torch
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from paloalto.optim.space import SearchSpace
from paloalto.utils import get_logger

logger = get_logger(__name__)

REF_POINT = torch.tensor([0.0, 0.0], dtype=torch.float64)


class ParetoBO:
    """Multi-objective BO with NEHVI acquisition."""

    def __init__(self, space: SearchSpace, seed: int = 42):
        self.space = space
        self.seed = seed
        self.bounds, self.param_names = space.to_botorch_bounds()

        self.X: List[torch.Tensor] = []
        self.Y: List[torch.Tensor] = []  # (scib_overall, scgraph_score)
        self.history: List[Dict] = []

    def suggest_initial(self, n: int = 5) -> List[Dict]:
        sobol = SobolEngine(dimension=self.bounds.shape[1], scramble=True, seed=self.seed)
        raw = sobol.draw(n).to(dtype=self.bounds.dtype)
        scaled = raw * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return [self.space.decode(scaled[i], self.param_names) for i in range(n)]

    def observe(self, params: Dict, scib_overall: float, scgraph_score: float):
        x = self.space.encode(params)
        self.X.append(torch.tensor(x, dtype=torch.float64))
        self.Y.append(torch.tensor([scib_overall, scgraph_score], dtype=torch.float64))
        self.history.append({
            "params": params,
            "scores": {"scib_overall": scib_overall, "scgraph_score": scgraph_score},
        })

    def suggest(self) -> Dict:
        train_X = torch.stack(self.X)
        train_Y = torch.stack(self.Y)

        # Fit independent GPs per objective
        models = []
        for i in range(2):
            gp = SingleTaskGP(train_X, train_Y[:, i:i+1])
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            models.append(gp)

        model = ModelListGP(*models)

        acq = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=REF_POINT,
            X_baseline=train_X,
            prune_baseline=True,
        )

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )

        return self.space.decode(candidate.squeeze(0), self.param_names)

    def pareto_front(self) -> List[Dict]:
        """Return non-dominated trials."""
        if not self.Y:
            return []
        Y = torch.stack(self.Y)
        mask = is_non_dominated(Y)
        return [self.history[i] for i in range(len(self.history)) if mask[i]]

    def get_history(self) -> List[Dict]:
        return self.history
