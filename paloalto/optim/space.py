"""Search space definitions for BO."""

import math
from typing import Dict, List, Tuple

import torch


class SearchSpace:
    """Defines the hyperparameter search space for BO."""

    def __init__(self, params: Dict):
        """
        Args:
            params: dict of param_name -> {
                "type": "float" | "int" | "categorical",
                "bounds": [lo, hi] for float/int,
                "choices": [...] for categorical,
                "log_scale": bool (optional),
            }
        """
        self.params = params

    def to_botorch_bounds(self) -> Tuple[torch.Tensor, List[str]]:
        """Convert to BoTorch bounds tensor. Categoricals become one-hot dims."""
        lowers, uppers, names = [], [], []

        for name, spec in self.params.items():
            if spec["type"] in ("float", "int"):
                lo, hi = spec["bounds"]
                if spec.get("log_scale"):
                    lo, hi = math.log(lo), math.log(hi)
                lowers.append(lo)
                uppers.append(hi)
                names.append(name)
            elif spec["type"] == "categorical":
                for choice in spec["choices"]:
                    lowers.append(0.0)
                    uppers.append(1.0)
                    names.append(f"{name}={choice}")

        bounds = torch.tensor([lowers, uppers], dtype=torch.float64)
        return bounds, names

    def decode(self, x: torch.Tensor, param_names: List[str]) -> Dict:
        """Decode a BoTorch tensor back to named hyperparameters."""
        result = {}
        x_np = x.detach().cpu().numpy()
        i = 0

        for name, spec in self.params.items():
            if spec["type"] == "float":
                val = float(x_np[i])
                if spec.get("log_scale"):
                    val = math.exp(val)
                result[name] = val
                i += 1
            elif spec["type"] == "int":
                val = float(x_np[i])
                if spec.get("log_scale"):
                    val = math.exp(val)
                result[name] = int(round(val))
                i += 1
            elif spec["type"] == "categorical":
                n_choices = len(spec["choices"])
                scores = x_np[i:i + n_choices]
                result[name] = spec["choices"][int(scores.argmax())]
                i += n_choices

        return result

    def encode(self, params: Dict) -> List[float]:
        """Encode a parameter dict to a flat list for BoTorch."""
        values = []
        for name, spec in self.params.items():
            val = params[name]
            if spec["type"] in ("float", "int"):
                v = float(val)
                if spec.get("log_scale"):
                    v = math.log(v)
                values.append(v)
            elif spec["type"] == "categorical":
                for choice in spec["choices"]:
                    values.append(1.0 if val == choice else 0.0)
        return values

    def prune(self, param_name: str, new_lo: float = None, new_hi: float = None):
        """Shrink bounds for a parameter (agent PRUNE action)."""
        spec = self.params[param_name]
        if spec["type"] not in ("float", "int"):
            return
        if new_lo is not None:
            spec["bounds"][0] = max(spec["bounds"][0], new_lo)
        if new_hi is not None:
            spec["bounds"][1] = min(spec["bounds"][1], new_hi)


def get_default_space(method: str) -> SearchSpace:
    """Return the default search space for a method."""
    if method == "numap":
        return SearchSpace({
            "n_neighbors": {"type": "int", "bounds": [5, 200], "log_scale": True},
            "min_dist": {"type": "float", "bounds": [0.001, 0.99]},
            "negative_sample_rate": {"type": "int", "bounds": [1, 20]},
            "metric": {"type": "categorical", "choices": ["euclidean", "cosine", "correlation"]},
            "se_dim": {"type": "int", "bounds": [2, 20]},
            "se_neighbors": {"type": "int", "bounds": [5, 100]},
        })
    elif method == "umap":
        return SearchSpace({
            "n_neighbors": {"type": "int", "bounds": [5, 200], "log_scale": True},
            "min_dist": {"type": "float", "bounds": [0.001, 0.99]},
        })
    elif method == "tsne":
        return SearchSpace({
            "perplexity": {"type": "float", "bounds": [5.0, 100.0], "log_scale": True},
            "dof": {"type": "float", "bounds": [0.2, 2.0]},
            "early_exaggeration": {"type": "float", "bounds": [4.0, 32.0], "log_scale": True},
            "metric": {"type": "categorical", "choices": ["euclidean", "cosine", "correlation"]},
            "learning_rate": {"type": "float", "bounds": [1e-4, 1e-1], "log_scale": True},
            "exaggeration": {"type": "float", "bounds": [1.0, 4.0]},
        })
    else:
        raise ValueError(f"Unknown method: {method}")
