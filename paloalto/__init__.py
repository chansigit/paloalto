"""PALO ALTO: Pareto-guided Automatic Layout Optimization for Aligning Latent Topology in Omics embeddings."""

__version__ = "0.1.0"

import copy
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from paloalto.data import load_and_validate, subsample_stratified
from paloalto.metrics import compute_all
from paloalto.embed import get_embedder
from paloalto.optim.space import get_default_space
from paloalto.optim.bo import WeightedBO
from paloalto.optim.mobo import ParetoBO
from paloalto.agent.judge import AgentJudge
from paloalto.report.html import generate_report
from paloalto.utils import get_logger

logger = get_logger(__name__)

# Parameters from the search space that each embedder actually accepts.
_EMBEDDER_PARAMS = {
    "numap": {"n_neighbors", "min_dist", "negative_sample_rate", "metric", "se_dim", "se_neighbors"},
    "tsne": {"perplexity", "dof", "early_exaggeration", "metric", "learning_rate", "exaggeration"},
}


def _filter_embedder_params(method: str, params: Dict) -> Dict:
    """Keep only the params that the embedder constructor accepts."""
    allowed = _EMBEDDER_PARAMS.get(method, set())
    return {k: v for k, v in params.items() if k in allowed}


def _evaluate_trial(
    adata,
    params: Dict,
    method: str,
    embedding_key: str,
    label_key: str,
    batch_key: str,
    subsample_idx: np.ndarray,
    embed_kwargs: Dict,
) -> Dict[str, float]:
    """Run one trial: embed, compute metrics, return scores dict."""
    # Build embedder with only the params it accepts, plus embed_kwargs
    embedder_params = _filter_embedder_params(method, params)
    merged = {**embedder_params, **embed_kwargs}
    embedder = get_embedder(method, **merged)

    # Fit and get 2D coordinates
    coords = embedder.fit(adata, embedding_key)

    # Store temporarily for metric computation
    adata.obsm["_paloalto_trial"] = coords

    # Subset to metric subsample
    adata_sub = adata[subsample_idx].copy()

    # Compute metrics on the subsample
    scores = compute_all(
        adata_sub,
        embed_key="_paloalto_trial",
        label_key=label_key,
        batch_key=batch_key,
    )

    # Clean up
    if "_paloalto_trial" in adata.obsm:
        del adata.obsm["_paloalto_trial"]

    return {
        "scib_overall": scores["scib_overall"],
        "scgraph_score": scores.get("scgraph_corr_weighted", 0.0),
    }


def optimize(
    adata,
    embedding_key: str = "X_pca",
    batch_key: str = "batch",
    label_key: str = "cell_type",
    method: str = "numap",
    bo_mode: str = "weighted",
    weights: Optional[List[float]] = None,
    n_trials: int = 30,
    n_initial: int = 5,
    use_agent: bool = True,
    agent_review_interval: int = 1,
    agent_model: str = "claude-sonnet-4-6",
    subsample_n: int = 10000,
    output_dir: str = "./results",
    seed: int = 42,
    search_space=None,
    embed_kwargs: Optional[Dict] = None,
) -> Dict:
    """Run Bayesian optimization of visualization hyperparameters.

    Parameters
    ----------
    adata : AnnData or str/Path
        Input data with embeddings, batch, and label annotations.
    embedding_key : str
        Key in ``adata.obsm`` for the source embedding (e.g. PCA).
    batch_key : str
        Column in ``adata.obs`` with batch labels.
    label_key : str
        Column in ``adata.obs`` with cell-type labels.
    method : str
        Embedding method: ``"numap"`` or ``"tsne"``.
    bo_mode : str
        ``"weighted"`` for single-objective or ``"pareto"`` for multi-objective.
    weights : list of float, optional
        Weights ``[w_scib, w_scgraph]`` for weighted mode. Default ``[0.5, 0.5]``.
    n_trials : int
        Total number of BO trials (including initial Sobol points).
    n_initial : int
        Number of Sobol quasi-random initialization points.
    use_agent : bool
        Whether to use the LLM agent judge for the agent arm.
    agent_review_interval : int
        Agent reviews every N-th trial after initialization.
    agent_model : str
        Model name for the agent judge.
    subsample_n : int
        Number of cells to subsample for metric computation.
    output_dir : str
        Directory for the HTML report and artifacts.
    seed : int
        Random seed for reproducibility.
    search_space : SearchSpace, optional
        Custom search space. Defaults to ``get_default_space(method)``.
    embed_kwargs : dict, optional
        Extra keyword arguments passed to the embedder (e.g. ``epochs``).

    Returns
    -------
    dict
        Result dictionary with keys: ``best_params``, ``best_scores``,
        ``trial_history``, ``baseline_history``, ``pareto_front``, ``report_path``.
    """
    start_time = time.time()

    # ---- 0. Setup ----
    embed_kwargs = dict(embed_kwargs or {})

    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Load & validate ----
    adata = load_and_validate(adata, embedding_key, batch_key, label_key)

    # ---- 2. Create fixed metric subsample ----
    subsample_idx = subsample_stratified(adata, label_key, batch_key, n=subsample_n, seed=seed)
    logger.info(f"Metric subsample: {len(subsample_idx)} cells")

    # ---- 3. Initialize BO ----
    if search_space is None:
        search_space = get_default_space(method)

    # Agent arm
    if bo_mode == "weighted":
        agent_bo = WeightedBO(space=search_space, weights=weights, seed=seed)
    else:
        agent_bo = ParetoBO(space=search_space, seed=seed)

    # Baseline arm (independent GP with its own copy of the space)
    baseline_space = get_default_space(method)
    if bo_mode == "weighted":
        baseline_bo = WeightedBO(space=baseline_space, weights=weights, seed=seed + 1)
    else:
        baseline_bo = ParetoBO(space=baseline_space, seed=seed + 1)

    # Agent judge (only used if use_agent=True)
    agent_judge = AgentJudge(model=agent_model) if use_agent else None

    dataset_summary = {
        "n_cells": adata.n_obs,
        "n_batches": adata.obs[batch_key].nunique(),
        "n_types": adata.obs[label_key].nunique(),
        "embedding_key": embedding_key,
        "embedding_dim": adata.obsm[embedding_key].shape[1],
    }

    # ---- 4. Sobol initialization (shared candidates) ----
    initial_candidates = agent_bo.suggest_initial(n=n_initial)
    logger.info(f"Running {n_initial} Sobol initialization trials...")

    for i, params in enumerate(initial_candidates):
        logger.info(f"  Init trial {i + 1}/{n_initial}: {params}")
        try:
            scores = _evaluate_trial(
                adata, params, method, embedding_key, label_key, batch_key,
                subsample_idx, embed_kwargs,
            )
        except Exception as e:
            logger.warning(f"  Trial {i + 1} failed: {e}")
            scores = {"scib_overall": 0.0, "scgraph_score": 0.0}

        # Observe on both arms (shared init)
        agent_bo.observe(params, scores["scib_overall"], scores["scgraph_score"])
        baseline_bo.observe(params, scores["scib_overall"], scores["scgraph_score"])

    # ---- 5. BO loop ----
    n_bo = n_trials - n_initial
    logger.info(f"Running {n_bo} BO trials...")

    for i in range(n_bo):
        trial_num = n_initial + i + 1
        logger.info(f"--- Trial {trial_num}/{n_trials} ---")

        # --- Agent arm ---
        try:
            agent_suggestion = agent_bo.suggest()
        except Exception as e:
            logger.warning(f"Agent BO suggest failed: {e}, using Sobol fallback")
            agent_suggestion = agent_bo.suggest_initial(1)[0]

        agent_params = agent_suggestion

        # Agent review
        if use_agent and agent_judge is not None and (i % agent_review_interval == 0):
            try:
                # Build best reference
                if bo_mode == "weighted":
                    current_best = agent_bo.best()
                else:
                    front = agent_bo.pareto_front()
                    if front:
                        current_best = max(
                            front,
                            key=lambda t: t["scores"]["scib_overall"] + t["scores"]["scgraph_score"],
                        )
                    else:
                        current_best = agent_bo.get_history()[-1] if agent_bo.get_history() else {}

                context = agent_judge.build_context(
                    trial_number=trial_num,
                    trial_history=agent_bo.get_history(),
                    current_best=current_best,
                    bo_suggestion=agent_suggestion,
                    search_space={k: v for k, v in search_space.params.items()},
                    dataset_summary=dataset_summary,
                )
                action = agent_judge.review(context)

                if action.action == "modify" and action.params:
                    agent_params = {**agent_suggestion, **action.params}
                elif action.action == "inject" and action.params:
                    agent_params = action.params
                elif action.action == "prune" and action.new_bounds:
                    for pname, (lo, hi) in action.new_bounds.items():
                        search_space.prune(pname, new_lo=lo, new_hi=hi)
                    # Re-suggest with pruned space
                    try:
                        agent_params = agent_bo.suggest()
                    except Exception:
                        agent_params = agent_suggestion
                elif action.action == "stop":
                    logger.info("Agent recommended STOP. Ending optimization early.")
                    break
                # else: accept -- use agent_suggestion as-is
            except Exception as e:
                logger.warning(f"Agent review failed: {e}, using BO suggestion as-is")

        # Evaluate agent arm
        logger.info(f"  Agent arm: {agent_params}")
        try:
            agent_scores = _evaluate_trial(
                adata, agent_params, method, embedding_key, label_key, batch_key,
                subsample_idx, embed_kwargs,
            )
        except Exception as e:
            logger.warning(f"  Agent trial failed: {e}")
            agent_scores = {"scib_overall": 0.0, "scgraph_score": 0.0}
        agent_bo.observe(agent_params, agent_scores["scib_overall"], agent_scores["scgraph_score"])

        # --- Baseline arm ---
        try:
            baseline_params = baseline_bo.suggest()
        except Exception as e:
            logger.warning(f"Baseline BO suggest failed: {e}, using Sobol fallback")
            baseline_params = baseline_bo.suggest_initial(1)[0]

        logger.info(f"  Baseline arm: {baseline_params}")
        try:
            baseline_scores = _evaluate_trial(
                adata, baseline_params, method, embedding_key, label_key, batch_key,
                subsample_idx, embed_kwargs,
            )
        except Exception as e:
            logger.warning(f"  Baseline trial failed: {e}")
            baseline_scores = {"scib_overall": 0.0, "scgraph_score": 0.0}
        baseline_bo.observe(
            baseline_params, baseline_scores["scib_overall"], baseline_scores["scgraph_score"],
        )

    wall_time = time.time() - start_time

    # ---- 6. Collect results ----
    agent_history = agent_bo.get_history()
    baseline_history = baseline_bo.get_history()

    if bo_mode == "weighted":
        best_agent = agent_bo.best()
        best_baseline = baseline_bo.best()
        pareto_front_result = None
    else:
        # For pareto mode, pick the trial with highest sum of objectives
        front = agent_bo.pareto_front()
        if front:
            best_agent = max(
                front,
                key=lambda t: t["scores"]["scib_overall"] + t["scores"]["scgraph_score"],
            )
        else:
            best_agent = max(
                agent_history,
                key=lambda t: t["scores"]["scib_overall"] + t["scores"]["scgraph_score"],
            )
        # Add objective key for report template compatibility
        best_agent.setdefault(
            "objective",
            best_agent["scores"]["scib_overall"] + best_agent["scores"]["scgraph_score"],
        )

        baseline_front = baseline_bo.pareto_front()
        if baseline_front:
            best_baseline = max(
                baseline_front,
                key=lambda t: t["scores"]["scib_overall"] + t["scores"]["scgraph_score"],
            )
        else:
            best_baseline = max(
                baseline_history,
                key=lambda t: t["scores"]["scib_overall"] + t["scores"]["scgraph_score"],
            )
        best_baseline.setdefault(
            "objective",
            best_baseline["scores"]["scib_overall"] + best_baseline["scores"]["scgraph_score"],
        )

        pareto_front_result = [
            {
                "params": t["params"],
                "scores": t["scores"],
            }
            for t in front
        ]

    # ---- 7. Re-fit best model for report visualization ----
    rng = np.random.RandomState(seed)

    def _refit_best(best_trial):
        """Re-fit the best params and return 2D coordinates."""
        try:
            bp = _filter_embedder_params(method, best_trial["params"])
            merged = {**bp, **embed_kwargs}
            embedder = get_embedder(method, **merged)
            coords = embedder.fit(adata, embedding_key)
            return coords
        except Exception as e:
            logger.warning(f"Re-fit failed: {e}, using random fallback")
            return rng.randn(adata.n_obs, 2)

    best_coords_agent = _refit_best(best_agent)
    best_coords_baseline = _refit_best(best_baseline)

    # ---- 8. Generate report ----
    # Ensure objective key exists on all history entries for the template
    for trial in agent_history:
        trial.setdefault(
            "objective",
            trial["scores"]["scib_overall"] + trial["scores"]["scgraph_score"],
        )
    for trial in baseline_history:
        trial.setdefault(
            "objective",
            trial["scores"]["scib_overall"] + trial["scores"]["scgraph_score"],
        )

    report_data = {
        "method": method,
        "bo_mode": bo_mode,
        "dataset_summary": dataset_summary,
        "agent_history": agent_history,
        "baseline_history": baseline_history,
        "best_agent": best_agent,
        "best_baseline": best_baseline,
        "best_coords_agent": best_coords_agent,
        "best_coords_baseline": best_coords_baseline,
        "labels": adata.obs[label_key].values,
        "batches": adata.obs[batch_key].values,
        "agent_log": agent_judge.get_log() if agent_judge else [],
        "wall_time_seconds": wall_time,
    }

    report_path = str(Path(output_dir) / "report.html")
    generate_report(report_data, report_path)

    # ---- 9. Return results ----
    result = {
        "best_params": best_agent["params"],
        "best_scores": {
            "scib_overall": best_agent["scores"]["scib_overall"],
            "scgraph_score": best_agent["scores"]["scgraph_score"],
        },
        "trial_history": agent_history,
        "baseline_history": baseline_history,
        "pareto_front": pareto_front_result,
        "report_path": report_path,
    }
    logger.info(f"Optimization complete. Report: {report_path}")
    logger.info(f"Best params: {result['best_params']}")
    logger.info(f"Best scores: {result['best_scores']}")
    return result
