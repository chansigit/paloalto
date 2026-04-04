"""PaloAlto: Pareto-guided Automatic Layout Optimization for Aligning Latent Topology in Omics embeddings."""

__version__ = "0.1.0"

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
    "umap": {"n_neighbors", "min_dist", "metric"},
    "tsne": {"perplexity", "dof", "early_exaggeration", "metric", "learning_rate", "exaggeration"},
}


def _filter_embedder_params(method: str, params: Dict) -> Dict:
    """Keep only the params that the embedder constructor accepts."""
    allowed = _EMBEDDER_PARAMS.get(method, set())
    return {k: v for k, v in params.items() if k in allowed}


def _clamp_to_bounds(params: Dict, space) -> Dict:
    """Clamp agent-provided params to search space bounds."""
    clamped = dict(params)
    for name, spec in space.params.items():
        if name not in clamped:
            continue
        if spec["type"] in ("float", "int"):
            lo, hi = spec["bounds"]
            val = clamped[name]
            clamped[name] = type(val)(max(lo, min(hi, val)))
        elif spec["type"] == "categorical":
            if clamped[name] not in spec["choices"]:
                clamped[name] = spec["choices"][0]
    return clamped


def _evaluate_trial(
    adata,
    params: Dict,
    method: str,
    embedding_key: str,
    label_key: str,
    batch_key: str,
    subsample_idx: np.ndarray,
    embed_kwargs: Dict,
    knn_cache=None,
    save_path: str = None,
) -> Dict:
    """Run one trial: embed, compute metrics, optionally save model.

    Returns dict with 'scib_overall', 'scgraph_score', 'coords', and 'embedder'.
    """
    # Build embedder with only the params it accepts, plus embed_kwargs
    embedder_params = _filter_embedder_params(method, params)
    merged = {**embedder_params, **embed_kwargs}
    embedder = get_embedder(method, **merged)

    # Fit and get 2D coordinates (pass knn_cache if available)
    fit_kwargs = {"embedding_key": embedding_key}
    if knn_cache is not None and method == "numap":
        fit_kwargs["knn_cache"] = knn_cache
    coords = embedder.fit(adata, **fit_kwargs)

    # Save model if path provided
    if save_path is not None:
        try:
            embedder.save(save_path)
        except Exception as e:
            logger.warning(f"  Failed to save model: {e}")

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
        "coords": coords,
        "embedder": embedder,
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

    # ---- 4. Pre-build kNN cache (NUMAP only) ----
    knn_cache = None
    if method == "numap":
        from paloalto.knn_cache import KNNCache
        X_embed = adata.obsm[embedding_key]
        if not isinstance(X_embed, np.ndarray):
            X_embed = np.asarray(X_embed)
        X_embed = X_embed.astype(np.float32)
        knn_cache = KNNCache(X_embed, max_k=200, seed=seed)

    # ---- 5. Sobol initialization (shared candidates) ----
    models_dir = str(Path(output_dir) / "models")
    os.makedirs(models_dir, exist_ok=True)

    # Track best embedder + coords across all trials (avoid re-fit at the end)
    _w = weights or [0.5, 0.5]
    def _score(s):
        """Compute scalar score consistent with BO objective."""
        return _w[0] * s["scib_overall"] + _w[1] * s["scgraph_score"]

    best_agent_embedder = None
    best_agent_coords = None
    best_agent_score = -float("inf")
    best_baseline_embedder = None
    best_baseline_coords = None
    best_baseline_score = -float("inf")

    initial_candidates = agent_bo.suggest_initial(n=n_initial)
    logger.info(f"Running {n_initial} Sobol initialization trials...")

    for i, params in enumerate(initial_candidates):
        trial_num = i + 1
        logger.info(f"  Init trial {trial_num}/{n_initial}: {params}")
        model_path = str(Path(models_dir) / f"trial_{trial_num:03d}.model")
        t_trial = time.time()
        try:
            result = _evaluate_trial(
                adata, params, method, embedding_key, label_key, batch_key,
                subsample_idx, embed_kwargs, knn_cache=knn_cache,
                save_path=model_path,
            )
            scores = {"scib_overall": result["scib_overall"], "scgraph_score": result["scgraph_score"]}
            trial_score = _score(scores)
            elapsed = time.time() - t_trial
            logger.info(
                f"  Init trial {trial_num}/{n_initial} done in {elapsed:.0f}s — "
                f"scIB={scores['scib_overall']:.4f} scGraph={scores['scgraph_score']:.4f} "
                f"obj={trial_score:.4f} best={max(best_agent_score, trial_score):.4f}"
            )
            if trial_score > best_agent_score:
                best_agent_score = trial_score
                best_agent_embedder = result["embedder"]
                best_agent_coords = result["coords"]
            if trial_score > best_baseline_score:
                best_baseline_score = trial_score
                best_baseline_embedder = result["embedder"]
                best_baseline_coords = result["coords"]
        except Exception as e:
            logger.warning(f"  Trial {trial_num} failed: {e} — skipping (not recorded in GP)")
            continue

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
                    agent_params = _clamp_to_bounds(agent_params, search_space)
                elif action.action == "inject" and action.params:
                    agent_params = _clamp_to_bounds(action.params, search_space)
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
        agent_model_path = str(Path(models_dir) / f"trial_{trial_num:03d}_agent.model")
        logger.info(f"  Agent arm: {agent_params}")
        t_trial = time.time()
        try:
            agent_result = _evaluate_trial(
                adata, agent_params, method, embedding_key, label_key, batch_key,
                subsample_idx, embed_kwargs, knn_cache=knn_cache,
                save_path=agent_model_path,
            )
            agent_scores = {"scib_overall": agent_result["scib_overall"], "scgraph_score": agent_result["scgraph_score"]}
            agent_trial_score = _score(agent_scores)
            if agent_trial_score > best_agent_score:
                best_agent_score = agent_trial_score
                best_agent_embedder = agent_result["embedder"]
                best_agent_coords = agent_result["coords"]
            agent_bo.observe(agent_params, agent_scores["scib_overall"], agent_scores["scgraph_score"])
            elapsed = time.time() - t_trial
            logger.info(
                f"  Agent {trial_num}/{n_trials} done in {elapsed:.0f}s — "
                f"scIB={agent_scores['scib_overall']:.4f} scGraph={agent_scores['scgraph_score']:.4f} "
                f"obj={agent_trial_score:.4f} best={best_agent_score:.4f}"
            )
        except Exception as e:
            logger.warning(f"  Agent trial failed: {e} — skipping")

        # --- Baseline arm ---
        baseline_model_path = str(Path(models_dir) / f"trial_{trial_num:03d}_baseline.model")
        try:
            baseline_params = baseline_bo.suggest()
        except Exception as e:
            logger.warning(f"Baseline BO suggest failed: {e}, using Sobol fallback")
            baseline_params = baseline_bo.suggest_initial(1)[0]

        logger.info(f"  Baseline arm: {baseline_params}")
        t_trial = time.time()
        try:
            baseline_result = _evaluate_trial(
                adata, baseline_params, method, embedding_key, label_key, batch_key,
                subsample_idx, embed_kwargs, knn_cache=knn_cache,
                save_path=baseline_model_path,
            )
            baseline_scores = {"scib_overall": baseline_result["scib_overall"], "scgraph_score": baseline_result["scgraph_score"]}
            baseline_trial_score = _score(baseline_scores)
            if baseline_trial_score > best_baseline_score:
                best_baseline_score = baseline_trial_score
                best_baseline_embedder = baseline_result["embedder"]
                best_baseline_coords = baseline_result["coords"]
            baseline_bo.observe(
                baseline_params, baseline_scores["scib_overall"], baseline_scores["scgraph_score"],
            )
            elapsed = time.time() - t_trial
            logger.info(
                f"  Baseline {trial_num}/{n_trials} done in {elapsed:.0f}s — "
                f"scIB={baseline_scores['scib_overall']:.4f} scGraph={baseline_scores['scgraph_score']:.4f} "
                f"obj={baseline_trial_score:.4f} best={best_baseline_score:.4f}"
            )
        except Exception as e:
            logger.warning(f"  Baseline trial failed: {e} — skipping")

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

    # ---- 7. Best model coords (tracked during trials, no re-fit needed) ----
    rng = np.random.RandomState(seed)
    if best_agent_coords is None:
        best_agent_coords = rng.randn(adata.n_obs, 2)
    if best_baseline_coords is None:
        best_baseline_coords = rng.randn(adata.n_obs, 2)

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
        "best_coords_agent": best_agent_coords,
        "best_coords_baseline": best_baseline_coords,
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
        "best_model": best_agent_embedder,  # has .transform(X_new) and .save(path)
        "trial_history": agent_history,
        "baseline_history": baseline_history,
        "pareto_front": pareto_front_result,
        "report_path": report_path,
        "models_dir": models_dir,
    }
    logger.info(f"Optimization complete. Report: {report_path}")
    logger.info(f"Best params: {result['best_params']}")
    logger.info(f"Best scores: {result['best_scores']}")
    return result
