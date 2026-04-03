# PALO ALTO Skill

Build an agentic Bayesian optimization workflow for single-cell embedding visualization tuning.

## Goal

Optimize visualization parameters (e.g. UMAP `min_dist`, `n_neighbors`, and t-SNE `perplexity`) against scIB/scGraph-style metrics, without requiring end users to install `scib` or `scgraph`.

## Scope

This skill should:
1. internalize metric computation code from scIB and scGraph,
2. load and validate AnnData inputs,
3. implement a Bayesian optimization loop,
4. generate a concise experiment report.

## Execution Plan

### Task 1 — Internalize metrics
Download the relevant scIB and scGraph metric computation code, extract only the needed functions, and vendor them into this skill’s internal codebase. Refactor as needed so the skill can compute metrics without asking the user to install `scib` or `scgraph`. Keep attribution and note any behavior changes.

### Task 2 — AnnData input
Implement robust AnnData loading and validation. Support `.h5ad` first. Validate required fields such as embeddings, batch labels, cell type labels, and any metadata needed by the selected metrics.

### Task 3 — Bayesian optimization
Implement a lightweight BO system for tuning visualization hyperparameters. Start with UMAP (`n_neighbors`, `min_dist`, optional `metric`), then optionally extend to t-SNE (`perplexity`, etc.). Support single-objective first, with future extension to multi-objective optimization.

### Task 4 — Report
Write a short final report summarizing:
- input dataset and embedding used,
- parameters searched,
- best trials,
- metric values,
- recommended visualization settings,
- important caveats about optimizing 2D projections.

## Principles

- Minimize external dependencies.
- Prefer reproducible, modular Python code.
- Keep metric logic transparent and auditable.
- Treat visualization tuning as separate from representation learning quality.
