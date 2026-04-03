"""HTML report generation with embedded plots."""

import base64
import io
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader

from paloalto.utils import get_logger

logger = get_logger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_report(results: Dict, output_path: str):
    """Generate a self-contained HTML report."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("report.html.j2")

    # Generate plots
    convergence_plot = _make_convergence_plot(
        results["agent_history"], results["baseline_history"], results.get("agent_log", [])
    )
    embedding_plot = _make_embedding_plot(
        results.get("best_coords_agent"),
        results.get("best_coords_baseline"),
        results.get("labels"),
        results.get("batches"),
    )

    html = template.render(
        **results,
        convergence_plot=convergence_plot,
        embedding_plot=embedding_plot,
        enumerate=enumerate,
    )

    Path(output_path).write_text(html)
    logger.info(f"Report saved to {output_path}")


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" />'


def _make_convergence_plot(agent_history, baseline_history, agent_log) -> str:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    def best_so_far(history):
        bsf = []
        best = -np.inf
        for t in history:
            s = t["scores"]["scib_overall"] + t["scores"]["scgraph_score"]
            best = max(best, s)
            bsf.append(best)
        return bsf

    agent_bsf = best_so_far(agent_history)
    baseline_bsf = best_so_far(baseline_history)
    trials = range(1, len(agent_bsf) + 1)

    ax.plot(trials, agent_bsf, "b-o", label="Agent-guided", markersize=4)
    ax.plot(trials, baseline_bsf, "r--s", label="Baseline (BoTorch)", markersize=4)

    # Mark agent interventions
    for entry in agent_log:
        t = entry.get("trial_number", 0)
        if 1 <= t <= len(agent_bsf):
            marker = {"modify": "^", "inject": "D", "prune": "v", "stop": "x"}.get(entry["action"])
            if marker:
                ax.plot(t, agent_bsf[t - 1], marker, color="green", markersize=10, zorder=5)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Score (scIB + scGraph)")
    ax.set_title("Convergence: Agent-guided vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _make_embedding_plot(coords_agent, coords_baseline, labels, batches) -> str:
    if coords_agent is None:
        return "<p>No embedding data available.</p>"

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    def scatter(ax, coords, colors, title):
        unique = np.unique(colors)
        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(unique))
        palette = {u: cmap(i) for i, u in enumerate(unique)}
        for u in unique:
            mask = colors == u
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[palette[u]], label=u, s=3, alpha=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    scatter(axes[0, 0], coords_agent, labels, "Agent Best — Cell Type")
    scatter(axes[0, 1], coords_agent, batches, "Agent Best — Batch")
    scatter(axes[1, 0], coords_baseline, labels, "Baseline Best — Cell Type")
    scatter(axes[1, 1], coords_baseline, batches, "Baseline Best — Batch")

    for ax in axes.flat:
        ax.legend(markerscale=3, fontsize=7, loc="best")

    fig.tight_layout()
    return _fig_to_base64(fig)
