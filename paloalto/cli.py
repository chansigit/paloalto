"""PaloAlto CLI."""

import json
import click

from paloalto.utils import get_logger

logger = get_logger(__name__)


@click.group()
def main():
    """PaloAlto: Pareto-guided Automatic Layout Optimization for Aligning Latent Topology in Omics embeddings."""
    pass


@main.command()
@click.option("--input", "input_path", required=True, help="Path to .h5ad file")
@click.option("--embedding", default="X_pca", help="Key in adata.obsm")
@click.option("--batch-key", default="batch", help="Batch column in adata.obs")
@click.option("--label-key", default="cell_type", help="Cell type column in adata.obs")
@click.option("--method", default="umap", type=click.Choice(["umap", "numap", "tsne"]))
@click.option("--bo-mode", default="weighted", type=click.Choice(["weighted", "pareto"]))
@click.option("--n-trials", default=30, type=int)
@click.option("--n-initial", default=5, type=int)
@click.option("--use-agent/--no-agent", default=True)
@click.option("--agent-review-interval", default=1, type=int)
@click.option("--subsample-n", default=10000, type=int)
@click.option("--output-dir", default="./results")
@click.option("--seed", default=42, type=int)
@click.option("--config", default=None, help="YAML config file")
@click.option("--embed-epochs", default=None, type=int, help="Override embedding training epochs")
def run(input_path, embedding, batch_key, label_key, method, bo_mode,
        n_trials, n_initial, use_agent, agent_review_interval,
        subsample_n, output_dir, seed, config, embed_epochs):
    """Run PaloAlto optimization."""
    if config:
        import yaml
        with open(config) as f:
            cfg = yaml.safe_load(f)
        input_path = cfg.get("input", input_path)
        embedding = cfg.get("embedding_key", embedding)
        batch_key = cfg.get("batch_key", batch_key)
        label_key = cfg.get("label_key", label_key)
        method = cfg.get("method", method)
        bo_mode = cfg.get("bo_mode", bo_mode)
        n_trials = cfg.get("n_trials", n_trials)
        n_initial = cfg.get("n_initial", n_initial)
        use_agent = cfg.get("use_agent", use_agent)
        agent_review_interval = cfg.get("agent_review_interval", agent_review_interval)
        subsample_n = cfg.get("subsample_n", subsample_n)
        seed = cfg.get("seed", seed)
        if embed_epochs is None:
            embed_epochs = cfg.get("embed_epochs")

    embed_kwargs = {}
    if embed_epochs is not None:
        embed_kwargs["epochs"] = embed_epochs

    from paloalto import optimize
    result = optimize(
        adata=input_path,
        embedding_key=embedding,
        batch_key=batch_key,
        label_key=label_key,
        method=method,
        bo_mode=bo_mode,
        n_trials=n_trials,
        n_initial=n_initial,
        use_agent=use_agent,
        agent_review_interval=agent_review_interval,
        subsample_n=subsample_n,
        output_dir=output_dir,
        seed=seed,
        embed_kwargs=embed_kwargs,
    )

    click.echo(f"\nBest parameters: {json.dumps(result['best_params'], indent=2)}")
    click.echo(f"Best scores: {json.dumps(result['best_scores'], indent=2)}")
    click.echo(f"Report: {result['report_path']}")


@main.command()
@click.option("--input", "input_path", required=True, help="Path to .h5ad file")
@click.option("--embedding", default="X_umap", help="Key in adata.obsm")
@click.option("--batch-key", default="batch")
@click.option("--label-key", default="cell_type")
def metrics(input_path, embedding, batch_key, label_key):
    """Compute metrics on an existing embedding."""
    from paloalto.data import load_and_validate
    from paloalto.metrics import compute_all

    adata = load_and_validate(input_path, embedding, batch_key, label_key)
    scores = compute_all(adata, embedding, label_key, batch_key)

    click.echo("\nMetric Results:")
    click.echo("-" * 40)
    for k, v in scores.items():
        click.echo(f"  {k:30s} {v:.4f}")
