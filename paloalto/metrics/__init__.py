"""Metrics aggregation: scIB overall + scGraph."""

from typing import Dict
import numpy as np

from paloalto.metrics.bio import cell_type_asw, nmi, ari, clisi
from paloalto.metrics.batch import batch_asw, ilisi, graph_connectivity
from paloalto.metrics.scgraph import corr_weighted


def scib_overall(bio: Dict[str, float], batch: Dict[str, float]) -> float:
    """Compute scIB overall score = 0.6 * Bio + 0.4 * Batch."""
    bio_score = float(np.mean(list(bio.values())))
    batch_score = float(np.mean(list(batch.values())))
    return 0.6 * bio_score + 0.4 * batch_score


def compute_all(
    adata,
    embed_key: str,
    label_key: str,
    batch_key: str,
) -> Dict[str, float]:
    """Compute all metrics and return a flat dict."""
    coords = adata.obsm[embed_key]
    labels = adata.obs[label_key].values
    batches = adata.obs[batch_key].values

    bio = {
        "cell_type_asw": cell_type_asw(coords, labels),
        "nmi": nmi(coords, labels),
        "ari": ari(coords, labels),
        "clisi": clisi(coords, labels),
    }
    batch_scores = {
        "batch_asw": batch_asw(coords, batches),
        "ilisi": ilisi(coords, batches),
        "graph_connectivity": graph_connectivity(coords, labels),
    }
    scgraph_score = corr_weighted(adata, embed_key, label_key, batch_key)

    bio_mean = float(np.mean(list(bio.values())))
    batch_mean = float(np.mean(list(batch_scores.values())))
    overall = scib_overall(bio, batch_scores)

    return {
        **bio,
        **batch_scores,
        "scgraph_corr_weighted": scgraph_score,
        "bio_score": bio_mean,
        "batch_score": batch_mean,
        "scib_overall": overall,
    }
