"""Bio-conservation metrics: Cell type ASW, NMI, ARI, cLISI.

Internalized from scIB. Computes bio-conservation scores that quantify how
well a 2D embedding preserves known cell type structure.

Attribution: Luecken et al. (2022) "Benchmarking atlas-level data integration
in single-cell genomics", Nature Methods. Original scIB code at
https://github.com/theislab/scib under BSD-3-Clause.

Behavior changes vs. scIB reference:
- cell_type_asw rescales silhouette from [-1, 1] to [0, 1].
- nmi and ari cluster via Leiden on a kNN graph built from 2D coords rather
  than from a pre-computed high-dimensional neighbors graph.
- clisi normalizes raw LISI to [0, 1] using the formula
  (n_types - median_lisi) / (n_types - 1).
"""

import numpy as np
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

from paloalto.metrics.lisi import compute_lisi


def cell_type_asw(coords: np.ndarray, labels: np.ndarray) -> float:
    """Average silhouette width on cell type labels, rescaled to [0, 1].

    Args:
        coords: (n_cells, n_dims) embedding coordinates.
        labels: (n_cells,) cell type labels.

    Returns:
        ASW score in [0, 1]. 1 = perfectly separated clusters.
    """
    asw = silhouette_score(coords, labels)
    return float((asw + 1) / 2)  # from [-1, 1] to [0, 1]


def nmi(coords: np.ndarray, labels: np.ndarray, resolution: float = 1.0) -> float:
    """NMI between Leiden clusters on 2D kNN graph and ground truth labels.

    Args:
        coords: (n_cells, n_dims) embedding coordinates.
        labels: (n_cells,) ground truth cell type labels.
        resolution: Leiden clustering resolution.

    Returns:
        NMI score in [0, 1]. 1 = perfect agreement.
    """
    clusters = _leiden_on_coords(coords, resolution=resolution)
    return float(normalized_mutual_info_score(labels, clusters))


def ari(coords: np.ndarray, labels: np.ndarray, resolution: float = 1.0) -> float:
    """ARI between Leiden clusters on 2D kNN graph and ground truth labels.

    Args:
        coords: (n_cells, n_dims) embedding coordinates.
        labels: (n_cells,) ground truth cell type labels.
        resolution: Leiden clustering resolution.

    Returns:
        ARI score in [0, 1]. 1 = perfect agreement.
    """
    clusters = _leiden_on_coords(coords, resolution=resolution)
    raw = float(adjusted_rand_score(labels, clusters))
    # ARI can be negative; clip to [0, 1] for consistent downstream use
    return max(0.0, raw)


def clisi(coords: np.ndarray, labels: np.ndarray, perplexity: float = 30.0) -> float:
    """Cell-type LISI: normalized so that 1 = perfect purity, 0 = fully mixed.

    Raw cLISI is in [1, n_types]. We normalize:
        score = (n_types - median_lisi) / (n_types - 1)

    Args:
        coords: (n_cells, n_dims) embedding coordinates.
        labels: (n_cells,) cell type labels.
        perplexity: Perplexity for kNN kernel bandwidth in LISI computation.

    Returns:
        Normalized cLISI in [0, 1]. 1 = each cell's neighborhood is
        dominated by a single cell type (high purity).
    """
    lisi_scores = compute_lisi(coords, labels, perplexity=perplexity)
    n_types = len(np.unique(labels))
    if n_types == 1:
        # Only one type — trivially pure
        return 1.0
    median_lisi = np.median(lisi_scores)
    score = (n_types - median_lisi) / (n_types - 1)
    return float(np.clip(score, 0.0, 1.0))


def _leiden_on_coords(
    coords: np.ndarray, n_neighbors: int = 15, resolution: float = 1.0
) -> np.ndarray:
    """Run Leiden clustering on a kNN graph built from 2D coords.

    Args:
        coords: (n_cells, n_dims) embedding coordinates.
        n_neighbors: Number of neighbors for kNN graph construction.
        resolution: Leiden clustering resolution.

    Returns:
        (n_cells,) array of cluster label strings.
    """
    import scanpy as sc
    import anndata as ad

    tmp = ad.AnnData(obsm={"X_embed": coords})
    sc.pp.neighbors(tmp, use_rep="X_embed", n_neighbors=n_neighbors)
    sc.tl.leiden(tmp, resolution=resolution, flavor="igraph", n_iterations=2)
    return tmp.obs["leiden"].values
