"""AnnData loading, validation, and subsampling."""

from pathlib import Path
from typing import Union

import anndata as ad
import numpy as np
import pandas as pd

from paloalto.utils import get_logger

logger = get_logger(__name__)


def load_and_validate(
    adata: Union[str, Path, ad.AnnData],
    embedding_key: str,
    batch_key: str,
    label_key: str,
) -> ad.AnnData:
    if not isinstance(adata, ad.AnnData):
        logger.info(f"Loading {adata}")
        adata = ad.read_h5ad(adata)

    if embedding_key not in adata.obsm:
        raise ValueError(
            f"embedding_key '{embedding_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    if batch_key not in adata.obs.columns:
        raise ValueError(
            f"batch_key '{batch_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if label_key not in adata.obs.columns:
        raise ValueError(
            f"label_key '{label_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    n_batches = adata.obs[batch_key].nunique()
    if n_batches < 2:
        raise ValueError(f"Need at least 2 batches, found {n_batches}")

    n_types = adata.obs[label_key].nunique()
    if n_types < 2:
        raise ValueError(f"Need at least 2 cell types, found {n_types}")

    if adata.n_obs < 500:
        logger.warning(f"Only {adata.n_obs} cells — metrics may be unreliable")

    logger.info(
        f"Validated: {adata.n_obs} cells, {n_batches} batches, "
        f"{n_types} cell types, embedding '{embedding_key}' dim={adata.obsm[embedding_key].shape[1]}"
    )
    return adata


def subsample_stratified(
    adata: ad.AnnData,
    label_key: str,
    batch_key: str,
    n: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    if adata.n_obs <= n:
        return np.arange(adata.n_obs)

    rng = np.random.RandomState(seed)
    groups = adata.obs.groupby([label_key, batch_key], observed=True)
    group_sizes = groups.size()
    total = group_sizes.sum()

    indices = []
    for (label, batch), size in group_sizes.items():
        group_idx = groups.groups[(label, batch)]
        k = max(1, round(size / total * n))
        k = min(k, len(group_idx))
        chosen = rng.choice(group_idx, size=k, replace=False)
        indices.extend(chosen)

    indices = np.array(indices)
    # Trim or pad to exact n
    if len(indices) > n:
        indices = rng.choice(indices, size=n, replace=False)
    indices.sort()
    return indices
