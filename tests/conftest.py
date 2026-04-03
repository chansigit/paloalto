import numpy as np
import pytest
import anndata as ad
import pandas as pd


@pytest.fixture
def mock_adata():
    """Small AnnData with PCA embedding, batch, and cell_type labels."""
    np.random.seed(42)
    n_cells = 400
    n_genes = 100
    n_pcs = 20
    n_batches = 2
    n_types = 4

    X = np.random.randn(n_cells, n_genes).astype(np.float32)
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(
                [f"batch_{i % n_batches}" for i in range(n_cells)]
            ),
            "cell_type": pd.Categorical(
                [f"type_{i % n_types}" for i in range(n_cells)]
            ),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    # Create clustered PCA so metrics are non-trivial
    pca = np.random.randn(n_cells, n_pcs).astype(np.float32)
    for t in range(n_types):
        mask = obs["cell_type"] == f"type_{t}"
        pca[mask] += np.random.randn(n_pcs) * 3

    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["X_pca"] = pca
    # Pre-compute a mock 2D embedding for metric tests
    adata.obsm["X_umap_mock"] = pca[:, :2]
    return adata


@pytest.fixture
def mock_adata_path(mock_adata, tmp_path):
    """Write mock AnnData to disk, return path."""
    path = tmp_path / "test.h5ad"
    mock_adata.write_h5ad(path)
    return str(path)
