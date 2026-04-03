import pytest
import numpy as np
import anndata as ad
import pandas as pd

from paloalto.data import load_and_validate, subsample_stratified


class TestLoadAndValidate:
    def test_loads_from_path(self, mock_adata_path):
        adata = load_and_validate(
            mock_adata_path, embedding_key="X_pca", batch_key="batch", label_key="cell_type"
        )
        assert adata.n_obs == 400
        assert "X_pca" in adata.obsm

    def test_accepts_anndata_object(self, mock_adata):
        adata = load_and_validate(
            mock_adata, embedding_key="X_pca", batch_key="batch", label_key="cell_type"
        )
        assert adata is mock_adata

    def test_rejects_missing_embedding(self, mock_adata):
        with pytest.raises(ValueError, match="embedding_key"):
            load_and_validate(mock_adata, "X_nonexistent", "batch", "cell_type")

    def test_rejects_missing_batch_key(self, mock_adata):
        with pytest.raises(ValueError, match="batch_key"):
            load_and_validate(mock_adata, "X_pca", "nonexistent", "cell_type")

    def test_rejects_missing_label_key(self, mock_adata):
        with pytest.raises(ValueError, match="label_key"):
            load_and_validate(mock_adata, "X_pca", "batch", "nonexistent")

    def test_rejects_single_batch(self, mock_adata):
        mock_adata.obs["batch"] = "batch_0"
        with pytest.raises(ValueError, match="at least 2"):
            load_and_validate(mock_adata, "X_pca", "batch", "cell_type")

    def test_rejects_single_cell_type(self, mock_adata):
        mock_adata.obs["cell_type"] = "type_0"
        with pytest.raises(ValueError, match="at least 2"):
            load_and_validate(mock_adata, "X_pca", "batch", "cell_type")


class TestSubsampleStratified:
    def test_returns_original_if_small(self, mock_adata):
        indices = subsample_stratified(mock_adata, "cell_type", "batch", n=10000, seed=42)
        assert len(indices) == 400  # all cells kept

    def test_subsamples_large_data(self, mock_adata):
        indices = subsample_stratified(mock_adata, "cell_type", "batch", n=100, seed=42)
        assert len(indices) == 100

    def test_preserves_proportions(self, mock_adata):
        indices = subsample_stratified(mock_adata, "cell_type", "batch", n=200, seed=42)
        sub = mock_adata[indices]
        orig_props = mock_adata.obs["cell_type"].value_counts(normalize=True).sort_index()
        sub_props = sub.obs["cell_type"].value_counts(normalize=True).sort_index()
        np.testing.assert_allclose(orig_props.values, sub_props.values, atol=0.1)

    def test_deterministic(self, mock_adata):
        i1 = subsample_stratified(mock_adata, "cell_type", "batch", n=100, seed=42)
        i2 = subsample_stratified(mock_adata, "cell_type", "batch", n=100, seed=42)
        np.testing.assert_array_equal(i1, i2)
