import numpy as np
import pytest
import torch

from paloalto.embed.ptsne import ParametricTSNE


class TestParametricTSNE:
    def test_fit_returns_2d(self, mock_adata):
        model = ParametricTSNE(perplexity=15, epochs=5, batch_size=64)
        coords = model.fit(mock_adata, embedding_key="X_pca")
        assert coords.shape == (400, 2)

    def test_transform(self, mock_adata):
        model = ParametricTSNE(perplexity=15, epochs=5, batch_size=64)
        model.fit(mock_adata, embedding_key="X_pca")
        new_coords = model.transform(mock_adata.obsm["X_pca"][:10])
        assert new_coords.shape == (10, 2)

    def test_multiscale_mode(self, mock_adata):
        model = ParametricTSNE(multiscale=True, epochs=5, batch_size=64)
        coords = model.fit(mock_adata, embedding_key="X_pca")
        assert coords.shape == (400, 2)

    def test_save_load(self, mock_adata, tmp_path):
        model = ParametricTSNE(perplexity=15, epochs=5, batch_size=64)
        model.fit(mock_adata, embedding_key="X_pca")
        path = str(tmp_path / "tsne_model.pt")
        model.save(path)

        model2 = ParametricTSNE.load(path)
        c1 = model.transform(mock_adata.obsm["X_pca"][:5])
        c2 = model2.transform(mock_adata.obsm["X_pca"][:5])
        np.testing.assert_allclose(c1, c2, atol=1e-5)

    def test_dof_parameter(self, mock_adata):
        model = ParametricTSNE(perplexity=15, dof=0.5, epochs=5, batch_size=64)
        coords = model.fit(mock_adata, embedding_key="X_pca")
        assert coords.shape == (400, 2)

    def test_get_params(self):
        model = ParametricTSNE(perplexity=30, dof=1.5, metric="cosine")
        params = model.get_params()
        assert params["perplexity"] == 30
        assert params["dof"] == 1.5
        assert params["metric"] == "cosine"
