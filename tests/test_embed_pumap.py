import numpy as np
import pytest
import torch

from paloalto.embed.pumap import NUMAPEmbedder


class TestNUMAPEmbedder:
    def test_fit_returns_2d(self, mock_adata):
        emb = NUMAPEmbedder(n_neighbors=10, min_dist=0.1, epochs=2)
        coords = emb.fit(mock_adata, embedding_key="X_pca")
        assert coords.shape == (400, 2)

    def test_transform(self, mock_adata):
        emb = NUMAPEmbedder(n_neighbors=10, min_dist=0.1, epochs=2)
        emb.fit(mock_adata, embedding_key="X_pca")
        new_coords = emb.transform(mock_adata.obsm["X_pca"][:10])
        assert new_coords.shape == (10, 2)

    def test_save_load(self, mock_adata, tmp_path):
        emb = NUMAPEmbedder(n_neighbors=10, min_dist=0.1, epochs=2)
        emb.fit(mock_adata, embedding_key="X_pca")
        path = str(tmp_path / "model.pt")
        emb.save(path)

        emb2 = NUMAPEmbedder.load(path)
        c1 = emb.transform(mock_adata.obsm["X_pca"][:5])
        c2 = emb2.transform(mock_adata.obsm["X_pca"][:5])
        np.testing.assert_allclose(c1, c2, atol=1e-5)

    def test_get_params(self):
        emb = NUMAPEmbedder(n_neighbors=15, min_dist=0.3, metric="cosine")
        params = emb.get_params()
        assert params["n_neighbors"] == 15
        assert params["min_dist"] == 0.3
        assert params["metric"] == "cosine"
