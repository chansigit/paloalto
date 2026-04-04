"""Embedding methods."""

from paloalto.embed.pumap import NUMAPEmbedder


def get_embedder(method: str, **kwargs):
    """Factory for embedding methods."""
    if method == "numap":
        return NUMAPEmbedder(**kwargs)
    elif method == "umap":
        from paloalto.embed.umap import UMAPEmbedder
        return UMAPEmbedder(**kwargs)
    elif method == "tsne":
        from paloalto.embed.ptsne import ParametricTSNE
        return ParametricTSNE(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'numap', 'umap', or 'tsne'.")
