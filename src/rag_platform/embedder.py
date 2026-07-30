"""
Local embedding model wrapper (BAAI/bge-small-en-v1.5 by default).

Loaded lazily so constructing an Embedder — e.g. in a test that never calls
.embed_documents() — doesn't pay the cost of loading transformer weights.
"""

from __future__ import annotations

from typing import Optional

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """
    Args:
        model_name: HuggingFace model id. BGE models require an instruction
            prefix on the *query* side only (not documents) — baked into
            embed_query() below, since forgetting this measurably hurts
            retrieval quality and is an easy thing to omit by hand.
        batch_size: Batch size for embed_documents().
        device: "cpu" or "cuda". Multi-tenant deployments on modest hardware
            should stick to "cpu" unless a GPU is guaranteed available.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32, device: str = "cpu"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document/passage chunks (no instruction prefix)."""
        if not texts:
            return []
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query, with the BGE instruction prefix applied automatically."""
        prefixed = BGE_QUERY_PREFIX + query if "bge" in self.model_name.lower() else query
        vector = self.model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)[0]
        return vector.tolist()
