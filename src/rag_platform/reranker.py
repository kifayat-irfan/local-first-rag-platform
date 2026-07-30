"""
Cross-encoder re-ranking of the hybrid retriever's candidate set.

Hybrid search (dense + BM25) is fast but its scores are still a proxy for
relevance. A cross-encoder scores (query, chunk) pairs jointly — slower,
so it only ever runs on the small candidate set hybrid search already
narrowed down, not the whole corpus.
"""

from __future__ import annotations

from typing import Any, Optional


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: Optional[int] = None
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, c["document"]) for c in candidates]
        scores = self.model.predict(pairs)

        reranked = [dict(c, rerank_score=float(s)) for c, s in zip(candidates, scores)]
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k] if top_k else reranked
