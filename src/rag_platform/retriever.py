"""
Hybrid retrieval: dense vector search (semantic) + BM25 (exact keyword
match), fused with Reciprocal Rank Fusion (RRF).

RRF, not a raw score blend, because cosine similarity and BM25 scores live
on incompatible scales — RRF fuses on rank position, sidestepping that.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from rag_platform.embedder import Embedder
from rag_platform.vector_store import VectorStore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """In-memory BM25 index, rebuilt from whatever's currently in the vector store."""

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._meta: list[dict[str, Any]] = []

    def build(self, documents: list[dict[str, Any]]) -> None:
        self._ids = [d["id"] for d in documents]
        self._docs = [d["document"] for d in documents]
        self._meta = [d.get("metadata", {}) for d in documents]
        tokenized = [_tokenize(doc) for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def is_built(self) -> bool:
        return self._bm25 is not None

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        if self._bm25 is None or not self._docs:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"id": self._ids[i], "document": self._docs[i], "metadata": self._meta[i], "score": float(scores[i])}
            for i in ranked
            if scores[i] > 0
        ]


class HybridRetriever:
    """
    Args:
        embedder: Query/document embedder for the dense side.
        vector_store: This tenant's isolated ChromaDB wrapper.
        top_k_dense / top_k_bm25: Candidates pulled from each side before fusion.
        top_k_final: Candidates returned after RRF fusion.
        rrf_k: RRF constant (higher = flatter weighting of rank position; 60 is the common default).
        use_bm25: If False, skip BM25 entirely and return dense-only results (still RRF-ranked for consistency).
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k_dense: int = 20,
        top_k_bm25: int = 20,
        top_k_final: int = 5,
        rrf_k: int = 60,
        use_bm25: bool = True,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = BM25Index()
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.top_k_final = top_k_final
        self.rrf_k = rrf_k
        self.use_bm25 = use_bm25

    def refresh_bm25(self) -> None:
        self.bm25_index.build(self.vector_store.get_all_documents())

    def search(self, query: str, where: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)
        dense_hits = self.vector_store.query(query_vector, top_k=self.top_k_dense, where=where)

        bm25_hits: list[dict[str, Any]] = []
        if self.use_bm25:
            if not self.bm25_index.is_built():
                self.refresh_bm25()
            bm25_hits = self.bm25_index.search(query, top_k=self.top_k_bm25)

        fused = self._reciprocal_rank_fusion(dense_hits, bm25_hits)
        return fused[: self.top_k_final]

    def _reciprocal_rank_fusion(
        self, dense_hits: list[dict[str, Any]], bm25_hits: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        payload: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (self.rrf_k + rank + 1)
            payload[hit["id"]] = hit

        for rank, hit in enumerate(bm25_hits):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (self.rrf_k + rank + 1)
            payload.setdefault(hit["id"], hit)

        ranked_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked_ids:
            item = dict(payload[i])
            item["rrf_score"] = scores[i]
            results.append(item)
        return results
