"""
Full retrieve -> rerank -> generate -> cite pipeline.

Every answer is generated strictly from retrieved chunks. Every chunk used
carries its source PDF filename and page number into the final result, so
"page 7 of handbook.pdf" is always available alongside the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rag_platform.llm import OllamaClient
from rag_platform.reranker import CrossEncoderReranker
from rag_platform.retriever import HybridRetriever

SYSTEM_PROMPT = (
    "You are an assistant that answers questions using ONLY the provided context. "
    "If the context does not contain the answer, say so plainly instead of guessing. "
    "Cite sources inline using [n] markers that map to the numbered context blocks."
)


@dataclass
class Citation:
    marker: int
    source_path: str
    page_number: Optional[int]
    chunk_text: str
    score: float


@dataclass
class QAResult:
    question: str
    answer: str
    citations: list[Citation]


class QAPipeline:
    """
    Args:
        retriever: This tenant's HybridRetriever (already scoped to the
            correct company's vector store — see scripts/query_company.py).
        reranker: Optional; pass None to skip reranking (faster, slightly
            lower precision — controlled by Settings.use_reranker).
        llm_client: LLM used for final answer generation.
        rerank_top_k: How many reranked chunks to feed the LLM.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker],
        llm_client: OllamaClient,
        rerank_top_k: int = 5,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.rerank_top_k = rerank_top_k

    async def answer(self, question: str, where: Optional[dict[str, Any]] = None) -> QAResult:
        candidates = self.retriever.search(question, where=where)

        if self.reranker is not None:
            ranked = self.reranker.rerank(question, candidates, top_k=self.rerank_top_k)
        else:
            ranked = candidates[: self.rerank_top_k]

        if not ranked:
            return QAResult(
                question=question,
                answer="I couldn't find any relevant indexed content to answer this question.",
                citations=[],
            )

        context_blocks = []
        citations: list[Citation] = []
        for i, hit in enumerate(ranked, start=1):
            meta = hit.get("metadata", {})
            source = meta.get("source_path", "unknown")
            page = meta.get("page_number")
            page_label = f", page {page}" if page is not None else ""
            context_blocks.append(f"[{i}] (source: {source}{page_label})\n{hit['document']}")
            citations.append(
                Citation(
                    marker=i,
                    source_path=source,
                    page_number=page,
                    chunk_text=hit["document"],
                    score=hit.get("rerank_score", hit.get("rrf_score", 0.0)),
                )
            )

        prompt = (
            f"Context:\n\n{chr(10).join(context_blocks)}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above, with [n] citation markers."
        )

        answer_text = await self.llm_client.generate(prompt, system=SYSTEM_PROMPT)
        return QAResult(question=question, answer=answer_text, citations=citations)
