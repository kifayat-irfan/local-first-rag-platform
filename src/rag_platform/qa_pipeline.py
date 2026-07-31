"""
Full retrieve -> rerank -> generate -> cite pipeline.

Every answer is generated strictly from retrieved chunks. Every chunk used
carries its source PDF filename and page number into the final result, so
"page 7 of handbook.pdf" is always available alongside the answer.

Two safeguards against a specific failure mode: an LLM connecting two
genuinely unrelated retrieved chunks into a false causal story (e.g.
mixing "attendance ownership" and "grade-editing permissions" — two
different features — because both chunks happened to be in the same
prompt). The system prompt explicitly forbids inferring relationships
that aren't stated within a single context block, and low-relevance
chunks (reranker score below min_relevance_score) are dropped before they
ever reach the LLM, rather than trusting the model to ignore them itself.
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
    "Each numbered context block [n] is an independent excerpt — treat blocks as "
    "unrelated to each other unless a single block explicitly connects them. "
    "Never infer a cause, relationship, or workaround that isn't directly stated in "
    "the context, even if it seems like an obvious or reasonable conclusion — if the "
    "steps or explanation aren't written in the context, say the documentation doesn't "
    "cover it rather than reasoning your way to an answer. "
    "Cite sources inline using [n] markers that map to the numbered context blocks."
)

DEFAULT_MIN_RELEVANCE_SCORE = -2.0  # cross-encoder scores below this are treated as noise, not context


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
        min_relevance_score: Chunks with a reranker score below this are
            dropped before generation, even if that leaves fewer than
            rerank_top_k chunks — feeding the LLM a chunk the reranker
            itself considers irrelevant is how unrelated topics get
            blended into one hallucinated answer. Only applied when a
            reranker is in use, since raw RRF scores aren't on a
            comparable scale. Set to None to disable this filter.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker],
        llm_client: OllamaClient,
        rerank_top_k: int = 5,
        min_relevance_score: Optional[float] = DEFAULT_MIN_RELEVANCE_SCORE,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.rerank_top_k = rerank_top_k
        self.min_relevance_score = min_relevance_score

    async def answer(self, question: str, where: Optional[dict[str, Any]] = None) -> QAResult:
        candidates = self.retriever.search(question, where=where)

        if self.reranker is not None:
            ranked = self.reranker.rerank(question, candidates, top_k=self.rerank_top_k)
            if self.min_relevance_score is not None:
                ranked = [r for r in ranked if r.get("rerank_score", 0.0) >= self.min_relevance_score]
        else:
            ranked = candidates[: self.rerank_top_k]

        if not ranked:
            return QAResult(
                question=question,
                answer="I couldn't find any sufficiently relevant indexed content to answer this question.",
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