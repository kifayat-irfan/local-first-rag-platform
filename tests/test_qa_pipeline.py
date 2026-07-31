"""
Tests for qa_pipeline.py's relevance-score filter — the safeguard against
an LLM blending two genuinely unrelated retrieved chunks into one
hallucinated answer (see min_relevance_score in QAPipeline).
"""

from __future__ import annotations

import pytest

from rag_platform.qa_pipeline import QAPipeline


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, question, where=None):
        return self.hits


class FakeReranker:
    """Assigns a high score to chunks mentioning 'marks', a strongly negative score to everything else."""

    def rerank(self, question, candidates, top_k=5):
        scored = []
        for c in candidates:
            score = 3.5 if "marks" in c["document"].lower() else -8.2
            scored.append({**c, "rerank_score": score})
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


class FakeLLM:
    def __init__(self):
        self.last_prompt = None
        self.last_system = None

    async def generate(self, prompt, system=None, temperature=0.2):
        self.last_prompt = prompt
        self.last_system = system
        return "fake answer"


RELEVANT_CHUNK = {
    "id": "1",
    "document": "Only the assigned teacher can edit marks for their own section.",
    "metadata": {"source_path": "gradebook.pdf", "page_number": 3},
}
UNRELATED_CHUNK = {
    "id": "2",
    "document": "Attendance ownership determines who can mark attendance for a class.",
    "metadata": {"source_path": "attendance.pdf", "page_number": 1},
}


@pytest.mark.asyncio
async def test_low_relevance_chunk_excluded_from_llm_prompt():
    """Reproduces the reported hallucination: an unrelated 'attendance' chunk
    must never reach the LLM for a 'marks editing' question, even if the
    retriever surfaced it."""
    llm = FakeLLM()
    qa = QAPipeline(
        retriever=FakeRetriever([RELEVANT_CHUNK, UNRELATED_CHUNK]),
        reranker=FakeReranker(),
        llm_client=llm,
        rerank_top_k=5,
    )

    result = await qa.answer("Why can't this teacher edit marks anymore?")

    assert len(result.citations) == 1
    assert result.citations[0].source_path == "gradebook.pdf"
    assert "attendance.pdf" not in llm.last_prompt
    assert "attendance" not in llm.last_prompt.lower()


@pytest.mark.asyncio
async def test_all_chunks_kept_when_all_relevant():
    llm = FakeLLM()
    two_relevant = [RELEVANT_CHUNK, {**RELEVANT_CHUNK, "id": "3", "document": "marks can be edited before the term closes."}]
    qa = QAPipeline(retriever=FakeRetriever(two_relevant), reranker=FakeReranker(), llm_client=llm, rerank_top_k=5)

    result = await qa.answer("When can marks be edited?")

    assert len(result.citations) == 2


@pytest.mark.asyncio
async def test_no_chunks_survive_filter_returns_graceful_message():
    llm = FakeLLM()
    qa = QAPipeline(retriever=FakeRetriever([UNRELATED_CHUNK]), reranker=FakeReranker(), llm_client=llm, rerank_top_k=5)

    result = await qa.answer("Why can't this teacher edit marks anymore?")

    assert result.citations == []
    assert "couldn't find" in result.answer.lower()


@pytest.mark.asyncio
async def test_relevance_filter_disabled_when_none():
    llm = FakeLLM()
    qa = QAPipeline(
        retriever=FakeRetriever([RELEVANT_CHUNK, UNRELATED_CHUNK]),
        reranker=FakeReranker(),
        llm_client=llm,
        rerank_top_k=5,
        min_relevance_score=None,
    )

    result = await qa.answer("Why can't this teacher edit marks anymore?")

    assert len(result.citations) == 2  # filter disabled, both chunks pass through


@pytest.mark.asyncio
async def test_system_prompt_forbids_cross_chunk_inference():
    """Guards against silently weakening the anti-hallucination instruction later."""
    llm = FakeLLM()
    qa = QAPipeline(retriever=FakeRetriever([RELEVANT_CHUNK]), reranker=None, llm_client=llm, rerank_top_k=5)

    await qa.answer("test question")

    assert "unrelated to each other" in llm.last_system
    assert "Never infer" in llm.last_system