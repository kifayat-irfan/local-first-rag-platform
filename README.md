"""
Tests for qa_pipeline.py's relative score-gap filter — the safeguard against
an LLM blending two genuinely unrelated retrieved chunks into one
hallucinated answer (see max_score_gap in QAPipeline).

The filter is relative (gap from this query's own top-ranked chunk), not an
absolute score floor, because real testing showed cross-encoder rerank
scores aren't on a fixed scale across queries — a genuinely relevant chunk
scored -7.63 for one question in production, while an irrelevant chunk
scored -8.2 for a different question. An absolute cutoff can't handle both.
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
    """Returns whatever score each chunk id was assigned, sorted descending."""

    def __init__(self, scores_by_id):
        self.scores_by_id = scores_by_id

    def rerank(self, question, candidates, top_k=5):
        scored = [{**c, "rerank_score": self.scores_by_id[c["id"]]} for c in candidates]
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
async def test_large_gap_excludes_unrelated_chunk():
    """Reproduces the original reported hallucination: top score 3.5 vs an
    unrelated chunk at -8.2 (an 11.7pt gap) must be excluded."""
    llm = FakeLLM()
    qa = QAPipeline(
        retriever=FakeRetriever([RELEVANT_CHUNK, UNRELATED_CHUNK]),
        reranker=FakeReranker({"1": 3.5, "2": -8.2}),
        llm_client=llm,
        rerank_top_k=5,
    )

    result = await qa.answer("Why can't this teacher edit marks anymore?")

    assert len(result.citations) == 1
    assert result.citations[0].source_path == "gradebook.pdf"
    assert "attendance" not in llm.last_prompt.lower()


@pytest.mark.asyncio
async def test_small_gap_keeps_all_chunks_even_when_all_scores_negative():
    """Reproduces the real production case that broke an absolute threshold:
    every chunk scored negative (-7.6 to -11.4), but the top match was
    genuinely the right answer and must not be filtered out just because
    the whole query's score distribution sits below zero."""
    llm = FakeLLM()
    hits = [
        {"id": "a", "document": "Guardian accounts are linked to one student.", "metadata": {"source_path": "guardian-guide.pdf"}},
        {"id": "b", "document": "GUARDIAN FLOW: admin links guardian to student.", "metadata": {"source_path": "guardian-guide.pdf"}},
        {"id": "c", "document": "Support and contact information.", "metadata": {"source_path": "support-contact.pdf"}},
        {"id": "d", "document": "More support contact info.", "metadata": {"source_path": "support-contact.pdf"}},
        {"id": "e", "document": "Platform admins manage schools.", "metadata": {"source_path": "getting-started.pdf"}},
    ]
    scores = {"a": -7.630, "b": -8.724, "c": -11.307, "d": -11.341, "e": -11.355}
    qa = QAPipeline(retriever=FakeRetriever(hits), reranker=FakeReranker(scores), llm_client=llm, rerank_top_k=5)

    result = await qa.answer("A guardian can't see their child.")

    assert len(result.citations) == 5
    assert result.citations[0].source_path == "guardian-guide.pdf"


@pytest.mark.asyncio
async def test_top_match_always_kept_even_alone():
    llm = FakeLLM()
    qa = QAPipeline(
        retriever=FakeRetriever([UNRELATED_CHUNK]),
        reranker=FakeReranker({"2": -20.0}),
        llm_client=llm,
        rerank_top_k=5,
    )

    result = await qa.answer("any question")

    assert len(result.citations) == 1  # the only candidate is always kept as "best available"


@pytest.mark.asyncio
async def test_gap_filter_disabled_when_none():
    llm = FakeLLM()
    qa = QAPipeline(
        retriever=FakeRetriever([RELEVANT_CHUNK, UNRELATED_CHUNK]),
        reranker=FakeReranker({"1": 3.5, "2": -8.2}),
        llm_client=llm,
        rerank_top_k=5,
        max_score_gap=None,
    )

    result = await qa.answer("Why can't this teacher edit marks anymore?")

    assert len(result.citations) == 2  # filter disabled, both chunks pass through


@pytest.mark.asyncio
async def test_custom_gap_threshold_is_respected():
    llm = FakeLLM()
    qa = QAPipeline(
        retriever=FakeRetriever([RELEVANT_CHUNK, UNRELATED_CHUNK]),
        reranker=FakeReranker({"1": 3.5, "2": -8.2}),
        llm_client=llm,
        rerank_top_k=5,
        max_score_gap=15.0,  # wide enough that both chunks now fit
    )

    result = await qa.answer("Why can't this teacher edit marks anymore?")

    assert len(result.citations) == 2


@pytest.mark.asyncio
async def test_system_prompt_forbids_cross_chunk_inference():
    """Guards against silently weakening the anti-hallucination instruction later."""
    llm = FakeLLM()
    qa = QAPipeline(retriever=FakeRetriever([RELEVANT_CHUNK]), reranker=None, llm_client=llm, rerank_top_k=5)

    await qa.answer("test question")

    assert "unrelated to each other" in llm.last_system
    assert "Never infer" in llm.last_system


class RecordingRetriever:
    """Like FakeRetriever, but remembers exactly what query string it was searched with."""

    def __init__(self, hits):
        self.hits = hits
        self.last_query = None

    def search(self, question, where=None):
        self.last_query = question
        return self.hits


class CondensingFakeLLM:
    """Returns a fixed condensed question when asked to condense, else a fixed answer."""

    def __init__(self, condensed_question: str, raises_on_condense: bool = False):
        self.condensed_question = condensed_question
        self.raises_on_condense = raises_on_condense
        self.last_prompt = None
        self.last_system = None

    async def generate(self, prompt, system=None, temperature=0.2):
        self.last_prompt = prompt
        self.last_system = system
        is_condense_call = system is not None and "standalone question" in system.lower()
        if is_condense_call:
            if self.raises_on_condense:
                raise RuntimeError("simulated LLM outage during condensing")
            return self.condensed_question
        return "final answer [1]"


@pytest.mark.asyncio
async def test_first_turn_has_no_history_and_is_not_condensed():
    retriever = RecordingRetriever([RELEVANT_CHUNK])
    llm = CondensingFakeLLM(condensed_question="should not be used")
    qa = QAPipeline(retriever=retriever, reranker=FakeReranker({"1": 3.5}), llm_client=llm, rerank_top_k=5)

    result = await qa.answer("Why can't this teacher edit marks anymore?")

    assert retriever.last_query == "Why can't this teacher edit marks anymore?"
    assert result.standalone_question is None


@pytest.mark.asyncio
async def test_followup_question_is_condensed_before_retrieval():
    retriever = RecordingRetriever([RELEVANT_CHUNK])
    llm = CondensingFakeLLM(condensed_question="My timetable says there is a room conflict. How do I fix it?")
    qa = QAPipeline(retriever=retriever, reranker=FakeReranker({"1": 3.5}), llm_client=llm, rerank_top_k=5)

    history = [("My timetable says there is a room conflict.", "Rooms are used for conflict checks.")]
    result = await qa.answer("How do I fix it?", conversation_history=history)

    assert retriever.last_query == "My timetable says there is a room conflict. How do I fix it?"
    assert result.standalone_question == "My timetable says there is a room conflict. How do I fix it?"
    # The original, un-rewritten question is still what's stored as .question for display/history purposes.
    assert result.question == "How do I fix it?"


@pytest.mark.asyncio
async def test_condensing_failure_falls_back_to_raw_question():
    """If the condensing LLM call fails (e.g. network hiccup), the pipeline must
    still answer using the raw follow-up rather than erroring out entirely."""
    retriever = RecordingRetriever([RELEVANT_CHUNK])
    llm = CondensingFakeLLM(condensed_question="unused", raises_on_condense=True)
    qa = QAPipeline(retriever=retriever, reranker=FakeReranker({"1": 3.5}), llm_client=llm, rerank_top_k=5)

    history = [("previous question", "previous answer")]
    result = await qa.answer("How do I fix it?", conversation_history=history)

    assert retriever.last_query == "How do I fix it?"  # fell back to the raw question
    assert result.answer == "final answer [1]"  # generation still succeeded


@pytest.mark.asyncio
async def test_conversation_history_included_in_generation_prompt():
    retriever = RecordingRetriever([RELEVANT_CHUNK])
    llm = CondensingFakeLLM(condensed_question="standalone version")
    qa = QAPipeline(retriever=retriever, reranker=FakeReranker({"1": 3.5}), llm_client=llm, rerank_top_k=5)

    history = [("earlier question", "earlier answer")]
    await qa.answer("follow-up", conversation_history=history)

    assert "Previous Q: earlier question" in llm.last_prompt
    assert "Previous A: earlier answer" in llm.last_prompt


@pytest.mark.asyncio
async def test_no_history_means_no_history_section_in_prompt():
    retriever = RecordingRetriever([RELEVANT_CHUNK])
    llm = CondensingFakeLLM(condensed_question="unused")
    qa = QAPipeline(retriever=retriever, reranker=FakeReranker({"1": 3.5}), llm_client=llm, rerank_top_k=5)

    await qa.answer("standalone question, no history")

    assert "Previous Q:" not in llm.last_prompt


@pytest.mark.asyncio
async def test_only_immediately_preceding_turn_included_not_older_unrelated_topics():
    """Regression test for a real reported bug: in a session with two unrelated
    topics (room conflicts, then CSV imports), a generic follow-up like "what
    should I do next?" pulled the much-earlier "room conflict" topic into the
    answer alongside the actually-relevant CSV topic. Only the single
    immediately-preceding turn must be included, not several turns back."""
    retriever = RecordingRetriever([RELEVANT_CHUNK])
    llm = CondensingFakeLLM(condensed_question="What should I do after a partial CSV import?")
    qa = QAPipeline(retriever=retriever, reranker=FakeReranker({"1": 3.5}), llm_client=llm, rerank_top_k=5)

    history = [
        ("My timetable says there is a room conflict.", "Rooms are used for conflict checks; not covered further."),
        ("How do I fix it?", "The documentation does not cover steps to resolve a room conflict."),
        ("I imported my CSV but half the students were not created.", "Errors in CSV rows likely caused this; download the error CSV."),
    ]
    await qa.answer("What should I do next?", conversation_history=history)

    assert "room conflict" not in llm.last_prompt.lower()
    assert "csv" in llm.last_prompt.lower()