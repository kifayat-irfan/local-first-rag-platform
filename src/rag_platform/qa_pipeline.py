"""
Full retrieve -> rerank -> generate -> cite pipeline, with multi-turn
conversation support.

Every answer is generated strictly from retrieved chunks. Every chunk used
carries its source PDF filename and page number into the final result, so
"page 7 of handbook.pdf" is always available alongside the answer.

Two safeguards against a specific failure mode: an LLM connecting two
genuinely unrelated retrieved chunks into a false causal story (e.g.
mixing "attendance ownership" and "grade-editing permissions" — two
different features — because both chunks happened to be in the same
prompt). The system prompt explicitly forbids inferring relationships
that aren't stated within a single context block, and chunks that score
far worse than the reranker's own best match are dropped before they
ever reach the LLM, rather than trusting the model to ignore them itself.

Why a relative gap, not an absolute score floor: cross-encoder rerank
scores are not on a fixed, comparable scale across different queries — the
same model can score a genuinely relevant chunk at -7.6 for one question
and an irrelevant chunk at -8.2 for a different question. An absolute
cutoff (e.g. "drop anything below -2.0") either lets real hallucination
cases through or wrongly rejects a question's only good match, depending
on where that query's whole score distribution happens to sit. Measuring
each chunk against *that same query's* top-ranked result is stable
regardless of where the absolute scale lands.

Multi-turn memory: a bare follow-up question like "how do I fix it?" embeds
and retrieves badly on its own — the embedding model has no idea what "it"
refers to. Before retrieval, a short conversation history (if any) is used
to rewrite the question into a standalone one that carries its own context
("how do I fix a room conflict in the timetable?"). This is the standard
"condense question" pattern: one extra, cheap LLM call, only made when
there's actual history to draw on.

Only the single immediately-preceding turn is included in the final
generation prompt (not several) — real testing showed that a 2-turn window
let the model blend two genuinely unrelated topics into one answer (e.g.
mentioning a much-earlier "room conflict" question while answering a new
one about CSV imports), because the second-most-recent turn just happened
to be about something else entirely. One turn is enough for legitimate
pronoun/context resolution ("it", "that", "what about X") without dragging
in whatever else the conversation covered several messages ago.
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
    "Cite sources inline using [n] markers that map to the numbered context blocks. "
    "If earlier conversation turns are included, use them only to understand what the "
    "user is currently asking about — never pull facts from your own prior answers "
    "into this one unless those facts are also present in the numbered context blocks. "
    "If the earlier turn was about a different topic than the current question, ignore "
    "it entirely; do not mention, list, or combine it with the current topic in your "
    "answer, even in passing."
)

CONDENSE_SYSTEM_PROMPT = (
    "Rewrite the follow-up question as a standalone question that includes any "
    "necessary context from the conversation history, so it makes sense on its "
    "own with no prior messages. If it's already standalone, return it unchanged. "
    "Respond with ONLY the rewritten question — no preamble, no quotes, no explanation."
)

DEFAULT_MAX_SCORE_GAP = 6.0  # chunks scoring this many points below THIS QUERY'S best match are dropped
DEFAULT_HISTORY_TURNS_FOR_CONDENSING = 3  # how many prior (question, answer) pairs to show the condenser
DEFAULT_HISTORY_TURNS_FOR_GENERATION = 1  # only the immediately preceding turn — see design note below

ConversationTurn = tuple[str, str]  # (question, answer)


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
    standalone_question: Optional[str] = None  # set only if a follow-up was rewritten for retrieval


class QAPipeline:
    """
    Args:
        retriever: This tenant's HybridRetriever (already scoped to the
            correct company's vector store — see scripts/query_company.py).
        reranker: Optional; pass None to skip reranking (faster, slightly
            lower precision — controlled by Settings.use_reranker).
        llm_client: LLM used for final answer generation (and query condensing).
        rerank_top_k: How many reranked chunks to feed the LLM.
        max_score_gap: Chunks scoring more than this many points below the
            top-ranked chunk *for this same query* are dropped, even if
            that leaves fewer than rerank_top_k chunks. The top match is
            always kept — this filters out chunks the reranker considers
            much worse than its own best answer, without relying on an
            absolute score threshold that doesn't hold across different
            queries. Only applied when a reranker is in use, since raw RRF
            scores aren't on a comparable scale. Set to None to disable.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker],
        llm_client: OllamaClient,
        rerank_top_k: int = 5,
        max_score_gap: Optional[float] = DEFAULT_MAX_SCORE_GAP,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.rerank_top_k = rerank_top_k
        self.max_score_gap = max_score_gap

    async def answer(
        self,
        question: str,
        where: Optional[dict[str, Any]] = None,
        conversation_history: Optional[list[ConversationTurn]] = None,
    ) -> QAResult:
        """
        Args:
            question: The user's current message, verbatim.
            where: Optional metadata filter passed through to retrieval.
            conversation_history: Prior (question, answer) turns in this
                chat session, oldest first. Pass None or [] for a
                single-shot question with no conversational context.
        """
        history = conversation_history or []

        standalone_question = question
        if history:
            standalone_question = await self._condense_question(question, history)

        candidates = self.retriever.search(standalone_question, where=where)

        if self.reranker is not None:
            ranked = self.reranker.rerank(standalone_question, candidates, top_k=self.rerank_top_k)
            if ranked and self.max_score_gap is not None:
                top_score = ranked[0].get("rerank_score", 0.0)
                floor = top_score - self.max_score_gap
                ranked = [r for r in ranked if r.get("rerank_score", 0.0) >= floor]
        else:
            ranked = candidates[: self.rerank_top_k]

        if not ranked:
            return QAResult(
                question=question,
                answer="I couldn't find any sufficiently relevant indexed content to answer this question.",
                citations=[],
                standalone_question=standalone_question if standalone_question != question else None,
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

        prompt_parts = []
        recent_history = history[-DEFAULT_HISTORY_TURNS_FOR_GENERATION:]
        if recent_history:
            history_text = "\n\n".join(f"Previous Q: {q}\nPrevious A: {a}" for q, a in recent_history)
            prompt_parts.append(f"Recent conversation (for context only, not a source of facts):\n{history_text}\n")

        prompt_parts.append(f"Context:\n\n{chr(10).join(context_blocks)}")
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Answer using only the numbered context blocks above, with [n] citation markers.")
        prompt = "\n\n".join(prompt_parts)

        answer_text = await self.llm_client.generate(prompt, system=SYSTEM_PROMPT)
        return QAResult(
            question=question,
            answer=answer_text,
            citations=citations,
            standalone_question=standalone_question if standalone_question != question else None,
        )

    async def _condense_question(self, question: str, history: list[ConversationTurn]) -> str:
        """Rewrites a possibly-context-dependent follow-up into a standalone question for retrieval."""
        recent = history[-DEFAULT_HISTORY_TURNS_FOR_CONDENSING:]
        history_text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in recent)
        prompt = f"Conversation history:\n{history_text}\n\nFollow-up question: {question}\n\nStandalone question:"

        try:
            rewritten = await self.llm_client.generate(prompt, system=CONDENSE_SYSTEM_PROMPT, temperature=0.0)
        except Exception:  # noqa: BLE001 — condensing is a nice-to-have; never let it break the whole answer
            return question

        rewritten = rewritten.strip().strip('"')
        return rewritten if rewritten else question