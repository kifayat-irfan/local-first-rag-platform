#!/usr/bin/env python3
"""
Query a single company's indexed corpus.

Usage:
    python scripts/query_company.py --company company_a --query "What is the deployment process?" --top-k 5
    python scripts/query_company.py --company company_a --query "..." --no-llm   # retrieval only, no answer generation
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_platform.company_registry import CompanyNotFoundError, CompanyRegistry
from rag_platform.config import get_settings
from rag_platform.config_loader import ConfigValidationError, load_company_config
from rag_platform.embedder import Embedder
from rag_platform.llm import OllamaClient
from rag_platform.logging_config import bind_company, configure_logging
from rag_platform.qa_pipeline import QAPipeline
from rag_platform.reranker import CrossEncoderReranker
from rag_platform.retriever import HybridRetriever
from rag_platform.vector_store import VectorStore


async def main(company_id: str, query: str, top_k: int, use_llm: bool) -> int:
    configure_logging()
    bind_company(company_id)

    settings = get_settings(company_id)

    try:
        company_config = load_company_config(company_id, settings.configs_dir)
    except ConfigValidationError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    registry = CompanyRegistry(settings.registry_path)
    try:
        registry.get(company_id)
    except CompanyNotFoundError:
        print(f"Warning: {company_id!r} is not registered (querying anyway).", file=sys.stderr)

    embedder = Embedder(model_name=company_config.embedding.model, device=company_config.embedding.device)
    vector_store = VectorStore(persist_dir=settings.vector_db_dir, collection_name=company_config.vector_store.collection_name)

    if vector_store.count() == 0:
        print(f"No indexed content for {company_id!r} yet. Run ingest_company_pdfs.py first.")
        return 0

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k_dense=company_config.retrieval.top_k,
        top_k_bm25=company_config.retrieval.top_k,
        top_k_final=top_k,
        use_bm25=company_config.retrieval.use_bm25,
    )

    if not use_llm:
        # Retrieval-only mode: skip LLM + reranker entirely, just show ranked chunks.
        hits = retriever.search(query)
        print(f"\n=== Top {len(hits)} chunks for: {query!r} ===\n")
        for i, hit in enumerate(hits, start=1):
            meta = hit.get("metadata", {})
            page = meta.get("page_number", -1)
            page_label = f", page {page}" if page and page > 0 else ""
            print(f"[{i}] {meta.get('source_path', 'unknown')}{page_label}  (rrf_score={hit.get('rrf_score', 0):.4f})")
            print(f"    {hit['document'][:200]}")
            print()
        return 0

    reranker = CrossEncoderReranker(model_name=company_config.retrieval.reranker_model) if company_config.retrieval.use_reranker else None
    llm = OllamaClient(model="llama3.1:8b")
    qa = QAPipeline(retriever=retriever, reranker=reranker, llm_client=llm, rerank_top_k=company_config.retrieval.rerank_top_k)

    result = await qa.answer(query)

    print("\n=== ANSWER ===\n")
    print(result.answer)
    print("\n=== SOURCES ===\n")
    for c in result.citations:
        page_label = f", page {c.page_number}" if c.page_number else ""
        print(f"[{c.marker}] {c.source_path}{page_label}  (score={c.score:.3f})")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-llm", action="store_true", help="Skip answer generation, just show retrieved chunks")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.company, args.query, args.top_k, use_llm=not args.no_llm)))
