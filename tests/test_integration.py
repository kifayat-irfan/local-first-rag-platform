"""
End-to-end integration test using a synthetically generated PDF (reportlab).

Uses a FakeEmbedder (deterministic random vectors) instead of the real BGE
model so this test runs fully offline and fast — it's exercising the
pipeline's plumbing (extraction -> chunking -> manifest -> vector store ->
retrieval), not embedding quality, which is the concern of a live-model
smoke test rather than CI.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `import scripts.*`

from rag_platform.chunker import RecursiveCharacterChunker
from rag_platform.company_registry import CompanyRegistry
from rag_platform.config_loader import load_company_config
from rag_platform.manifest import ManifestStore
from rag_platform.retriever import HybridRetriever
from rag_platform.vector_store import VectorStore

import scripts.ingest_company_pdfs as ingest_module


class FakeEmbedder:
    """Deterministic-enough fake: same text -> same vector, different text -> different vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vec(query)

    @staticmethod
    def _vec(text: str) -> list[float]:
        # Cheap deterministic hash-based embedding — good enough to prove the
        # plumbing works without downloading a real model.
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:16]]


TEMPLATE_YAML = """
company:
  id: company_xxx
  name: "Company Name"
sources:
  - type: pdf
    path: {raw_pdfs_dir}
    glob: "*.pdf"
processing:
  chunk_size: 300
  chunk_overlap: 40
  min_chunk_length: 20
embedding:
  model: BAAI/bge-small-en-v1.5
  batch_size: 32
vector_store:
  type: chromadb
  path: {vector_db_dir}
  collection_name: test_co_docs
retrieval:
  top_k: 10
  rerank_top_k: 5
"""


@pytest.fixture
def company_setup(tmp_settings, sample_pdf: Path):
    """Wires up configs/template.yaml + configs/test_co.yaml and drops sample_pdf into raw_pdfs/."""
    tmp_settings.configs_dir.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE_YAML.format(raw_pdfs_dir=tmp_settings.raw_pdfs_dir, vector_db_dir=tmp_settings.vector_db_dir)
    (tmp_settings.configs_dir / "template.yaml").write_text(template)
    (tmp_settings.configs_dir / "test_co.yaml").write_text(
        yaml.dump({"company": {"id": "test_co", "name": "Test Co"}})
    )

    dest = tmp_settings.raw_pdfs_dir / "sample.pdf"
    shutil.copy(sample_pdf, dest)

    registry = CompanyRegistry(tmp_settings.registry_path)
    registry.add("test_co", "Test Co", str(tmp_settings.config_path), str(tmp_settings.data_dir))

    return tmp_settings, dest


@pytest.mark.asyncio
async def test_full_ingestion_pipeline(company_setup):
    settings, pdf_path = company_setup
    company_config = load_company_config("test_co", settings.configs_dir)

    manifest = ManifestStore(settings.manifest_path)
    manifest.load()
    chunker = RecursiveCharacterChunker(
        chunk_size=company_config.processing.chunk_size,
        chunk_overlap=company_config.processing.chunk_overlap,
        min_chunk_length=company_config.processing.min_chunk_length,
    )
    vector_store = VectorStore(settings.vector_db_dir, company_config.vector_store.collection_name)
    embedder = FakeEmbedder()

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=1) as executor:
        result = await ingest_module.process_one_pdf(
            pdf_path, "test_co", settings, company_config, manifest, chunker, embedder, vector_store, executor
        )

    assert result["status"] == "ok"
    assert result["chunk_count"] > 0
    assert vector_store.count() == result["chunk_count"]

    entry = manifest.get(str(pdf_path))
    assert entry is not None
    assert entry.status == "ok"
    assert Path(entry.clean_path).exists()

    # Local-first: re-running against the unchanged file must not require reprocessing.
    assert manifest.needs_processing(pdf_path) is False


@pytest.mark.asyncio
async def test_retrieval_after_ingestion(company_setup):
    settings, pdf_path = company_setup
    company_config = load_company_config("test_co", settings.configs_dir)

    manifest = ManifestStore(settings.manifest_path)
    manifest.load()
    chunker = RecursiveCharacterChunker(
        chunk_size=company_config.processing.chunk_size,
        chunk_overlap=company_config.processing.chunk_overlap,
        min_chunk_length=company_config.processing.min_chunk_length,
    )
    vector_store = VectorStore(settings.vector_db_dir, company_config.vector_store.collection_name)
    embedder = FakeEmbedder()

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=1) as executor:
        await ingest_module.process_one_pdf(
            pdf_path, "test_co", settings, company_config, manifest, chunker, embedder, vector_store, executor
        )

    retriever = HybridRetriever(
        embedder=embedder, vector_store=vector_store, top_k_dense=10, top_k_bm25=10, top_k_final=5, use_bm25=True
    )
    hits = retriever.search("rollback procedure")

    assert len(hits) > 0
    # BM25 keyword match should surface the rollback chunk specifically for this query.
    assert any("rollback" in h["document"].lower() for h in hits)
    # Page numbers must have made it through extraction -> chunking -> indexing intact.
    assert all(h["metadata"]["page_number"] in (1, 2) for h in hits)


@pytest.mark.asyncio
async def test_scanned_pdf_is_skipped_not_indexed(company_setup, blank_pdf: Path):
    settings, _existing_pdf = company_setup
    scanned_dest = settings.raw_pdfs_dir / "scanned.pdf"
    shutil.copy(blank_pdf, scanned_dest)

    company_config = load_company_config("test_co", settings.configs_dir)
    manifest = ManifestStore(settings.manifest_path)
    manifest.load()
    chunker = RecursiveCharacterChunker(chunk_size=300, chunk_overlap=40, min_chunk_length=20)
    vector_store = VectorStore(settings.vector_db_dir, company_config.vector_store.collection_name)
    embedder = FakeEmbedder()

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=1) as executor:
        result = await ingest_module.process_one_pdf(
            scanned_dest, "test_co", settings, company_config, manifest, chunker, embedder, vector_store, executor
        )

    assert result["status"] == "scanned"
    assert result["chunk_count"] == 0
    entry = manifest.get(str(scanned_dest))
    assert entry.status == "skipped_scanned"
