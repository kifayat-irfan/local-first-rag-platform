"""
Streamlit UI for the multi-tenant RAG platform — portfolio-ready front end.

Deployment note (read this before deploying to Streamlit Community Cloud):

  - Retrieval (dense + BM25 + RRF + cross-encoder rerank) works everywhere,
    including on Streamlit Cloud, because it's fully local (small models,
    no external LLM call). This is the core of the demo and needs nothing
    special to deploy.

  - LLM answer generation talks to Ollama over HTTP. On your own machine
    that's http://localhost:11434 and it just works. On Streamlit Cloud,
    "localhost" is the *server's* localhost, not yours — there is no Ollama
    there. The Ollama URL is exposed as a sidebar field specifically so
    you can point it at a reachable Ollama instance (e.g. one you've
    tunneled with ngrok/Cloudflare Tunnel) if you want live LLM answers on
    the hosted demo. Without that, ship the deployed version with "Generate
    LLM answer" off by default and let retrieval + citations carry the demo
    — which is still a complete, honest showcase of the pipeline.

  - Streamlit Cloud's filesystem is ephemeral (wiped on every redeploy/
    restart). Any company you create or PDF you ingest through this UI on
    the hosted version will NOT persist across restarts. For a stable
    hosted demo, commit a small pre-ingested company's data/companies/{id}/
    directory (including vector_store/) to the repo so it's there on boot.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import scripts.ingest_company_pdfs as ingest_module  # noqa: E402
from rag_platform.cache import cache_key, cached_call, get_cache  # noqa: E402
from rag_platform.chat_history import ChatHistoryStore  # noqa: E402
from rag_platform.chunker import RecursiveCharacterChunker  # noqa: E402
from rag_platform.company_registry import (  # noqa: E402
    CompanyAlreadyExistsError,
    CompanyRegistry,
)
from rag_platform.config import Settings, get_settings  # noqa: E402
from rag_platform.config_loader import ConfigValidationError, load_company_config  # noqa: E402
from rag_platform.embedder import Embedder  # noqa: E402
from rag_platform.llm import GroqLLMClient, OllamaClient  # noqa: E402
from rag_platform.manifest import ManifestStore  # noqa: E402
from rag_platform.qa_pipeline import QAPipeline  # noqa: E402
from rag_platform.reranker import CrossEncoderReranker  # noqa: E402
from rag_platform.retriever import HybridRetriever  # noqa: E402
from rag_platform.vector_store import VectorStore  # noqa: E402
from rag_platform.voice import GroqTranscriptionError, GroqWhisperClient  # noqa: E402

st.set_page_config(page_title="Local-First RAG Platform", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------- #
# Theme: dark glass + neon accents (cyan / pink / purple), combining a
# fintech-style stat dashboard with a neon glass chat surface.
# ---------------------------------------------------------------------- #
THEME_CSS = """
<style>
:root {
    --bg-deep: #0A0A12;
    --bg-panel: #12121C;
    --bg-card: #15151F;
    --neon-cyan: #22D3EE;
    --neon-pink: #F472B6;
    --neon-purple: #A78BFA;
    --text-hi: #F1F1F4;
    --text-lo: #9CA3AF;
}
.stApp { background: var(--bg-deep); }
section[data-testid="stSidebar"] { background: var(--bg-panel); border-right: 1px solid rgba(255,255,255,0.06); }

.rag-stat-row { display: flex; gap: 14px; margin: 4px 0 22px 0; flex-wrap: wrap; }
.rag-stat-card {
    flex: 1; min-width: 150px; background: var(--bg-card); border-radius: 14px;
    padding: 14px 16px; border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.02);
}
.rag-stat-card .icon { font-size: 20px; }
.rag-stat-card .label { color: var(--text-lo); font-size: 12.5px; margin-top: 6px; }
.rag-stat-card .value { color: var(--text-hi); font-size: 26px; font-weight: 700; margin-top: 2px; }
.rag-stat-card.cyan { box-shadow: 0 0 14px -4px var(--neon-cyan); border-color: rgba(34,211,238,0.35); }
.rag-stat-card.pink { box-shadow: 0 0 14px -4px var(--neon-pink); border-color: rgba(244,114,182,0.35); }
.rag-stat-card.purple { box-shadow: 0 0 14px -4px var(--neon-purple); border-color: rgba(167,139,250,0.35); }
.rag-stat-card.amber { box-shadow: 0 0 14px -4px #FBBF24; border-color: rgba(251,191,36,0.35); }

.rag-company-card {
    border-radius: 12px; padding: 10px 12px; margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.08); background: var(--bg-card);
    display: flex; align-items: center; gap: 10px;
}
.rag-company-card.active { border-color: var(--neon-cyan); box-shadow: 0 0 12px -3px var(--neon-cyan); }
.rag-avatar {
    width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-weight: 700; font-size: 13px; color: #0A0A12; flex-shrink: 0;
}
.rag-company-name { color: var(--text-hi); font-size: 14px; font-weight: 600; }
.rag-company-count { color: var(--text-lo); font-size: 11.5px; }

section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background: var(--bg-card); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; color: var(--text-hi); text-align: left; width: 100%;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    border-color: var(--neon-cyan); box-shadow: 0 0 10px -3px var(--neon-cyan);
}

div[data-testid="stChatMessage"] { background: var(--bg-card); border-radius: 14px; border: 1px solid rgba(255,255,255,0.06); }

.rag-citation-card {
    border: 1px solid rgba(34,211,238,0.3); border-radius: 10px; padding: 8px 12px;
    margin-bottom: 6px; background: rgba(34,211,238,0.05);
}
.rag-citation-badge {
    display: inline-block; background: var(--neon-cyan); color: #05202A; font-weight: 700;
    font-size: 11px; border-radius: 999px; padding: 1px 7px; margin-right: 6px;
}

div[data-baseweb="radio"] label {
    border: 1px solid rgba(255,255,255,0.15); border-radius: 999px; padding: 4px 14px; margin-right: 6px;
}

.rag-history-row {
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.06); padding: 10px 4px;
}
.rag-badge-pill {
    background: rgba(167,139,250,0.15); color: var(--neon-purple); border-radius: 999px;
    padding: 2px 10px; font-size: 11.5px; font-weight: 600;
}
</style>
"""

AVATAR_PALETTE = [
    ("#22D3EE", "cyan"), ("#F472B6", "pink"), ("#A78BFA", "purple"), ("#FBBF24", "amber"),
]


def avatar_color(company_id: str) -> tuple[str, str]:
    idx = sum(ord(c) for c in company_id) % len(AVATAR_PALETTE)
    return AVATAR_PALETTE[idx]


TEMPLATE_YAML = """company:
  id: {company_id}
  name: "{company_name}"
sources:
  - type: pdf
    path: ./data/companies/{company_id}/raw_pdfs/
    glob: "*.pdf"
processing:
  chunk_size: 512
  chunk_overlap: 50
  min_chunk_length: 100
embedding:
  model: BAAI/bge-small-en-v1.5
  batch_size: 32
  device: cpu
vector_store:
  type: chromadb
  path: ./data/companies/{company_id}/vector_store/
  collection_name: {company_id}_docs
retrieval:
  top_k: 10
  rerank_top_k: 5
  use_bm25: true
  use_reranker: true
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""


# ---------------------------------------------------------------------- #
# Cached resources — loaded once per process, not on every rerun/click.
# ---------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_embedder(model_name: str, device: str) -> Embedder:
    return Embedder(model_name=model_name, device=device)


@st.cache_resource(show_spinner="Loading reranker model (first run only)...")
def get_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=model_name)


@st.cache_resource
def get_chat_store() -> ChatHistoryStore:
    return ChatHistoryStore(get_settings().chat_history_db_path)


def get_registry() -> CompanyRegistry:
    return CompanyRegistry(get_settings().registry_path)


# ---------------------------------------------------------------------- #
# Sidebar: company selection, onboarding, PDF upload + ingest
# ---------------------------------------------------------------------- #
def render_sidebar() -> str | None:
    st.sidebar.title("📚 Companies")
    registry = get_registry()
    companies = registry.list_companies()

    if not companies:
        st.sidebar.info("No companies yet — add one below.")
        selected_id = None
    else:
        if "selected_company" not in st.session_state or st.session_state.selected_company not in [c.company_id for c in companies]:
            st.session_state.selected_company = companies[0].company_id

        for c in companies:
            color, _name = avatar_color(c.company_id)
            initials = "".join(w[0] for w in c.name.split()[:2]).upper()
            is_active = c.company_id == st.session_state.selected_company
            active_class = "active" if is_active else ""
            st.sidebar.markdown(
                f"""<div class="rag-company-card {active_class}">
                    <div class="rag-avatar" style="background:{color};">{initials}</div>
                    <div>
                        <div class="rag-company-name">{c.name}</div>
                        <div class="rag-company-count">{c.total_pdfs} PDFs · {c.total_chunks} chunks</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.sidebar.button(f"Select {c.name}", key=f"pick_{c.company_id}", use_container_width=True):
                st.session_state.selected_company = c.company_id
                st.rerun()

        selected_id = st.session_state.selected_company

    with st.sidebar.expander("➕ Add a new company"):
        new_id = st.text_input("Company ID (lowercase, no spaces)", key="new_company_id")
        new_name = st.text_input("Display name", key="new_company_name")
        if st.button("Register company"):
            if not new_id or not new_name:
                st.warning("Both fields are required.")
            else:
                try:
                    settings = get_settings(new_id)
                    settings.ensure_dirs()
                    registry.add(new_id, new_name, str(settings.config_path), str(settings.data_dir))
                    settings.config_path.write_text(
                        TEMPLATE_YAML.format(company_id=new_id, company_name=new_name)
                    )
                    st.success(f"Registered {new_id!r}. Select it above.")
                    st.rerun()
                except CompanyAlreadyExistsError:
                    st.error(f"{new_id!r} is already registered.")
                except ValueError as exc:
                    st.error(f"Invalid company id: {exc}")

    if selected_id:
        render_upload_and_ingest(selected_id)

    return selected_id


def render_upload_and_ingest(company_id: str) -> None:
    settings = get_settings(company_id)
    with st.sidebar.expander("📄 Upload PDFs & ingest", expanded=False):
        uploaded = st.file_uploader(
            "Upload PDF documents", type=["pdf"], accept_multiple_files=True, key=f"upload_{company_id}"
        )
        if uploaded and st.button("Save uploaded files", key=f"save_{company_id}"):
            settings.ensure_dirs()
            for f in uploaded:
                (settings.raw_pdfs_dir / f.name).write_bytes(f.getbuffer())
            st.success(f"Saved {len(uploaded)} file(s) to raw_pdfs/. Click Ingest below.")

        if st.button("▶️ Run ingestion", key=f"ingest_{company_id}"):
            run_ingestion_ui(company_id)


def run_ingestion_ui(company_id: str) -> None:
    settings = get_settings(company_id)
    try:
        company_config = load_company_config(company_id, settings.configs_dir)
    except ConfigValidationError as exc:
        st.error(f"Config error: {exc}")
        return

    pdfs = ingest_module.find_pdfs(company_config)
    if not pdfs:
        st.warning("No PDFs found in raw_pdfs/. Upload some first.")
        return

    manifest = ManifestStore(settings.manifest_path)
    manifest.load()
    to_process = [p for p in pdfs if manifest.needs_processing(p)]

    if not to_process:
        st.info("Everything is already up to date (local-first skip).")
        return

    chunker = RecursiveCharacterChunker(
        chunk_size=company_config.processing.chunk_size,
        chunk_overlap=company_config.processing.chunk_overlap,
        min_chunk_length=company_config.processing.min_chunk_length,
    )
    embedder = get_embedder(company_config.embedding.model, company_config.embedding.device)
    vector_store = VectorStore(settings.vector_db_dir, company_config.vector_store.collection_name)

    progress = st.progress(0.0, text="Starting ingestion...")
    ok = scanned = errors = total_chunks = 0

    async def run_all():
        nonlocal ok, scanned, errors, total_chunks
        with ProcessPoolExecutor(max_workers=settings.max_pdf_workers) as executor:
            for i, pdf_path in enumerate(to_process):
                progress.progress((i) / len(to_process), text=f"Processing {pdf_path.name}...")
                result = await ingest_module.process_one_pdf(
                    pdf_path, company_id, settings, company_config, manifest, chunker, embedder, vector_store, executor
                )
                if result["status"] == "ok":
                    ok += 1
                    total_chunks += result["chunk_count"]
                elif result["status"] == "scanned":
                    scanned += 1
                else:
                    errors += 1

    asyncio.run(run_all())
    progress.progress(1.0, text="Done.")

    registry = get_registry()
    if registry.exists(company_id):
        registry.update_stats(
            company_id,
            total_pdfs=len(manifest.all_entries()),
            total_chunks=sum(e.chunk_count or 0 for e in manifest.all_entries()),
            mark_ingestion_now=True,
        )

    st.success(f"Ingested {ok} PDF(s), {total_chunks} chunks. Scanned/skipped: {scanned}. Errors: {errors}.")
    st.rerun()


# ---------------------------------------------------------------------- #
# Main chat area
# ---------------------------------------------------------------------- #
def answer_question(
    company_id: str,
    company_config,
    vector_store: VectorStore,
    question: str,
    top_k: int,
    use_llm: bool,
    llm_mode: str = "local",
    ollama_url: str = "",
    ollama_model: str = "",
    groq_api_key: str = "",
    groq_model: str = "",
    use_cache: bool = True,
) -> tuple[str, list[dict]]:
    """
    Shared retrieve -> rerank -> (optionally) generate path.

    llm_mode: "local" uses Ollama (default everywhere in this platform);
    "public" uses Groq's hosted LLM instead — for situations where a local
    Ollama isn't reachable (e.g. a deployed demo). Voice transcription
    always uses Groq regardless of this toggle; this setting only affects
    which backend generates the final answer.
    """
    cache = get_cache(maxsize=256, ttl_seconds=3600)
    key = cache_key(company_id, question, top_k, use_llm, llm_mode, ollama_model, groq_model)

    def compute():
        embedder = get_embedder(company_config.embedding.model, company_config.embedding.device)
        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            top_k_dense=company_config.retrieval.top_k,
            top_k_bm25=company_config.retrieval.top_k,
            top_k_final=top_k,
            use_bm25=company_config.retrieval.use_bm25,
        )
        hits = retriever.search(question)
        reranker = get_reranker(company_config.retrieval.reranker_model) if company_config.retrieval.use_reranker else None
        ranked = reranker.rerank(question, hits, top_k=top_k) if reranker else hits
        citations = build_citations(ranked)

        if not use_llm:
            answer = "Here are the most relevant passages I found (enable **Generate LLM answer** for a synthesized response):"
            return {"answer": answer, "citations": citations}

        if llm_mode == "public":
            llm = GroqLLMClient(api_key=groq_api_key, model=groq_model)
            backend_label = f"Groq ({groq_model})"
        else:
            llm = OllamaClient(base_url=ollama_url, model=ollama_model)
            backend_label = f"Ollama ({ollama_model}) at {ollama_url}"

        qa = QAPipeline(retriever=retriever, reranker=reranker, llm_client=llm, rerank_top_k=top_k)
        try:
            result = asyncio.run(qa.answer(question))
            answer = result.answer
        except Exception as exc:  # noqa: BLE001
            answer = (
                f"⚠️ Couldn't reach {backend_label} ({exc}). "
                "Showing retrieved chunks instead — check the connection, "
                "or turn off 'Generate LLM answer'."
            )
        return {"answer": answer, "citations": citations}

    if use_cache:
        result, was_hit = cached_call(cache, key, compute)
    else:
        result, was_hit = compute(), False

    return result["answer"], result["citations"]


def render_voice_input(company_id: str) -> Optional[str]:
    """Records audio via the browser mic and transcribes it with Groq Whisper. Returns transcribed text, or None."""
    settings = get_settings()
    audio = st.audio_input("🎤 Or ask by voice", key=f"voice_{company_id}")
    if audio is None:
        return None

    if st.button("Transcribe & ask", key=f"transcribe_{company_id}"):
        with st.spinner("Transcribing with Groq Whisper..."):
            client = GroqWhisperClient(api_key=settings.groq_api_key, model=settings.groq_whisper_model)
            try:
                text = asyncio.run(client.transcribe(audio.getvalue(), filename="query.wav"))
                return text
            except GroqTranscriptionError as exc:
                st.error(str(exc))
                return None
    return None


def render_chat(company_id: str) -> None:
    settings = get_settings(company_id)
    try:
        company_config = load_company_config(company_id, settings.configs_dir)
    except ConfigValidationError as exc:
        st.error(f"Config error: {exc}")
        return

    vector_store = VectorStore(settings.vector_db_dir, company_config.vector_store.collection_name)
    registry = get_registry()
    record = registry.get(company_id)

    st.caption(
        f"{record.total_pdfs} PDF(s) indexed · {record.total_chunks} chunks · "
        f"isolated vector store at `data/companies/{company_id}/vector_store/`"
    )

    if vector_store.count() == 0:
        st.info("No documents indexed yet for this company. Upload PDFs and run ingestion from the sidebar.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        use_llm = st.checkbox("Generate LLM answer", value=False, key=f"llm_{company_id}")
    with col2:
        top_k = st.slider("Chunks to retrieve", min_value=3, max_value=15, value=5, key=f"topk_{company_id}")

    ollama_url, ollama_model, groq_model = settings.ollama_base_url, settings.ollama_model, settings.groq_llm_model
    llm_mode = "local"
    if use_llm:
        llm_mode = st.radio(
            "LLM mode",
            options=["local", "public"],
            format_func=lambda m: "🔒 Local (Ollama)" if m == "local" else "🌐 Public (Groq)",
            horizontal=True,
            key=f"mode_{company_id}",
        )
        if llm_mode == "local":
            st.caption("Runs fully on your machine. Needs Ollama running locally.")
            with st.expander("Ollama settings"):
                ollama_url = st.text_input("Ollama base URL", value=settings.ollama_base_url, key=f"url_{company_id}")
                ollama_model = st.text_input("Ollama model", value=settings.ollama_model, key=f"model_{company_id}")
        else:
            st.caption("Uses Groq's hosted LLM (free tier) — works even without a local Ollama, e.g. on a deployed demo.")
            with st.expander("Groq settings"):
                groq_model = st.text_input("Groq model", value=settings.groq_llm_model, key=f"groqmodel_{company_id}")
                if not settings.groq_api_key:
                    st.warning("No GROQ_API_KEY set — add one to your .env to use Public mode.")

    history_key = f"messages_{company_id}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])

    voice_text = render_voice_input(company_id)
    question = voice_text or st.chat_input("Ask a question about this company's documentation...")
    if not question:
        return

    st.session_state[history_key].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, citations = answer_question(
                company_id, company_config, vector_store, question, top_k, use_llm,
                llm_mode=llm_mode, ollama_url=ollama_url, ollama_model=ollama_model,
                groq_api_key=settings.groq_api_key, groq_model=groq_model,
            )
        st.markdown(answer)
        render_citations(citations)

    st.session_state[history_key].append({"role": "assistant", "content": answer, "citations": citations})
    get_chat_store().add(company_id, question, answer, citations)


def render_compare() -> None:
    st.caption("Ask the same question across multiple companies at once and compare answers side by side.")
    registry = get_registry()
    companies = registry.list_companies()

    if len(companies) < 2:
        st.info("Register at least 2 companies to use comparison mode.")
        return

    labels = {c.company_id: c.name for c in companies}
    selected = st.multiselect(
        "Companies to compare", options=list(labels.keys()), default=list(labels.keys())[:2], format_func=lambda cid: labels[cid]
    )
    top_k = st.slider("Chunks to retrieve per company", min_value=3, max_value=15, value=5, key="compare_topk")
    question = st.text_input("Question to ask all selected companies")

    if not (selected and question and st.button("Compare")):
        return

    cols = st.columns(len(selected))
    for col, company_id in zip(cols, selected):
        with col:
            st.subheader(labels[company_id])
            settings = get_settings(company_id)
            try:
                company_config = load_company_config(company_id, settings.configs_dir)
            except ConfigValidationError as exc:
                st.error(f"Config error: {exc}")
                continue

            vector_store = VectorStore(settings.vector_db_dir, company_config.vector_store.collection_name)
            if vector_store.count() == 0:
                st.info("No documents indexed yet.")
                continue

            with st.spinner("Retrieving..."):
                answer, citations = answer_question(
                    company_id, company_config, vector_store, question, top_k, use_llm=False
                )
            st.markdown(answer)
            render_citations(citations)
            get_chat_store().add(company_id, question, answer, citations)


def render_history(company_id: str) -> None:
    store = get_chat_store()
    search = st.text_input("Search past questions", key=f"search_{company_id}")
    entries = store.list_for_company(company_id, limit=50, search=search or None)

    if not entries:
        st.info("No saved conversations yet for this company.")
        return

    for entry in entries:
        star_icon = "⭐" if entry.starred else "☆"
        n_sources = len(entry.citations)
        st.markdown(
            f'<span class="rag-badge-pill">Sources {n_sources}</span>',
            unsafe_allow_html=True,
        )
        with st.expander(f"{star_icon} {entry.question}  ·  {entry.created_at[:19]}"):
            st.markdown(entry.answer)
            if entry.citations:
                render_citations(entry.citations)
            c1, c2 = st.columns([1, 1])
            if c1.button("Toggle star", key=f"star_{entry.id}"):
                store.toggle_star(entry.id)
                st.rerun()
            if c2.button("Delete", key=f"del_{entry.id}"):
                store.delete(entry.id)
                st.rerun()


def build_citations(ranked: list[dict]) -> list[dict]:
    citations = []
    for i, hit in enumerate(ranked, start=1):
        meta = hit.get("metadata", {})
        citations.append(
            {
                "marker": i,
                "source": Path(meta.get("source_path", "unknown")).name,
                "page": meta.get("page_number"),
                "snippet": hit["document"][:300],
                "score": hit.get("rerank_score", hit.get("rrf_score", 0.0)),
            }
        )
    return citations


def render_citations(citations: list[dict]) -> None:
    with st.expander(f"📎 Sources ({len(citations)})"):
        for c in citations:
            page_label = f", page {c['page']}" if c.get("page") and c["page"] > 0 else ""
            st.markdown(
                f"""<div class="rag-citation-card">
                    <span class="rag-citation-badge">{c['marker']}</span>
                    <strong>{c['source']}{page_label}</strong>
                    <span style="color:var(--text-lo); font-size:12px;"> · score {c['score']:.3f}</span>
                    <div style="color:var(--text-lo); font-size:13px; margin-top:4px;">{c['snippet']}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------- #
def render_stat_row() -> None:
    registry = get_registry()
    companies = registry.list_companies()
    total_pdfs = sum(c.total_pdfs for c in companies)
    total_chunks = sum(c.total_chunks for c in companies)
    total_companies = len(companies)

    store = get_chat_store()
    query_volume = sum(len(store.list_for_company(c.company_id, limit=100000)) for c in companies)

    cards = [
        ("📄", "PDFs indexed", f"{total_pdfs:,}", "cyan"),
        ("🧩", "Total chunks", f"{total_chunks:,}", "pink"),
        ("🏢", "Companies", f"{total_companies:,}", "purple"),
        ("🔍", "Query volume", f"{query_volume:,}", "amber"),
    ]
    html = '<div class="rag-stat-row">'
    for icon, label, value, color in cards:
        html += (
            f'<div class="rag-stat-card {color}"><div class="icon">{icon}</div>'
            f'<div class="label">{label}</div><div class="value">{value}</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------- #
def main() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    selected_id = render_sidebar()

    if selected_id is None:
        st.title("📚 Local-First Multi-Tenant RAG Platform")
        st.markdown(
            """
            Register a company in the sidebar to get started. Each company gets:
            - Its own isolated data directory and vector store (zero cross-tenant leakage)
            - Local-first PDF ingestion (SHA-256 dedup, resumable)
            - Hybrid search (dense BGE embeddings + BM25, fused with RRF) + cross-encoder reranking
            - Optional LLM answer generation with page-accurate citations, via Ollama or Groq
            - Persistent chat history, cross-company comparison, and voice input (Groq Whisper)
            """
        )
        return

    registry = get_registry()
    record = registry.get(selected_id)
    st.title(f"💬 {record.name}")
    render_stat_row()

    tab_chat, tab_compare, tab_history = st.tabs(["💬 Chat", "🔍 Compare Companies", "📜 History"])
    with tab_chat:
        render_chat(selected_id)
    with tab_compare:
        render_compare()
    with tab_history:
        render_history(selected_id)


if __name__ == "__main__":
    main()
