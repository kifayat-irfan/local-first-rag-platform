# syntax=docker/dockerfile:1

# --- Builder: install dependencies into a venv, kept separate from runtime ---
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --upgrade pip \
    && /venv/bin/pip install --no-cache-dir -r requirements.txt

# --- Runtime: copy only the venv + source, no build toolchain ---
FROM python:3.11-slim AS runtime

# libgomp1 is required by onnxruntime (a chromadb dependency); the rest of
# the stack is pure Python + prebuilt wheels, so nothing else is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /app
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY streamlit_app.py .

# data/ is a mount point, not baked into the image — see docker-compose.yml.
RUN mkdir -p /app/data/companies

ENV PYTHONPATH=/app/src
ENV RAG_OLLAMA_BASE_URL=http://ollama:11434

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
