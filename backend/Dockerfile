# ResearchMind AI — backend production image
#
# Single-stage build (see docs/architecture-decisions.md's Phase 10 ADR for
# why a multi-stage build was considered and rejected: this project has no
# compiled-extension build step of its own that would benefit from a
# separate builder stage — the only "build" work is `pip install`, and pip
# wheels for every dependency here (fastapi, chromadb, sentence-transformers,
# torch-cpu, reportlab, ...) are already prebuilt, so a second stage would
# only add Dockerfile complexity without shrinking the final image).
#
# Base image: slim, not alpine. `sentence-transformers`/`torch`/`chromadb`
# pull in compiled extensions (numpy, tokenizers, onnxruntime-adjacent libs
# via chromadb) that ship musl-incompatible manylinux wheels — alpine would
# force building several of these from source, trading a larger, slower,
# less-tested build for a marginally smaller image. `python:3.11-slim`
# matches this project's own pinned interpreter (see pyproject.toml's
# `target-version = ["py311"]`) on a glibc base, so every wheel already
# validated in development installs the same prebuilt way in the image.
FROM python:3.11-slim

# System packages needed only to *build* a couple of dependencies' native
# extensions (chromadb's own transitive deps occasionally need this) --
# removed from the apt cache in the same layer to avoid bloating the image
# with package-index metadata this image never needs again after install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Dependency installation -------------------------------------------
#
# CPU-only PyTorch, installed explicitly and *before* requirements.txt:
# `sentence-transformers==5.6.1` only declares `torch>=1.11.0` (a loose
# transitive constraint), so if pip resolves torch while installing
# requirements.txt it pulls PyPI's default wheel, which bundles the full
# CUDA runtime (nvidia-cublas, nvidia-cudnn, nvidia-cufft, ...) -- 1.5+ GB
# of libraries this project never uses, since EMBEDDING__DEVICE=cpu is the
# only supported/configured device end to end. Installing the CPU-only
# wheel from PyTorch's own index first means pip's resolver sees `torch`
# already satisfied by the time it processes requirements.txt's
# `sentence-transformers` line, and never reaches for the CUDA build.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# --- Bake the embedding model into the image (Decision B) ---------------
#
# Downloads and caches `all-MiniLM-L6-v2` (EMBEDDING__MODEL_NAME's default,
# see .env.example) into the image's Hugging Face cache at build time, so
# container startup never depends on network reachability to
# huggingface.co -- it loads instantly from the local cache baked into this
# layer instead. This is the same model `SentenceTransformerEmbeddingProvider`
# (app/rag/embedding_provider.py) constructs at DI-composition time
# (app.core.dependencies.get_embedding_provider), so baking it in here
# guarantees the exact model the running container will use is already
# present -- there is no separate "warm the cache later" step that could
# drift from what's actually configured.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# --- Application code -----------------------------------------------------
COPY app ./app

# --- Runtime directories & non-root user ----------------------------------
#
# Matches Settings.ensure_runtime_directories()'s own list (vector store
# persist dir, report output dir, sqlite parent dir) plus the unified
# /app/data mount point (Decision D) -- created here so the non-root user
# below owns them from the start rather than needing a runtime chown.
# REPORT__OUTPUT_DIR is reconfigured to live under /app/data via the
# environment (docker-compose.yml sets REPORT__OUTPUT_DIR=/app/data/reports),
# not via a code change -- ensure_runtime_directories() already creates
# whatever path is configured, so this is a deployment-time config choice,
# not an application change.
RUN mkdir -p /app/data/chroma /app/data/reports /app/data/cache \
    && groupadd --system researchmind \
    && useradd --system --gid researchmind --home-dir /app --no-create-home researchmind \
    && chown -R researchmind:researchmind /app

USER researchmind

# Runtime configuration defaults appropriate for a container -- all
# overridable via docker-compose's `environment:`/`env_file:`, per this
# project's Settings.__init__ subclass -> BaseSettings env-var precedence.
# Not baked as ENV values that shadow docker-compose's own environment,
# except where a container-specific default genuinely differs from
# .env.example's local-dev default (the three below).
ENV APP__API_HOST=0.0.0.0 \
    APP__API_PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Uses the /health route built earlier in Phase 10 -- reflects readiness via
# HTTP status code (503 when not ready), so `curl -f` alone (no body
# parsing) is enough for Docker to fail the healthcheck on real
# unreadiness, not just process death.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
