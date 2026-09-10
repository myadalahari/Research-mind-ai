# ResearchMind AI — frontend production image
#
# Deliberately separate from backend/Dockerfile, mirroring the project's
# standing "frontend and backend are independently deployable, no shared
# imports" rule (see docs/architecture-decisions.md, Phase 9 ADR): this
# image is built from its own build context (frontend/), has its own
# .dockerignore, and shares nothing with the backend image beyond both
# starting from the same base Python version for consistency.
#
# Lightweight by construction, not by Dockerfile trickery: frontend/requirements.txt
# has exactly four dependencies (streamlit, requests, pydantic,
# pydantic-settings) -- none of them pull in torch, chromadb, or any
# compiled ML extension, so there is no CPU-only-torch concern here the way
# there is for the backend image, and no benefit to a multi-stage build.
FROM python:3.11-slim

# curl is not part of python:3.11-slim by default -- installed only for the
# HEALTHCHECK below (`curl -f` against Streamlit's own liveness endpoint),
# same justification/cleanup pattern as backend/Dockerfile's own apt-get step.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# core/, state/, ui/, pages/, app.py -- everything streamlit needs to run
# this multipage app. Excludes pyproject.toml/requirements-dev.txt via
# .dockerignore (dev-only tooling, never needed at runtime).
COPY app.py .
COPY core ./core
COPY state ./state
COPY ui ./ui
COPY pages ./pages

RUN groupadd --system researchmind \
    && useradd --system --gid researchmind --home-dir /app --no-create-home researchmind \
    && chown -R researchmind:researchmind /app

USER researchmind

# RESEARCHMIND_API_BASE_URL is overridden by docker-compose.yml to
# http://backend:8000/api/v1 (the backend's in-network service name, not
# localhost) -- this ENV default only applies if the image is ever run
# standalone, outside compose, against a backend on the same host.
ENV RESEARCHMIND_API_BASE_URL=http://localhost:8000/api/v1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

# Streamlit ships its own liveness endpoint (`_stcore/health`, part of the
# framework itself, not a route this project owns) -- reused here rather
# than building a bespoke one, since the frontend has no external
# dependencies of its own to report on beyond "the Streamlit process is
# serving requests" (its one real dependency, the backend API, is already
# monitored by the backend's own /health, not duplicated here).
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
