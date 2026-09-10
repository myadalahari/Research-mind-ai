# Docker Deployment

This document covers running ResearchMind AI's full stack (Ollama, backend,
frontend) with Docker Compose, and the one manual step Compose can't do for
you: pulling the LLM model into the Ollama container (Decision A, see
`docs/architecture-decisions.md`'s Phase 10 entry).

## Prerequisites

* Docker Engine with the Compose plugin (`docker compose version`).
* At least ~4 GB free disk for the `qwen3` model plus the backend image's
  baked-in embedding model.
* A `.env` file in the project root (copy `.env.example` and fill in
  `REQUIRED` values -- for a default local run, no value is strictly
  required unless `FEATURES__ENABLE_WEB_SEARCH=true`, in which case
  `SEARCH__TAVILY_API_KEY` must be set).

```bash
cp .env.example .env
# edit .env as needed
```

## Build and start the stack

```bash
docker compose up --build
```

This builds `researchmind-backend` and `researchmind-frontend` from their
respective `Dockerfile`s and starts all three services (`ollama`, `backend`,
`frontend`) on one Compose-managed network. Because `docker-compose.override.yml`
sits next to `docker-compose.yml`, Compose auto-merges it -- you get live
source bind-mounts and `--reload`/debug settings by default. For a
production-like run without those dev conveniences:

```bash
docker compose -f docker-compose.yml up --build -d
```

## One-time step: pulling the LLM model (Decision A)

The `ollama` service starts an empty Ollama runtime -- it does not ship any
model inside the `ollama/ollama` image, and this project does not build a
custom Ollama image to bake one in (that tradeoff was considered and
rejected in favor of the simpler, standard approach: pull the model once
into the named `ollama_data` volume, where it persists across
`docker compose down`/`up` cycles exactly like any other Ollama installation).

After the stack is up, run:

```bash
docker exec -it researchmind-ollama ollama pull qwen3
```

This only needs to be done once per `ollama_data` volume -- the model
persists on that volume, so subsequent `docker compose up` runs (even after
`down`, as long as the volume isn't removed with `-v`) start with the model
already present. Verify with:

```bash
docker exec -it researchmind-ollama ollama list
```

Until this step is run, `GET /api/v1/health` will report `llm.status:
unhealthy` and the overall response will be `ready: false` (HTTP 503) --
this is expected and matches the health route's documented behavior (see
`backend/app/api/routes/health.py`), not a startup failure.

## Verifying the stack

* Backend health: `curl http://localhost:8000/api/v1/health` -- expect
  HTTP 200 with `"ready": true` once the model is pulled and every
  dependency (Ollama, ChromaDB, SQLite) is reachable.
* Frontend: open `http://localhost:8501` in a browser.
* Backend API docs (FastAPI's built-in Swagger UI): `http://localhost:8000/docs`.

## Stopping and resetting

```bash
docker compose down           # stop containers, keep volumes (data, model)
docker compose down -v        # stop containers AND remove volumes (full reset)
```

## Notes on the two Docker-related architectural decisions baked into the images

* **Decision B -- embedding model baked into the backend image at build
  time.** `backend/Dockerfile` runs the `SentenceTransformer('all-MiniLM-L6-v2')`
  constructor during the image build, caching the model inside the image
  layer. This means the backend container never needs network access to
  `huggingface.co` at runtime -- only at *build* time, when the image is
  built on a machine with normal internet access. (This is also why the
  image cannot be built inside network-sandboxed environments that block
  `huggingface.co`, such as certain CI runners or restricted development
  sandboxes -- build on a host with unrestricted outbound access, or adapt
  this step to use a pre-downloaded model cache mounted at build time.)
* **Decision D -- one unified `/app/data` volume.** The backend container's
  SQLite database, ChromaDB persistence directory, generated reports, and
  optional on-disk cache all live under `/app/data`, backed by the single
  `backend_data` named volume in `docker-compose.yml`. This is a
  configuration choice (`REPORT__OUTPUT_DIR`, `VECTOR_STORE__PERSIST_DIR`,
  `DATABASE__URL`, `CACHE__DIR` are all set via `environment:` in
  `docker-compose.yml`), not a code change -- every one of those paths was
  already configurable via `Settings`, and `ensure_runtime_directories()`
  already creates whatever directories those settings point to.
