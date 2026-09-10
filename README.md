ResearchMind AI

An enterprise-style, multi-agent research assistant. ResearchMind AI takes a research question, plans an investigation, gathers evidence from your own documents and the live web, drafts a cited report, fact-checks and reviews its own output, and hands back a polished, sourced answer — with a downloadable Markdown or PDF report on request.

It is built as a small but production-shaped system: a FastAPI backend orchestrating an eight-agent LangGraph workflow, a Streamlit frontend, a local-first Retrieval-Augmented Generation (RAG) pipeline, and a Docker Compose stack that runs the whole thing — including the LLM — entirely on your own machine.

Table of Contents
Overview
Key Features
Architecture
Tech Stack
Project Structure
Getting Started
Run with Docker Compose (recommended)
Run without Docker
Configuration
API Overview
Example Usage
Documentation
Roadmap
Overview

ResearchMind AI is organized around a single idea: a research question deserves more than one pass through a language model. Instead of a single prompt-and-response, a query is routed through a coordinated team of specialized agents — each responsible for one step of the research process — before a final, reviewed answer reaches the user, drawing on live web search today and designed from the ground up for retrieval over your own documents as that pipeline is completed (see Roadmap).

The system supports two workflow modes:

Research mode — the full multi-agent pipeline: planning, retrieval, web search, synthesis, writing, fact-checking, and review.
Chat mode — a lighter-weight conversational mode for quick, direct exchanges that don't need the full pipeline.

Generated research can also be exported as a structured report (Markdown or PDF) for sharing outside the application.

Key Features
Eight-agent research pipeline — Coordinator, Planner, Retriever, Search, Researcher, Writer, Fact Checker, and Reviewer agents, orchestrated as a LangGraph StateGraph with conditional routing and a bounded revision loop.
Retrieval-Augmented Generation (RAG) pipeline — document extraction (PDF, DOCX, TXT, Markdown), chunking, embedding, and vector retrieval are implemented end-to-end (app/rag), so retrieved context can be woven into a response alongside live web results. The HTTP upload endpoint that feeds documents into this pipeline is scoped for a later phase (see Roadmap).
Live web search — optional, pluggable web search (via Tavily) that the Planner can invoke when a query needs current information beyond the local knowledge base.
Self-review loop — every draft passes through an automated Fact Checker and Reviewer before being finalized, with a capped retry loop back to the Writer to prevent runaway cycles.
Report export — generate a Markdown or PDF report from any research turn and download it via the API.
Local-first LLM — runs against a self-hosted Ollama model (Qwen3) by default, with provider interfaces already in place for OpenAI, Anthropic, and Azure OpenAI.
Pluggable providers throughout — LLM, embeddings, vector store, and search are all defined behind interfaces (app.core.interfaces), so swapping an implementation is a configuration change, not a code change.
Structured, observable by design — structured JSON logging, request/trace IDs, per-dependency health checks, and an optional step-by-step agent execution trace surfaced in the UI.
Feature flags — web search, RAG, memory, report export, and agent tracing can each be toggled independently and are designed to degrade gracefully when disabled.
Architecture
                    ┌───────────────────────────┐
                    │    Streamlit Frontend     │
                    │   (Chat & Report pages)   │
                    └───────────────────────────┘
                                  │ HTTP (REST)
                    ┌───────────────────────────┐
                    │      FastAPI Backend      │
                    │  /api/v1/chat, /report,   │
                    │          /health          │
                    └───────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌──────────▼──────────┐   ┌──────────▼──────────┐   ┌──────────▼──────────┐
│      LangGraph      │   │    RAG Pipeline     │   │     Persistence     │
│   Agent Workflow    │   │ (Chroma + Sentence  │   │     (SQLite via     │
│                     │   │    Transformers)    │   │     SQLAlchemy)     │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘

START → (mode == CHAT) → coordinator_chat → END
START → planner → [retriever?] [search?] → researcher → writer
        → fact_checker → reviewer
              ├─ approved → coordinator_finalize → END
              ├─ rejected, retries remain → writer (revision loop)
              └─ retries exhausted → error

Agent responsibilities

Agent	Role
Coordinator	Entry/exit point; routes chat-mode turns directly, and finalizes approved research turns.
Planner	Breaks the query into a research plan and decides whether retrieval, web search, or both are needed.
Retriever	Pulls relevant chunks from the vector store (uploaded documents).
Search	Queries the web search provider for current, external information.
Researcher	Synthesizes retrieved and searched material into structured findings.
Writer	Drafts the response/report from the research findings.
Fact Checker	Verifies claims in the draft against the gathered sources.
Reviewer	Approves the draft or sends it back to the Writer for revision (bounded by AGENT__MAX_REVIEWER_RETRIES).

Three independently deployable services make up the running system: Ollama (self-hosted LLM runtime), the FastAPI backend, and the Streamlit frontend — wired together by Docker Compose on a shared network.

Tech Stack
Layer	Technology
Backend framework	FastAPI, Uvicorn
Agent orchestration	LangGraph
LLM runtime	Ollama (Qwen3) — pluggable OpenAI / Anthropic / Azure OpenAI
Embeddings	Sentence Transformers (all-MiniLM-L6-v2)
Vector store	ChromaDB
Web search	Tavily
Database	SQLite (via SQLAlchemy, async) — swappable for Postgres with no code changes
Document ingestion	pypdf, python-docx
Report export	ReportLab (PDF), Markdown
Frontend	Streamlit
Validation/config	Pydantic v2, pydantic-settings
Containerization	Docker, Docker Compose
Tooling	Black, Ruff, mypy
Project Structure
researchmind-ai/
├── backend/
│   ├── app/
│   │   ├── agents/                # The 8 LangGraph agents + graph assembly
│   │   ├── api/                   # Routes (chat, report, health) & middleware
│   │   ├── core/                  # Settings, DI, logging, exceptions, interfaces
│   │   ├── database/              # Async session management, repositories
│   │   ├── memory/                # Conversation memory & compaction
│   │   ├── models/                # SQLAlchemy models & persistence-layer enums
│   │   ├── rag/                   # Chunking, extraction, embeddings, retrieval
│   │   ├── reports/               # Report building & Markdown/PDF exporters
│   │   ├── schemas/               # Pydantic request/response contracts
│   │   ├── services/              # Chat, report, health, history, memory services
│   │   └── main.py                # Application composition root
│   ├── data/                      # SQLite DB, Chroma persistence, generated reports
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py                     # Chat / Research page (entry point)
│   ├── pages/1_Report.py          # Report generation page
│   ├── core/                      # API client & shared models
│   ├── state/                     # Session state management
│   ├── ui/                        # Reusable rendering components
│   ├── Dockerfile
│   └── requirements.txt
├── docs/
│   ├── architecture-decisions.md  # Full ADR log
│   └── docker-deployment.md       # Deployment walkthrough
├── docker-compose.yml             # Production-safe baseline
├── docker-compose.override.yml    # Dev conveniences (auto-merged)
└── .env.example
Getting Started
Run with Docker Compose (recommended)

Prerequisites

Docker Engine with the Compose plugin (docker compose version)
~4 GB free disk space (for the Qwen3 model and the backend's embedding model)

1. Configure environment variables

bash
cp .env.example .env
# edit .env if needed — no value is strictly required for a default local run
# unless FEATURES__ENABLE_WEB_SEARCH=true, which needs SEARCH__TAVILY_API_KEY

2. Build and start the stack

bash
docker compose up --build

This starts three services — ollama, backend, and frontend — on a shared Compose network. docker-compose.override.yml is auto-merged for local development (live source mounts, reload). For a production-like run without dev conveniences:

bash
docker compose -f docker-compose.yml up --build -d

3. Pull the LLM model (one-time step)

The Ollama container starts empty — the model isn't baked into the image, so it's pulled once into the persistent ollama_data volume:

bash
docker exec -it researchmind-ollama ollama pull qwen3

This only needs to be run once per volume; the model persists across docker compose down/up cycles (as long as the volume isn't removed with -v).

4. Open the app

Service	URL
Frontend (Streamlit)	http://localhost:8501
Backend API	http://localhost:8000/api/v1
Health check	http://localhost:8000/api/v1/health

See docs/docker-deployment.md for the full deployment walkthrough.

Run without Docker

Each service can also be run directly for development:

bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (in a separate terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py

You'll need a locally running Ollama instance (or another configured LLM provider) reachable at the URL set in LLM__OLLAMA__BASE_URL.

Configuration

All configuration is environment-variable driven (see backend/app/core/config.py), grouped by concern, with __ as the nesting delimiter — e.g. LLM__OLLAMA__BASE_URL maps to Settings.llm.ollama.base_url. Copy .env.example to .env as a starting point. Key groups:

Group	Controls
APP__*	Application metadata, host/port, CORS origins
LLM__*	Provider selection (Ollama / OpenAI / Anthropic / Azure OpenAI) and per-provider settings
EMBEDDING__*	Sentence Transformers model, device, batching
VECTOR_STORE__*	ChromaDB persistence and collection settings
SEARCH__*	Web search provider (Tavily) and API key
DATABASE__*	Connection URL (SQLite by default; Postgres-ready)
RAG__*	Chunking, retrieval top-k, upload limits
AGENT__*	Reviewer retry cap, per-agent timeout, chat history depth
REPORT__*	Output directory, default export format
FEATURES__*	Feature flags for web search, RAG, memory, report export, agent tracing

Every setting has a safe default except SEARCH__TAVILY_API_KEY, which is only required if FEATURES__ENABLE_WEB_SEARCH=true.

API Overview

All routes are mounted under APP__API_PREFIX (default /api/v1).

Method	Path	Description
POST	/chat	Send a chat or research request; returns the agent response (and, optionally, the execution trace).
POST	/report	Generate a research report from a completed session.
GET	/report/{report_id}/download	Stream the exported report file (Markdown or PDF).
GET	/health	Liveness/readiness check, including per-dependency status.
Example Usage

The examples below are the actual request/response schemas declared in the codebase (app/schemas/*.py), not illustrative pseudo-JSON.

Send a research query — POST /api/v1/chat

Request

json
{
  "query": "Compare Llama 4 and Qwen3 for enterprise deployment",
  "session_id": null,
  "mode": "research",
  "stream": false,
  "retrieval_options": { "session_scope": true, "top_k": 6 },
  "conversation_title": null
}

Response — 200 OK

json
{
  "data": {
    "session_id": "sess-7f3a1c9d",
    "message_id": "msg-4e5b4a1e",
    "mode": "research",
    "answer": "Qwen3 and Llama 4 differ primarily in...[1][2]",
    "citations": [],
    "conversation_title": "Llama 4 vs Qwen3 comparison",
    "follow_up_suggestions": [
      "How do their licensing terms compare?",
      "Which performs better on long-context tasks?"
    ],
    "research_metadata": {
      "llm_provider": "ollama",
      "llm_model": "qwen3",
      "rag_enabled": true,
      "web_search_enabled": true,
      "retrieved_chunk_count": 6,
      "web_result_count": 4,
      "documents_considered": ["doc-42"],
      "total_latency_ms": 4820.5,
      "token_usage": { "prompt_tokens": 3120, "completion_tokens": 812, "total_tokens": 3932 }
    },
    "execution_trace": null,
    "created_at": "2026-07-29T10:15:03Z"
  }
}

mode also accepts "chat" for lightweight conversational turns that don't need the full agent pipeline.

Generate a report — POST /api/v1/report

Request

json
{
  "query": "Create a report on Quantum Computing",
  "session_id": null,
  "retrieval_options": { "session_scope": true, "top_k": 8 },
  "export_format": "pdf",
  "conversation_title": "Quantum Computing Overview"
}

Response — 200 OK

json
{
  "data": {
    "report_id": "rpt-9c4e5b4a",
    "session_id": "sess-7f3a1c9d",
    "status": "completed",
    "export_format": "pdf",
    "download_url": "/api/v1/report/rpt-9c4e5b4a/download",
    "file_size_bytes": 184320,
    "generation_latency_ms": 18420.0,
    "token_usage": { "prompt_tokens": 9820, "completion_tokens": 2140, "total_tokens": 11960 },
    "created_at": "2026-07-29T10:15:03Z"
  }
}

Fetch the finished file with GET /api/v1/report/rpt-9c4e5b4a/download.

Check system health — GET /api/v1/health

Response — 200 OK

json
{
  "data": {
    "status": "healthy",
    "live": true,
    "ready": true,
    "app_version": "0.1.0",
    "environment": "local",
    "started_at": "2026-07-29T08:00:00Z",
    "uptime_seconds": 8103.5,
    "checked_at": "2026-07-29T10:15:03Z",
    "llm": { "status": "healthy", "provider": "ollama", "model": "qwen3", "latency_ms": 12.4 },
    "vector_store": {
      "status": "healthy",
      "provider": "chroma",
      "collection_name": "researchmind_documents",
      "embedding_model": "all-MiniLM-L6-v2",
      "document_count": 1248,
      "latency_ms": 3.1
    },
    "search_provider": { "status": "healthy", "provider": "tavily", "enabled": true, "latency_ms": 210.8 },
    "database": { "status": "healthy", "provider": "sqlite", "latency_ms": 0.9 }
  }
}

status/ready roll up to "unhealthy" if a critical dependency (LLM, vector store, or database) is down; a disabled or unreachable web search provider only degrades the result, since it isn't on the critical path for every request.

Documentation
docs/architecture-decisions.md — the full architecture decision record (ADR) log, covering the reasoning behind every major design and dependency choice made throughout the project's build phases.
docs/docker-deployment.md — step-by-step Docker Compose deployment guide.
Roadmap

The following are scoped but not yet built:

Document upload endpoint (POST /upload) — the schemas (app/schemas/upload.py) and the full ingestion pipeline (app/rag/ingest.py) already exist; the route wiring it to the API is the remaining piece.
Conversation history endpoint (GET /history) — the service layer (app/services/history_service.py) exists; the route is not yet exposed.
