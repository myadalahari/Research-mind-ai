"""
Frontend configuration for ResearchMind AI's Streamlit application.

Deliberately small and independent of the backend's ``app.core.config.Settings``
(see ``docs/architecture-decisions.md``'s Phase 9 entry for the full reasoning):
this frontend treats the backend purely as an HTTP API, never as an importable
Python package, so the two can be built, versioned, and deployed as separate
services (relevant once Phase 10 puts them in separate Docker images). The only
backend-related value the frontend needs to know is where the API lives --
everything else (which LLM provider is configured, whether web search is
enabled, retrieval defaults, ...) is the backend's own concern, discovered
per-request from its responses, never hardcoded here.

Environment variables use a flat ``RESEARCHMIND_`` prefix (not the backend's
nested ``APP__``/``LLM__OLLAMA__`` convention) since this file has no nested
setting groups to disambiguate -- see ``.env.example`` at the project root for
the documented list.

This module is run as part of a Streamlit app started via
``streamlit run frontend/app.py``. Streamlit adds the entry script's own
directory (``frontend/``) to ``sys.path``, so every module under ``frontend/``
imports its siblings with a bare top-level name (``from core.config import
get_settings``), not a ``frontend.``-prefixed path.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Streamlit frontend."""

    api_base_url: str = Field(
        default="http://localhost:8000/api/v1",
        description=(
            "Base URL of the ResearchMind AI backend API, including its path prefix "
            "(matches the backend's own APP__API_PREFIX, default '/api/v1'). "
            "Every request api_client.py makes is built by joining an endpoint path onto this."
        ),
    )
    request_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        description=(
            "Timeout for a single backend request. Report generation runs the full multi-agent "
            "pipeline synchronously and can take tens of seconds (see "
            "ReportGenerationResponse.generation_latency_ms's own docstring on the backend) -- "
            "the default here is deliberately generous rather than the short timeout that would "
            "be appropriate for a typical low-latency HTTP call."
        ),
    )

    model_config = SettingsConfigDict(env_prefix="RESEARCHMIND_", extra="ignore")

    @field_validator("api_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        """
        Normalize away a trailing slash so ``api_client.py`` can always join
        paths as ``f"{settings.api_base_url}/chat"`` without risking a
        double slash if a user's env var happens to include one
        (e.g. ``RESEARCHMIND_API_BASE_URL=http://localhost:8000/api/v1/``).
        """
        return value.rstrip("/")


@lru_cache
def get_settings() -> Settings:
    """
    Return the process-wide cached ``Settings`` instance.

    Mirrors the backend's own ``core.dependencies.get_settings`` caching
    pattern: constructed once per process (here, once per Streamlit session
    process) and reused, rather than re-reading environment variables on
    every rerun of the script (Streamlit reruns the whole script top-to-bottom
    on every user interaction, so this matters more here than in a typical
    request-scoped backend call).
    """
    return Settings()
