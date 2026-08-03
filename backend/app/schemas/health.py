"""
Request/response schemas for ``GET /health``.

This schema stays provider-agnostic by design: the Phase 4 router
populates ``LLMHealthInfo``/``VectorStoreHealthInfo``/
``SearchProviderHealthInfo``/``DatabaseHealthInfo`` by calling each
injected dependency's own ``health_check()`` method — defined on every
``core.interfaces`` ABC (``LLMService.health_check()``,
``VectorStore.health_check()``, ``SearchProvider.health_check()``, and an
analogous check for the database session) — never by importing a concrete
adapter (Ollama, Chroma, Tavily) directly. Swapping any provider later
changes nothing about this schema or how the health endpoint is wired.

Liveness and readiness are modeled as distinct booleans on one response
rather than two separate response types, so this single ``GET /health``
endpoint (per the project's endpoint list) already carries what a future
container-orchestration split into ``/health/live`` and ``/health/ready``
would need — that split, if ever added, would read straight off these same
fields rather than requiring a schema change.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Environment
from app.utils.time import utc_now

# =============================================================================
# Status
# =============================================================================


class ServiceStatus(str, Enum):
    """
    Health status of the service or a single dependency.

    * ``healthy`` — fully operational.
    * ``degraded`` — operational but with reduced functionality (e.g. web
      search is down but chat/RAG still work, since search is not on the
      critical path for every request).
    * ``unhealthy`` — not operational; requests depending on this
      component will fail.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# =============================================================================
# Per-dependency health
# =============================================================================


class DependencyHealth(BaseModel):
    """
    Shared fields for a single dependency's health check result.

    Specialized by each concrete dependency's info model below rather than
    used directly, so provider-specific metadata (model name, collection
    name, ...) can sit alongside the common status/latency/message fields
    without a generic untyped ``details`` bag.
    """

    status: ServiceStatus = Field(..., description="Health status of this dependency.")
    latency_ms: Optional[float] = Field(
        default=None, ge=0, description="Time taken to perform this health check, in milliseconds."
    )
    message: Optional[str] = Field(
        default=None, description="Human-readable detail, populated when status is not 'healthy'."
    )
    checked_at: datetime = Field(default_factory=utc_now, description="When this dependency was last checked.")


class LLMHealthInfo(DependencyHealth):
    """Health and identity of the configured LLM backend."""

    provider: str = Field(..., description="LLM provider in use, e.g. 'ollama'.", examples=["ollama"])
    model: str = Field(..., description="Concrete model identifier, e.g. 'qwen3'.", examples=["qwen3"])


class VectorStoreHealthInfo(DependencyHealth):
    """Health and metadata of the configured vector store."""

    provider: str = Field(..., description="Vector store provider in use, e.g. 'chroma'.")
    collection_name: str = Field(..., description="Active collection name.")
    embedding_model: str = Field(..., description="Embedding model used to populate this collection.")
    document_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of indexed chunks, if available. Omitted when counting would be too expensive for a lightweight check.",
    )


class SearchProviderHealthInfo(DependencyHealth):
    """Health and configuration of the configured web search provider."""

    provider: str = Field(..., description="Search provider in use, e.g. 'tavily'.")
    enabled: bool = Field(
        ...,
        description="Whether web search is enabled via FEATURES__ENABLE_WEB_SEARCH. When false, status reflects configuration, not reachability.",
    )


class DatabaseHealthInfo(DependencyHealth):
    """Health of the relational database backing memory/history/reports metadata."""

    provider: str = Field(..., description="Database dialect in use, e.g. 'sqlite' or 'postgresql'.")


# =============================================================================
# Build metadata
# =============================================================================


class BuildInfo(BaseModel):
    """Optional build/deployment metadata, populated when available in the runtime environment."""

    git_commit: Optional[str] = Field(
        default=None, description="Short git commit SHA the running image was built from."
    )
    git_branch: Optional[str] = Field(default=None, description="Git branch the running image was built from.")
    build_time: Optional[datetime] = Field(default=None, description="When the running image was built.")


# =============================================================================
# Response
# =============================================================================


class HealthCheckResponse(BaseModel):
    """Response payload for ``GET /health``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "healthy",
                    "live": True,
                    "ready": True,
                    "app_version": "0.1.0",
                    "environment": "local",
                    "build_info": {"git_commit": "a1b2c3d", "git_branch": "main", "build_time": "2026-07-28T09:00:00Z"},
                    "started_at": "2026-07-29T08:00:00Z",
                    "uptime_seconds": 8103.5,
                    "checked_at": "2026-07-29T10:15:03Z",
                    "llm": {"status": "healthy", "provider": "ollama", "model": "qwen3", "latency_ms": 12.4},
                    "vector_store": {
                        "status": "healthy",
                        "provider": "chroma",
                        "collection_name": "researchmind_documents",
                        "embedding_model": "all-MiniLM-L6-v2",
                        "document_count": 1248,
                        "latency_ms": 3.1,
                    },
                    "search_provider": {
                        "status": "healthy",
                        "provider": "tavily",
                        "enabled": True,
                        "latency_ms": 210.8,
                    },
                    "database": {"status": "healthy", "provider": "sqlite", "latency_ms": 0.9},
                }
            ]
        }
    )

    status: ServiceStatus = Field(..., description="Overall service health, rolled up from all checked dependencies.")
    live: bool = Field(
        ...,
        description="Liveness: whether the process itself is running and able to respond. True whenever this endpoint returns at all.",
    )
    ready: bool = Field(
        ...,
        description="Readiness: whether the service is able to serve requests. False when a critical dependency (LLM, vector store, database) is unhealthy.",
    )
    app_version: str = Field(..., description="Application version (AppSettings.version).")
    environment: Environment = Field(..., description="Deployment environment this instance is running in.")
    build_info: Optional[BuildInfo] = Field(default=None, description="Build/deployment metadata, if available.")
    started_at: datetime = Field(..., description="When this process started.")
    uptime_seconds: float = Field(..., ge=0, description="Seconds since process startup.")
    checked_at: datetime = Field(default_factory=utc_now, description="When this health check was performed.")
    llm: LLMHealthInfo = Field(..., description="LLM backend health.")
    vector_store: VectorStoreHealthInfo = Field(..., description="Vector store health.")
    search_provider: SearchProviderHealthInfo = Field(..., description="Web search provider health.")
    database: DatabaseHealthInfo = Field(..., description="Database health.")

    @classmethod
    def assemble(
        cls,
        *,
        app_version: str,
        environment: Environment,
        started_at: datetime,
        llm: LLMHealthInfo,
        vector_store: VectorStoreHealthInfo,
        search_provider: SearchProviderHealthInfo,
        database: DatabaseHealthInfo,
        build_info: Optional[BuildInfo] = None,
        checked_at: Optional[datetime] = None,
    ) -> "HealthCheckResponse":
        """
        Build a ``HealthCheckResponse`` from individual dependency checks,
        computing the overall ``status``/``ready`` rollup consistently
        rather than leaving the router to reimplement that logic.

        ``started_at`` must be timezone-aware (capture it once at process
        startup with ``app.utils.time.utc_now()`` and store it, e.g. on
        ``app.state``) — it's diffed against ``checked_at``/``utc_now()``
        to compute ``uptime_seconds``, and mixing a naive datetime into
        that subtraction raises ``TypeError`` rather than silently
        producing a wrong value.

        Rollup rule: the overall status is the worst status among LLM,
        vector store, and database (these are always on the critical
        path); the search provider only counts toward the rollup when
        ``search_provider.enabled`` is true, since a disabled search
        provider being unreachable is expected configuration, not a
        degradation. ``ready`` is true unless the overall status is
        ``unhealthy`` — a ``degraded`` service (e.g. web search down) is
        still considered ready to serve the requests it can handle.
        """
        resolved_checked_at = checked_at or utc_now()
        uptime_seconds = max(0.0, (resolved_checked_at - started_at).total_seconds())

        critical_statuses: List[ServiceStatus] = [llm.status, vector_store.status, database.status]
        if search_provider.enabled:
            critical_statuses.append(search_provider.status)

        if ServiceStatus.UNHEALTHY in critical_statuses:
            overall_status = ServiceStatus.UNHEALTHY
        elif ServiceStatus.DEGRADED in critical_statuses:
            overall_status = ServiceStatus.DEGRADED
        else:
            overall_status = ServiceStatus.HEALTHY

        return cls(
            status=overall_status,
            live=True,
            ready=overall_status != ServiceStatus.UNHEALTHY,
            app_version=app_version,
            environment=environment,
            build_info=build_info,
            started_at=started_at,
            uptime_seconds=uptime_seconds,
            checked_at=resolved_checked_at,
            llm=llm,
            vector_store=vector_store,
            search_provider=search_provider,
            database=database,
        )
