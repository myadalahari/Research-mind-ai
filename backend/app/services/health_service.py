"""
HealthService — assembles the ``GET /health`` response.

Framework-agnostic: this module has no FastAPI import anywhere. It depends
only on ``app.core.interfaces`` abstractions (never a concrete Ollama/
Chroma/Tavily/Sentence-Transformers client) plus ``Settings``, so the
router that eventually calls this service stays a thin adapter — parse the
request, call ``HealthService.check_health()``, return the result.

Every dependency's health is checked through the same ``health_check()``
contract each ``core.interfaces`` ABC already defines (``LLMService``,
``VectorStore``, ``SearchProvider``, ``EmbeddingProvider``) — this service
adds no provider-specific logic of its own; swapping Ollama for OpenAI, or
Chroma for a hosted vector store, changes nothing here.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from app.core.config import LLMProvider, Settings
from app.core.interfaces.embedding_provider import EmbeddingProvider
from app.core.interfaces.llm_service import LLMService
from app.core.interfaces.search_provider import SearchProvider
from app.core.interfaces.vector_store import VectorStore
from app.core.logging import bind_context, get_logger, measure_latency_ms
from app.schemas.health import (
    BuildInfo,
    DatabaseHealthInfo,
    HealthCheckResponse,
    LLMHealthInfo,
    SearchProviderHealthInfo,
    ServiceStatus,
    VectorStoreHealthInfo,
)

logger = get_logger(__name__)


@runtime_checkable
class DatabaseHealthChecker(Protocol):
    """
    Minimal structural contract for checking database reachability.

    Defined locally rather than in ``core.interfaces`` because the
    database/repository layer doesn't exist yet — this ``Protocol`` is the
    seam ``HealthService`` depends on today. Whatever session-backed
    implementation is built alongside ``database/session.py`` only needs
    to expose an async ``health_check() -> bool`` method (typically
    "execute ``SELECT 1``") to satisfy this contract and be injected here;
    ``HealthService`` never imports SQLAlchemy or any specific database
    client directly. Using ``Protocol`` (structural typing) rather than an
    ABC means the eventual implementation doesn't even need to inherit
    from this type explicitly — it just needs the matching method.
    """

    async def health_check(self) -> bool: ...


class HealthService:
    """
    Aggregates health/readiness information across every external
    dependency ResearchMind AI relies on.

    All dependency checks run concurrently (not sequentially) so a slow or
    hanging dependency doesn't multiply ``GET /health``'s latency by the
    number of dependencies checked.
    """

    def __init__(
        self,
        *,
        llm_service: LLMService,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        search_provider: SearchProvider,
        database_health_checker: DatabaseHealthChecker,
        settings: Settings,
        started_at: datetime,
    ) -> None:
        """
        Args:
            started_at: Process startup time, timezone-aware (captured
                once via ``app.utils.time.utc_now()`` at application
                startup and passed in — not read from ``settings``, since
                it's runtime process state, not configuration).
        """
        self._llm_service = llm_service
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._search_provider = search_provider
        self._database_health_checker = database_health_checker
        self._settings = settings
        self._started_at = started_at
        self._build_info = _read_build_info()

    async def check_health(self) -> HealthCheckResponse:
        """
        Run every dependency health check concurrently and assemble the
        overall ``HealthCheckResponse``, including the status/readiness
        rollup (delegated to ``HealthCheckResponse.assemble``, which owns
        that rollup logic — this method's job is purely gathering the
        per-dependency results).
        """
        llm_health, vector_store_health, search_health, database_health = await asyncio.gather(
            self._check_llm(),
            self._check_vector_store(),
            self._check_search_provider(),
            self._check_database(),
        )

        response = HealthCheckResponse.assemble(
            app_version=self._settings.app.version,
            environment=self._settings.app.environment,
            started_at=self._started_at,
            llm=llm_health,
            vector_store=vector_store_health,
            search_provider=search_health,
            database=database_health,
            build_info=self._build_info,
        )

        with bind_context(overall_status=response.status.value, ready=response.ready):
            logger.info("Health check completed")

        return response

    # ------------------------------------------------------------------ #
    # Per-dependency checks
    # ------------------------------------------------------------------ #

    async def _check_llm(self) -> LLMHealthInfo:
        provider = self._settings.llm.provider.value
        model = _resolve_llm_model_name(self._settings)
        with measure_latency_ms() as elapsed:
            try:
                is_healthy = await self._llm_service.health_check()
                message = None
            except Exception as exc:  # defensive: health_check() should never raise per its interface contract
                is_healthy = False
                message = f"{type(exc).__name__}: {exc}"
        return LLMHealthInfo(
            status=ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY,
            provider=provider,
            model=model,
            latency_ms=round(elapsed(), 2),
            message=message,
        )

    async def _check_vector_store(self) -> VectorStoreHealthInfo:
        """
        Checks both the vector store and the embedding provider, since
        retrieval genuinely requires both to be up — a healthy Chroma
        instance is useless for retrieval if the embedding model that
        turns queries into vectors is unreachable, so this section's
        status reflects retrieval capability as a whole rather than
        Chroma reachability alone.
        """
        with measure_latency_ms() as elapsed:
            try:
                vector_store_ok, embedding_ok = await asyncio.gather(
                    self._vector_store.health_check(),
                    self._embedding_provider.health_check(),
                )
                is_healthy = vector_store_ok and embedding_ok
                if is_healthy:
                    message = None
                elif not vector_store_ok and not embedding_ok:
                    message = "Vector store and embedding provider are both unreachable."
                elif not vector_store_ok:
                    message = "Vector store is unreachable."
                else:
                    message = "Embedding provider is unreachable."
            except Exception as exc:
                is_healthy = False
                message = f"{type(exc).__name__}: {exc}"

        document_count: Optional[int] = None
        if is_healthy:
            try:
                document_count = await self._vector_store.count()
            except Exception:
                # Best-effort only: a failure to count indexed chunks should
                # not itself mark the vector store unhealthy.
                document_count = None

        return VectorStoreHealthInfo(
            status=ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY,
            provider=self._settings.vector_store.provider.value,
            collection_name=self._settings.vector_store.collection_name,
            embedding_model=self._settings.embedding.model_name,
            document_count=document_count,
            latency_ms=round(elapsed(), 2),
            message=message,
        )

    async def _check_search_provider(self) -> SearchProviderHealthInfo:
        enabled = self._settings.features.enable_web_search
        provider = self._settings.search.provider.value

        if not enabled:
            # Skip the network call entirely when search is deliberately
            # disabled (e.g. no Tavily key configured) — an unreachable
            # provider that was never supposed to be called isn't a
            # health problem, it's expected configuration.
            return SearchProviderHealthInfo(
                status=ServiceStatus.HEALTHY,
                provider=provider,
                enabled=False,
                message="Web search is disabled (FEATURES__ENABLE_WEB_SEARCH=false).",
            )

        with measure_latency_ms() as elapsed:
            try:
                is_healthy = await self._search_provider.health_check()
                message = None
            except Exception as exc:
                is_healthy = False
                message = f"{type(exc).__name__}: {exc}"

        return SearchProviderHealthInfo(
            status=ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY,
            provider=provider,
            enabled=True,
            latency_ms=round(elapsed(), 2),
            message=message,
        )

    async def _check_database(self) -> DatabaseHealthInfo:
        provider = _resolve_database_dialect(self._settings)
        with measure_latency_ms() as elapsed:
            try:
                is_healthy = await self._database_health_checker.health_check()
                message = None
            except Exception as exc:
                is_healthy = False
                message = f"{type(exc).__name__}: {exc}"
        return DatabaseHealthInfo(
            status=ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY,
            provider=provider,
            latency_ms=round(elapsed(), 2),
            message=message,
        )


# =============================================================================
# Helpers
# =============================================================================


def _resolve_llm_model_name(settings: Settings) -> str:
    """Resolve the active model identifier for whichever LLM provider is configured."""
    provider = settings.llm.provider
    if provider == LLMProvider.OLLAMA:
        return settings.llm.ollama.model
    if provider == LLMProvider.OPENAI:
        return settings.llm.openai.model
    if provider == LLMProvider.ANTHROPIC:
        return settings.llm.anthropic.model
    if provider == LLMProvider.AZURE_OPENAI:
        return settings.llm.azure_openai.deployment_name or "unknown"
    return "unknown"


def _resolve_database_dialect(settings: Settings) -> str:
    """Extract a short dialect name (e.g. 'sqlite', 'postgresql') from DATABASE__URL."""
    url = settings.database.url
    scheme = url.split("://", 1)[0] if "://" in url else url
    return scheme.split("+", 1)[0]


def _read_build_info() -> Optional[BuildInfo]:
    """
    Read optional build/deployment metadata from environment variables
    (typically injected by CI at image build time: ``GIT_COMMIT``,
    ``GIT_BRANCH``, ``BUILD_TIME``). Returns ``None`` if none are set,
    rather than a ``BuildInfo`` with every field empty.
    """
    git_commit = os.environ.get("GIT_COMMIT") or os.environ.get("GIT_SHA")
    git_branch = os.environ.get("GIT_BRANCH")
    build_time_raw = os.environ.get("BUILD_TIME")

    if not any([git_commit, git_branch, build_time_raw]):
        return None

    build_time: Optional[datetime] = None
    if build_time_raw:
        try:
            build_time = datetime.fromisoformat(build_time_raw.replace("Z", "+00:00"))
        except ValueError:
            build_time = None

    return BuildInfo(git_commit=git_commit, git_branch=git_branch, build_time=build_time)
