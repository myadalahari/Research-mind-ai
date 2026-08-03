"""
Dependency-injection wiring: constructs every singleton adapter/service
this application needs and exposes them as zero-argument, memoized
provider functions.

Every other module written so far refers forward to this file by name --
``app.core.interfaces.llm_service.LLMService``'s own docstring says an
``LLMService`` is "injected via ``app.core.dependencies.get_llm_service``",
and the same pattern repeats across ``get_embedding_provider``,
``get_vector_store``, ``get_search_provider``, and ``core.dependencies.
get_settings``. This file is where those forward references resolve.

Design: every provider function is decorated with ``functools.lru_cache``
(no arguments, so it acts purely as a "compute once, memoize forever"
singleton cache -- the same pattern ``app.core.config.get_settings``
already established) and composes *other* provider functions via plain
Python calls, never ``fastapi.Depends``. This is a deliberate choice, not
an oversight: every service/adapter in this codebase is already
framework-agnostic (no FastAPI import anywhere under ``app.services``,
``app.rag``, ``app.agents``, ``app.database``) specifically so it can be
constructed and tested without a running ASGI app. Wiring this module
around ``Depends(...)`` chains would only work inside an actual FastAPI
request and would force importing ``fastapi`` here for no benefit --
every provider function below needs zero arguments and is trivially
usable both as a plain call (startup scripts, tests) and as a route
parameter default (``Depends(get_chat_service)``, in the not-yet-built API
layer) without changing anything here.

Each dispatch-based provider (``get_llm_service``, ``get_embedding_provider``,
``get_vector_store``, ``get_search_provider`` -- the ones that must choose
a concrete adapter class based on a ``*Settings.provider`` enum) is split
into two layers:

* A pure ``_build_*(settings: ...) -> Interface`` function containing the
  actual dispatch logic, taking its settings object as an explicit
  argument. Independently testable with a hand-built settings instance
  (e.g. pointed at a local fake server), with no dependency on the global
  ``get_settings()`` singleton or any monkeypatched environment state.
* A cached, zero-argument ``get_*() -> Interface`` function that calls the
  ``_build_*`` function with ``get_settings()``'s relevant section --
  this is the one actually referenced by every forward-declaring docstring
  above, and the one a route/startup script uses.

Only ``LLMProvider.OLLAMA``, ``VectorStoreProvider.CHROMA``, and
``SearchProviderName.TAVILY`` have concrete adapters today -- every
``_build_*`` function raises ``ConfigurationError`` (``RM-API-004``) for
any other configured provider, a fail-fast application-wiring concern
distinct from ``Settings``' own field-level validation (which validates
that a provider's *own* configuration is internally consistent, e.g.
credentials present, not that an adapter class exists for it).
``EmbeddingSettings`` has no ``provider`` field at all (there is exactly
one embedding backend, Sentence Transformers) so ``_build_embedding_provider``
has no dispatch to do.

``get_report_exporter`` is this module's one deliberate exception to the
"every provider is zero-argument" rule stated above: it takes an explicit
``export_format: ReportExportFormat`` parameter, because -- unlike every
other adapter here, which is chosen once from configuration for the
whole process -- callers (``ReportService``, and eventually the
``GET /report/{id}/download`` route) need to pick a format per request
(a user can ask for either Markdown or PDF for the same report). Dispatch
is a plain ``if``/``elif`` on the two ``ReportExportFormat`` members, not
a registry or plugin-lookup mechanism -- there are exactly two formats,
both known at compile time, and nothing in this project's scope suggests
a third arriving via anything other than a new ``if`` branch someone adds
by hand. Still ``@lru_cache``-memoized (each of the two formats' exporter
instances is a genuine singleton, just keyed by the extra argument rather
than by nothing).

Deliberately NOT included here: calling
``Settings.ensure_runtime_directories()`` or ``Database.create_all()`` --
both are documented, in their own files, as ``app.main``'s startup-lifecycle
responsibility, not this module's. This file only builds objects; deciding
*when* to create directories, create tables, or register exception
handlers belongs to the application entrypoint.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph.state import CompiledStateGraph

from app.agents.graph import build_research_graph
from app.core.config import (
    EmbeddingSettings,
    LLMProvider,
    LLMSettings,
    ReportExportFormat,
    SearchProviderName,
    SearchSettings,
    VectorStoreProvider,
    VectorStoreSettings,
    get_settings,
)
from app.core.exceptions import ConfigurationError
from app.core.interfaces.embedding_provider import EmbeddingProvider
from app.core.interfaces.llm_service import LLMService
from app.core.interfaces.report_exporter import ReportExporter
from app.core.interfaces.search_provider import SearchProvider
from app.core.interfaces.vector_store import VectorStore
from app.core.logging import get_logger
from app.database.session import Database
from app.rag.embedding_provider import SentenceTransformerEmbeddingProvider
from app.rag.vector_store import ChromaVectorStore
from app.reports.exporters.markdown_exporter import MarkdownReportExporter
from app.reports.exporters.pdf_exporter import PDFReportExporter
from app.services.chat_service import ChatService
from app.services.health_service import HealthService
from app.services.history_service import HistoryService
from app.services.llm.ollama_llm_service import OllamaLLMService
from app.services.memory_service import MemoryService
from app.services.report_service import ReportService
from app.services.search.tavily_search_provider import TavilySearchProvider
from app.utils.time import utc_now

logger = get_logger(__name__)

# Captured once, at import time (i.e. once per process), for HealthService's
# uptime reporting -- mirrors every other "constructed once, reused for the
# process's lifetime" adapter in this module, just as a plain timestamp
# rather than an object.
_PROCESS_STARTED_AT = utc_now()

# Re-exported so `app.core.dependencies.get_settings` (the name every
# other module's docstring already references) resolves to something real,
# without duplicating the singleton or moving it out of `app.core.config`
# (which already owns it, is already relied upon directly by that module's
# own tests, and has no reason to change for this file's sake).
__all__ = [
    "get_settings",
    "get_database",
    "get_llm_service",
    "get_embedding_provider",
    "get_vector_store",
    "get_search_provider",
    "get_research_graph",
    "get_memory_service",
    "get_chat_service",
    "get_history_service",
    "get_health_service",
    "get_report_exporter",
    "get_report_service",
]


# =============================================================================
# Dispatch-based adapters: pure `_build_*` (testable, explicit settings)
# + cached `get_*` (the singleton every caller actually uses)
# =============================================================================


def _build_llm_service(settings: LLMSettings) -> LLMService:
    if settings.provider == LLMProvider.OLLAMA:
        return OllamaLLMService(settings)
    raise ConfigurationError(
        f"LLM__PROVIDER={settings.provider.value!r} has no concrete LLMService implementation yet."
    )


@lru_cache
def get_llm_service() -> LLMService:
    return _build_llm_service(get_settings().llm)


def _build_embedding_provider(settings: EmbeddingSettings) -> EmbeddingProvider:
    # No provider switch: EmbeddingSettings has no `provider` field, since
    # there is exactly one embedding backend in this codebase today. See
    # this module's own docstring for why that makes a dispatch layer here
    # premature.
    return SentenceTransformerEmbeddingProvider(settings)


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return _build_embedding_provider(get_settings().embedding)


def _build_vector_store(settings: VectorStoreSettings) -> VectorStore:
    if settings.provider == VectorStoreProvider.CHROMA:
        return ChromaVectorStore(settings)
    raise ConfigurationError(
        f"VECTOR_STORE__PROVIDER={settings.provider.value!r} has no concrete VectorStore implementation yet."
    )


@lru_cache
def get_vector_store() -> VectorStore:
    return _build_vector_store(get_settings().vector_store)


def _build_search_provider(settings: SearchSettings) -> SearchProvider:
    if settings.provider == SearchProviderName.TAVILY:
        return TavilySearchProvider(settings)
    raise ConfigurationError(
        f"SEARCH__PROVIDER={settings.provider.value!r} has no concrete SearchProvider implementation yet."
    )


@lru_cache
def get_search_provider() -> SearchProvider:
    return _build_search_provider(get_settings().search)


def _build_report_exporter(export_format: ReportExportFormat) -> ReportExporter:
    if export_format == ReportExportFormat.MARKDOWN:
        return MarkdownReportExporter()
    if export_format == ReportExportFormat.PDF:
        return PDFReportExporter()
    raise ConfigurationError(
        f"Report export format {export_format.value!r} has no concrete ReportExporter implementation yet."
    )


@lru_cache
def get_report_exporter(export_format: ReportExportFormat) -> ReportExporter:
    """
    Explicit per-format singleton, not a per-process default like the
    other cached adapters above -- see this module's own docstring for
    why ``get_report_exporter`` takes an argument at all. Neither concrete
    exporter (``MarkdownReportExporter``, ``PDFReportExporter``) has a
    settings object to read, unlike ``_build_llm_service`` and friends --
    both are stateless renderers, so this skips the ``_build_*``/``get_*``
    split those use (there's no meaningful "pure, explicit-settings"
    layer to separate out when there are no settings involved).
    """
    return _build_report_exporter(export_format)


def _resolve_llm_model_name(settings: LLMSettings) -> str:
    """
    The configured model identifier for whichever provider is active --
    needed by both ``get_research_graph`` and ``get_chat_service``, which
    otherwise have no way to know which of ``LLMSettings``'s per-provider
    sub-settings (``ollama``, ``openai``, ...) holds the active model
    name. Mirrors ``_build_llm_service``'s own dispatch on
    ``settings.provider`` -- kept as a separate function rather than
    folded into it because this is a plain string lookup, not adapter
    construction, and every future provider added to one dispatch will
    need a matching branch here regardless of whether that provider's
    concrete adapter exists yet.
    """
    if settings.provider == LLMProvider.OLLAMA:
        return settings.ollama.model
    if settings.provider == LLMProvider.OPENAI:
        return settings.openai.model
    if settings.provider == LLMProvider.ANTHROPIC:
        return settings.anthropic.model
    raise ConfigurationError(f"LLM__PROVIDER={settings.provider.value!r} has no known model-name field to resolve.")


# =============================================================================
# Composed singletons
# =============================================================================


@lru_cache
def get_database() -> Database:
    return Database(get_settings())


@lru_cache
def get_research_graph() -> CompiledStateGraph:
    settings = get_settings()
    return build_research_graph(
        llm_service=get_llm_service(),
        llm_model_name=_resolve_llm_model_name(settings.llm),
        feature_flags=settings.features,
        embedding_provider=get_embedding_provider(),
        vector_store=get_vector_store(),
        rag_settings=settings.rag,
        search_provider=get_search_provider(),
        search_settings=settings.search,
        agent_settings=settings.agent,
    )


@lru_cache
def get_memory_service() -> MemoryService:
    settings = get_settings()
    return MemoryService(get_database(), get_llm_service(), settings.agent)


@lru_cache
def get_chat_service() -> ChatService:
    settings = get_settings()
    return ChatService(
        database=get_database(),
        memory_service=get_memory_service(),
        graph=get_research_graph(),
        llm_provider_name=settings.llm.provider.value,
        llm_model_name=_resolve_llm_model_name(settings.llm),
    )


@lru_cache
def get_report_service() -> ReportService:
    """
    Composes both concrete exporters into the ``{format: exporter}``
    mapping ``ReportService`` expects -- built here via plain calls to
    ``get_report_exporter()``, the same "compose other provider functions
    via plain Python calls" pattern every other composed singleton above
    already uses. ``ReportService`` itself never imports this module or
    calls ``get_report_exporter`` directly (see that class's own
    docstring) -- this function is the one place those two facts meet.
    """
    return ReportService(
        database=get_database(),
        chat_service=get_chat_service(),
        report_exporters={
            fmt: get_report_exporter(fmt) for fmt in (ReportExportFormat.MARKDOWN, ReportExportFormat.PDF)
        },
        settings=get_settings(),
    )


@lru_cache
def get_history_service() -> HistoryService:
    return HistoryService(get_database())


@lru_cache
def get_health_service() -> HealthService:
    return HealthService(
        llm_service=get_llm_service(),
        vector_store=get_vector_store(),
        embedding_provider=get_embedding_provider(),
        search_provider=get_search_provider(),
        database_health_checker=get_database(),
        settings=get_settings(),
        started_at=_PROCESS_STARTED_AT,
    )
