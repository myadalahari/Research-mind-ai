"""
Application configuration for ResearchMind AI.

All runtime configuration is centralized here using Pydantic Settings so that:
  * every value is type-validated at process startup (fail fast, not three
    layers deep in the RAG pipeline at 2am),
  * configuration is organized into cohesive, nested groups instead of one
    flat namespace, which keeps the file navigable as the system grows,
  * provider-specific settings (Ollama, OpenAI, Anthropic, Azure OpenAI,
    Tavily, Chroma, ...) are fully isolated from the business logic that
    consumes them through the LLMService / SearchProvider / VectorStore /
    EmbeddingProvider / ReportExporter abstractions defined in
    ``app.core.interfaces``. Nothing outside ``app/services/llm`` and
    ``app/services/search`` should ever read ``settings.llm.ollama`` or
    ``settings.search.tavily_api_key`` directly.

Environment variables use ``__`` as the nested delimiter, e.g.::

    APP__DEBUG=true
    LLM__PROVIDER=ollama
    LLM__OLLAMA__BASE_URL=http://localhost:11434
    SEARCH__TAVILY_API_KEY=tvly-xxxxxxxx

See ``.env.example`` at the project root for the full, documented list of
supported variables and their defaults.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from typing_extensions import Annotated

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class Environment(str, Enum):
    """Deployment environment the app is running in."""

    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    """Supported LLM backends.

    The application depends exclusively on the ``LLMService`` interface
    (see ``app.core.interfaces.llm_service``); this enum only selects which
    concrete adapter ``core.dependencies.get_llm_service`` constructs.
    """

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"


class SearchProviderName(str, Enum):
    """Supported web search backends, selected the same way as LLMProvider."""

    TAVILY = "tavily"


class VectorStoreProvider(str, Enum):
    """Supported vector store backends."""

    CHROMA = "chroma"


class LogFormat(str, Enum):
    """Log rendering format."""

    JSON = "json"
    CONSOLE = "console"


class ReportExportFormat(str, Enum):
    """Default report export format."""

    MARKDOWN = "markdown"
    PDF = "pdf"


# --------------------------------------------------------------------------- #
# Nested settings groups
# --------------------------------------------------------------------------- #


class AppSettings(BaseSettings):
    """General application and HTTP server metadata."""

    name: str = "ResearchMind AI"
    version: str = "0.1.0"
    environment: Environment = Environment.LOCAL
    debug: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # NoDecode: pydantic-settings otherwise attempts to JSON-decode any List[...]
    # field pulled from an env var *before* validators run, which would raise a
    # JSONDecodeError on a plain comma-separated string like "a,b,c". NoDecode
    # hands the raw string to our validator below instead.
    cors_origins: Annotated[List[str], NoDecode] = Field(default_factory=lambda: ["http://localhost:8501"])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow ``CORS_ORIGINS`` to be provided as a comma-separated string."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    model_config = SettingsConfigDict(env_prefix="APP__", extra="ignore")


class OllamaSettings(BaseSettings):
    """Configuration specific to the local Ollama LLM adapter."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen3"
    temperature: float = 0.3
    keep_alive: str = "5m"

    model_config = SettingsConfigDict(env_prefix="LLM__OLLAMA__", extra="ignore")


class OpenAISettings(BaseSettings):
    """Configuration specific to the OpenAI LLM adapter (future provider)."""

    api_key: Optional[SecretStr] = None
    model: str = "gpt-4o-mini"
    organization: Optional[str] = None
    base_url: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="LLM__OPENAI__", extra="ignore")


class AnthropicSettings(BaseSettings):
    """Configuration specific to the Anthropic LLM adapter (future provider)."""

    api_key: Optional[SecretStr] = None
    model: str = "claude-sonnet-4-5"

    model_config = SettingsConfigDict(env_prefix="LLM__ANTHROPIC__", extra="ignore")


class AzureOpenAISettings(BaseSettings):
    """Configuration specific to the Azure OpenAI LLM adapter (future provider)."""

    api_key: Optional[SecretStr] = None
    endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    api_version: str = "2024-08-01-preview"

    model_config = SettingsConfigDict(env_prefix="LLM__AZURE_OPENAI__", extra="ignore")


class LLMSettings(BaseSettings):
    """
    Provider-agnostic LLM configuration.

    ``provider`` is the single switch that determines which concrete
    ``LLMService`` implementation ``core.dependencies.get_llm_service``
    constructs. Everything downstream (agents, services) depends only on
    the ``LLMService`` interface and never inspects this object directly.
    """

    provider: LLMProvider = LLMProvider.OLLAMA
    max_retries: int = 3
    timeout_seconds: int = 60
    max_context_tokens: int = 8192

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)

    model_config = SettingsConfigDict(env_prefix="LLM__", extra="ignore")

    @model_validator(mode="after")
    def _validate_provider_credentials(self) -> "LLMSettings":
        """Fail fast at startup if the selected provider is missing required credentials."""
        if self.provider == LLMProvider.OPENAI and self.openai.api_key is None:
            raise ValueError("LLM__PROVIDER=openai requires LLM__OPENAI__API_KEY to be set.")
        if self.provider == LLMProvider.ANTHROPIC and self.anthropic.api_key is None:
            raise ValueError("LLM__PROVIDER=anthropic requires LLM__ANTHROPIC__API_KEY to be set.")
        if self.provider == LLMProvider.AZURE_OPENAI:
            missing = [
                name
                for name, val in (
                    ("LLM__AZURE_OPENAI__API_KEY", self.azure_openai.api_key),
                    ("LLM__AZURE_OPENAI__ENDPOINT", self.azure_openai.endpoint),
                    ("LLM__AZURE_OPENAI__DEPLOYMENT_NAME", self.azure_openai.deployment_name),
                )
                if not val
            ]
            if missing:
                raise ValueError(
                    "LLM__PROVIDER=azure_openai requires the following settings to be " f"set: {', '.join(missing)}."
                )
        return self


class EmbeddingSettings(BaseSettings):
    """Configuration for the embedding model used by the RAG pipeline."""

    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True

    model_config = SettingsConfigDict(env_prefix="EMBEDDING__", extra="ignore")


class VectorStoreSettings(BaseSettings):
    """Configuration for the vector store backing document retrieval."""

    provider: VectorStoreProvider = VectorStoreProvider.CHROMA
    persist_dir: str = "./data/chroma"
    collection_name: str = "researchmind_documents"
    distance_metric: str = "cosine"
    telemetry_enabled: bool = False

    model_config = SettingsConfigDict(env_prefix="VECTOR_STORE__", extra="ignore")


class SearchSettings(BaseSettings):
    """Configuration for the web search tool used by the Search Agent."""

    provider: SearchProviderName = SearchProviderName.TAVILY
    tavily_api_key: Optional[SecretStr] = None
    max_results: int = 5
    max_retries: int = 3
    timeout_seconds: int = 30
    base_url: Optional[str] = Field(
        default=None,
        description=(
            "Override the search provider's API base URL. None uses the provider "
            "SDK's own default (Tavily: https://api.tavily.com). Primarily for "
            "testing against a local fake server, or pointing at a self-hosted/"
            "enterprise-proxied endpoint."
        ),
    )

    model_config = SettingsConfigDict(env_prefix="SEARCH__", extra="ignore")

    @model_validator(mode="after")
    def _validate_tavily_credentials(self) -> "SearchSettings":
        """
        Only require an API key if web search is actually reachable at the
        provider level; the ``ENABLE_WEB_SEARCH`` feature flag (see
        ``FeatureFlags``) is checked separately at the service layer so a
        missing key doesn't block startup when search is disabled entirely.
        """
        return self


class DatabaseSettings(BaseSettings):
    """
    Configuration for the relational store backing memory/history/reports metadata.

    ``url`` must use an async-capable driver (``sqlite+aiosqlite://``, or
    ``postgresql+asyncpg://`` for a future Postgres deployment), since
    ``database.session.Database`` is built on SQLAlchemy's async engine to
    match the async-everywhere pattern used by every other interface
    (``LLMService``, ``VectorStore``, ...). A bare ``sqlite://`` URL uses
    the sync driver and will fail when handed to the async engine.
    """

    url: str = "sqlite+aiosqlite:///./data/researchmind.db"
    echo: bool = False
    pool_pre_ping: bool = True

    model_config = SettingsConfigDict(env_prefix="DATABASE__", extra="ignore")


class LoggingSettings(BaseSettings):
    """Structured logging configuration."""

    level: str = "INFO"
    format: LogFormat = LogFormat.JSON
    include_trace_id: bool = True
    include_request_id: bool = True

    model_config = SettingsConfigDict(env_prefix="LOGGING__", extra="ignore")

    @field_validator("level")
    @classmethod
    def _validate_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper()
        if normalized not in allowed:
            raise ValueError(f"LOGGING__LEVEL must be one of {sorted(allowed)}, got {value!r}.")
        return normalized


class RAGSettings(BaseSettings):
    """Configuration for the document ingestion and retrieval pipeline."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.3
    max_upload_size_mb: int = 25
    allowed_upload_extensions: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md"]
    )
    max_documents_per_session: int = 20
    max_upload_files: int = 10

    model_config = SettingsConfigDict(env_prefix="RAG__", extra="ignore")

    @field_validator("allowed_upload_extensions", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [ext.strip().lower() for ext in value.split(",") if ext.strip()]
        return value

    @model_validator(mode="after")
    def _validate_chunking(self) -> "RAGSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "RAG__CHUNK_OVERLAP must be smaller than RAG__CHUNK_SIZE "
                f"(got chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})."
            )
        return self


class AgentSettings(BaseSettings):
    """Configuration for the LangGraph multi-agent workflow."""

    max_reviewer_retries: int = 2
    agent_timeout_seconds: int = 120
    max_chat_history: int = 20

    model_config = SettingsConfigDict(env_prefix="AGENT__", extra="ignore")


class ReportSettings(BaseSettings):
    """Configuration for report generation and export."""

    output_dir: str = "./reports"
    default_export_format: ReportExportFormat = ReportExportFormat.MARKDOWN

    model_config = SettingsConfigDict(env_prefix="REPORT__", extra="ignore")


class FeatureFlags(BaseSettings):
    """
    Runtime feature toggles.

    These let the graph and API degrade gracefully in constrained
    environments (e.g. CI, offline demos, or a reviewer with no Tavily key)
    without code changes: a disabled feature short-circuits at the service
    layer rather than failing on a missing credential deep in a node.
    """

    enable_web_search: bool = True
    enable_rag: bool = True
    enable_memory: bool = True
    enable_report_export: bool = True
    enable_agent_tracing: bool = True

    model_config = SettingsConfigDict(env_prefix="FEATURES__", extra="ignore")


class CacheSettings(BaseSettings):
    """Optional on-disk caching (e.g. for embeddings or search results)."""

    enabled: bool = False
    dir: str = "./data/cache"
    ttl_seconds: int = 3600

    model_config = SettingsConfigDict(env_prefix="CACHE__", extra="ignore")


class HTTPSettings(BaseSettings):
    """Default HTTP client behavior shared by outbound integrations."""

    timeout_seconds: int = 30

    model_config = SettingsConfigDict(env_prefix="HTTP__", extra="ignore")


# --------------------------------------------------------------------------- #
# Root settings object
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """
    Root application settings, composed of nested, cohesive settings groups.

    A single instance is constructed once (see ``get_settings``) and shared
    across the process via FastAPI dependency injection
    (``core.dependencies.get_settings``). Never instantiate ``Settings()``
    directly outside of ``get_settings`` / tests, so the whole app agrees on
    one configuration snapshot for the lifetime of the process.
    """

    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    report: ReportSettings = Field(default_factory=ReportSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    http: HTTPSettings = Field(default_factory=HTTPSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def _validate_cross_cutting_constraints(self) -> "Settings":
        """Validate constraints that span more than one settings group."""
        if self.features.enable_web_search and self.search.provider == SearchProviderName.TAVILY:
            if self.search.tavily_api_key is None and self.app.environment != Environment.LOCAL:
                raise ValueError(
                    "FEATURES__ENABLE_WEB_SEARCH=true requires SEARCH__TAVILY_API_KEY "
                    "to be set in non-local environments. Set FEATURES__ENABLE_WEB_SEARCH=false "
                    "to run without web search."
                )
        if self.agent.max_chat_history < 1:
            raise ValueError("AGENT__MAX_CHAT_HISTORY must be at least 1.")
        return self

    def ensure_runtime_directories(self) -> None:
        """
        Create the on-disk directories the app writes to at runtime
        (uploads, reports, vector store, cache, sqlite file location).

        Called once at application startup (see ``app.main``) so routers and
        services can assume these paths already exist rather than each
        having to defensively ``mkdir`` before every write.
        """
        directories = [
            Path(self.vector_store.persist_dir),
            Path(self.report.output_dir),
        ]
        if self.cache.enabled:
            directories.append(Path(self.cache.dir))
        for sqlite_prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
            if self.database.url.startswith(sqlite_prefix):
                db_path = Path(self.database.url.replace(sqlite_prefix, "", 1))
                directories.append(db_path.parent)
                break
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Return the process-wide ``Settings`` singleton.

    Cached with ``lru_cache`` so the (relatively expensive, validation-heavy)
    ``Settings`` construction happens exactly once per process. Tests that
    need a different configuration should call ``get_settings.cache_clear()``
    after setting environment variables, or construct ``Settings(...)``
    directly rather than relying on this singleton.
    """
    return Settings()
