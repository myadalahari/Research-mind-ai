"""
HTTP contract models for the ResearchMind AI frontend.

These are **wire-contract models owned by this frontend**, not shared
domain models imported from the backend. Every class here exists solely to
validate and type a specific JSON shape ``api_client.py`` sends to or
receives from the backend's ``/chat`` and ``/report`` endpoints -- see
``docs/architecture-decisions.md``'s Phase 9 entry for why the frontend
keeps its own copies instead of importing ``app.schemas.*`` from the
backend package (independent deployability; a backend contract change
becomes a visible Pydantic validation failure here rather than a silent
runtime drift).

Two deliberate consequences of "mirror only what's consumed":

* Fields the current frontend pages never read or send are omitted, even
  where the backend's schema defines them (e.g. ``Citation.document_id``,
  ``RetrievalOptions.document_ids`` -- both tied to document upload, which
  is out of scope for this phase; see the Phase 9 scoping discussion).
* Two backend types that happen to both be called "Citation" on the
  backend (``app.schemas.common.Citation`` used by ``/chat``, and
  ``app.core.interfaces.report_exporter.Citation`` embedded in
  ``ReportDocument`` returned by ``/report``) are genuinely different
  wire shapes with different fields. They get two distinct models here
  (``Citation`` and ``ReportCitation``) rather than one merged/loosened
  model, so each stays an accurate, fail-fast description of what that
  specific endpoint actually sends.

Enum values and field names are preserved exactly as they appear on the
wire (e.g. ``ChatMode.RESEARCH == "research"``) so round-tripping a value
back to the backend (a session's ``mode``, an ``export_format`` choice)
never needs a translation step.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field
from typing_extensions import Annotated

# A research query, mirroring the length bounds ChatRequest.query /
# ReportGenerationRequest.query enforce on the backend (app.schemas.chat /
# app.schemas.report) -- validating client-side lets the UI reject an
# empty or oversized query before a network round trip, not just relay
# whatever 422 the backend would eventually return.
QueryText = Annotated[str, Field(min_length=1, max_length=4000)]

# =============================================================================
# Shared enums
# =============================================================================


class ChatMode(str, Enum):
    """Mirrors ``app.schemas.chat.ChatMode``. Selects which workflow a ``/chat`` request runs."""

    CHAT = "chat"
    RESEARCH = "research"
    REPORT = "report"


class SourceType(str, Enum):
    """Mirrors ``app.schemas.common.SourceType``. Where a citation's evidence came from."""

    DOCUMENT = "document"
    WEB = "web"


class ExecutionStepStatus(str, Enum):
    """Mirrors ``app.schemas.common.ExecutionStepStatus``. Lifecycle status of one agent step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReportExportFormat(str, Enum):
    """Mirrors ``app.schemas.report.ReportExportFormat``. Output format for a generated report."""

    MARKDOWN = "markdown"
    PDF = "pdf"


class ReportGenerationStatus(str, Enum):
    """Mirrors ``app.schemas.report.ReportGenerationStatus``. Lifecycle of a report generation request."""

    QUEUED = "queued"
    GENERATING = "generating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Shared value types
# =============================================================================


class TokenUsage(BaseModel):
    """Mirrors ``app.schemas.common.TokenUsage``. LLM token consumption for a call or a run."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class Citation(BaseModel):
    """
    A single source backing a ``/chat`` answer.

    Mirrors ``app.schemas.common.Citation``, minus ``document_id`` --
    that field exists on the backend purely to correlate a citation back
    to an uploaded ``Document`` row, which this frontend has no use for
    without a document-management UI (out of scope for this phase).
    """

    citation_id: str
    source_type: SourceType
    title: str
    url: Optional[str] = None
    source_filename: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    excerpt: Optional[str] = None
    score: Optional[float] = None
    published_date: Optional[datetime] = None
    accessed_at: Optional[datetime] = None


class ResearchMetadata(BaseModel):
    """Mirrors ``app.schemas.chat.ResearchMetadata``. Summary of providers/retrieval used for a ``/chat`` response."""

    llm_provider: str
    llm_model: str
    rag_enabled: bool
    web_search_enabled: bool
    retrieved_chunk_count: int = 0
    web_result_count: int = 0
    documents_considered: List[str] = Field(default_factory=list)
    total_latency_ms: float
    token_usage: TokenUsage


class ErrorBody(BaseModel):
    """
    Mirrors ``app.schemas.common.ErrorBody``.

    Used both as the payload of a top-level ``ErrorEnvelope`` (a failed
    HTTP request) and, embedded, as ``AgentExecutionStep.error`` (a single
    failed step within an otherwise-successful trace) -- the backend
    reuses one shape for both, and this model does the same.
    """

    error_code: str
    message: str
    retryable: bool
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ErrorEnvelope(BaseModel):
    """Mirrors ``app.schemas.common.ErrorEnvelope``. Top-level shape of every backend error response."""

    error: ErrorBody


class AgentExecutionStep(BaseModel):
    """Mirrors ``app.schemas.common.AgentExecutionStep``. One agent/graph-node execution, for the trace viewer."""

    step_id: str
    agent_name: str
    graph_node: str
    status: ExecutionStepStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    latency_ms: Optional[float] = None
    model_name: Optional[str] = None
    tool_name: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
    summary: Optional[str] = None
    error: Optional[ErrorBody] = None


class AgentExecutionTrace(BaseModel):
    """Mirrors ``app.schemas.common.AgentExecutionTrace``. Full step-by-step trace for one ``/chat`` run."""

    trace_id: str
    session_id: str
    steps: List[AgentExecutionStep] = Field(default_factory=list)
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_latency_ms: Optional[float] = None
    total_token_usage: Optional[TokenUsage] = None


# =============================================================================
# /chat request/response
# =============================================================================


class RetrievalOptions(BaseModel):
    """
    Subset of ``app.schemas.chat.RetrievalOptions`` this frontend can
    actually set: a retrieval-count override and a web-search toggle.

    ``document_ids`` and ``session_scope`` are omitted -- both only matter
    once documents can be uploaded and selected in the UI, which is out of
    scope for this phase (see the Phase 9 scoping discussion). Omitting a
    field here simply means the backend falls back to its own default for
    it, exactly as if the field were never sent.
    """

    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    include_web_search: Optional[bool] = None


class ChatRequest(BaseModel):
    """
    Request body for ``POST /chat``.

    Deliberately excludes ``app.schemas.chat.ChatRequest.stream``: that
    field's own backend docstring says requesting ``stream=true``
    currently has no effect on the response shape (SSE delivery isn't
    implemented yet), so this frontend never sends it rather than
    implying a capability that doesn't exist.
    """

    query: QueryText
    session_id: Optional[str] = None
    mode: ChatMode = ChatMode.RESEARCH
    retrieval_options: Optional[RetrievalOptions] = None
    conversation_title: Optional[str] = Field(default=None, max_length=200)


class ChatResponse(BaseModel):
    """Mirrors the fields of ``app.schemas.chat.ChatResponse`` this frontend renders."""

    session_id: str
    message_id: str
    mode: ChatMode
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    conversation_title: Optional[str] = None
    follow_up_suggestions: Optional[List[str]] = None
    research_metadata: ResearchMetadata
    execution_trace: Optional[AgentExecutionTrace] = None
    created_at: datetime


# =============================================================================
# /report request/response
# =============================================================================


class ReportCitation(BaseModel):
    """
    A single source cited within a generated report.

    Mirrors ``app.core.interfaces.report_exporter.Citation`` -- the
    narrower citation shape embedded in ``ReportDocument`` -- which is
    *not* the same wire shape as this module's ``Citation`` (used by
    ``/chat``). See the module docstring for why these stay two models.
    """

    citation_id: str
    source_type: str
    title: str
    url: Optional[str] = None
    source_filename: Optional[str] = None
    page_number: Optional[int] = None
    accessed_at: Optional[datetime] = None


class ReportTable(BaseModel):
    """Mirrors ``app.core.interfaces.report_exporter.ReportTable``. A table embedded within a report section."""

    caption: Optional[str] = None
    headers: List[str]
    rows: List[List[str]]


class ReportSection(BaseModel):
    """Mirrors ``app.core.interfaces.report_exporter.ReportSection``. A section of a report, recursively nested."""

    heading: str
    content: str
    tables: List[ReportTable] = Field(default_factory=list)
    subsections: List["ReportSection"] = Field(default_factory=list)


ReportSection.model_rebuild()


class ReportDocument(BaseModel):
    """Mirrors ``app.core.interfaces.report_exporter.ReportDocument``. The full structured content of a report."""

    title: str
    session_id: str
    generated_at: datetime
    executive_summary: str
    key_findings: List[str] = Field(default_factory=list)
    sections: List[ReportSection] = Field(default_factory=list)
    conclusion: str
    citations: List[ReportCitation] = Field(default_factory=list)


class ReportGenerationRequest(BaseModel):
    """Request body for ``POST /report``. Mirrors ``app.schemas.report.ReportGenerationRequest``."""

    query: QueryText
    session_id: Optional[str] = None
    retrieval_options: Optional[RetrievalOptions] = None
    export_format: ReportExportFormat = ReportExportFormat.PDF
    conversation_title: Optional[str] = Field(default=None, max_length=200)


class ReportGenerationResponse(BaseModel):
    """Mirrors ``app.schemas.report.ReportGenerationResponse``. Response payload for ``POST /report``."""

    report_id: str
    session_id: str
    status: ReportGenerationStatus
    export_format: ReportExportFormat
    download_url: Optional[str] = None
    report: Optional[ReportDocument] = None
    file_size_bytes: Optional[int] = None
    generation_latency_ms: Optional[float] = None
    token_usage: Optional[TokenUsage] = None
    error: Optional[ErrorBody] = None
    created_at: datetime


# =============================================================================
# Success response envelope
# =============================================================================

T = TypeVar("T", bound=BaseModel)


class ResponseMeta(BaseModel):
    """Mirrors ``app.schemas.common.ResponseMeta``. Correlation/timing metadata on every successful response."""

    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    latency_ms: Optional[float] = None


class DataEnvelope(BaseModel, Generic[T]):
    """
    Mirrors ``app.schemas.common.DataResponse``. Generic success envelope:
    ``{"data": ..., "meta": {...}}``.

    ``api_client.py`` parses every successful response through
    ``DataEnvelope[ChatResponse]``/``DataEnvelope[ReportGenerationResponse]``
    rather than reaching into raw JSON, so a malformed or unexpectedly
    shaped ``data`` payload fails fast as a validation error instead of
    surfacing later as a confusing ``AttributeError`` deep in a UI component.
    """

    data: T
    meta: ResponseMeta
