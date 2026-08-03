"""
Request/response schemas for ``POST /report``.

``POST /report`` is a thin, report-specific convenience wrapper around the
same underlying research workflow ``POST /chat`` runs with
``mode="report"`` (see ``schemas/chat.py``) — it accepts the same query/
session/retrieval vocabulary rather than duplicating it, and additionally
handles export-format selection and produces a downloadable file rather
than a conversational answer.

The actual report *content* model (title, executive summary, sections,
tables, conclusion, citations) is not redefined here — it already exists
as ``app.core.interfaces.report_exporter.ReportDocument``, the shared
contract between the Writer/Reviewer agents and the ``ReportExporter``
implementations (Phase 8). Redefining an equivalent shape in the API layer
would let the two drift apart; instead this module imports and reuses it
directly as the API-facing report content type, plus its own
request/wrapper types.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated

from app.core.interfaces.report_exporter import ReportDocument
from app.schemas.chat import RetrievalOptions
from app.schemas.common import ErrorBody, TokenUsage
from app.utils.time import utc_now

# =============================================================================
# Request
# =============================================================================


class ReportExportFormat(str, Enum):
    """Output format for a generated report."""

    MARKDOWN = "markdown"
    PDF = "pdf"


class ReportGenerationRequest(BaseModel):
    """
    Request body for ``POST /report``.

    Deliberately mirrors ``ChatRequest``'s ``query``/``session_id``/
    ``retrieval_options`` fields rather than reusing ``ChatRequest``
    itself by inheritance — a report request has no ``mode`` (it's always
    report mode) and no ``stream`` (a report is never streamed), so
    inheriting would mean immediately overriding/hiding fields that don't
    apply. Composing the shared pieces explicitly keeps this schema
    self-describing in the OpenAPI docs without inherited-but-irrelevant
    fields showing up.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "Create a report on Quantum Computing",
                    "session_id": None,
                    "retrieval_options": {"session_scope": True, "top_k": 8},
                    "export_format": "pdf",
                    "conversation_title": "Quantum Computing Overview",
                }
            ]
        }
    )

    query: Annotated[str, Field(min_length=1, max_length=4000)] = Field(
        ...,
        description="The research topic or question the report should address.",
        examples=["Create a report on Quantum Computing"],
    )
    session_id: Optional[str] = Field(
        default=None, description="Existing research session to generate this report from. Omit to start a new session."
    )
    retrieval_options: Optional[RetrievalOptions] = Field(
        default=None, description="Configurable retrieval behavior for this request. Omit to use RAGSettings defaults."
    )
    export_format: ReportExportFormat = Field(
        default=ReportExportFormat.PDF, description="Output format for the generated report file."
    )
    conversation_title: Optional[Annotated[str, Field(max_length=200)]] = Field(
        default=None,
        description="Client-supplied title for a new session. Ignored if session_id refers to an existing session.",
    )


# =============================================================================
# Response
# =============================================================================


class ReportGenerationStatus(str, Enum):
    """
    Lifecycle of a report generation request.

    Report generation runs the full multi-agent pipeline (potentially
    tens of seconds), so unlike ``/chat`` this response models an
    asynchronous-capable lifecycle even though the current implementation
    may complete synchronously within the request: a client can always
    check ``status`` rather than assuming ``completed``.
    """

    QUEUED = "queued"
    GENERATING = "generating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class ReportGenerationResponse(BaseModel):
    """Response payload for ``POST /report`` (wrapped in ``DataResponse[ReportGenerationResponse]``)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "report_id": "rpt-9c4e5b4a",
                    "session_id": "sess-7f3a1c9d",
                    "status": "completed",
                    "export_format": "pdf",
                    "download_url": "/api/v1/report/rpt-9c4e5b4a/download",
                    "report": {},
                    "file_size_bytes": 184320,
                    "generation_latency_ms": 18420.0,
                    "token_usage": {"prompt_tokens": 9820, "completion_tokens": 2140, "total_tokens": 11960},
                    "created_at": "2026-07-29T10:15:03Z",
                }
            ]
        }
    )

    report_id: str = Field(..., description="Unique identifier for this generated report.")
    session_id: str = Field(..., description="Research session this report was generated from.")
    status: ReportGenerationStatus = Field(..., description="Current lifecycle status of report generation.")
    export_format: ReportExportFormat = Field(..., description="Format the report was (or will be) exported to.")
    download_url: Optional[str] = Field(
        default=None,
        description="URL to download the exported file. Populated once status='completed'.",
    )
    report: Optional[ReportDocument] = Field(
        default=None,
        description=(
            "The structured report content (title, executive summary, sections, citations, conclusion), "
            "reusing app.core.interfaces.report_exporter.ReportDocument. Populated once status is "
            "'exporting' or 'completed'; omitted while still 'queued'/'generating' to avoid returning a "
            "partially-written document."
        ),
    )
    file_size_bytes: Optional[int] = Field(
        default=None, ge=0, description="Size of the exported file in bytes. Populated once status='completed'."
    )
    generation_latency_ms: Optional[float] = Field(
        default=None, ge=0, description="End-to-end time to generate and export the report."
    )
    token_usage: Optional[TokenUsage] = Field(
        default=None, description="Aggregate LLM token usage for this report's generation."
    )
    error: Optional[ErrorBody] = Field(default=None, description="Populated when status='failed'.")
    created_at: datetime = Field(default_factory=utc_now, description="When report generation was requested.")

    @model_validator(mode="after")
    def _validate_status_consistency(self) -> "ReportGenerationResponse":
        if self.status == ReportGenerationStatus.FAILED and self.error is None:
            raise ValueError("error must be set when status='failed'.")
        if self.status == ReportGenerationStatus.COMPLETED:
            if self.download_url is None:
                raise ValueError("download_url must be set when status='completed'.")
            if self.report is None:
                raise ValueError("report must be set when status='completed'.")
        return self


class ReportListItem(BaseModel):
    """
    Lightweight summary of a previously generated report, for listing
    (e.g. within ``GET /history`` or a future ``GET /reports``) without
    the cost of returning each report's full content.
    """

    report_id: str
    session_id: str
    title: str
    export_format: ReportExportFormat
    status: ReportGenerationStatus
    created_at: datetime
