"""
Shared, cross-cutting API schemas.

Every endpoint-specific schema module (``chat.py``, ``upload.py``,
``report.py``, ``history.py``, ``sources.py``) builds on top of the types
defined here rather than redefining them, so the API presents one
consistent vocabulary for citations, token usage, pagination, and response
envelopes regardless of which endpoint a client is calling.

This module is the canonical source for the error response envelope
(``ErrorBody``/``ErrorEnvelope``) — ``app.api.middleware.error_handler``
imports these types rather than defining its own copy, so the documented
OpenAPI error schema and the actual runtime response shape can never drift
apart.

These are API/DTO-layer models, distinct from the domain models in
``app.core.interfaces`` (e.g. ``core.interfaces.report_exporter.Citation``).
They intentionally look similar in places — a citation is a citation — but
are kept as separate types because they serve different contracts: the
interface models are what agents and services pass to each other
internally, while these are what actually gets serialized over the wire,
versioned, and documented in the OpenAPI schema. Collapsing the two would
mean an internal refactor could silently change the public API contract.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated

from app.core.logging import get_request_id, get_trace_id
from app.utils.time import utc_now

T = TypeVar("T")


# =============================================================================
# Citations & sources
# =============================================================================


class SourceType(str, Enum):
    """Where a piece of grounding evidence came from."""

    DOCUMENT = "document"
    WEB = "web"


class Citation(BaseModel):
    """
    A single reference backing a claim in a chat answer or report.

    Every research answer ResearchMind produces is expected to cite its
    sources — this is the shape those citations take on the wire, whether
    they originated from an uploaded document chunk (via the Retriever
    agent) or a web search result (via the Search agent).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "citation_id": "1",
                    "source_type": "document",
                    "title": "qwen3_technical_report.pdf",
                    "source_filename": "qwen3_technical_report.pdf",
                    "document_id": "doc-42",
                    "page_number": 4,
                    "section_title": "3.2 Architecture",
                    "excerpt": "Qwen3 adopts a mixture-of-experts architecture with...",
                    "score": 0.87,
                    "url": None,
                    "published_date": None,
                    "accessed_at": "2026-07-29T10:15:03Z",
                }
            ]
        }
    )

    citation_id: str = Field(
        ..., description="Stable identifier used for inline citation markers, e.g. '[1]'.", examples=["1"]
    )
    source_type: SourceType = Field(..., description="Whether this citation is from an uploaded document or the web.")
    title: str = Field(..., description="Title of the cited source.")
    url: Optional[str] = Field(default=None, description="URL of the source, present for web citations.")
    source_filename: Optional[str] = Field(
        default=None, description="Original filename of the source, present for document citations."
    )
    document_id: Optional[str] = Field(
        default=None,
        description=(
            "Identifier of the source document, present for document citations. Kept distinct from "
            "source_filename (a display-only name) -- this is what ChatService persists onto "
            "TurnCitation.document_id, the column HistoryService's documents_considered reconstruction "
            "reads back. Always None for web citations, which have no backing Document row."
        ),
    )
    page_number: Optional[int] = Field(
        default=None, description="Page number within the source document, if applicable."
    )
    section_title: Optional[str] = Field(default=None, description="Nearest section/heading the citation falls under.")
    excerpt: Optional[str] = Field(
        default=None, description="Short quoted excerpt of the cited text, shown in the frontend's source viewer."
    )
    score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Relevance/similarity score for this citation, if available."
    )
    published_date: Optional[datetime] = Field(default=None, description="Publication date, for web citations.")
    accessed_at: Optional[datetime] = Field(default=None, description="When this source was retrieved by ResearchMind.")


# =============================================================================
# Token usage & latency
# =============================================================================


class TokenUsage(BaseModel):
    """
    LLM token consumption for a single call or an aggregate across a run.

    Powers the token-usage-tracking observability requirement; surfaced
    per agent step in ``AgentExecutionStep`` and aggregated at the run
    level in ``AgentExecutionTrace``.
    """

    prompt_tokens: Optional[int] = Field(default=None, ge=0, description="Tokens consumed by the prompt.")
    completion_tokens: Optional[int] = Field(default=None, ge=0, description="Tokens generated in the completion.")
    total_tokens: Optional[int] = Field(
        default=None, ge=0, description="Total tokens consumed; computed from prompt + completion if not supplied."
    )

    @model_validator(mode="after")
    def _compute_total_if_missing(self) -> "TokenUsage":
        if self.total_tokens is None and self.prompt_tokens is not None and self.completion_tokens is not None:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self

    @classmethod
    def aggregate(cls, usages: List["TokenUsage"]) -> "TokenUsage":
        """Sum a list of per-call token usages into one run-level total."""
        prompt = sum(u.prompt_tokens or 0 for u in usages) or None
        completion = sum(u.completion_tokens or 0 for u in usages) or None
        total = sum(u.total_tokens or 0 for u in usages) or None
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


# =============================================================================
# Agent execution trace (powers the frontend's Agent Execution Viewer)
# =============================================================================


class ExecutionStepStatus(str, Enum):
    """Lifecycle status of a single agent/node execution step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentExecutionStep(BaseModel):
    """
    A single agent/graph-node execution within a research run.

    One of these is emitted per LangGraph node invocation (Planner,
    Researcher, Retriever, Search, Fact Checker, Writer, Reviewer,
    Coordinator), giving the frontend's Agent Execution Viewer a
    step-by-step account of what happened, how long it took, and what it
    cost in tokens — built from the same correlation ids
    (``app.core.logging``) used in the structured operational logs, so a
    step shown in the UI can be cross-referenced with server-side logs.
    """

    step_id: str = Field(..., description="Unique identifier for this execution step.")
    agent_name: str = Field(..., description="Name of the agent that ran, e.g. 'planner'.", examples=["planner"])
    graph_node: str = Field(..., description="LangGraph node name this step corresponds to.", examples=["plan"])
    status: ExecutionStepStatus = Field(..., description="Lifecycle status of this step.")
    started_at: datetime = Field(..., description="When this step began executing.")
    completed_at: Optional[datetime] = Field(default=None, description="When this step finished, if it has.")
    latency_ms: Optional[float] = Field(
        default=None, ge=0, description="Wall-clock duration of this step, in milliseconds."
    )
    model_name: Optional[str] = Field(default=None, description="LLM model used by this step, if it called one.")
    tool_name: Optional[str] = Field(default=None, description="Tool used by this step (e.g. 'tavily_search'), if any.")
    token_usage: Optional[TokenUsage] = Field(
        default=None, description="Token usage for this step, if it called an LLM."
    )
    summary: Optional[str] = Field(
        default=None,
        description="Short human-readable description of what this step did, e.g. 'Retrieved 5 relevant chunks'.",
    )
    error: Optional["ErrorBody"] = Field(default=None, description="Populated when status is 'failed'.")


class AgentExecutionTrace(BaseModel):
    """
    The full execution trace for a single research run (one ``/chat`` or
    ``/report`` invocation), composed of ordered ``AgentExecutionStep``s.

    This is what the frontend's Agent Execution Viewer renders directly,
    and what ``GET /history`` returns alongside each past conversation
    turn so prior runs remain inspectable, not just their final answer.
    """

    trace_id: str = Field(..., description="Correlates this run's steps to server-side structured logs.")
    session_id: str = Field(..., description="Research session this run belongs to.")
    steps: List[AgentExecutionStep] = Field(default_factory=list)
    started_at: datetime = Field(..., description="When the run began.")
    completed_at: Optional[datetime] = Field(default=None, description="When the run finished, if it has.")
    total_latency_ms: Optional[float] = Field(default=None, ge=0, description="Total wall-clock duration of the run.")
    total_token_usage: Optional[TokenUsage] = Field(default=None, description="Aggregate token usage across all steps.")

    @classmethod
    def from_steps(
        cls,
        *,
        trace_id: str,
        session_id: str,
        steps: List[AgentExecutionStep],
        started_at: datetime,
        completed_at: Optional[datetime] = None,
    ) -> "AgentExecutionTrace":
        """Build a trace from its steps, computing the aggregate fields rather than requiring callers to."""
        usages = [step.token_usage for step in steps if step.token_usage is not None]
        total_latency = sum(step.latency_ms or 0 for step in steps) or None
        return cls(
            trace_id=trace_id,
            session_id=session_id,
            steps=steps,
            started_at=started_at,
            completed_at=completed_at,
            total_latency_ms=total_latency,
            total_token_usage=TokenUsage.aggregate(usages) if usages else None,
        )


# =============================================================================
# Pagination
# =============================================================================


class PaginationParams(BaseModel):
    """
    Standard page/page_size query parameters.

    Used as a FastAPI dependency (``Depends()``) on any endpoint that
    returns a list, e.g. ``GET /history``, so pagination behavior and
    validation bounds are defined once rather than per-router.
    """

    page: Annotated[int, Field(default=1, ge=1, description="1-indexed page number.")]
    page_size: Annotated[int, Field(default=20, ge=1, le=100, description="Number of items per page (max 100).")]


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated list response wrapper."""

    items: List[T] = Field(..., description="The page of items.")
    total: int = Field(..., ge=0, description="Total number of items across all pages.")
    page: int = Field(..., ge=1, description="The current page number.")
    page_size: int = Field(..., ge=1, description="Number of items per page.")
    total_pages: int = Field(..., ge=1, description="Total number of pages.")
    has_next: bool = Field(..., description="Whether a subsequent page exists.")
    has_previous: bool = Field(..., description="Whether a preceding page exists.")

    @classmethod
    def create(cls, *, items: List[T], total: int, page: int, page_size: int) -> "PaginatedResponse[T]":
        """
        Build a ``PaginatedResponse`` from raw query results, computing
        ``total_pages``/``has_next``/``has_previous`` consistently rather
        than leaving each call site to compute (and potentially get wrong)
        that arithmetic independently.
        """
        total_pages = max(1, math.ceil(total / page_size)) if page_size else 1
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )


# =============================================================================
# Success response envelope
# =============================================================================


class ResponseMeta(BaseModel):
    """Metadata attached to every successful API response."""

    request_id: Optional[str] = Field(default=None, description="Correlates this response to server-side logs.")
    trace_id: Optional[str] = Field(default=None, description="Correlates this response across a multi-request run.")
    timestamp: datetime = Field(default_factory=utc_now, description="When this response was generated.")
    latency_ms: Optional[float] = Field(default=None, ge=0, description="Server-side processing time for this request.")


class DataResponse(BaseModel, Generic[T]):
    """
    Generic success response envelope: ``{"data": ..., "meta": {...}}``.

    Mirrors the shape of ``ErrorEnvelope`` (``{"error": {...}}``) so a
    client can always look at the top-level key to know whether a response
    succeeded, and every successful response carries the same correlation
    metadata an error response would.
    """

    data: T
    meta: ResponseMeta

    @classmethod
    def wrap(cls, data: T, *, latency_ms: Optional[float] = None) -> "DataResponse[T]":
        """
        Build a ``DataResponse`` around ``data``, auto-populating
        ``request_id``/``trace_id`` from the current logging context so
        routers don't need to thread them through by hand.
        """
        return cls(
            data=data,
            meta=ResponseMeta(request_id=get_request_id(), trace_id=get_trace_id(), latency_ms=latency_ms),
        )


# =============================================================================
# Error response envelope
#
# Canonical definitions — app.api.middleware.error_handler imports these
# rather than defining its own copies, so the documented OpenAPI error
# schema is guaranteed to match what the middleware actually returns.
# =============================================================================


class ErrorBody(BaseModel):
    """The ``error`` object inside every error response envelope."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error_code": "RM-RAG-001",
                    "message": "Vector store operation failed.",
                    "retryable": True,
                    "request_id": "3f2a1c9d4e5b4a1e9c7d2f8a1b6c9d0e",
                    "trace_id": "9b7e0a11f3c24d5a8b1e6f0a2c3d4e5f",
                    "details": {"document_id": "doc-42"},
                }
            ]
        }
    )

    error_code: str = Field(
        ..., description="Stable, machine-readable error identifier, e.g. 'RM-RAG-001'.", examples=["RM-RAG-001"]
    )
    message: str = Field(..., description="Human-readable, client-safe description of what went wrong.")
    retryable: bool = Field(..., description="Whether retrying the same request may succeed.")
    request_id: Optional[str] = Field(default=None, description="Correlates this error to server-side logs.")
    trace_id: Optional[str] = Field(default=None, description="Correlates this error across a multi-request run.")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extra diagnostic context. Omitted entirely in production for unexpected errors.",
    )


class ErrorEnvelope(BaseModel):
    """Top-level shape of every error response body: ``{"error": {...}}``."""

    error: ErrorBody


# Resolve the forward reference to ErrorBody used in AgentExecutionStep,
# which is defined earlier in this module (above ErrorBody) because it
# reads more naturally grouped with the other execution-trace models.
AgentExecutionStep.model_rebuild()
