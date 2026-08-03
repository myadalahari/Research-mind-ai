"""
Request/response schemas for ``GET /history``.

History is modeled around two distinct levels of detail, matching how a
client actually wants to consume it:

* ``ConversationSession`` — a session-level summary (title, timestamps,
  counts, status, a short preview, a reference to the session's most
  recent report). This is what ``GET /history`` returns by default, as
  ``PaginatedResponse[ConversationSession]`` (see ``schemas.common``) — a
  session list view should be cheap to render and cheap to fetch, so it
  never carries full turn content or execution traces unless asked for.
* ``ConversationTurn`` — a single exchange (one query/answer pair) within
  a session, carrying its own citations, research metadata, and an
  *optional* ``AgentExecutionTrace`` for step-by-step inspection.

``ConversationSession.turns`` is ``None`` by default; a client requesting
detail for one session (``HistoryQueryParams.include_turns=True``) gets it
populated. Similarly, ``ConversationTurn.execution_trace`` is ``None``
unless ``HistoryQueryParams.include_trace=True``. This mirrors the same
"lightweight by default, detailed on demand" principle already used for
``ChatResponse.execution_trace`` in ``schemas/chat.py`` — a returning user
scrolling their session list should not pay the serialization cost of
every past run's full agent trace just to see what they talked about.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated

from app.schemas.chat import ChatMode, ResearchMetadata
from app.schemas.common import AgentExecutionTrace, Citation, PaginationParams, TokenUsage
from app.schemas.report import ReportListItem

# =============================================================================
# Status
# =============================================================================


class SessionStatus(str, Enum):
    """Lifecycle status of a research session."""

    ACTIVE = "active"
    ARCHIVED = "archived"


# =============================================================================
# Turn (a single exchange within a session)
# =============================================================================


class ConversationTurn(BaseModel):
    """
    A single query/answer exchange within a research session.

    Shares its ``citations``/``research_metadata`` shapes with
    ``schemas.chat.ChatResponse`` since a turn *is* the persisted record
    of a past ``/chat`` response — the two are kept as separate models
    (rather than one reused for both) because a live response and a
    historical record have different required framing (a turn always
    belongs to a specific, already-resolved session and has a fixed
    ``sequence_number``, neither of which applies to an in-flight
    response).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "turn_id": "msg-4e5b4a1e",
                    "session_id": "sess-7f3a1c9d",
                    "sequence_number": 0,
                    "mode": "research",
                    "query": "Compare Llama 4 and Qwen3 for enterprise deployment",
                    "answer": "Qwen3 and Llama 4 differ primarily in...[1][2]",
                    "citations": [],
                    "research_metadata": {},
                    "execution_trace": None,
                    "created_at": "2026-07-29T10:15:03Z",
                }
            ]
        }
    )

    turn_id: str = Field(..., description="Unique identifier for this exchange.")
    session_id: str = Field(..., description="Research session this turn belongs to.")
    sequence_number: Annotated[int, Field(ge=0)] = Field(
        ..., description="0-indexed position of this turn within its session, for chronological ordering."
    )
    mode: ChatMode = Field(..., description="The workflow that produced this turn.")
    query: str = Field(..., description="The user's query for this turn.")
    answer: str = Field(..., description="The generated answer, with inline citation markers.")
    citations: List[Citation] = Field(default_factory=list, description="Sources backing this turn's answer.")
    research_metadata: ResearchMetadata = Field(
        ..., description="Summary of providers/models/retrieval used for this turn."
    )
    execution_trace: Optional[AgentExecutionTrace] = Field(
        default=None,
        description="Full step-by-step agent execution trace for this turn. Populated only when explicitly requested.",
    )
    created_at: datetime = Field(..., description="When this turn was generated.")


# =============================================================================
# Session summary
# =============================================================================


class ConversationSession(BaseModel):
    """
    Session-level summary, the primary unit ``GET /history`` returns.

    ``turns`` is omitted (``None``) in the default, paginated session-list
    view and populated only when a client requests detail for a specific
    session — see the module docstring.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "session_id": "sess-7f3a1c9d",
                    "title": "Llama 4 vs Qwen3 comparison",
                    "status": "active",
                    "created_at": "2026-07-29T10:15:03Z",
                    "updated_at": "2026-07-29T10:22:41Z",
                    "turn_count": 3,
                    "document_count": 2,
                    "report_count": 1,
                    "preview": "Compare Llama 4 and Qwen3 for enterprise deployment",
                    "latest_report": None,
                    "total_token_usage": {"prompt_tokens": 9820, "completion_tokens": 2140, "total_tokens": 11960},
                    "turns": None,
                }
            ]
        }
    )

    session_id: str = Field(..., description="Unique identifier for this research session.")
    title: Optional[str] = Field(
        default=None, description="Session title, client-supplied or auto-generated from the first query."
    )
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Lifecycle status of this session.")
    created_at: datetime = Field(..., description="When this session was created.")
    updated_at: datetime = Field(..., description="When this session was last active (its most recent turn or upload).")
    turn_count: Annotated[int, Field(ge=0)] = Field(..., description="Total number of exchanges in this session.")
    document_count: Annotated[int, Field(ge=0)] = Field(
        default=0, description="Number of documents uploaded into this session."
    )
    report_count: Annotated[int, Field(ge=0)] = Field(
        default=0, description="Number of reports generated from this session."
    )
    preview: Optional[str] = Field(
        default=None, description="Short excerpt (typically the first query) shown in a session list UI."
    )
    latest_report: Optional[ReportListItem] = Field(
        default=None, description="Reference to the most recently generated report for this session, if any."
    )
    total_token_usage: Optional[TokenUsage] = Field(
        default=None, description="Aggregate LLM token usage across every turn in this session."
    )
    turns: Optional[List[ConversationTurn]] = Field(
        default=None,
        description=(
            "Full turn history for this session. None in the default paginated session-list view; "
            "populated when a client requests detail for this specific session."
        ),
    )

    @model_validator(mode="after")
    def _validate_turns_consistency(self) -> "ConversationSession":
        # A single bound covers both cases: turns is a subset/page of the
        # session's full history, so it may never exceed turn_count —
        # including the turn_count == 0 case, where any non-empty turns
        # list already violates this same inequality.
        if self.turns is not None and len(self.turns) > self.turn_count:
            raise ValueError("turns cannot contain more items than turn_count.")
        return self


# =============================================================================
# Query parameters
# =============================================================================


class HistoryQueryParams(PaginationParams):
    """
    Query parameters for ``GET /history``.

    Extends the standard ``PaginationParams`` (``page``/``page_size``) with
    history-specific filters and detail-level toggles, used as a single
    FastAPI dependency (``Depends()``) on the route.
    """

    session_id: Optional[str] = Field(
        default=None,
        description="Return only this session. When set, include_turns defaults to true at the service layer.",
    )
    status: Optional[SessionStatus] = Field(default=None, description="Filter sessions by lifecycle status.")
    include_turns: bool = Field(
        default=False, description="Populate ConversationSession.turns for each returned session."
    )
    include_trace: bool = Field(
        default=False,
        description="When include_turns is also true, populate each turn's execution_trace. Ignored otherwise.",
    )
