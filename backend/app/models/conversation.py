"""
ORM models for the conversation aggregate: ``ResearchSession`` (the
aggregate root), ``ConversationTurn``, ``AgentExecutionStep``, and
``TurnCitation``.

See the database layer design discussion for the full entity-relationship
justification (indexes, cascade behavior, uniqueness constraints). Summary
of what's encoded here:

* ``ResearchSession`` -> ``ConversationTurn`` -> {``AgentExecutionStep``,
  ``TurnCitation``} is a strict ownership hierarchy: nothing in this file
  has a lifecycle independent of its parent, so every FK in this module
  cascades on delete.
* Aggregate fields that would appear on ``schemas.history.ConversationSession``
  (``turn_count``, ``document_count``, ``total_token_usage``, ``preview``)
  are deliberately NOT columns here — they're computed by
  ``ConversationRepository`` via aggregate queries, to avoid denormalized-
  counter drift bugs. See that repository for the query implementations.
* ``ResearchMetadata`` (provider, model, retrieval/search flags and
  counts, latency, token usage) is flattened directly onto
  ``ConversationTurn`` rather than given its own table, since it's 1:1
  with the turn, always present, and has no independent lifecycle.
* ``ResearchSession.conversation_summary`` / ``conversation_summary_through_sequence``
  (Phase 7, Memory) are a paired watermark: the compacted account of
  older turns, and the highest ``ConversationTurn.sequence_number``
  already folded into it. Flattened onto the session for the same 1:1,
  no-independent-lifecycle reason as ``ResearchMetadata`` above, and
  always written together (see ``ConversationRepository.update_summary``)
  so compaction is idempotent -- see ADR-031.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base
from app.database.types import UTCDateTime
from app.models.enums import ChatMode, CitationSourceType, ExecutionStepStatus, SessionStatus
from app.utils.time import utc_now


class ResearchSession(Base):
    """The aggregate root: one bounded research investigation."""

    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_status", "status"),
        Index("ix_sessions_updated_at", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    status: Mapped[SessionStatus] = mapped_column(
        SAEnum(
            SessionStatus, name="session_status", values_callable=lambda enum_cls: [member.value for member in enum_cls]
        ),
        nullable=False,
        default=SessionStatus.ACTIVE,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now, onupdate=utc_now)

    # --- Memory (Phase 7): compacted account of turns older than the
    # recent-history window (see app.memory.compaction / ADR-031). Both
    # columns are always written together via
    # ConversationRepository.update_summary() -- never independently --
    # so they can never drift out of sync with each other. That pairing is
    # the concrete mechanism that makes compaction idempotent: a caller
    # determines "which turns still need folding in" by comparing a
    # turn's sequence_number against conversation_summary_through_sequence,
    # so a turn at or below the watermark is excluded by construction,
    # even across retries or repeated requests -- there is no state where
    # the watermark advanced but the summary text didn't (or vice versa)
    # that could cause the same turn to be folded in twice.
    conversation_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    conversation_summary_through_sequence: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    turns: Mapped[List["ConversationTurn"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ConversationTurn.sequence_number"
    )
    documents: Mapped[List["Document"]] = relationship(  # type: ignore[name-defined]  # noqa: F821 - forward ref to document.py
        back_populates="session", cascade="all, delete-orphan", order_by="Document.created_at"
    )
    reports: Mapped[List["Report"]] = relationship(  # type: ignore[name-defined]  # noqa: F821 - forward ref to report.py
        back_populates="session", cascade="all, delete-orphan", order_by="Report.created_at"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"<ResearchSession id={self.id!r} status={self.status.value} title={self.title!r}>"


class ConversationTurn(Base):
    """One query/answer exchange within a session."""

    __tablename__ = "conversation_turns"
    __table_args__ = (
        UniqueConstraint("session_id", "sequence_number", name="uq_turns_session_sequence"),
        Index("ix_turns_session_id", "session_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    mode: Mapped[ChatMode] = mapped_column(
        SAEnum(ChatMode, name="turn_mode", values_callable=lambda enum_cls: [member.value for member in enum_cls]),
        nullable=False,
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)

    # --- Flattened ResearchMetadata (see module docstring for rationale) ---
    llm_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(100), nullable=False)
    rag_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    web_search_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retrieved_chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    web_result_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)

    session: Mapped["ResearchSession"] = relationship(back_populates="turns")
    citations: Mapped[List["TurnCitation"]] = relationship(
        back_populates="turn", cascade="all, delete-orphan", order_by="TurnCitation.citation_id"
    )
    execution_steps: Mapped[List["AgentExecutionStep"]] = relationship(
        back_populates="turn", cascade="all, delete-orphan", order_by="AgentExecutionStep.started_at"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ConversationTurn id={self.id!r} session_id={self.session_id!r} seq={self.sequence_number}>"


class TurnCitation(Base):
    """
    One citation attached to a turn's answer.

    Denormalizes display fields (title, excerpt, page_number, ...) rather
    than relying solely on a join to ``documents`` — a citation is a
    historical record of what was cited at generation time, and must
    remain fully displayable even if its source ``Document`` is later
    deleted (hence ``document_id`` is nullable with ``ON DELETE SET NULL``,
    not ``CASCADE``).
    """

    __tablename__ = "turn_citations"
    __table_args__ = (
        UniqueConstraint("turn_id", "citation_id", name="uq_citations_turn_marker"),
        Index("ix_citations_turn_id", "turn_id"),
        Index("ix_citations_document_id", "document_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    turn_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversation_turns.id", ondelete="CASCADE"), nullable=False
    )
    citation_id: Mapped[str] = mapped_column(
        String(20), nullable=False, doc="The inline citation marker within this turn, e.g. '1'."
    )
    source_type: Mapped[CitationSourceType] = mapped_column(
        SAEnum(
            CitationSourceType,
            name="citation_source_type",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    url: Mapped[Optional[str]] = mapped_column(String(2000), nullable=True)
    source_filename: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc=(
            "Original filename of the cited document, for document citations. "
            "Kept distinct from title -- Document.title is an optional, "
            "client-supplied display title that may differ from the file's "
            "actual name, so a citation must denormalize both independently "
            "to remain accurate even after its source Document is deleted."
        ),
    )
    document_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_title: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    excerpt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    published_date: Mapped[Optional[datetime]] = mapped_column(UTCDateTime(), nullable=True)
    accessed_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime(), nullable=True)

    turn: Mapped["ConversationTurn"] = relationship(back_populates="citations")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<TurnCitation turn_id={self.turn_id!r} citation_id={self.citation_id!r} title={self.title!r}>"


class AgentExecutionStep(Base):
    """One LangGraph node execution within a turn's agent run."""

    __tablename__ = "agent_execution_steps"
    __table_args__ = (Index("ix_execution_steps_turn_id_started_at", "turn_id", "started_at"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    turn_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversation_turns.id", ondelete="CASCADE"), nullable=False
    )
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False)
    graph_node: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[ExecutionStepStatus] = mapped_column(
        SAEnum(
            ExecutionStepStatus,
            name="execution_step_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime(), nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    tool_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    turn: Mapped["ConversationTurn"] = relationship(back_populates="execution_steps")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AgentExecutionStep turn_id={self.turn_id!r} agent={self.agent_name!r} status={self.status.value}>"
