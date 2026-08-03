"""
ORM model for generated research reports.

See the database layer design discussion (and the Architecture Decision
Log, docs/architecture-decisions.md) for the full justification. Summary:

* ``Report`` belongs to exactly one ``ResearchSession`` and has no
  lifecycle independent of it -- like ``ConversationTurn`` and
  ``Document``, it cascade-deletes with its parent session.
* The recursive/nested parts of ``ReportDocument``
  (``key_findings``, ``sections`` -- which themselves recurse into
  ``subsections`` and embed ``ReportTable`` rows, and ``citations``) are
  stored as JSON rather than normalized into relational tables. Nothing
  in this project ever queries *into* a report's structure (e.g. "find
  all sections titled X across reports") -- a report is always written
  and read as a single whole document -- so relational modeling here
  would add real complexity (self-referential FK + ordering columns for
  sections, a child table for tables, a child table for citations) for a
  query pattern that doesn't exist. JSON preserves
  ``core.interfaces.report_exporter.ReportDocument``'s exact shape, so a
  stored row round-trips losslessly through
  ``ReportDocument.model_validate(...)``.
* The exported file itself (Markdown/PDF bytes) lives on disk under
  ``ReportSettings.output_dir``, not in the database -- this row stores
  ``file_path``/``file_size_bytes``/``mime_type`` metadata about it, the
  same catalog/payload split used by virtually any production
  file-serving system.
* Token usage and generation latency are flattened directly onto the row
  (1:1, always present once generation starts, no independent
  lifecycle) -- the same pattern already used for
  ``ConversationTurn``'s flattened ``ResearchMetadata`` and
  ``Document``'s flattened ``ProcessingStatistics``.
* The JSON metadata column is named ``report_metadata``, not
  ``metadata`` -- ``metadata`` is reserved on every SQLAlchemy
  ``DeclarativeBase`` subclass (``Base.metadata`` is the schema
  registry), so a column of that name would shadow it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base
from app.database.types import UTCDateTime
from app.models.enums import ReportExportFormat, ReportGenerationStatus
from app.utils.time import utc_now


class Report(Base):
    """One generated research report, belonging to exactly one session."""

    __tablename__ = "reports"
    __table_args__ = (
        Index("ix_reports_session_id", "session_id"),
        Index("ix_reports_status", "status"),
        Index("ix_reports_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)

    title: Mapped[str] = mapped_column(String(300), nullable=False)
    status: Mapped[ReportGenerationStatus] = mapped_column(
        SAEnum(
            ReportGenerationStatus,
            name="report_generation_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=ReportGenerationStatus.QUEUED,
    )
    export_format: Mapped[ReportExportFormat] = mapped_column(
        SAEnum(
            ReportExportFormat,
            name="report_export_format",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
    )

    # --- Report content (see module docstring for the JSON-vs-relational rationale) ---
    executive_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_findings: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    sections: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSON, nullable=True, doc="Serialized List[core.interfaces.report_exporter.ReportSection]."
    )
    conclusion: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    citations: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSON, nullable=True, doc="Serialized List[core.interfaces.report_exporter.Citation]."
    )
    report_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True, doc="Serialized ReportDocument.metadata (arbitrary extra fields)."
    )
    generated_at: Mapped[Optional[datetime]] = mapped_column(
        UTCDateTime(), nullable=True, doc="When the report content itself was produced (ReportDocument.generated_at)."
    )

    # --- Exported file metadata (payload lives on disk, not in the DB) ---
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # --- Flattened generation metadata (mirrors ConversationTurn's ResearchMetadata pattern) ---
    generation_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    error_code: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now, onupdate=utc_now)

    session: Mapped["ResearchSession"] = relationship(  # type: ignore[name-defined]  # noqa: F821 - forward ref to conversation.py
        back_populates="reports"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Report id={self.id!r} session_id={self.session_id!r} status={self.status.value}>"
