"""
ORM model for uploaded documents.

See the database layer design discussion for the full justification.
Summary: session-scoped (not a cross-session library — see that
discussion for why), flattens ``ProcessingStatistics`` directly onto the
row (1:1, always present once ingestion completes, no independent
lifecycle), and deliberately does NOT model individual chunks — those
live exclusively in ChromaDB, with this table tracking only the
aggregate ``chunk_count``.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base
from app.database.types import UTCDateTime
from app.models.enums import IngestionStatus, UploadStatus
from app.utils.time import utc_now


class Document(Base):
    """One uploaded document, belonging to exactly one session."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("session_id", "checksum_sha256", name="uq_documents_session_checksum"),
        Index("ix_documents_session_id", "session_id"),
        Index("ix_documents_ingestion_status", "ingestion_status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)

    # --- File-level metadata (mirrors schemas.upload.DocumentMetadata) ---
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_extension: Mapped[str] = mapped_column(String(20), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)

    # --- Status ---
    upload_status: Mapped[UploadStatus] = mapped_column(
        SAEnum(UploadStatus, name="upload_status", values_callable=lambda enum_cls: [m.value for m in enum_cls]),
        nullable=False,
        default=UploadStatus.PENDING,
    )
    ingestion_status: Mapped[IngestionStatus] = mapped_column(
        SAEnum(IngestionStatus, name="ingestion_status", values_callable=lambda enum_cls: [m.value for m in enum_cls]),
        nullable=False,
        default=IngestionStatus.NOT_STARTED,
    )
    indexed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # --- Flattened ProcessingStatistics (1:1, populated once ingestion progresses) ---
    processing_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    extraction_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    chunking_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    embedding_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    extracted_text_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    average_chunk_size: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    embedding_dimension: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    error_code: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now, onupdate=utc_now)

    session: Mapped["ResearchSession"] = relationship(  # type: ignore[name-defined]  # noqa: F821 - forward ref to conversation.py
        back_populates="documents"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Document id={self.id!r} filename={self.filename!r} ingestion_status={self.ingestion_status.value}>"
