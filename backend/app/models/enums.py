"""
Persistence-layer enums.

Deliberately independent from the structurally-identical enums in
``app.schemas.*`` (``SessionStatus``, ``ChatMode``, ``UploadStatus``,
``IngestionStatus``, ``ReportExportFormat``, ``ReportGenerationStatus``,
``SourceType``, ``ExecutionStepStatus``) — not a shortcut duplication, but
a deliberate application of Clean Architecture's dependency rule: the
persistence layer must not depend on the outer API layer. Importing the
schema enums here would mean a change to the public API's enum values
could silently break persisted data semantics, and vice versa. The two
families are allowed — expected — to diverge independently as each layer
evolves; today they share the same values because that's the correct
mapping right now, not because they're the same type.

Every enum here is stored via SQLAlchemy's ``Enum`` type configured with
``values_callable`` (see ``models/conversation.py`` etc.) so the database
stores the string value (``"active"``), not the Python member name
(``"ACTIVE"``) — this keeps stored data human-readable in raw SQL and
stable across any future renaming of the enum member itself.
"""

from __future__ import annotations

from enum import Enum


class SessionStatus(str, Enum):
    """Lifecycle status of a research session."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class ChatMode(str, Enum):
    """Which workflow produced a given conversation turn."""

    CHAT = "chat"
    RESEARCH = "research"
    REPORT = "report"


class UploadStatus(str, Enum):
    """Lifecycle of a document's raw file transfer/storage."""

    PENDING = "pending"
    RECEIVED = "received"
    VALIDATED = "validated"
    STORED = "stored"
    REJECTED = "rejected"
    FAILED = "failed"


class IngestionStatus(str, Enum):
    """Lifecycle of the RAG ingestion pipeline for a stored document."""

    NOT_STARTED = "not_started"
    QUEUED = "queued"
    EXTRACTING = "extracting"
    CLEANING = "cleaning"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class ReportExportFormat(str, Enum):
    """Output format a report was (or will be) exported to."""

    MARKDOWN = "markdown"
    PDF = "pdf"


class ReportGenerationStatus(str, Enum):
    """Lifecycle of a report generation request."""

    QUEUED = "queued"
    GENERATING = "generating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class CitationSourceType(str, Enum):
    """Where a citation's underlying evidence came from."""

    DOCUMENT = "document"
    WEB = "web"


class ExecutionStepStatus(str, Enum):
    """Lifecycle status of a single agent/graph-node execution step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
