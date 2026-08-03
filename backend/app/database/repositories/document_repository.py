"""
Repository for ``Document`` (uploaded, session-scoped files).

See the database layer design discussion and the Architecture Decision
Log for the full rationale. Summary of what's encoded here:

* ``Document`` has no owned children (unlike ``ResearchSession``), so this
  repository is narrower than ``ConversationRepository`` -- mostly query
  methods over the generic CRUD ``BaseRepository`` already provides.
* ``get_by_checksum()`` is a friendly pre-check for the same
  ``(session_id, checksum_sha256)`` uniqueness the database already
  enforces (ADR-008: documents are deliberately deduplicated per-session,
  not globally). It is a UX convenience, not the source of truth --
  ``add()``'s ``RecordConflictError`` translation (from
  ``base_repository.py``) remains the race-safe guarantee if two uploads
  for the same file land concurrently.
* ``list_by_session()``/``count_by_session()`` exist for two already-
  identified consumers: ``schemas.upload.DocumentMetadata``'s module
  docstring explicitly calls out reuse by a future ``GET /sources``
  listing, and ``RAGSettings.MAX_DOCUMENTS_PER_SESSION`` needs a cheap
  count to enforce. No status-transition helpers (e.g. a
  "mark ingestion completed" convenience) are added here -- that's
  ``IngestionService``'s (Phase 5) business-rule orchestration once its
  actual pipeline shape exists, not something to guess at from the
  persistence layer.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select

from app.database.repositories.base_repository import BaseRepository
from app.models.document import Document
from app.models.enums import IngestionStatus


class DocumentRepository(BaseRepository[Document]):
    """Repository for ``Document`` rows."""

    model = Document

    async def get_by_checksum(self, session_id: str, checksum_sha256: str) -> Optional[Document]:
        """
        Return the document with this checksum within this session, or
        ``None``.

        Deliberately scoped to ``session_id`` -- the same file uploaded
        into two different sessions is not a duplicate (ADR-008: documents
        are session-scoped, not a cross-session library), so this must
        never check the checksum globally.
        """
        stmt = select(Document).where(
            Document.session_id == session_id,
            Document.checksum_sha256 == checksum_sha256,
        )
        result = await self._execute(stmt, f"Failed to look up document by checksum in session {session_id!r}.")
        return result.scalar_one_or_none()

    async def list_by_session(
        self,
        session_id: str,
        *,
        ingestion_status: Optional[IngestionStatus] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Document]:
        """
        Return this session's documents, oldest first (matching
        ``ResearchSession.documents``'s relationship ordering), optionally
        filtered by ingestion status and paginated.
        """
        stmt = select(Document).where(Document.session_id == session_id)
        if ingestion_status is not None:
            stmt = stmt.where(Document.ingestion_status == ingestion_status)
        stmt = stmt.order_by(Document.created_at.asc())
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self._execute(stmt, f"Failed to list documents for session {session_id!r}.")
        return list(result.scalars().all())

    async def count_by_session(self, session_id: str) -> int:
        """
        Return the number of documents in this session.

        Used to enforce ``RAGSettings.MAX_DOCUMENTS_PER_SESSION`` before
        accepting a new upload.
        """
        return await self.count(Document.session_id == session_id)
