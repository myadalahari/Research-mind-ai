"""
Repository for ``Report`` (generated research reports).

See the database layer design discussion and the Architecture Decision
Log for the full rationale. Summary of what's encoded here:

* ``Report`` has no owned children, so this is the leanest repository in
  the layer -- CRUD (inherited) plus two query methods for the only
  concretely identified consumers today: ``get_latest_by_session()`` for
  ``schemas.history.ConversationSession.latest_report``, and
  ``list_by_session()`` (with its necessary pagination companion
  ``count_by_session()``) for the documented-but-not-yet-built
  ``GET /reports`` listing (see ``schemas.report.ReportListItem``'s
  docstring).
* ``list_by_session()`` orders oldest-first for consistency with
  ``DocumentRepository.list_by_session()`` and with
  ``Report.created_at`` ascending, already the declared order on
  ``ResearchSession.reports`` -- one convention across sibling
  repositories, rather than a per-file guess at UI ordering.
  ``get_latest_by_session()`` is a separate, purpose-built query (most
  recent first, limit 1) rather than a reuse of ``list_by_session()``
  with a direction flag, since it has one specific, different need.
* No status-transition or content-write helpers, confirmed unnecessary
  now that ``ReportService`` (Phase 8, complete) actually exists:
  it writes a ``Report`` row exactly once per request, via the inherited
  ``BaseRepository.add()``, only after the final outcome (``COMPLETED``
  or ``FAILED``) is already known -- never an intermediate
  ``QUEUED``/``GENERATING`` row, and never an in-place status transition
  on an existing row. ``queued``/``generating``/``exporting`` remain on
  ``ReportGenerationStatus`` only for a possible future asynchronous/
  queued implementation (see that enum's own docstring); this repository
  still has no code path that produces or mutates a row into any of
  those three statuses, so no transition helper was added speculatively
  for them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from sqlalchemy import select

from app.database.repositories.base_repository import BaseRepository
from app.models.report import Report


class ReportRepository(BaseRepository[Report]):
    """Repository for ``Report`` rows."""

    model = Report

    async def get_latest_by_session(self, session_id: str) -> Optional[Report]:
        """
        Return this session's most recently created report, or ``None`` if
        it has none.

        Backs ``schemas.history.ConversationSession.latest_report``.
        """
        stmt = select(Report).where(Report.session_id == session_id).order_by(Report.created_at.desc()).limit(1)
        result = await self._execute(stmt, f"Failed to load latest report for session {session_id!r}.")
        return result.scalar_one_or_none()

    async def get_latest_by_sessions(self, session_ids: Sequence[str]) -> Dict[str, Report]:
        """
        Batch version of ``get_latest_by_session()`` for a page of sessions
        at once: returns a ``{session_id: latest Report}`` mapping,
        omitting sessions with no reports.

        Exists specifically so ``HistoryService`` can build
        ``ConversationSession.latest_report`` for a whole page of sessions
        (``GET /history``) in one query instead of one ``get_latest_by_session()``
        call per session -- the same N+1 concern
        ``ConversationRepository.list_summaries()`` (ADR-017) already
        exists to avoid, applied here to full report rows instead of
        counts. Implemented as a single ``ORDER BY session_id,
        created_at DESC`` scan reduced client-side (first row seen per
        ``session_id`` is that session's most recent report, since rows
        for the same session are contiguous and already ordered newest
        first) rather than a window-function query, keeping it portable
        across SQLite and Postgres without relying on dialect-specific SQL.
        """
        if not session_ids:
            return {}
        stmt = (
            select(Report)
            .where(Report.session_id.in_(session_ids))
            .order_by(Report.session_id, Report.created_at.desc())
        )
        result = await self._execute(stmt, "Failed to batch-load latest reports for a page of sessions.")
        latest: Dict[str, Report] = {}
        for report in result.scalars().all():
            latest.setdefault(report.session_id, report)
        return latest

    async def list_by_session(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Report]:
        """Return this session's reports, oldest first, optionally paginated."""
        stmt = select(Report).where(Report.session_id == session_id).order_by(Report.created_at.asc())
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self._execute(stmt, f"Failed to list reports for session {session_id!r}.")
        return list(result.scalars().all())

    async def count_by_session(self, session_id: str) -> int:
        """
        Return the number of reports in this session.

        The necessary pagination companion to ``list_by_session()`` --
        without it, a caller building a ``PaginatedResponse`` for a
        session's reports would have no way to compute ``total_items``.
        """
        return await self.count(Report.session_id == session_id)
