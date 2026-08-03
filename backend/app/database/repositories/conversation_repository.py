"""
Repository for the conversation aggregate: ``ResearchSession`` (the
aggregate root) and its children ``ConversationTurn``, ``TurnCitation``,
and ``AgentExecutionStep``.

See the database layer design discussion and the Architecture Decision
Log for the full rationale. Summary of what's encoded here:

* ``list_summaries()`` computes ``turn_count``/``document_count``/
  ``report_count``/``total_token_usage``/``preview`` for a whole page of
  sessions in a single query, using independent correlated scalar
  subqueries per aggregate rather than a join + GROUP BY. Joining
  ``conversation_turns``, ``documents``, and ``reports`` together would
  multiply rows (a session with 3 turns and 2 documents produces 6 joined
  rows before any grouping), corrupting naive ``COUNT()`` results;
  correlated subqueries avoid that fan-out entirely.
* ``document_count``/``report_count`` are computed here even though
  ``Document``/``Report`` belong to other model modules -- a deliberate,
  narrow exception to per-model repository boundaries. These are
  read-only counts against the aggregate root's own children, and
  computing them anywhere else would reintroduce the N+1 problem this
  method exists to avoid. Fetching a *full* related row (e.g.
  ``ConversationSession.latest_report``'s title/format/status) is
  different in kind, not just degree, and stays ``ReportRepository``'s
  job -- ``HistoryService`` composes the two.
* ``add_turn()`` assigns citations/execution steps via the ORM
  relationship and flushes once; SQLAlchemy's existing
  ``cascade="all, delete-orphan"`` (declared in ``models/conversation.py``)
  persists the children in the same flush as the parent.
* ``touch_session()`` exists because inserting a turn does NOT update its
  parent session's ``updated_at`` -- ``onupdate=utc_now`` only fires when
  the session row itself is part of an UPDATE, never as a side effect of
  a child insert. Since ``ConversationSession.updated_at`` and
  ``ix_sessions_updated_at`` are both meant to reflect "most recent
  activity," callers must call this explicitly alongside ``add_turn()``.
* ``get_turns_in_sequence_range()`` / ``update_summary()`` (Phase 7,
  Memory -- see ADR-031) exist so ``MemoryService`` can compute and
  persist compaction deltas without ever re-fetching a session's full
  turn history or updating the summary text and its watermark out of
  step with each other -- ``update_summary()`` refuses (``ValueError``)
  to move the watermark backward, which is what makes compaction
  idempotent across retries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.database.repositories.base_repository import BaseRepository
from app.models.conversation import AgentExecutionStep, ConversationTurn, ResearchSession, TurnCitation
from app.models.document import Document
from app.models.enums import SessionStatus
from app.models.report import Report
from app.utils.time import utc_now


@dataclass(frozen=True)
class SessionSummary:
    """
    One row of ``list_summaries()``/``get_summary()``: a ``ResearchSession``
    plus its computed aggregates.

    Kept separate from ``ResearchSession`` itself (rather than bolting
    these fields onto the ORM model) because these are query-time
    aggregates, not persisted columns -- exactly the denormalized-counter-
    drift problem the database design discussion called out for
    ``schemas.history.ConversationSession``.
    """

    session: ResearchSession
    turn_count: int
    document_count: int
    report_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    preview: Optional[str]


class ConversationRepository(BaseRepository[ResearchSession]):
    """Repository for ``ResearchSession`` and its owned children."""

    model = ResearchSession

    # -------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------

    async def create_session(self, session_id: str, *, title: Optional[str] = None) -> ResearchSession:
        """Create and persist a new, active research session."""
        return await self.add(ResearchSession(id=session_id, title=title, status=SessionStatus.ACTIVE))

    async def touch_session(self, session: ResearchSession) -> None:
        """
        Bump ``session.updated_at`` to now and flush.

        Must be called explicitly whenever a session gains new activity
        that doesn't itself update the session row (e.g. after
        ``add_turn()``) -- see the module docstring for why this can't be
        automatic.
        """
        session.updated_at = utc_now()
        await self.flush()

    # -------------------------------------------------------------------
    # Turns
    # -------------------------------------------------------------------

    async def next_sequence_number(self, session_id: str) -> int:
        """
        Return the next 0-indexed ``sequence_number`` for a new turn in
        this session (one past the current maximum, or 0 if the session
        has no turns yet).

        Exposed so callers (the future ``ChatService``) don't have to
        track sequence numbers themselves -- and so the
        ``(session_id, sequence_number)`` uniqueness invariant is computed
        in exactly one place. Note this project is single-user/single-
        process (see the database design discussion), so the trivial
        read-then-write race here is an accepted, documented limitation,
        not an oversight: a concurrent write would surface as a
        ``RecordConflictError`` from ``add_turn()`` rather than silently
        corrupting data.
        """
        stmt = select(func.coalesce(func.max(ConversationTurn.sequence_number), -1) + 1).where(
            ConversationTurn.session_id == session_id
        )
        result = await self._execute(stmt, f"Failed to compute next sequence number for session {session_id!r}.")
        return int(result.scalar_one())

    async def add_turn(
        self,
        turn: ConversationTurn,
        *,
        citations: Sequence[TurnCitation] = (),
        execution_steps: Sequence[AgentExecutionStep] = (),
    ) -> ConversationTurn:
        """
        Insert ``turn`` together with its citations and execution steps in
        one flush.

        Does NOT call ``touch_session()`` -- composing that call is the
        caller's responsibility, since a caller inserting several turns in
        one unit of work (unlikely today, but not precluded) should only
        need to touch the session once.
        """
        turn.citations = list(citations)
        turn.execution_steps = list(execution_steps)
        self._session.add(turn)
        await self._flush(f"Failed to create turn for session {turn.session_id!r}.")
        return turn

    async def get_turns(
        self,
        session_id: str,
        *,
        with_citations: bool = True,
        with_execution_steps: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ConversationTurn]:
        """
        Return this session's turns in chronological order, eagerly
        loading the requested relationships.

        Eager loading (``selectinload``) is required, not optional: these
        rows are read within one ``async with database.session()`` block
        and then handed back to a service/router that maps them to
        Pydantic schemas *after* this method returns. A lazy-loaded
        relationship accessed outside its originating async context raises
        ``MissingGreenlet``, so anything the caller needs must be loaded
        up front.
        """
        stmt = (
            select(ConversationTurn)
            .where(ConversationTurn.session_id == session_id)
            .order_by(ConversationTurn.sequence_number)
        )
        if with_citations:
            stmt = stmt.options(selectinload(ConversationTurn.citations))
        if with_execution_steps:
            stmt = stmt.options(selectinload(ConversationTurn.execution_steps))
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self._execute(stmt, f"Failed to list turns for session {session_id!r}.")
        return list(result.scalars().unique().all())

    async def get_turns_in_sequence_range(
        self,
        session_id: str,
        *,
        min_sequence_number: Optional[int] = None,
        max_sequence_number: Optional[int] = None,
    ) -> List[ConversationTurn]:
        """
        Return this session's turns whose ``sequence_number`` falls within
        ``[min_sequence_number, max_sequence_number]`` (either bound
        omitted means unbounded on that side), in chronological order.

        Built for ``MemoryService`` (Phase 7, see ADR-031): one flexible
        range query serves both "the current recent-history window"
        (``min_sequence_number=<turn_count - max_chat_history>``) and "the
        turns that newly aged out of it since the summary was last
        updated" (``min_sequence_number=<watermark + 1>``,
        ``max_sequence_number=<window's lower bound - 1>``) -- deliberately
        not a reuse of ``get_turns()``, which defaults to eager-loading
        ``citations``/``execution_steps`` via ``selectinload`` for the
        history API's needs. Memory only ever reads ``query``/``answer``/
        ``created_at`` (plain columns, not relationships), so eager-loading
        those relationships here would be pure wasted joins.
        """
        stmt = select(ConversationTurn).where(ConversationTurn.session_id == session_id)
        if min_sequence_number is not None:
            stmt = stmt.where(ConversationTurn.sequence_number >= min_sequence_number)
        if max_sequence_number is not None:
            stmt = stmt.where(ConversationTurn.sequence_number <= max_sequence_number)
        stmt = stmt.order_by(ConversationTurn.sequence_number)
        result = await self._execute(stmt, f"Failed to list turns in sequence range for session {session_id!r}.")
        return list(result.scalars().all())

    async def update_summary(self, session: ResearchSession, *, summary: str, through_sequence_number: int) -> None:
        """
        Persist a compacted conversation summary and advance its watermark
        together, in one call -- the mechanism that makes compaction
        idempotent (see ADR-031 and the field docstrings in
        ``models/conversation.py``).

        ``through_sequence_number`` must be the highest ``sequence_number``
        among the turns actually folded into ``summary`` this call, and
        must not move the watermark backward: a caller (``MemoryService``)
        determines which turns still need folding in by comparing against
        the *previous* value of ``conversation_summary_through_sequence``,
        so a regression here would make an already-covered turn look
        uncovered again and get folded into the summary a second time --
        exactly the double-summarization this method exists to prevent.
        Raises ``ValueError`` immediately if that invariant would be
        violated, rather than silently accepting a regression.
        """
        current = session.conversation_summary_through_sequence
        if current is not None and through_sequence_number <= current:
            raise ValueError(
                f"update_summary() called with through_sequence_number={through_sequence_number}, which does not "
                f"advance past the existing watermark ({current}) for session {session.id!r}. This would make "
                "already-summarized turns look uncovered again, causing them to be folded into the summary twice."
            )
        session.conversation_summary = summary
        session.conversation_summary_through_sequence = through_sequence_number
        await self.flush()

    # -------------------------------------------------------------------
    # Session summaries (see module docstring for the aggregation design)
    # -------------------------------------------------------------------

    async def list_summaries(
        self,
        *,
        status: Optional[SessionStatus] = None,
        limit: int,
        offset: int,
    ) -> List[SessionSummary]:
        """
        Return a page of sessions (most recently active first) with their
        aggregate counts and token usage, in one query.
        """
        stmt = self._summary_statement()
        if status is not None:
            stmt = stmt.where(ResearchSession.status == status)
        stmt = stmt.order_by(ResearchSession.updated_at.desc()).limit(limit).offset(offset)
        result = await self._execute(stmt, "Failed to list session summaries.")
        return [self._row_to_summary(row) for row in result.all()]

    async def get_summary(self, session_id: str) -> Optional[SessionSummary]:
        """Return one session's summary, or ``None`` if it doesn't exist."""
        stmt = self._summary_statement().where(ResearchSession.id == session_id)
        result = await self._execute(stmt, f"Failed to load summary for session {session_id!r}.")
        row = result.first()
        return self._row_to_summary(row) if row is not None else None

    async def count_sessions(self, *, status: Optional[SessionStatus] = None) -> int:
        """Return the total number of sessions matching ``status``, for pagination."""
        filters = () if status is None else (ResearchSession.status == status,)
        return await self.count(*filters)

    def _summary_statement(self):
        """
        Build the shared ``SELECT`` used by both ``list_summaries()`` and
        ``get_summary()``: ``ResearchSession`` plus its aggregate columns,
        unfiltered and unordered -- callers add ``.where()``/``.order_by()``/
        ``.limit()``/``.offset()`` as needed.
        """
        turn_count = (
            select(func.count(ConversationTurn.id))
            .where(ConversationTurn.session_id == ResearchSession.id)
            .correlate(ResearchSession)
            .scalar_subquery()
        )
        document_count = (
            select(func.count(Document.id))
            .where(Document.session_id == ResearchSession.id)
            .correlate(ResearchSession)
            .scalar_subquery()
        )
        report_count = (
            select(func.count(Report.id))
            .where(Report.session_id == ResearchSession.id)
            .correlate(ResearchSession)
            .scalar_subquery()
        )
        prompt_tokens = (
            select(func.coalesce(func.sum(ConversationTurn.prompt_tokens), 0))
            .where(ConversationTurn.session_id == ResearchSession.id)
            .correlate(ResearchSession)
            .scalar_subquery()
        )
        completion_tokens = (
            select(func.coalesce(func.sum(ConversationTurn.completion_tokens), 0))
            .where(ConversationTurn.session_id == ResearchSession.id)
            .correlate(ResearchSession)
            .scalar_subquery()
        )
        total_tokens = (
            select(func.coalesce(func.sum(ConversationTurn.total_tokens), 0))
            .where(ConversationTurn.session_id == ResearchSession.id)
            .correlate(ResearchSession)
            .scalar_subquery()
        )
        # The first turn's query (lowest sequence_number) doubles as a
        # session preview -- matches schemas.history.ConversationSession.preview's
        # documented default ("typically the first query").
        preview = (
            select(ConversationTurn.query)
            .where(ConversationTurn.session_id == ResearchSession.id)
            .order_by(ConversationTurn.sequence_number.asc())
            .limit(1)
            .correlate(ResearchSession)
            .scalar_subquery()
        )
        return select(
            ResearchSession,
            turn_count.label("turn_count"),
            document_count.label("document_count"),
            report_count.label("report_count"),
            prompt_tokens.label("prompt_tokens"),
            completion_tokens.label("completion_tokens"),
            total_tokens.label("total_tokens"),
            preview.label("preview"),
        )

    @staticmethod
    def _row_to_summary(row) -> SessionSummary:
        return SessionSummary(
            session=row[0],
            turn_count=int(row.turn_count),
            document_count=int(row.document_count),
            report_count=int(row.report_count),
            prompt_tokens=int(row.prompt_tokens),
            completion_tokens=int(row.completion_tokens),
            total_tokens=int(row.total_tokens),
            preview=row.preview,
        )
