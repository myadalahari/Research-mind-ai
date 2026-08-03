"""
``MemoryService`` -- the sole database-aware entry point into the Memory
subsystem (``app.memory``, see that package's own docstring and ADR-031).

Framework-agnostic like ``HistoryService`` (no FastAPI import), and like
that service, has no ABC in front of it: there is exactly one persistence
technology in this project, so a swappable-backend interface here would
be an abstraction with nothing to abstract over. Opens its own
``database.session()`` unit of work per call, mirroring
``HistoryService``'s own pattern, rather than participating in a
caller-managed transaction.

Owns all of the delta bookkeeping ``app.memory.compaction`` deliberately
does not: figuring out exactly which turns have newly aged out of the
recent-history window since the summary was last updated. That
computation is always driven directly from the persisted watermark
(``ResearchSession.conversation_summary_through_sequence``) via
``ConversationRepository.get_turns_in_sequence_range`` -- never inferred
from turn counts, timestamps, or window size alone -- so it stays correct
even if retention/windowing policy changes later.
"""

from __future__ import annotations

from typing import List, Optional

from app.core.config import AgentSettings
from app.core.interfaces.llm_service import LLMService
from app.core.logging import get_logger
from app.database.repositories.conversation_repository import ConversationRepository
from app.database.session import Database
from app.memory.compaction import build_memory_context
from app.memory.types import MemoryContext, MemoryTurn
from app.models.conversation import ConversationTurn

logger = get_logger(__name__)


class MemoryService:
    """
    Loads a session's recent conversation history, compacting older turns
    into a running summary when necessary, and returns a ``MemoryContext``
    ready for ``ChatService`` to inject into the graph.
    """

    def __init__(self, database: Database, llm_service: LLMService, agent_settings: AgentSettings) -> None:
        self._database = database
        self._llm_service = llm_service
        self._agent_settings = agent_settings

    async def load_context(self, session_id: Optional[str]) -> MemoryContext:
        """
        Load (and, if needed, compact) the given session's conversation
        history.

        Args:
            session_id: The session to load history for. ``None`` means a
                brand-new conversation -- guaranteed to have zero prior
                turns, so this short-circuits to an empty ``MemoryContext``
                without touching the database at all, the same "a session
                that doesn't exist yet has no history" reasoning
                ``retriever_agent.py``'s ``_resolve_scope`` already
                established for document retrieval.

        Returns:
            A ``MemoryContext`` with the bounded recent-turns window and
            (if applicable) an up-to-date compacted summary. Never raises
            over a missing or empty session -- both degrade to an empty
            ``MemoryContext``, since providing context is this method's
            job, not enforcing that a session exists.
        """
        if session_id is None:
            return MemoryContext()

        async with self._database.session() as session:
            repo = ConversationRepository(session)
            session_row = await repo.get_by_id(session_id)
            if session_row is None:
                logger.warning(
                    "MemoryService.load_context called with an unknown session_id", extra={"session_id": session_id}
                )
                return MemoryContext()

            watermark = session_row.conversation_summary_through_sequence
            since_watermark = await repo.get_turns_in_sequence_range(
                session_id, min_sequence_number=(watermark + 1 if watermark is not None else 0)
            )

            max_recent = self._agent_settings.max_chat_history
            if len(since_watermark) <= max_recent:
                newly_aged_out_rows: List[ConversationTurn] = []
                recent_rows = since_watermark
            else:
                split = len(since_watermark) - max_recent
                newly_aged_out_rows = since_watermark[:split]
                recent_rows = since_watermark[split:]

            recent_turns = [_to_memory_turn(row) for row in recent_rows]
            newly_aged_out_turns = [_to_memory_turn(row) for row in newly_aged_out_rows]

            outcome = await build_memory_context(
                recent_turns=recent_turns,
                newly_aged_out_turns=newly_aged_out_turns,
                existing_summary=session_row.conversation_summary,
                llm_service=self._llm_service,
            )

            if outcome.summary_changed:
                # Only ever advance the watermark when the summary text
                # itself was genuinely updated (CompactionOutcome.summary_changed) --
                # advancing it on a no-op or a gracefully-degraded failure
                # would mark these turns "covered" without their content
                # ever having been folded in, permanently losing it. The
                # boundary is read directly off the ORM rows that were
                # actually summarized, never recomputed or inferred.
                through_sequence_number = max(row.sequence_number for row in newly_aged_out_rows)
                # By construction (see build_memory_context/_summarize in
                # app.memory.compaction), summary_changed=True only comes
                # from a successful LLM summarization call, which always
                # produces a real summary string -- never from the
                # existing-summary passthrough branch, where it's paired
                # with summary_changed=False instead. So context.summary
                # is never None here at runtime; this assert documents
                # that invariant explicitly and lets the type checker
                # narrow Optional[str] to str, rather than the two facts
                # (summary_changed and context.summary's nullability)
                # living only in two different, unconnected files.
                assert outcome.context.summary is not None
                await repo.update_summary(
                    session_row, summary=outcome.context.summary, through_sequence_number=through_sequence_number
                )
                # Deliberately no touch_session() here: compaction is
                # internal bookkeeping, not user-facing conversational
                # activity, and updated_at's documented purpose (sorting
                # the history list by "most recent activity") shouldn't be
                # perturbed by a background summarization side effect.

            return outcome.context


def _to_memory_turn(row: ConversationTurn) -> MemoryTurn:
    return MemoryTurn(query=row.query, answer=row.answer, created_at=row.created_at)
