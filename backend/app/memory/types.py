"""
Types for the Memory subsystem's output: a bounded, ready-to-inject
account of a session's prior conversation.

Deliberately not a reuse of ``schemas.history.ConversationTurn`` -- that
type carries citations, ``ResearchMetadata``, and an optional execution
trace, none of which a model needs in order to remember what was said.
This mirrors the same judgment call already made for
``app.agents.fact_checker._FactCheckOutput`` and
``app.agents.coordinator._FollowUpSuggestions``: reuse an existing schema
when it fits, define a new minimal one when it doesn't.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class MemoryTurn(BaseModel):
    """
    The minimal record of one prior turn needed for LLM context: what was
    asked and what was answered. Order within a ``MemoryContext.recent_turns``
    list conveys chronology (oldest first) -- no separate sequence number
    is carried, since nothing downstream keys off of one.
    """

    query: str = Field(..., description="The user's query for this prior turn.")
    answer: str = Field(..., description="The final answer given for this prior turn.")
    created_at: datetime = Field(..., description="When this turn was generated.")


class MemoryContext(BaseModel):
    """
    A session's conversation history, already bounded and ready to inject
    into a prompt: the most recent turns verbatim, plus an optional
    compacted summary of everything older than that window.

    Ordering contract: ``recent_turns`` is always chronological,
    oldest-to-newest -- the same order a transcript reads in, and the
    order a prompt should present them in so "the most recent turn" is
    whatever appears last. This is a contract of the type itself, not an
    incidental property of how ``app.memory.compaction`` happens to build
    it today: any producer of a ``MemoryContext`` (compaction now,
    something else later) must preserve it, and any consumer
    (``app.agents.planner``, ``app.agents.coordinator``) may rely on it
    without re-sorting or re-deriving recency from ``created_at`` itself.

    ``summary`` is ``None`` until a session has produced enough turns to
    need compaction (see ``app.memory.compaction``) -- most sessions never
    reach that point, and this field distinguishes "no summary exists yet"
    from "the summary is an empty string."
    """

    recent_turns: List[MemoryTurn] = Field(
        default_factory=list,
        description="The most recent turns of this session, in chronological order (oldest first), kept verbatim.",
    )
    summary: Optional[str] = Field(
        default=None, description="Compacted account of turns older than recent_turns, if compaction has occurred."
    )

    @property
    def is_empty(self) -> bool:
        """
        True when there is nothing to include -- a brand-new session, or
        (defensively) a session whose turns somehow produced neither
        recent turns nor a summary. Both consumers of ``MemoryContext``
        (``app.agents.planner``, ``app.agents.coordinator``) need this
        identical check before deciding whether to add a history section
        to their prompt at all.
        """
        return not self.recent_turns and not self.summary
