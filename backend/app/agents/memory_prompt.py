"""
Shared prompt-formatting helper for folding a ``MemoryContext`` (Phase 7,
``app.memory.types``) into an agent's prompt text.

This module exists because of an *observed*, not anticipated, duplication:
``app.agents.planner`` was implemented first with its own local
``_format_conversation_history``/truncation-bound logic. While implementing
the same need in ``app.agents.coordinator`` (``coordinator_chat_node``,
the only other approved consumer of ``conversation_history``), the turn/
summary *rendering* step -- chronological "User: .../Assistant: ..." pairs
with a bounded per-turn answer length -- came out materially identical
between the two: same reliance on ``MemoryContext.recent_turns``'s
documented oldest-first ordering contract, same reasoning for why a prior
answer must be bounded before entering a second prompt. Per this project's
stated preference (let a second implementation determine whether an
abstraction is warranted, rather than introducing one speculatively), this
module was only introduced once that duplication was directly observed.

What is deliberately NOT shared here: the surrounding prompt template
(section headers, how the current query is introduced, any node-specific
instructions). Those differ by node purpose -- the Planner needs a
different framing ("context only, still plan for the current query") than
the Coordinator's lightweight chat node ("respond to this new message") --
and forcing them into one shared template would be exactly the kind of
premature, one-size-fits-all abstraction this project's process avoids.
"""

from __future__ import annotations

from typing import Optional

from app.memory.types import MemoryContext

# The bound applied to each prior turn's answer before it enters a second
# prompt. Mirrors app.memory.compaction's own per-turn truncation bound
# (a distinct concern -- bounding a *consuming* agent's prompt size, not
# compaction's own summarization input -- that happens to reuse the same
# limit; not a value the two modules are required to stay in lockstep on).
MAX_HISTORY_ANSWER_CHARS = 300


def format_conversation_history(conversation_history: MemoryContext) -> Optional[str]:
    """
    Render a ``MemoryContext`` as plain text for prompt inclusion.

    Returns ``None`` when there is nothing to render (``conversation_history.
    is_empty``), so callers can cheaply branch to their unmodified,
    pre-Phase-7 prompt in the common single-turn case -- rather than every
    caller re-checking ``is_empty`` themselves before calling in.

    Relies directly on ``MemoryContext.recent_turns``'s documented
    chronological (oldest-first) ordering contract rather than re-sorting.
    """
    if conversation_history.is_empty:
        return None

    sections = []
    if conversation_history.summary:
        sections.append(f"Summary of earlier turns: {conversation_history.summary}")
    for turn in conversation_history.recent_turns:
        answer = turn.answer.strip()
        if len(answer) > MAX_HISTORY_ANSWER_CHARS:
            answer = answer[:MAX_HISTORY_ANSWER_CHARS].rstrip() + "..."
        sections.append(f"User: {turn.query}\nAssistant: {answer}")
    return "\n\n".join(sections)
