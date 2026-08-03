"""
Compacts turns that have aged out of a session's recent-history window
into a running summary, so a long session's memory stays bounded without
ever discarding what happened earlier in the conversation.

Framework- and database-agnostic, mirroring ``app.rag.ingest``'s own
design: this module has no FastAPI or SQLAlchemy import anywhere, and
depends only on the ``LLMService`` interface (injected), never a concrete
provider SDK -- the same "framework-agnostic means no web/DB framework
coupling, not no interface dependency" precedent ``ingest_document``
already established.

Deliberately incremental, not "re-summarize everything older than the
window from scratch every request": ``AgentSettings.max_chat_history``
defines a fixed-size sliding window, so once a session has exceeded it,
exactly one turn ages out of that window per new turn added, forever.
Re-summarizing the entire (ever-growing) older-turns bucket on every
request would both make the persisted summary pointless (why store
something always regenerated?) and require fetching a session's whole
unbounded turn history every request. Folding in only the newly
aged-out turns keeps each compaction call cheap and its cost independent
of session length. Figuring out *which* turns newly aged out is
``app.services.memory_service.MemoryService``'s job (it owns the
DB-specific bookkeeping); this module only does the pure "fold these new
turns into the existing summary" step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.core.exceptions import LLMServiceError
from app.core.interfaces.llm_service import LLMService
from app.core.logging import get_logger
from app.memory.types import MemoryContext, MemoryTurn

logger = get_logger(__name__)

_TEMPERATURE = 0.2
_MAX_ANSWER_CHARS_PER_TURN = 500

_SYSTEM_PROMPT = (
    "You maintain a running summary of an ongoing conversation between a user and a "
    "research assistant, so the assistant can remember earlier context without "
    "re-reading the entire transcript every time. Given the summary so far (if any) "
    "and the next batch of turns that happened after it, produce an updated summary "
    "that preserves every important fact, decision, or topic from both -- concise, "
    "in plain prose, not a transcript. Do not lose information from the prior "
    "summary; extend it, don't replace it with only the new turns."
)


@dataclass(frozen=True)
class CompactionOutcome:
    """
    ``build_memory_context``'s result: the ``MemoryContext`` to inject
    into a request, plus whether the summary text actually changed this
    call.

    ``summary_changed`` exists because ``MemoryContext.summary`` alone
    cannot distinguish "nothing needed folding in" or "summarization
    failed and gracefully kept the old value" from "summarization
    succeeded and genuinely updated the summary" -- comparing the
    returned text against the previous value by equality would be a
    fragile proxy (a real update could coincidentally produce identical
    text). ``MemoryService`` needs this signal to decide whether it is
    safe to advance ``ResearchSession.conversation_summary_through_sequence``:
    advancing it when the summary text was NOT actually updated would
    mark turns as covered that were never truly folded in, permanently
    losing their content (the watermark's own idempotency guarantee,
    ADR-031, cuts both ways -- it must never advance on a no-op, or it
    would silently corrupt memory instead of protecting it).
    """

    context: MemoryContext
    summary_changed: bool


async def build_memory_context(
    *,
    recent_turns: List[MemoryTurn],
    newly_aged_out_turns: List[MemoryTurn],
    existing_summary: Optional[str],
    llm_service: LLMService,
) -> CompactionOutcome:
    """
    Build the ``MemoryContext`` to inject into a request, updating the
    summary only if turns have newly aged out of the recent window since
    it was last computed.

    Args:
        recent_turns: The bounded, verbatim recent-history window
            (chronological, oldest first) -- passed through unchanged;
            this function never trims or reorders it, that's the caller's
            responsibility (mirroring ``MemoryContext.recent_turns``'s own
            ordering contract).
        newly_aged_out_turns: Turns that fell out of the recent window
            since the summary was last updated, chronological (oldest
            first). Empty on the common-case request (a session still
            within its recent window, or one where the window hasn't
            shifted since last time) -- no LLM call happens in that case.
        existing_summary: Whatever summary already covers everything
            older than ``newly_aged_out_turns``, or ``None`` if none
            exists yet.
        llm_service: Injected ``LLMService``, used only when there is
            something new to fold in.

    Returns:
        A ``CompactionOutcome``. ``summary_changed`` is ``False`` whenever
        ``context.summary`` is a passthrough of ``existing_summary``
        (nothing to fold in, or summarization failed), and ``True`` only
        when a real, successful summarization call produced it.
    """
    if not newly_aged_out_turns:
        return CompactionOutcome(
            context=MemoryContext(recent_turns=recent_turns, summary=existing_summary), summary_changed=False
        )

    updated_summary, changed = await _summarize(newly_aged_out_turns, existing_summary, llm_service)
    return CompactionOutcome(
        context=MemoryContext(recent_turns=recent_turns, summary=updated_summary), summary_changed=changed
    )


async def _summarize(
    newly_aged_out_turns: List[MemoryTurn],
    existing_summary: Optional[str],
    llm_service: LLMService,
) -> Tuple[Optional[str], bool]:
    prompt = _build_prompt(newly_aged_out_turns, existing_summary)
    try:
        response = await llm_service.generate(prompt, system_prompt=_SYSTEM_PROMPT, temperature=_TEMPERATURE)
    except LLMServiceError:
        # Compaction is request-lifecycle housekeeping, not a step in the
        # user's own request -- a failure here must not break the chat
        # request over a memory bookkeeping problem. Keep whatever
        # summary already existed rather than losing it or raising, and
        # report summary_changed=False so the caller knows not to advance
        # its watermark over turns that were never actually folded in.
        logger.warning(
            "Memory compaction failed; keeping the existing summary unchanged",
            extra={"newly_aged_out_turn_count": len(newly_aged_out_turns)},
        )
        return existing_summary, False

    return response.content.strip(), True


def _build_prompt(newly_aged_out_turns: List[MemoryTurn], existing_summary: Optional[str]) -> str:
    summary_section = existing_summary if existing_summary else "(no summary yet -- this is the first compaction)"
    turns_block = "\n".join(f"User: {turn.query}\nAssistant: {_truncate(turn.answer)}" for turn in newly_aged_out_turns)
    return (
        f"Summary so far:\n{summary_section}\n\n"
        f"New turns to fold in:\n{turns_block}\n\n"
        "Produce the updated summary now."
    )


def _truncate(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= _MAX_ANSWER_CHARS_PER_TURN:
        return stripped
    return stripped[:_MAX_ANSWER_CHARS_PER_TURN].rstrip() + "..."
