"""
The Coordinator agent: the research graph's entry point (dispatching a
lightweight response for ``ChatMode.CHAT`` instead of running the full
pipeline) and exit point (assembling the final answer/citations once the
Reviewer has approved a draft).

Built as two separate closure factories rather than one, since the two
jobs run at two different points in the graph with different inputs and
different failure semantics -- see each function's own docstring.
``LLMService`` is captured once at graph-build time, mirroring every
other LLM-backed node in this package.

Scope note, per this file's review requirements: this module intentionally
contains no routing/dispatch logic (no check of ``state["mode"]``, no
decision about which node runs next). *Which* of these two node functions
``graph.py`` calls, and when, is a conditional-edge concern that belongs
entirely to ``graph.py`` -- this file only provides the work each path
does once ``graph.py`` has already decided to run it. It also never reads
``verified_claims`` or ``review_feedback``: judging the draft is the
Reviewer's job, already done by the time ``finalize`` runs; the Coordinator
trusts that verdict rather than re-inspecting it.

Phase 7 (Memory): ``coordinator_chat_node`` is the second of exactly two
nodes (the other is the Planner) the approved Memory design names as a
consumer of ``ResearchGraphState["conversation_history"]`` -- it's the
one place ``ChatMode.CHAT``'s own docstring ("may be served from memory")
is actually realized. ``coordinator_finalize_node`` deliberately does NOT
read it: by the time finalize runs, the full pipeline (including the
Planner, which already had its own look at the history) has produced
``draft_answer``, so folding history in a second time here would be
redundant, not additive. Turn/summary rendering is shared with the
Planner via ``app.agents.memory_prompt`` (see that module's docstring);
the surrounding chat prompt template is local to this file, since a
lightweight conversational reply needs different framing than the
Planner's decomposition task.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable, List, Set, Tuple

from pydantic import BaseModel, Field

from app.agents.memory_prompt import format_conversation_history
from app.agents.state import AgentName, ResearchGraphState, track_step
from app.core.exceptions import CoordinatorError, LLMServiceError
from app.core.interfaces.llm_service import LLMService
from app.core.logging import get_logger
from app.memory.types import MemoryContext
from app.schemas.common import Citation

logger = get_logger(__name__)

_CHAT_TEMPERATURE = 0.5
_SUGGESTION_TEMPERATURE = 0.5
_MAX_SUGGESTIONS = 3

_CITATION_MARKER_PATTERN = re.compile(r"\[(\d+)\]")

_CHAT_SYSTEM_PROMPT = (
    "You are ResearchMind, a helpful research assistant. The user has sent a "
    "lightweight conversational message that does not require the full research "
    "pipeline (document retrieval, web search, multi-step verification) -- answer "
    "directly and conversationally. Do not fabricate citations; this response has "
    "no evidence sources attached to it. You may also be given earlier turns from "
    "this conversation, for context only -- use them to understand references and "
    "stay consistent with what was already said, but respond to the new message "
    "specifically, not to the earlier turns."
)

_SUGGESTIONS_SYSTEM_PROMPT = (
    "Given a user's query and the answer they were given, suggest natural, "
    "specific follow-up questions the user might want to ask next. Suggest up to "
    f"{_MAX_SUGGESTIONS}; suggest fewer (or none) if the topic doesn't naturally "
    "lead to further questions. Each suggestion should be a complete question a "
    "user could send as-is, not a topic label."
)


class _FollowUpSuggestions(BaseModel):
    """
    ``generate_structured`` envelope for the follow-up suggestions list.

    Not part of ``ResearchGraphState`` -- the graph stores
    ``follow_up_suggestions: List[str]`` directly; this wrapper exists only
    because ``generate_structured`` requires a single ``BaseModel`` schema,
    matching the same reasoning behind ``fact_checker.py``'s
    ``_FactCheckOutput``.
    """

    suggestions: List[str] = Field(default_factory=list, description="Up to a few natural follow-up questions.")


def build_coordinator_chat_node(
    llm_service: LLMService,
    model_name: str,
) -> Callable[[ResearchGraphState], Awaitable[dict]]:
    """
    Build the Coordinator's lightweight-chat node function.

    Runs instead of the full Planner -> ... -> Reviewer pipeline when
    ``graph.py``'s entry routing decides ``ChatMode.CHAT`` doesn't warrant
    it. A single direct LLM call, no retrieval, no search, no citations --
    matching ``ChatMode.CHAT``'s own docstring ("may be served from
    memory/a single LLM call for simple follow-ups").

    Args:
        llm_service: Injected ``LLMService``.
        model_name: The configured model identifier, stamped onto this
            node's ``AgentExecutionStep.model_name``.

    Returns:
        An ``async def coordinator_chat_node(state) -> dict`` suitable for
        ``StateGraph.add_node``.
    """

    async def coordinator_chat_node(state: ResearchGraphState) -> dict:
        query = state["query"]
        conversation_history = state.get("conversation_history", MemoryContext())

        async with track_step(AgentName.COORDINATOR, "chat_response", model_name=model_name) as rec:
            try:
                response = await llm_service.generate(
                    _build_chat_prompt(query, conversation_history),
                    system_prompt=_CHAT_SYSTEM_PROMPT,
                    temperature=_CHAT_TEMPERATURE,
                )
            except LLMServiceError as exc:
                # The entire value of this path is the direct answer --
                # unlike `finalize`'s follow-up suggestions, there is
                # nothing to degrade to. Fatal, matching the Planner's/
                # Writer's "nothing usable without it" reasoning.
                raise CoordinatorError.wrap(
                    exc, f"Coordinator failed to produce a chat response for query: {query!r}"
                ) from exc

            suggestions = await _try_generate_suggestions(llm_service, query, response.content)
            rec.summary = f"Chat response: {len(response.content)} character(s)"

        return {
            "final_answer": response.content,
            "citations": [],
            "follow_up_suggestions": suggestions,
            "execution_steps": [rec.step],
        }

    return coordinator_chat_node


def build_coordinator_finalize_node(
    llm_service: LLMService,
    model_name: str,
) -> Callable[[ResearchGraphState], Awaitable[dict]]:
    """
    Build the Coordinator's finalize node function.

    Runs once ``graph.py``'s routing has decided a draft is ready to ship
    (the Reviewer approved it, or -- a decision that also belongs to
    ``graph.py`` -- the bounded revision loop was exhausted and the best
    available draft is being accepted anyway). This node does not
    re-evaluate that decision; it only assembles the final response from
    whatever draft is present.

    Args:
        llm_service: Injected ``LLMService``.
        model_name: The configured model identifier, stamped onto this
            node's ``AgentExecutionStep.model_name``.

    Returns:
        An ``async def coordinator_finalize_node(state) -> dict`` suitable
        for ``StateGraph.add_node``.
    """

    async def coordinator_finalize_node(state: ResearchGraphState) -> dict:
        query = state["query"]
        draft_answer = state.get("draft_answer")

        if not draft_answer:
            # The graph is wired so the Writer (and Reviewer) always run
            # before finalize -- a missing draft here means the graph
            # itself is miswired, not an expected runtime state. Fatal,
            # matching the Fact Checker's/Reviewer's identical guard.
            raise CoordinatorError(f"Coordinator has no draft answer to finalize for query: {query!r}")

        citations = state.get("citations") or []

        async with track_step(AgentName.COORDINATOR, "finalize", model_name=model_name) as rec:
            final_citations, unmatched_markers = _filter_cited_sources(draft_answer, citations)
            if unmatched_markers:
                logger.warning(
                    "Draft answer references citation marker(s) not present in the available citations",
                    extra={"query": query, "unmatched_markers": sorted(unmatched_markers)},
                )

            suggestions = await _try_generate_suggestions(llm_service, query, draft_answer)
            rec.summary = f"Finalized answer citing {len(final_citations)} of {len(citations)} available source(s)"

        return {
            "final_answer": draft_answer,
            "citations": final_citations,
            "follow_up_suggestions": suggestions,
            "execution_steps": [rec.step],
        }

    return coordinator_finalize_node


async def _try_generate_suggestions(llm_service: LLMService, query: str, answer: str) -> List[str]:
    """
    Generate follow-up suggestions, degrading to an empty list on failure
    without ever raising out of this function.

    Called from inside a ``track_step`` block, this is that helper's own
    documented second pattern: catch a sub-call's exception *inside* the
    block, before it reaches ``track_step`` at all, when the situation is
    expected/acceptable rather than a genuine node failure -- producing
    the final answer is this node's job, and that already succeeded by
    the time this is called; missing follow-up suggestions is a soft
    degradation of a secondary feature, not a failure of the node itself.
    Because the exception is caught here and never propagates, the
    enclosing ``track_step`` block completes normally and its step comes
    out COMPLETED, exactly as that pattern intends.
    """
    prompt = f"User's query:\n{query}\n\nAnswer given:\n{answer}\n\nSuggest follow-up questions now."
    try:
        output = await llm_service.generate_structured(
            prompt, _FollowUpSuggestions, system_prompt=_SUGGESTIONS_SYSTEM_PROMPT, temperature=_SUGGESTION_TEMPERATURE
        )
        return output.suggestions[:_MAX_SUGGESTIONS]
    except LLMServiceError:
        logger.warning("Follow-up suggestion generation failed; continuing without suggestions", extra={"query": query})
        return []


def _build_chat_prompt(query: str, conversation_history: MemoryContext) -> str:
    """
    Build ``coordinator_chat_node``'s user-turn prompt, prefixing it with
    prior-turn context only when there is any.

    When ``conversation_history`` is empty, this returns ``query``
    unchanged -- byte-for-byte identical to the pre-Phase-7 behavior of
    calling ``llm_service.generate(query, ...)`` directly, so the common
    single-turn chat case is unaffected.
    """
    history_text = format_conversation_history(conversation_history)
    if history_text is None:
        return query

    return f"Conversation so far in this session:\n{history_text}\n\nUser's new message:\n{query}"


def _filter_cited_sources(draft_answer: str, citations: List[Citation]) -> Tuple[List[Citation], Set[str]]:
    """
    Narrow ``citations`` (every source the Researcher consolidated) down
    to just the ones ``draft_answer`` actually cites via a ``[n]`` marker
    -- matching ``ChatResponse.citations``'s own documented contract
    ("Sources backing the answer's citation markers"), not "every source
    that was ever retrieved."

    Returns the filtered list (in ``citations``' original order) and the
    set of marker numbers referenced in the text that don't correspond to
    any available citation -- a real (if rare) inconsistency between the
    Writer's output and the evidence it was given, worth surfacing rather
    than silently swallowing.
    """
    referenced_ids = set(_CITATION_MARKER_PATTERN.findall(draft_answer))
    available_ids = {c.citation_id for c in citations}
    unmatched_markers = referenced_ids - available_ids
    filtered = [c for c in citations if c.citation_id in referenced_ids]
    return filtered, unmatched_markers
