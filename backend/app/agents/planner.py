"""
The Planner agent: the research graph's entry-point node (once the
Coordinator has decided the full pipeline is warranted for this request).

Decomposes the user's query into concrete sub-questions and decides which
evidence sources -- document retrieval, web search -- are worth running.
Built as a closure factory (``build_planner_node``) rather than a class or
a bare module-level function, so its dependencies (``LLMService``,
``FeatureFlags``) are captured once at graph-build time and never smuggled
through ``ResearchGraphState`` -- the pattern ``app.agents.state`` already
documents and requires of every node in this graph.

Phase 7 (Memory): the Planner is one of exactly two nodes (the other is
``coordinator_chat_node``) the approved Memory design names as a consumer
of ``ResearchGraphState["conversation_history"]``. Prior turns are folded
into the prompt as context -- e.g. so "what about its cost?" can be
decomposed against whatever "it" referred to earlier in the session --
never into ``_SYSTEM_PROMPT`` itself, since the history is per-request
content, not a standing instruction. When ``conversation_history`` is
empty (the common case: a session's first turn, or Memory not yet wired
by a given caller/test), the prompt is byte-for-byte identical to the
pre-Phase-7 prompt -- no empty section header, no behavior change for the
single-turn case this node already handled correctly. The turn/summary
rendering step itself is shared with ``coordinator_chat_node`` via
``app.agents.memory_prompt`` (see that module's docstring for why it was
extracted only after both implementations existed); the surrounding
prompt template below -- specific to how the Planner should use this
context -- remains local to this file.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from app.agents.memory_prompt import format_conversation_history
from app.agents.state import AgentName, ResearchGraphState, ResearchPlan, track_step
from app.core.config import FeatureFlags
from app.core.exceptions import LLMServiceError, PlannerError
from app.core.interfaces.llm_service import LLMService
from app.core.logging import get_logger
from app.memory.types import MemoryContext
from app.schemas.chat import RetrievalOptions

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are the Planner agent in a multi-agent research assistant. Given a "
    "user's research query, decompose it into concrete, answerable "
    "sub-questions, and judge whether answering it well would benefit from "
    "(a) retrieving relevant excerpts from documents the user has uploaded "
    "into this session, and/or (b) searching the current web for up-to-date "
    "or external information. Recommend both sources when genuinely useful, "
    "but avoid recommending a source that clearly would not help -- e.g. a "
    "question entirely about a user's uploaded documents has no need for web "
    "search, and a question with no plausible connection to any uploaded "
    "document has no need for retrieval. Keep sub_questions concrete and "
    "specific enough that each could be researched independently, and keep "
    "reasoning brief -- one or two sentences. You may also be given earlier "
    "turns from this conversation, for context only -- use them to resolve "
    "references (e.g. 'it', 'that report') in the current query, but the "
    "plan you produce must still address the current query specifically, "
    "not the earlier turns."
)


def build_planner_node(
    llm_service: LLMService,
    feature_flags: FeatureFlags,
    model_name: str,
) -> Callable[[ResearchGraphState], Awaitable[dict]]:
    """
    Build the Planner's LangGraph node function.

    Args:
        llm_service: Injected ``LLMService``.
        feature_flags: Governs whether ``use_web_search`` / ``use_retrieval``
            can ever end up ``True`` regardless of what the LLM itself
            judges -- see the module-level design note on this in the
            file's docstring and the ADR log.
        model_name: The configured model identifier (e.g.
            ``LLMSettings.ollama.model``), stamped onto this node's
            ``AgentExecutionStep.model_name`` for observability.
            ``generate_structured`` returns a validated schema instance
            directly rather than a response wrapper (by interface design),
            so there is no response object to read a model name from.

    Returns:
        An ``async def planner_node(state) -> dict`` suitable for
        ``StateGraph.add_node``.
    """

    async def planner_node(state: ResearchGraphState) -> dict:
        query = state["query"]
        retrieval_options = state.get("retrieval_options")
        conversation_history = state.get("conversation_history", MemoryContext())

        async with track_step(AgentName.PLANNER, "plan", model_name=model_name) as rec:
            try:
                plan = await llm_service.generate_structured(
                    _build_prompt(query, conversation_history),
                    ResearchPlan,
                    system_prompt=_SYSTEM_PROMPT,
                    temperature=0.2,
                )
            except LLMServiceError as exc:
                raise PlannerError.wrap(exc, f"Planner failed to produce a research plan for query: {query!r}") from exc

            plan = _apply_source_overrides(plan, retrieval_options, feature_flags)

            rec.summary = (
                f"Planned {len(plan.sub_questions)} sub-question(s); "
                f"retrieval={plan.use_retrieval}, web_search={plan.use_web_search}"
            )

        return {"plan": plan, "execution_steps": [rec.step]}

    return planner_node


def _apply_source_overrides(
    plan: ResearchPlan, retrieval_options: Optional[RetrievalOptions], feature_flags: FeatureFlags
) -> ResearchPlan:
    """
    Reconcile the LLM's own judgment with per-request overrides and global
    feature flags -- neither of which the LLM was told about, since both
    are this project's own runtime configuration, not something to explain
    in a prompt and trust the model to honor correctly.

    ``RetrievalOptions.include_web_search`` (see its own docstring) is an
    explicit per-request override: when set, it wins outright over the
    LLM's judgment. ``use_retrieval`` has no equivalent per-request field,
    so it's only ever gated by ``FeatureFlags.enable_rag``. Both are always
    gated by their respective feature flag regardless of the LLM's
    judgment -- a disabled feature must never be routed to.
    """
    if retrieval_options is not None and retrieval_options.include_web_search is not None:
        use_web_search = retrieval_options.include_web_search
    else:
        use_web_search = plan.use_web_search and feature_flags.enable_web_search

    use_retrieval = plan.use_retrieval and feature_flags.enable_rag

    if use_web_search == plan.use_web_search and use_retrieval == plan.use_retrieval:
        return plan
    return plan.model_copy(update={"use_retrieval": use_retrieval, "use_web_search": use_web_search})


def _build_prompt(query: str, conversation_history: MemoryContext) -> str:
    """
    Build the Planner's user-turn prompt, prefixing it with prior-turn
    context only when there is any.

    Kept as a plain string-building function, not folded into
    ``planner_node`` itself, so it's independently testable without
    driving the whole node (matching ``app.memory.compaction._build_prompt``'s
    own reason for existing as a separate function).
    """
    history_text = format_conversation_history(conversation_history)
    if history_text is None:
        return f"Research query:\n{query}\n\nProduce a research plan for this query."

    return (
        f"Conversation so far in this session (context only -- see the system "
        f"instructions on how to use this):\n{history_text}\n\n"
        f"Current research query:\n{query}\n\nProduce a research plan for this query."
    )
