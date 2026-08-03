"""
Shared state for the LangGraph research workflow, and a small helper for
recording per-node execution steps.

Graph shape (implemented across the rest of Phase 6, wired together in
``app.agents.graph``)::

    Coordinator -> Planner -> [Retriever, Search] (parallel) -> Researcher
        -> Writer -> Fact Checker -> Reviewer -> (approved) -> Coordinator
                                   \\-> (needs revision, retries remain) -> Writer

``Coordinator`` is both the entry point (dispatches on ``ChatMode`` --
``ChatMode.CHAT`` may bypass the full pipeline for a lightweight response)
and the exit point (assembles the final answer/citations once the Reviewer
approves). The Reviewer -> Writer edge is the one cycle in this graph,
bounded by ``AgentSettings.max_reviewer_retries`` (see
``MaxRetriesExceededError`` in ``core.exceptions``, which already
documents this exact loop).

``ResearchGraphState`` is a ``TypedDict``, not a Pydantic model -- verified
directly against the installed ``langgraph`` that both are accepted as a
graph's state schema, but ``TypedDict`` is the framework-idiomatic choice:
LangGraph merges each node's *partial* return dict onto the running state
via its own reducer machinery, which fits a plain mapping far more
naturally than a Pydantic model (which validates a *complete* object on
construction). Every value stored inside the state is still strongly
typed -- either an existing Pydantic model reused directly, or a small new
one defined below.

Reuse over reinvention: ``RetrievedChunk`` / ``SearchResult`` (the raw
per-source evidence) and ``Citation`` / ``AgentExecutionStep`` / ``ChatMode``
/ ``RetrievalOptions`` (already API-facing shapes) are used as-is rather
than wrapped in parallel "agent-internal" types. This extends the same
precedent already established for ``DocumentChunk``/``RetrievedChunk`` in
Phase 5, and mirrors how ``HistoryService`` already works directly in
``app.schemas`` types -- ``app.agents`` sits at the same layer relative to
the schema layer that ``app.services`` does. ``MemoryContext`` (Phase 7,
``app.memory.types``) follows the same rule: it's stored here exactly as
``MemoryService`` produces it, not adapted into a graph-local shape.

Deliberately excluded from this state (see the design discussion this
file was approved against):

* Error-as-data. A node's genuinely fatal failure raises its own
  ``AgentError`` subclass (``core.exceptions``) and halts the graph run,
  consistent with how the rest of this codebase handles errors --
  exceptions, not error fields threaded through a data structure. What
  *is* data is the per-step execution trace (including failed/degraded
  steps), which is what ``track_step`` below builds.
"""

from __future__ import annotations

import operator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import AsyncIterator, List, Optional, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from app.core.exceptions import ResearchMindError
from app.core.interfaces.search_provider import SearchResult
from app.core.interfaces.vector_store import RetrievedChunk
from app.core.logging import get_logger
from app.memory.types import MemoryContext
from app.schemas.chat import ChatMode, RetrievalOptions
from app.schemas.common import AgentExecutionStep, Citation, ErrorBody, ExecutionStepStatus, TokenUsage
from app.utils.time import utc_now

logger = get_logger(__name__)


class AgentName(str, Enum):
    """
    The eight agents in the research workflow, in the same lowercase form
    used by ``AgentExecutionStep.agent_name`` (see its own docstring's
    example, ``"planner"``). Centralized here -- rather than each node
    file hardcoding its own string literal -- because 7 separate node
    files (each written and reviewed on its own turn across this
    project's process) referencing an identical literal is exactly the
    kind of drift a single source of truth exists to prevent.
    """

    COORDINATOR = "coordinator"
    PLANNER = "planner"
    RETRIEVER = "retriever"
    SEARCH = "search"
    RESEARCHER = "researcher"
    WRITER = "writer"
    FACT_CHECKER = "fact_checker"
    REVIEWER = "reviewer"


# =============================================================================
# Structured outputs produced by individual agents
# =============================================================================


class ResearchPlan(BaseModel):
    """
    The Planner's output: how the rest of the run should proceed.

    ``use_retrieval`` / ``use_web_search`` let the Planner skip a source
    entirely when it's clearly not relevant (e.g. a question with nothing
    to do with any uploaded document skips RAG retrieval), rather than the
    Retriever/Search nodes always running and returning empty-but-costly
    results.
    """

    sub_questions: List[str] = Field(
        ..., min_length=1, description="The query decomposed into concrete research sub-questions."
    )
    use_retrieval: bool = Field(..., description="Whether the Retriever agent (RAG) should run for this query.")
    use_web_search: bool = Field(..., description="Whether the Search agent (web) should run for this query.")
    reasoning: str = Field(..., description="The Planner's brief rationale for this plan, surfaced for observability.")


class VerifiedClaim(BaseModel):
    """
    The Fact Checker's per-claim verdict: does a specific claim in the
    Writer's draft actually hold up against the citations it references?
    """

    claim_text: str = Field(..., description="The claim as it appears (or paraphrased) in the draft.")
    citation_ids: List[str] = Field(
        default_factory=list, description="Which of the run's Citation.citation_id values this claim relies on."
    )
    is_supported: bool = Field(..., description="Whether the cited source(s) actually support this claim.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="The Fact Checker's confidence in this verdict.")
    notes: Optional[str] = Field(default=None, description="Explanation, especially when is_supported is False.")


class ReviewFeedback(BaseModel):
    """The Reviewer's verdict on a Writer draft: approve, or send back for revision."""

    approved: bool = Field(..., description="Whether this draft is ready to become the final answer.")
    feedback: str = Field(
        ..., description="Actionable feedback for the Writer if not approved; a brief summary if approved."
    )
    quality_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Overall quality score, if the Reviewer produces one."
    )
    issues: List[str] = Field(
        default_factory=list, description="Specific problems found, e.g. unsupported claims, missing citations."
    )


# =============================================================================
# Graph state
# =============================================================================


class ResearchGraphState(TypedDict, total=False):
    """
    The state threaded through every node of the research graph.

    ``total=False``: not every key is present at every point in the run --
    e.g. ``plan`` doesn't exist until the Planner has run, ``final_answer``
    doesn't exist until the Coordinator finalizes. Nodes must read via
    ``state.get(...)`` with an explicit default, never direct subscripting,
    for exactly this reason.
    """

    # --- Inputs, set once at graph invocation ---
    query: str
    mode: ChatMode
    session_id: Optional[str]
    retrieval_options: Optional[RetrievalOptions]
    started_at: datetime

    # --- Memory (Phase 7): this session's prior turns, loaded once by
    # MemoryService.load_context() *before* the graph is invoked and
    # passed in as part of the initial state -- never written by any
    # node, so (like the other inputs above) it needs no reducer. Per the
    # approved Phase 7 design, only the Planner and coordinator_chat_node
    # read this key; every other node ignores it. That restriction is
    # each of those node's own responsibility to honor -- a TypedDict
    # key is visible to every node, this comment is guidance, not
    # enforcement.
    conversation_history: MemoryContext

    # --- Planner ---
    plan: ResearchPlan

    # --- Retriever / Search (run in parallel; disjoint keys, no reducer needed) ---
    retrieved_chunks: List[RetrievedChunk]
    search_results: List[SearchResult]

    # --- Researcher ---
    citations: List[Citation]

    # --- Writer / Fact Checker / Reviewer ---
    draft_answer: str
    verified_claims: List[VerifiedClaim]
    review_feedback: ReviewFeedback
    revision_count: int

    # --- Coordinator (finalized output) ---
    final_answer: str
    follow_up_suggestions: List[str]

    # --- Observability, written by every node; the one field with more
    # than one writer per superstep (Retriever and Search write
    # concurrently), hence the reducer. ---
    execution_steps: Annotated[List[AgentExecutionStep], operator.add]


def create_initial_state(
    *,
    query: str,
    mode: ChatMode,
    session_id: Optional[str],
    retrieval_options: Optional[RetrievalOptions] = None,
    conversation_history: Optional[MemoryContext] = None,
) -> ResearchGraphState:
    """
    Build a well-formed initial state to invoke the graph with.

    Exists so every caller (today, whatever test drives ``graph.py``;
    eventually ``ChatService``) constructs the same correctly-defaulted
    shape -- ``revision_count=0``, empty lists ready for accumulation --
    rather than each hand-rolling the initial dict literal.

    ``conversation_history`` defaults to an empty ``MemoryContext()``
    (equivalent to "no prior turns") rather than being required, so every
    existing caller/test that predates Phase 7 keeps working unchanged.
    ``ChatService`` is the only caller expected to ever pass a non-empty
    one, populated from ``MemoryService.load_context()``.
    """
    return ResearchGraphState(
        query=query,
        mode=mode,
        session_id=session_id,
        retrieval_options=retrieval_options,
        started_at=utc_now(),
        conversation_history=conversation_history if conversation_history is not None else MemoryContext(),
        retrieved_chunks=[],
        search_results=[],
        citations=[],
        verified_claims=[],
        revision_count=0,
        follow_up_suggestions=[],
        execution_steps=[],
    )


# =============================================================================
# Per-node execution step recording
# =============================================================================


@dataclass
class StepRecorder:
    """
    Mutable handle a node uses, inside a ``track_step`` block, to annotate
    the ``AgentExecutionStep`` being built for that node's run. ``step`` is
    populated by ``track_step`` on exit (success or failure) -- read it
    after the ``async with`` block, not during it.
    """

    agent_name: AgentName
    graph_node: str
    model_name: Optional[str] = None
    tool_name: Optional[str] = None
    summary: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
    step: Optional[AgentExecutionStep] = field(default=None, init=False)


@asynccontextmanager
async def track_step(
    agent_name: AgentName,
    graph_node: str,
    *,
    model_name: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> AsyncIterator[StepRecorder]:
    """
    Time a node's work and build the resulting ``AgentExecutionStep``.

    Usage::

        async with track_step(AgentName.SEARCH, "search") as rec:
            response = await search_provider.search(query)
            rec.summary = f"Found {len(response.results)} results"
        # rec.step is now a COMPLETED AgentExecutionStep

    On success, ``rec.step`` is COMPLETED. On any exception raised inside
    the block, ``rec.step`` is built as FAILED (with ``error`` populated
    from the exception) *before* the exception is re-raised unchanged --
    this function never swallows an exception. That means a node has two
    valid patterns available, and the choice between them is the node's,
    not this helper's:

    * Let the exception propagate past the node entirely (a genuinely
      fatal failure) -- the FAILED step is built but never makes it into
      ``execution_steps`` for this run, since the graph invocation itself
      aborts. Accepted as a known limitation rather than solved with
      partial-state capture machinery (e.g. LangGraph checkpointing) that
      nothing in this project needs yet.
    * Catch the exception around the ``async with`` block (``rec`` stays
      bound in the enclosing scope even though the exception propagated
      out of the block), read ``rec.step`` for the FAILED record, and
      return gracefully with degraded results -- e.g. web search failing
      shouldn't necessarily kill a run that still has RAG results.

    A node that wants to degrade *without* ever marking the step FAILED
    (because the situation is expected/acceptable, not an error worth
    flagging) should catch its own sub-call's exception *inside* the
    block, before it would reach this context manager at all, and set
    ``rec.summary`` instead -- the step then comes out COMPLETED.
    """
    recorder = StepRecorder(agent_name=agent_name, graph_node=graph_node, model_name=model_name, tool_name=tool_name)
    step_id = f"step-{uuid4().hex[:8]}"
    started_at = utc_now()
    try:
        yield recorder
    except Exception as exc:
        completed_at = utc_now()
        latency_ms = (completed_at - started_at).total_seconds() * 1000.0
        if isinstance(exc, ResearchMindError):
            error = ErrorBody(
                error_code=exc.error_code,
                message=exc.message,
                retryable=exc.retryable,
                request_id=exc.request_id,
                trace_id=exc.trace_id,
            )
        else:
            error = ErrorBody(error_code="RM-GEN-000", message=str(exc) or type(exc).__name__, retryable=False)
        recorder.step = AgentExecutionStep(
            step_id=step_id,
            agent_name=agent_name.value,
            graph_node=graph_node,
            status=ExecutionStepStatus.FAILED,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            model_name=recorder.model_name,
            tool_name=recorder.tool_name,
            token_usage=recorder.token_usage,
            summary=recorder.summary,
            error=error,
        )
        logger.warning(
            "Agent step failed",
            extra={"agent_name": agent_name.value, "graph_node": graph_node, "error_code": error.error_code},
        )
        raise
    else:
        completed_at = utc_now()
        latency_ms = (completed_at - started_at).total_seconds() * 1000.0
        recorder.step = AgentExecutionStep(
            step_id=step_id,
            agent_name=agent_name.value,
            graph_node=graph_node,
            status=ExecutionStepStatus.COMPLETED,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            model_name=recorder.model_name,
            tool_name=recorder.tool_name,
            token_usage=recorder.token_usage,
            summary=recorder.summary,
            error=None,
        )
        logger.debug(
            "Agent step completed",
            extra={"agent_name": agent_name.value, "graph_node": graph_node, "latency_ms": round(latency_ms, 2)},
        )
