"""
``ChatService`` -- orchestrates a single ``POST /chat`` request end-to-end.

Per the approved Phase 7 design, this is deliberately the *only*
orchestration layer: every actual decision (what to research, how to
compact history, how to route through the agent graph) already lives
elsewhere --

* ``app.memory.MemoryService`` owns loading and, if needed, compacting a
  session's conversation history.
* ``app.agents.graph.build_research_graph`` (built once at application
  startup, injected here as an already-compiled ``CompiledStateGraph``)
  owns every research/generation decision.

``ChatService`` only resolves or creates the session, invokes
``MemoryService``, runs the graph, persists the resulting turn, and maps
the graph's output into a ``ChatResponse``. It is framework-agnostic (no
FastAPI import), mirroring ``HistoryService``/``MemoryService``'s own
scope.

Two independent ``database.session()`` units of work happen per request
(resolve-or-create the session; persist the completed turn afterward) --
never one spanning the whole request. This mirrors ``MemoryService``'s own
reasoning: the research graph run in between can take many seconds (real
LLM calls, retrieval, web search), and holding a single DB transaction
open across that for no reason would be an unbounded, unnecessary
lock/connection hold. Consequently, no ORM object obtained from the first
unit of work is ever reused in the second -- a detached instance from a
closed ``AsyncSession`` cannot be safely mutated against a different
session's identity map. ``_persist_turn`` re-fetches the session row by id
instead of holding onto the one ``_resolve_session`` returned.

A ``MaxRetriesExceededError`` (or any other ``ResearchMindError`` raised
by a node) out of ``graph.ainvoke()`` is deliberately allowed to propagate
unchanged. This service does not catch or downgrade a fatal agent failure
into a partial response -- that decision belongs to Phase 6 (see
``graph.py``'s own docstring on the bounded revision loop) and this file
has no reason to revisit it.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph

from app.agents.state import ResearchGraphState, create_initial_state
from app.core.exceptions import GraphExecutionError, SessionNotFoundError
from app.core.logging import get_logger
from app.database.repositories.conversation_repository import ConversationRepository
from app.database.session import Database
from app.models.conversation import AgentExecutionStep as AgentExecutionStepRow
from app.models.conversation import ConversationTurn as ConversationTurnRow
from app.models.conversation import TurnCitation as TurnCitationRow
from app.models.enums import ChatMode as ModelChatMode
from app.models.enums import CitationSourceType as ModelCitationSourceType
from app.models.enums import ExecutionStepStatus as ModelExecutionStepStatus
from app.schemas.chat import ChatMode as SchemaChatMode
from app.schemas.chat import ChatRequest, ChatResponse, ResearchMetadata
from app.schemas.common import AgentExecutionStep, AgentExecutionTrace, Citation, TokenUsage
from app.services.memory_service import MemoryService
from app.utils.time import utc_now

logger = get_logger(__name__)


class ChatService:
    """
    Orchestrates a single ``POST /chat`` request: resolve/create the
    session, load conversation history, run the research graph, persist
    the turn, map the response.

    Constructed once at application startup, mirroring every other
    adapter/service in this codebase (``Database``, ``OllamaLLMService``,
    ``MemoryService``). ``graph`` is injected already-compiled -- building
    it is ``app.agents.graph.build_research_graph``'s job, done once by
    ``core.dependencies``, not on every request.
    """

    def __init__(
        self,
        *,
        database: Database,
        memory_service: MemoryService,
        graph: CompiledStateGraph,
        llm_provider_name: str,
        llm_model_name: str,
    ) -> None:
        self._database = database
        self._memory_service = memory_service
        self._graph = graph
        self._llm_provider_name = llm_provider_name
        self._llm_model_name = llm_model_name

    async def send_message(self, request: ChatRequest, *, include_trace: bool = False) -> ChatResponse:
        """
        Run one full ``/chat`` request and return its response.

        Args:
            request: The validated request body.
            include_trace: Whether to populate ``ChatResponse.execution_trace``.
                Defaults to ``False`` so ordinary chat responses aren't
                forced to pay the trace's serialization cost -- per
                ``ChatResponse.execution_trace``'s own docstring, this is
                meant to be driven by a route-level request (e.g. an
                ``include_trace`` query parameter), which is why it's a
                parameter here rather than a field ``ChatService`` decides
                on its own.
        """
        session_id, conversation_title = await self._resolve_session(request)
        conversation_history = await self._memory_service.load_context(session_id)

        initial_state = create_initial_state(
            query=request.query,
            mode=request.mode,
            session_id=session_id,
            retrieval_options=request.retrieval_options,
            conversation_history=conversation_history,
        )

        final_state = await self._graph.ainvoke(initial_state)
        completed_at = utc_now()

        turn_row = await self._persist_turn(
            session_id=session_id, request=request, final_state=final_state, completed_at=completed_at
        )

        return _map_response(
            turn_row=turn_row,
            final_state=final_state,
            conversation_title=conversation_title,
            include_trace=include_trace,
        )

    async def _resolve_session(self, request: ChatRequest) -> Tuple[str, Optional[str]]:
        """
        Return ``(session_id, title)`` for this request.

        ``request.session_id`` set means "continue this session" -- it
        must already exist, or this raises ``SessionNotFoundError`` (a
        client error: silently starting a new session under a client-
        supplied id it didn't actually create would hide a real mistake,
        e.g. a stale/mistyped id). ``request.session_id`` omitted means
        "start a new session", created here with a fresh id.
        """
        async with self._database.session() as session:
            repo = ConversationRepository(session)
            if request.session_id is not None:
                row = await repo.get_by_id(request.session_id)
                if row is None:
                    raise SessionNotFoundError(f"No research session found with id {request.session_id!r}.")
                return row.id, row.title

            new_session_id = f"sess-{uuid4().hex[:8]}"
            row = await repo.create_session(new_session_id, title=request.conversation_title)
            return row.id, row.title

    async def _persist_turn(
        self,
        *,
        session_id: str,
        request: ChatRequest,
        final_state: ResearchGraphState,
        completed_at: datetime,
    ) -> ConversationTurnRow:
        """
        Build and persist the completed turn (with its citations and
        execution steps) from the graph's final state.

        Reads defensively via ``.get(...)`` throughout, matching every
        node's own convention for a ``total=False`` ``TypedDict`` --
        even though a *successful* graph run is expected to have set
        every key this method reads.
        """
        final_answer = final_state.get("final_answer")
        if not final_answer:
            # Both coordinator_chat_node and coordinator_finalize_node set
            # final_answer before a run completes successfully (see their
            # own docstrings) -- a missing value here means the graph
            # itself is miswired, not an expected runtime state. Mirrors
            # coordinator_finalize_node's identical guard on draft_answer.
            raise GraphExecutionError(
                f"Research graph completed without producing a final_answer for query: {request.query!r}"
            )

        citations = final_state.get("citations") or []
        execution_steps = final_state.get("execution_steps") or []
        plan = final_state.get("plan")
        started_at = final_state.get("started_at") or completed_at
        mode = final_state.get("mode", request.mode)

        usages = [step.token_usage for step in execution_steps if step.token_usage is not None]
        token_usage = TokenUsage.aggregate(usages) if usages else None

        turn_row = ConversationTurnRow(
            id=f"msg-{uuid4().hex[:8]}",
            session_id=session_id,
            sequence_number=0,  # overwritten below, inside the same unit of work as next_sequence_number()
            mode=ModelChatMode(mode.value),
            query=request.query,
            answer=final_answer,
            llm_provider=self._llm_provider_name,
            llm_model=self._llm_model_name,
            rag_enabled=bool(plan.use_retrieval) if plan is not None else False,
            web_search_enabled=bool(plan.use_web_search) if plan is not None else False,
            retrieved_chunk_count=len(final_state.get("retrieved_chunks") or []),
            web_result_count=len(final_state.get("search_results") or []),
            total_latency_ms=(completed_at - started_at).total_seconds() * 1000.0,
            prompt_tokens=token_usage.prompt_tokens if token_usage else None,
            completion_tokens=token_usage.completion_tokens if token_usage else None,
            total_tokens=token_usage.total_tokens if token_usage else None,
        )
        citation_rows = [_to_citation_row(c) for c in citations]
        step_rows = [_to_execution_step_row(s) for s in execution_steps]

        async with self._database.session() as session:
            repo = ConversationRepository(session)
            turn_row.sequence_number = await repo.next_sequence_number(session_id)
            await repo.add_turn(turn_row, citations=citation_rows, execution_steps=step_rows)
            session_row = await repo.get_by_id(session_id)
            if session_row is not None:
                # Always true in practice (the session existed moments ago,
                # in _resolve_session, and nothing in this project deletes
                # sessions concurrently) -- guarded rather than asserted,
                # since a missing session here would otherwise raise
                # AttributeError deep inside touch_session() with a
                # confusing traceback instead of failing cleanly.
                await repo.touch_session(session_row)

        logger.info(
            "Persisted conversation turn",
            extra={
                "session_id": session_id,
                "turn_id": turn_row.id,
                "sequence_number": turn_row.sequence_number,
                "mode": mode.value,
                "citation_count": len(citation_rows),
                "execution_step_count": len(step_rows),
            },
        )
        return turn_row


# =============================================================================
# schema -> ORM mapping (pure, stateless -- module-level, mirroring
# HistoryService's own ORM -> schema mapping functions, just the reverse
# direction)
# =============================================================================


def _to_citation_row(citation: Citation) -> TurnCitationRow:
    return TurnCitationRow(
        id=f"cit-{uuid4().hex[:8]}",
        citation_id=citation.citation_id,
        source_type=ModelCitationSourceType(citation.source_type.value),
        title=citation.title,
        url=citation.url,
        source_filename=citation.source_filename,
        document_id=citation.document_id,
        page_number=citation.page_number,
        section_title=citation.section_title,
        excerpt=citation.excerpt,
        score=citation.score,
        published_date=citation.published_date,
        accessed_at=citation.accessed_at,
    )


def _to_execution_step_row(step: AgentExecutionStep) -> AgentExecutionStepRow:
    return AgentExecutionStepRow(
        # Reuses the schema step's own step_id as this row's primary key,
        # rather than generating a fresh one -- AgentExecutionStep's own
        # docstring documents that these ids correlate a step to
        # server-side structured logs; generating a new id here would
        # sever that correlation for every persisted step.
        id=step.step_id,
        agent_name=step.agent_name,
        graph_node=step.graph_node,
        status=ModelExecutionStepStatus(step.status.value),
        started_at=step.started_at,
        completed_at=step.completed_at,
        latency_ms=step.latency_ms,
        model_name=step.model_name,
        tool_name=step.tool_name,
        prompt_tokens=step.token_usage.prompt_tokens if step.token_usage else None,
        completion_tokens=step.token_usage.completion_tokens if step.token_usage else None,
        total_tokens=step.token_usage.total_tokens if step.token_usage else None,
        summary=step.summary,
        error_code=step.error.error_code if step.error else None,
        error_message=step.error.message if step.error else None,
    )


def _map_response(
    *,
    turn_row: ConversationTurnRow,
    final_state: ResearchGraphState,
    conversation_title: Optional[str],
    include_trace: bool,
) -> ChatResponse:
    """
    Build the ``ChatResponse`` from the just-persisted turn row (the
    source of truth for anything already normalized/generated -- id,
    sequence_number, timestamps) plus whatever ``final_state`` carries
    that isn't persisted (``follow_up_suggestions`` has no ORM column;
    ephemeral to the response only, matching ``schemas.history.
    ConversationTurn`` -- which also has no such field).
    """
    citations = final_state.get("citations") or []
    documents_considered = sorted({c.document_id for c in citations if c.document_id})

    research_metadata = ResearchMetadata(
        llm_provider=turn_row.llm_provider,
        llm_model=turn_row.llm_model,
        rag_enabled=turn_row.rag_enabled,
        web_search_enabled=turn_row.web_search_enabled,
        retrieved_chunk_count=turn_row.retrieved_chunk_count,
        web_result_count=turn_row.web_result_count,
        documents_considered=documents_considered,
        total_latency_ms=turn_row.total_latency_ms,
        token_usage=TokenUsage(
            prompt_tokens=turn_row.prompt_tokens,
            completion_tokens=turn_row.completion_tokens,
            total_tokens=turn_row.total_tokens,
        ),
    )

    execution_trace: Optional[AgentExecutionTrace] = None
    if include_trace:
        steps: List[AgentExecutionStep] = final_state.get("execution_steps") or []
        if steps:
            execution_trace = AgentExecutionTrace.from_steps(
                # Mirrors HistoryService._map_turn's identical choice: no
                # request-time trace_id is persisted (no column for it),
                # so the turn's own id serves as a stable per-turn
                # identifier here too, keeping a live response and its
                # later historical replay consistent.
                trace_id=turn_row.id,
                session_id=turn_row.session_id,
                steps=steps,
                started_at=steps[0].started_at,
                completed_at=steps[-1].completed_at,
            )

    return ChatResponse(
        session_id=turn_row.session_id,
        message_id=turn_row.id,
        # Read back off the persisted row (the authoritative "mode that
        # actually produced this response") rather than final_state a
        # second time, so this can never diverge from what was stored.
        mode=SchemaChatMode(turn_row.mode.value),
        answer=turn_row.answer,
        citations=citations,
        conversation_title=conversation_title,
        follow_up_suggestions=final_state.get("follow_up_suggestions"),
        research_metadata=research_metadata,
        execution_trace=execution_trace,
        created_at=turn_row.created_at,
    )
