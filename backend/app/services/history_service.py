"""
HistoryService — assembles ``GET /history`` responses.

Framework-agnostic: no FastAPI import anywhere. Depends only on
``app.database.session.Database`` (itself framework-agnostic) and
constructs concrete repositories against it per unit of work -- unlike
``HealthService``'s external-provider dependencies (``LLMService``,
``VectorStore``, ...), repositories have no ABC in front of them, since
there is exactly one persistence technology in this project and the
repository classes are already the abstraction over it. Introducing a
repository *interface* with no second implementation ever planned would
be an abstraction with nothing to abstract over.

Composes ``ConversationRepository`` and ``ReportRepository`` within a
single ``database.session()`` unit of work per call, following the
pattern established across the repository layer (see the Architecture
Decision Log, ADR-015 onward).
"""

from __future__ import annotations

from typing import List, Optional

from app.core.logging import get_logger
from app.database.repositories.conversation_repository import ConversationRepository, SessionSummary
from app.database.repositories.report_repository import ReportRepository
from app.database.session import Database
from app.models.conversation import AgentExecutionStep as AgentExecutionStepRow
from app.models.conversation import ConversationTurn as ConversationTurnRow
from app.models.conversation import TurnCitation as TurnCitationRow
from app.models.enums import ExecutionStepStatus as ModelExecutionStepStatus
from app.models.enums import SessionStatus as ModelSessionStatus
from app.models.report import Report as ReportRow
from app.schemas.chat import ChatMode as SchemaChatMode
from app.schemas.chat import ResearchMetadata
from app.schemas.common import AgentExecutionStep as SchemaAgentExecutionStep
from app.schemas.common import AgentExecutionTrace, Citation, ErrorBody, PaginatedResponse, SourceType, TokenUsage
from app.schemas.common import ExecutionStepStatus as SchemaExecutionStepStatus
from app.schemas.history import ConversationSession
from app.schemas.history import ConversationTurn as SchemaConversationTurn
from app.schemas.history import HistoryQueryParams
from app.schemas.history import SessionStatus as SchemaSessionStatus
from app.schemas.report import ReportExportFormat as SchemaReportExportFormat
from app.schemas.report import ReportGenerationStatus as SchemaReportGenerationStatus
from app.schemas.report import ReportListItem

logger = get_logger(__name__)


class HistoryService:
    """Builds ``GET /history``'s paginated session list (and single-session detail view)."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def list_sessions(self, params: HistoryQueryParams) -> PaginatedResponse[ConversationSession]:
        """
        Return a page of sessions, or -- when ``params.session_id`` is set
        -- that one session's detail as a single-item page.

        Per ``HistoryQueryParams.session_id``'s documented contract,
        setting it forces ``include_turns=True`` regardless of the
        client's ``include_turns`` value, and the result is a normal
        (possibly empty) filtered page rather than a 404: a session_id
        that matches nothing behaves like any other filter that matches
        nothing, not like a required lookup that failed.
        """
        async with self._database.session() as session:
            conversation_repo = ConversationRepository(session)
            report_repo = ReportRepository(session)

            if params.session_id is not None:
                summary = await conversation_repo.get_summary(params.session_id)
                summaries: List[SessionSummary] = [summary] if summary is not None else []
                total = len(summaries)
                include_turns = True
                page, page_size = 1, 1
            else:
                offset = (params.page - 1) * params.page_size
                model_status = _to_model_status(params.status)
                summaries = await conversation_repo.list_summaries(
                    status=model_status, limit=params.page_size, offset=offset
                )
                total = await conversation_repo.count_sessions(status=model_status)
                include_turns = params.include_turns
                page, page_size = params.page, params.page_size

            # Batched across the whole page -- see module docstring and the
            # design discussion for why this one is batched but full turn
            # listing below is not.
            latest_reports = await report_repo.get_latest_by_sessions([summary.session.id for summary in summaries])

            items: List[ConversationSession] = []
            for summary in summaries:
                turns: Optional[List[SchemaConversationTurn]] = None
                if include_turns:
                    turn_rows = await conversation_repo.get_turns(
                        summary.session.id,
                        with_citations=True,
                        with_execution_steps=params.include_trace,
                    )
                    turns = [_map_turn(row, include_trace=params.include_trace) for row in turn_rows]
                items.append(_map_session(summary, turns=turns, latest_report=latest_reports.get(summary.session.id)))

        logger.info(
            "Listed session history",
            extra={"returned": len(items), "total": total, "filtered_to_session": params.session_id is not None},
        )
        return PaginatedResponse.create(items=items, total=total, page=page, page_size=page_size)


# =============================================================================
# ORM -> schema mapping (pure, stateless -- module-level rather than methods)
# =============================================================================


def _to_model_status(status: Optional[SchemaSessionStatus]) -> Optional[ModelSessionStatus]:
    return ModelSessionStatus(status.value) if status is not None else None


def _map_session(
    summary: SessionSummary,
    *,
    turns: Optional[List[SchemaConversationTurn]],
    latest_report: Optional[ReportRow],
) -> ConversationSession:
    session = summary.session
    # None (not a zeroed TokenUsage) when the session has no turns yet --
    # "no data" and "zero usage" are different facts, the same convention
    # ProcessingStatistics already uses elsewhere in this codebase.
    total_token_usage = (
        TokenUsage(
            prompt_tokens=summary.prompt_tokens,
            completion_tokens=summary.completion_tokens,
            total_tokens=summary.total_tokens,
        )
        if summary.turn_count > 0
        else None
    )
    return ConversationSession(
        session_id=session.id,
        title=session.title,
        status=SchemaSessionStatus(session.status.value),
        created_at=session.created_at,
        updated_at=session.updated_at,
        turn_count=summary.turn_count,
        document_count=summary.document_count,
        report_count=summary.report_count,
        preview=summary.preview,
        latest_report=_map_report_list_item(latest_report) if latest_report is not None else None,
        total_token_usage=total_token_usage,
        turns=turns,
    )


def _map_report_list_item(report: ReportRow) -> ReportListItem:
    return ReportListItem(
        report_id=report.id,
        session_id=report.session_id,
        title=report.title,
        export_format=SchemaReportExportFormat(report.export_format.value),
        status=SchemaReportGenerationStatus(report.status.value),
        created_at=report.created_at,
    )


def _map_turn(turn: ConversationTurnRow, *, include_trace: bool) -> SchemaConversationTurn:
    citations = [_map_citation(c) for c in turn.citations]
    # documents_considered isn't a persisted column (ADR-012 deliberately
    # flattened only 1:1 scalars onto ConversationTurn, not list fields) --
    # reconstructed correctly from this turn's own already-loaded citations
    # rather than left silently empty.
    documents_considered = sorted({c.document_id for c in turn.citations if c.document_id})
    research_metadata = ResearchMetadata(
        llm_provider=turn.llm_provider,
        llm_model=turn.llm_model,
        rag_enabled=turn.rag_enabled,
        web_search_enabled=turn.web_search_enabled,
        retrieved_chunk_count=turn.retrieved_chunk_count,
        web_result_count=turn.web_result_count,
        documents_considered=documents_considered,
        total_latency_ms=turn.total_latency_ms,
        token_usage=TokenUsage(
            prompt_tokens=turn.prompt_tokens,
            completion_tokens=turn.completion_tokens,
            total_tokens=turn.total_tokens,
        ),
    )

    execution_trace: Optional[AgentExecutionTrace] = None
    if include_trace and turn.execution_steps:
        mapped_steps = [_map_execution_step(step) for step in turn.execution_steps]
        execution_trace = AgentExecutionTrace.from_steps(
            # No request-time trace_id is persisted yet -- that would be
            # ChatService's (not yet built) job to capture at generation
            # time. The turn's own id is used as a stable per-turn
            # identifier instead; it is not a guarantee of matching any
            # specific server log line.
            trace_id=turn.id,
            session_id=turn.session_id,
            steps=mapped_steps,
            started_at=mapped_steps[0].started_at,
            completed_at=mapped_steps[-1].completed_at,
        )

    return SchemaConversationTurn(
        turn_id=turn.id,
        session_id=turn.session_id,
        sequence_number=turn.sequence_number,
        mode=SchemaChatMode(turn.mode.value),
        query=turn.query,
        answer=turn.answer,
        citations=citations,
        research_metadata=research_metadata,
        execution_trace=execution_trace,
        created_at=turn.created_at,
    )


def _map_citation(citation: TurnCitationRow) -> Citation:
    return Citation(
        citation_id=citation.citation_id,
        source_type=SourceType(citation.source_type.value),
        title=citation.title,
        url=citation.url,
        source_filename=citation.source_filename,
        page_number=citation.page_number,
        section_title=citation.section_title,
        excerpt=citation.excerpt,
        score=citation.score,
        published_date=citation.published_date,
        accessed_at=citation.accessed_at,
    )


def _map_execution_step(step: AgentExecutionStepRow) -> SchemaAgentExecutionStep:
    token_usage: Optional[TokenUsage] = None
    if step.prompt_tokens is not None or step.completion_tokens is not None or step.total_tokens is not None:
        token_usage = TokenUsage(
            prompt_tokens=step.prompt_tokens, completion_tokens=step.completion_tokens, total_tokens=step.total_tokens
        )

    error: Optional[ErrorBody] = None
    if step.status == ModelExecutionStepStatus.FAILED:
        error = ErrorBody(
            error_code=step.error_code or "RM-GEN-000",
            message=step.error_message or "An unknown error occurred.",
            # retryable is a live, in-the-moment question ("should a caller
            # retry right now?"); it isn't a fact about a past run, and
            # nothing persists it per-step. Defaulting False for a
            # historical record is honest -- error_code (the canonical
            # registry key) is what a reader would use to look up whether
            # this class of error is retryable in general.
            retryable=False,
        )

    return SchemaAgentExecutionStep(
        step_id=step.id,
        agent_name=step.agent_name,
        graph_node=step.graph_node,
        status=SchemaExecutionStepStatus(step.status.value),
        started_at=step.started_at,
        completed_at=step.completed_at,
        latency_ms=step.latency_ms,
        model_name=step.model_name,
        tool_name=step.tool_name,
        token_usage=token_usage,
        summary=step.summary,
        error=error,
    )
