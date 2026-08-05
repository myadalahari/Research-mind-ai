"""
Renders the ResearchMind AI conversation.

Purely presentational: this module takes ``ChatTurn`` objects (from
``state.session``) and draws Streamlit widgets from them. It never makes an
HTTP request and never reads or writes ``st.session_state`` -- ``app.py``
owns both of those. The one piece of user interaction this module can
produce (clicking a follow-up suggestion) is reported back to the caller as
a plain return value, the same pattern ``st.chat_input`` itself uses,
rather than this module reaching into ``api_client``/``session`` to act on
it directly.

Every turn -- whether it's the first message in the conversation or the one
that was just appended a moment ago -- is drawn by the same
``_render_turn`` call. The only thing that varies between turns is whether
follow-up suggestions are shown, which is a data-availability condition
(only the most recent turn's suggestions are still relevant to click) of
the same kind already applied to citations and the execution trace, not a
structurally different rendering path for "the latest message."
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

from core.models import AgentExecutionTrace, Citation, ExecutionStepStatus
from state.session import ChatTurn

_STATUS_ICONS = {
    ExecutionStepStatus.COMPLETED: "✅",  # check mark
    ExecutionStepStatus.FAILED: "❌",  # cross mark
    ExecutionStepStatus.RUNNING: "\U0001f504",  # cycle arrows
    ExecutionStepStatus.PENDING: "⏳",  # hourglass
    ExecutionStepStatus.SKIPPED: "⏭️",  # skip
}


def render_conversation(turns: List[ChatTurn]) -> Optional[str]:
    """
    Render every turn in ``turns``, oldest first.

    Returns the text of a follow-up suggestion the user clicked during
    this render, or ``None`` if none was clicked. The caller (``app.py``)
    is responsible for treating that returned text as a new query through
    its normal request flow -- this function only reports the click, it
    never acts on it.
    """
    if not turns:
        st.info("Ask a research question to get started.")
        return None

    clicked_suggestion: Optional[str] = None
    last_index = len(turns) - 1
    for index, turn in enumerate(turns):
        result = _render_turn(turn, show_suggestions=index == last_index)
        if result is not None:
            clicked_suggestion = result
    return clicked_suggestion


def _render_turn(turn: ChatTurn, *, show_suggestions: bool) -> Optional[str]:
    """Render one query/answer exchange. Returns a clicked suggestion's text, or ``None``."""
    with st.chat_message("user"):
        st.markdown(turn.query)

    clicked_suggestion: Optional[str] = None
    with st.chat_message("assistant"):
        st.markdown(turn.response.answer)

        if turn.response.citations:
            _render_citations(turn.response.citations)

        if turn.response.execution_trace is not None:
            _render_execution_trace(turn.response.execution_trace)

        if show_suggestions and turn.response.follow_up_suggestions:
            clicked_suggestion = _render_follow_up_suggestions(
                turn.response.follow_up_suggestions, message_id=turn.response.message_id
            )

    return clicked_suggestion


def _render_citations(citations: List[Citation]) -> None:
    with st.expander(f"Sources ({len(citations)})"):
        for citation in citations:
            title = f"[{citation.citation_id}] {citation.title}"
            if citation.url:
                st.markdown(f"**[{title}]({citation.url})**")
            else:
                st.markdown(f"**{title}**")

            source_line = _citation_source_line(citation)
            if source_line:
                st.caption(source_line)

            if citation.excerpt:
                st.markdown(f"> {citation.excerpt}")

            if citation.score is not None:
                st.caption(f"Relevance score: {citation.score:.2f}")

            st.divider()


def _citation_source_line(citation: Citation) -> Optional[str]:
    if citation.source_type.value == "document":
        parts = ["Document"]
        if citation.page_number is not None:
            parts.append(f"p. {citation.page_number}")
        if citation.section_title:
            parts.append(citation.section_title)
        return " · ".join(parts)
    if citation.source_type.value == "web":
        parts = ["Web"]
        if citation.accessed_at is not None:
            parts.append(f"accessed {citation.accessed_at.date().isoformat()}")
        return " · ".join(parts)
    return None


def _render_execution_trace(trace: AgentExecutionTrace) -> None:
    summary_parts = [f"{len(trace.steps)} step{'s' if len(trace.steps) != 1 else ''}"]
    if trace.total_latency_ms is not None:
        summary_parts.append(f"{trace.total_latency_ms:.0f} ms")
    with st.expander(f"Agent execution trace ({', '.join(summary_parts)})", expanded=False):
        for step in trace.steps:
            icon = _STATUS_ICONS.get(step.status, "")
            header = f"{icon} **{step.agent_name}** -- {step.status.value}"
            if step.latency_ms is not None:
                header += f" ({step.latency_ms:.0f} ms)"
            st.markdown(header)

            if step.summary:
                st.caption(step.summary)

            detail_parts = []
            if step.model_name:
                detail_parts.append(f"model: {step.model_name}")
            if step.tool_name:
                detail_parts.append(f"tool: {step.tool_name}")
            if step.token_usage is not None and step.token_usage.total_tokens is not None:
                detail_parts.append(f"tokens: {step.token_usage.total_tokens}")
            if detail_parts:
                st.caption(" · ".join(detail_parts))

            if step.status == ExecutionStepStatus.FAILED and step.error is not None:
                st.error(f"{step.error.message} (`{step.error.error_code}`)")

            st.divider()


def _render_follow_up_suggestions(suggestions: List[str], *, message_id: str) -> Optional[str]:
    st.caption("What would you like to ask next?")
    clicked: Optional[str] = None
    columns = st.columns(len(suggestions))
    # strict=True: columns is constructed as exactly len(suggestions) columns above,
    # so the two are always the same length by construction -- strict=True documents
    # that invariant and fails loudly rather than silently truncating if it's ever
    # violated by a future change, the same reasoning behind the backend's own
    # zip(..., strict=True) in app.rag.vector_store.
    for index, (column, suggestion) in enumerate(zip(columns, suggestions, strict=True)):
        with column:
            if st.button(suggestion, key=f"followup-{message_id}-{index}", use_container_width=True):
                clicked = suggestion
    return clicked
