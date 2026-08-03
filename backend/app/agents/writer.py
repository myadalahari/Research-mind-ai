"""
The Writer agent: synthesizes the grounded draft answer from the
Researcher's consolidated citations (and, on a revision pass, the
Reviewer's feedback on the previous draft).

Built as a closure factory (``build_writer_node``), mirroring
``app.agents.planner`` -- ``LLMService`` is captured once at graph-build
time. Uses ``LLMService.generate`` (free-form text), not
``generate_structured``: the draft answer is prose (Markdown), not a
schema-shaped object -- the Fact Checker (next node) is what extracts and
verifies discrete claims from it.
"""

from __future__ import annotations

from typing import Awaitable, Callable, List, Optional

from app.agents.state import AgentName, ReviewFeedback, ResearchGraphState, ResearchPlan, track_step
from app.core.exceptions import LLMServiceError, WriterError
from app.core.interfaces.llm_service import LLMResponse, LLMService
from app.core.logging import get_logger
from app.schemas.chat import ChatMode
from app.schemas.common import Citation, TokenUsage

logger = get_logger(__name__)

_TEMPERATURE = 0.4

_BASE_SYSTEM_PROMPT = (
    "You are the Writer agent in a multi-agent research assistant. Given a "
    "user's research query, a set of numbered evidence sources, and (optionally) "
    "sub-questions to address, write a clear, well-organized, accurate answer.\n\n"
    "Grounding rules:\n"
    "- Every factual claim that is supported by one of the numbered sources must "
    "cite it inline using the source's number in square brackets, e.g. [1], "
    "immediately after the claim. A claim may cite more than one source, e.g. [1][2].\n"
    "- Never invent a source number that was not provided.\n"
    "- Do not fabricate facts, quotes, or citations that aren't backed by the "
    "provided evidence.\n"
    "- If the evidence is insufficient to fully answer part of the query, say so "
    "plainly rather than guessing."
)

_REPORT_MODE_ADDENDUM = (
    "\n\nThis answer will be used as a formal report. Structure it in Markdown with "
    "these sections, in order: a brief executive summary; one or more detailed "
    "findings sections (using '## ' headings) that address each sub-question; and a "
    "concluding section. Use Markdown tables where they would clarify comparisons."
)

_RESEARCH_MODE_ADDENDUM = (
    "\n\nThis answer will be shown directly in a conversational research assistant. "
    "Write it as a concise, well-organized response -- short paragraphs and, where "
    "useful, bullet points. Do not pad it with report-style front matter."
)

_NO_EVIDENCE_NOTE = (
    "No external sources were retrieved for this query (no relevant uploaded "
    "documents or web results). Answer from your own general knowledge, and do not "
    "invent citations -- simply answer without bracketed source numbers."
)


def build_writer_node(
    llm_service: LLMService,
    model_name: str,
) -> Callable[[ResearchGraphState], Awaitable[dict]]:
    """
    Build the Writer's LangGraph node function.

    Args:
        llm_service: Injected ``LLMService``.
        model_name: The configured model identifier, stamped onto this
            node's ``AgentExecutionStep.model_name``, matching
            ``build_planner_node``'s same parameter.

    Returns:
        An ``async def writer_node(state) -> dict`` suitable for
        ``StateGraph.add_node``.
    """

    async def writer_node(state: ResearchGraphState) -> dict:
        query = state["query"]
        mode = state["mode"]
        plan = state.get("plan")
        citations = state.get("citations") or []
        previous_draft = state.get("draft_answer")
        review_feedback = state.get("review_feedback")

        system_prompt = _build_system_prompt(mode)
        user_prompt = _build_user_prompt(
            query=query,
            plan=plan,
            citations=citations,
            previous_draft=previous_draft,
            review_feedback=review_feedback,
        )

        async with track_step(AgentName.WRITER, "write", model_name=model_name) as rec:
            try:
                response = await llm_service.generate(
                    user_prompt, system_prompt=system_prompt, temperature=_TEMPERATURE
                )
            except LLMServiceError as exc:
                raise WriterError.wrap(exc, f"Writer failed to synthesize a draft answer for query: {query!r}") from exc

            rec.token_usage = _to_token_usage(response)
            revision_note = " (revision)" if previous_draft is not None else ""
            rec.summary = f"Drafted answer{revision_note}: {len(response.content)} character(s), citing {len(citations)} source(s)"

        return {"draft_answer": response.content, "execution_steps": [rec.step]}

    return writer_node


def _build_system_prompt(mode: ChatMode) -> str:
    if mode == ChatMode.REPORT:
        return _BASE_SYSTEM_PROMPT + _REPORT_MODE_ADDENDUM
    return _BASE_SYSTEM_PROMPT + _RESEARCH_MODE_ADDENDUM


def _build_user_prompt(
    *,
    query: str,
    plan: Optional[ResearchPlan],
    citations: List[Citation],
    previous_draft: Optional[str],
    review_feedback: Optional[ReviewFeedback],
) -> str:
    sections = [f"Research query:\n{query}"]

    if plan is not None and len(plan.sub_questions) > 1:
        numbered_sub_questions = "\n".join(f"- {sq}" for sq in plan.sub_questions)
        sections.append(f"Sub-questions to address:\n{numbered_sub_questions}")

    if citations:
        sections.append(f"Evidence sources:\n{_format_citations(citations)}")
    else:
        sections.append(_NO_EVIDENCE_NOTE)

    if previous_draft is not None and review_feedback is not None:
        issues_block = (
            "\n".join(f"- {issue}" for issue in review_feedback.issues) if review_feedback.issues else "(none listed)"
        )
        sections.append(
            "This is a revision. Your previous draft was reviewed and needs improvement.\n\n"
            f"Previous draft:\n{previous_draft}\n\n"
            f"Reviewer feedback:\n{review_feedback.feedback}\n\n"
            f"Specific issues to fix:\n{issues_block}\n\n"
            "Write an improved draft that addresses this feedback. Do not just restate "
            "the previous draft -- make the changes needed to resolve the issues above."
        )

    sections.append("Write the answer now.")
    return "\n\n".join(sections)


def _format_citations(citations: List[Citation]) -> str:
    lines = []
    for citation in citations:
        source = citation.source_filename or citation.url or citation.title
        excerpt = citation.excerpt or "(no excerpt available)"
        lines.append(f"[{citation.citation_id}] {citation.title} ({source}): {excerpt}")
    return "\n".join(lines)


def _to_token_usage(response: LLMResponse) -> Optional[TokenUsage]:
    if response.prompt_tokens is None and response.completion_tokens is None and response.total_tokens is None:
        return None
    return TokenUsage(
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
    )
