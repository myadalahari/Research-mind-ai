"""
The Reviewer agent: the research graph's quality gate. Judges the
Writer's draft (using the Fact Checker's per-claim verdicts, not by
re-deriving grounding itself) and either approves it or sends it back for
revision.

Built as a closure factory (``build_reviewer_node``), mirroring
``app.agents.planner``/``app.agents.writer``/``app.agents.fact_checker`` --
``LLMService`` is captured once at graph-build time. Uses
``LLMService.generate_structured`` against ``ReviewFeedback`` (already
defined in ``app.agents.state``, reused as-is).

Scope note: this node increments ``revision_count`` when it sends a draft
back for revision, since that's the only place in the graph where the
decision "this draft needs another pass" is made. It deliberately does
*not* enforce ``AgentSettings.max_reviewer_retries`` or raise
``MaxRetriesExceededError`` -- LangGraph's conditional-edge routing
functions are pure reads of state that pick the next node without
mutating it, so *which* node runs next once revision_count is updated
(back to the Writer, on to the Coordinator, or a hard failure once
retries are exhausted) is ``app.agents.graph``'s responsibility, not this
node's. This node only produces the data that routing decision reads.
"""

from __future__ import annotations

from typing import Awaitable, Callable, List, Optional

from app.agents.state import AgentName, ResearchGraphState, ResearchPlan, ReviewFeedback, VerifiedClaim, track_step
from app.core.exceptions import LLMServiceError, ReviewerError
from app.core.interfaces.llm_service import LLMService
from app.core.logging import get_logger

logger = get_logger(__name__)

_TEMPERATURE = 0.1

_SYSTEM_PROMPT = (
    "You are the Reviewer agent in a multi-agent research assistant -- the final "
    "quality gate before a draft answer is shown to the user. Given the original "
    "query, the draft answer, and the Fact Checker's per-claim verdicts, decide "
    "whether the draft is ready to ship.\n\n"
    "Approve (approved=true) only if the draft:\n"
    "- Actually answers the query (and every sub-question, if given).\n"
    "- Has no claims the Fact Checker marked unsupported -- an unsupported claim is "
    "grounds for rejection, not just a note.\n"
    "- Is clear, well-organized, and free of obvious contradictions or filler.\n\n"
    "If you reject (approved=false), 'feedback' must give the Writer clear, "
    "actionable direction for the rewrite, and 'issues' must list the specific "
    "problems found (e.g. 'Claim about X is unsupported by the cited source', "
    "'Does not address the sub-question about Y'). Set quality_score (0.0 to 1.0) "
    "as your overall assessment either way."
)


def build_reviewer_node(
    llm_service: LLMService,
    model_name: str,
) -> Callable[[ResearchGraphState], Awaitable[dict]]:
    """
    Build the Reviewer's LangGraph node function.

    Args:
        llm_service: Injected ``LLMService``.
        model_name: The configured model identifier, stamped onto this
            node's ``AgentExecutionStep.model_name``, matching every other
            LLM-backed node's same parameter.

    Returns:
        An ``async def reviewer_node(state) -> dict`` suitable for
        ``StateGraph.add_node``.
    """

    async def reviewer_node(state: ResearchGraphState) -> dict:
        query = state["query"]
        draft_answer = state.get("draft_answer")

        if not draft_answer:
            # The graph is wired so the Writer always runs before the
            # Reviewer -- a missing draft here means the graph itself is
            # miswired, not an expected runtime state. Fatal, matching the
            # Fact Checker's own identical guard.
            raise ReviewerError(f"Reviewer has no draft answer to review for query: {query!r}")

        plan = state.get("plan")
        verified_claims = state.get("verified_claims") or []
        revision_count = state.get("revision_count", 0)
        prompt = _build_prompt(query, plan, draft_answer, verified_claims)

        try:
            # As with fact_checker.py: every branch returns only after the
            # `async with` block exits, never from inside it, so `rec.step`
            # is always fully populated by the time it's read.
            async with track_step(AgentName.REVIEWER, "review", model_name=model_name) as rec:
                feedback = await llm_service.generate_structured(
                    prompt, ReviewFeedback, system_prompt=_SYSTEM_PROMPT, temperature=_TEMPERATURE
                )
                score_part = (
                    f", quality_score={feedback.quality_score:.2f}" if feedback.quality_score is not None else ""
                )
                rec.summary = f"Review: {'approved' if feedback.approved else 'needs revision'}{score_part}"
        except LLMServiceError:
            # The Reviewer's own call failing doesn't make the Writer's
            # draft unusable -- there is still a complete, usable answer
            # sitting in state. Auto-approving rather than blocking the
            # entire run on an unrelated infrastructure failure applies
            # the same graceful-degrade-over-fatal reasoning already used
            # for the Retriever, Search, and Fact Checker agents (as
            # opposed to the Planner/Writer, where failure leaves nothing
            # usable downstream at all). The FAILED step (built by
            # track_step, preserving the real LLMServiceError subclass's
            # error_code) is still recorded.
            logger.warning(
                "Reviewer failed; auto-approving the draft rather than blocking the run", extra={"query": query}
            )
            feedback = ReviewFeedback(
                approved=True,
                feedback="Automated review could not be completed due to an internal error; the draft was accepted as-is.",
                quality_score=None,
                issues=[],
            )
            return {"review_feedback": feedback, "execution_steps": [rec.step]}
        except Exception as exc:
            raise ReviewerError.wrap(exc, f"Reviewer failed unexpectedly for query: {query!r}") from exc

        new_revision_count = revision_count if feedback.approved else revision_count + 1
        return {
            "review_feedback": feedback,
            "revision_count": new_revision_count,
            "execution_steps": [rec.step],
        }

    return reviewer_node


def _build_prompt(
    query: str,
    plan: Optional[ResearchPlan],
    draft_answer: str,
    verified_claims: List[VerifiedClaim],
) -> str:
    sections = [f"Original query:\n{query}"]

    if plan is not None and len(plan.sub_questions) > 1:
        numbered_sub_questions = "\n".join(f"- {sq}" for sq in plan.sub_questions)
        sections.append(f"Sub-questions the answer should address:\n{numbered_sub_questions}")

    sections.append(f"Draft answer:\n{draft_answer}")

    if verified_claims:
        sections.append(f"Fact Checker verdicts:\n{_format_claims(verified_claims)}")
    else:
        sections.append(
            "The Fact Checker found no claims to verify (no evidence sources were used for this "
            "query). Judge the draft on whether it answers the query clearly and correctly from "
            "general knowledge, not on citation grounding."
        )

    sections.append("Review the draft now.")
    return "\n\n".join(sections)


def _format_claims(verified_claims: List[VerifiedClaim]) -> str:
    lines = []
    for claim in verified_claims:
        verdict = "SUPPORTED" if claim.is_supported else "UNSUPPORTED"
        notes = f" -- {claim.notes}" if claim.notes else ""
        lines.append(f"[{verdict}] {claim.claim_text}{notes}")
    return "\n".join(lines)
