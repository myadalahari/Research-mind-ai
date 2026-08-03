"""
The Fact Checker agent: verifies the Writer's draft answer against the
citations the Researcher consolidated -- nothing more. It does not
re-query the vector store or search provider, does not fetch new
evidence, and does not rewrite the draft; it only produces a per-claim
verdict (``VerifiedClaim``) the Reviewer uses to judge the draft's
quality.

Built as a closure factory (``build_fact_checker_node``), mirroring
``app.agents.planner``/``app.agents.writer`` -- ``LLMService`` is captured
once at graph-build time. Uses ``LLMService.generate_structured`` (like
the Planner) since the output here is machine-parseable verdicts, not
prose.
"""

from __future__ import annotations

from typing import Awaitable, Callable, List

from pydantic import BaseModel, Field

from app.agents.state import AgentName, ResearchGraphState, VerifiedClaim, track_step
from app.core.exceptions import FactCheckerError, LLMServiceError
from app.core.interfaces.llm_service import LLMService
from app.core.logging import get_logger
from app.schemas.common import Citation

logger = get_logger(__name__)

_TEMPERATURE = 0.1

_SYSTEM_PROMPT = (
    "You are the Fact Checker agent in a multi-agent research assistant. Given a "
    "draft answer and the numbered evidence sources it was written from, extract "
    "every distinct factual claim in the draft and verify each one against the "
    "sources.\n\n"
    "For each claim:\n"
    "- claim_text: the claim as it appears (or closely paraphrased) in the draft.\n"
    '- citation_ids: the source number(s) (as strings, e.g. "1") the draft cited '
    "for this claim. Use exactly the numbers the draft used -- do not invent one.\n"
    "- is_supported: true only if the cited source(s) actually contain evidence for "
    "the claim. If the draft cited no source for this claim, or cited a source that "
    "doesn't actually support it, set this to false.\n"
    "- confidence: your confidence in this verdict, 0.0 to 1.0.\n"
    "- notes: a brief explanation, especially when is_supported is false.\n\n"
    "Only evaluate claims against the sources provided -- you have no other "
    "information about whether a claim is true. A claim can be well-written and "
    "still unsupported if the cited source doesn't actually back it up."
)


class _FactCheckOutput(BaseModel):
    """
    ``generate_structured`` envelope for a list of ``VerifiedClaim``.

    Not part of ``ResearchGraphState`` -- the graph stores
    ``verified_claims: List[VerifiedClaim]`` directly (see
    ``app.agents.state``); this wrapper exists only because
    ``generate_structured`` requires a single ``BaseModel`` schema, and a
    bare ``List[VerifiedClaim]`` isn't one.
    """

    claims: List[VerifiedClaim] = Field(
        default_factory=list, description="Every distinct factual claim found in the draft, with its verdict."
    )


def build_fact_checker_node(
    llm_service: LLMService,
    model_name: str,
) -> Callable[[ResearchGraphState], Awaitable[dict]]:
    """
    Build the Fact Checker's LangGraph node function.

    Args:
        llm_service: Injected ``LLMService``.
        model_name: The configured model identifier, stamped onto this
            node's ``AgentExecutionStep.model_name``, matching
            ``build_planner_node``'s/``build_writer_node``'s same parameter.

    Returns:
        An ``async def fact_checker_node(state) -> dict`` suitable for
        ``StateGraph.add_node``.
    """

    async def fact_checker_node(state: ResearchGraphState) -> dict:
        query = state["query"]
        draft_answer = state.get("draft_answer")
        citations = state.get("citations") or []

        if not draft_answer:
            # The graph is wired so the Writer always runs before the Fact
            # Checker -- a missing draft here means the graph itself is
            # miswired, not an expected runtime state. Fatal, like the
            # Planner's/Writer's own "nothing to work with" failures.
            raise FactCheckerError(f"Fact Checker has no draft answer to verify for query: {query!r}")

        try:
            # NOTE: every branch below sets `claims` and returns only
            # *after* the `async with` block exits -- never from inside
            # it. Returning from inside the block would read `rec.step`
            # before `track_step`'s normal-exit path has populated it
            # (an early `return` still triggers __aexit__, but the
            # returned dict literal is evaluated first), which would ship
            # `None` as the execution step. `retriever_agent.py`'s
            # skip_entirely case avoids this the same way.
            async with track_step(AgentName.FACT_CHECKER, "verify", model_name=model_name) as rec:
                if not citations:
                    # Nothing was cited (the Planner decided neither
                    # retrieval nor search was useful for this query) --
                    # there is nothing to verify claims against, so there
                    # is nothing to gain from an LLM call here. Mirrors the
                    # Writer's own no-evidence case.
                    rec.summary = "No citations available; nothing to verify."
                    claims = []
                else:
                    prompt = _build_prompt(draft_answer, citations)
                    output = await llm_service.generate_structured(
                        prompt, _FactCheckOutput, system_prompt=_SYSTEM_PROMPT, temperature=_TEMPERATURE
                    )
                    claims = output.claims
                    unsupported = sum(1 for c in claims if not c.is_supported)
                    rec.summary = f"Verified {len(claims)} claim(s), {unsupported} unsupported"
        except LLMServiceError:
            # Verification failing doesn't make the draft unusable -- the
            # Reviewer can still evaluate it, just without claim-level
            # verdicts for this pass. Degrade rather than abort, matching
            # the Retriever's/Search agent's precedent: an external call
            # failing shouldn't necessarily kill a run that otherwise has
            # a usable draft. The FAILED step (built by track_step,
            # preserving the real LLMServiceError subclass's error_code)
            # is still recorded before this exception propagated out of
            # the `async with` block.
            logger.warning("Fact Checker failed; continuing without claim verification", extra={"query": query})
            return {"verified_claims": [], "execution_steps": [rec.step]}
        except Exception as exc:
            raise FactCheckerError.wrap(exc, f"Fact Checker failed unexpectedly for query: {query!r}") from exc

        return {"verified_claims": claims, "execution_steps": [rec.step]}

    return fact_checker_node


def _build_prompt(draft_answer: str, citations: List[Citation]) -> str:
    citations_block = "\n".join(
        f"[{c.citation_id}] {c.title}: {c.excerpt or '(no excerpt available)'}" for c in citations
    )
    return (
        f"Draft answer to verify:\n{draft_answer}\n\n"
        f"Evidence sources it was written from:\n{citations_block}\n\n"
        "Extract and verify every distinct factual claim now."
    )
