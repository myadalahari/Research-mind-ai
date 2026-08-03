"""
The Search agent: a thin graph-node adapter over the already-complete
``SearchProvider`` interface (``app.core.interfaces.search_provider``).

Built as a closure factory, mirroring ``app.agents.retriever_agent`` --
dependencies (``SearchProvider``, ``SearchSettings``) are captured once at
graph-build time, never smuggled through ``ResearchGraphState``.

Unlike the Retriever agent, there is no session-scoping ambiguity to
resolve here -- a web search has no notion of "this session's documents."
The only non-trivial decision in this file is which ``SearchProviderError``
failures are recoverable (degrade to empty results, matching
``track_step``'s own docstring example of exactly this agent) versus
genuinely unexpected (wrapped as a fatal ``SearchAgentError``).
"""

from __future__ import annotations

from typing import Awaitable, Callable, List

from app.agents.state import AgentName, ResearchGraphState, track_step
from app.core.config import SearchSettings
from app.core.exceptions import SearchAgentError, SearchProviderError
from app.core.interfaces.search_provider import SearchProvider, SearchResult
from app.core.logging import get_logger

logger = get_logger(__name__)


def build_search_node(
    search_provider: SearchProvider,
    settings: SearchSettings,
) -> Callable[[ResearchGraphState], Awaitable[dict]]:
    """
    Build the Search agent's LangGraph node function.

    Args:
        search_provider: Injected ``SearchProvider`` (``TavilySearchProvider``
            today; any other implementation works unchanged, per the
            interface's own docstring).
        settings: Supplies ``max_results`` when a request doesn't override
            it. Currently unused beyond that -- ``SearchProvider.search``
            already reads its own default from the adapter's stored
            ``SearchSettings`` when ``max_results`` is omitted, so this
            parameter is kept for symmetry with ``build_retriever_node``
            and to leave room for a future per-request override without
            changing this function's signature. Not speculative machinery:
            it costs nothing to accept and not yet use, versus a signature
            change later that would ripple into ``app.agents.graph``.

    Returns:
        An ``async def search_node(state) -> dict`` suitable for
        ``StateGraph.add_node``.
    """

    async def search_node(state: ResearchGraphState) -> dict:
        query = state["query"]

        try:
            async with track_step(AgentName.SEARCH, "search", tool_name=settings.provider.value) as rec:
                response = await search_provider.search(query)
                results: List[SearchResult] = response.results
                rec.summary = f"Found {len(results)} result(s)"
        except SearchProviderError:
            # A well-understood, already-defined failure mode of the search
            # provider (timeout, quota exceeded, provider unavailable, or a
            # non-retryable request rejection that survived the adapter's
            # own internal retries) -- degrade gracefully rather than kill
            # a run that may still succeed via the Retriever agent's RAG
            # results. This is track_step's own docstring example of the
            # catch-around-the-block pattern. The FAILED step (built by
            # track_step before this exception propagated out of the
            # `async with` block) is still recorded.
            logger.warning("Search agent failed; continuing without web results", extra={"query": query})
            return {"search_results": [], "execution_steps": [rec.step]}
        except Exception as exc:
            # Anything else is not a known search-provider failure mode --
            # a genuine bug, not something to silently paper over.
            raise SearchAgentError.wrap(exc, f"Search agent failed unexpectedly for query: {query!r}") from exc

        return {"search_results": results, "execution_steps": [rec.step]}

    return search_node
