"""
The Researcher agent: consolidates the Retriever agent's document chunks
and the Search agent's web results into a single, uniformly-numbered list
of ``Citation``\\ s -- the shape every downstream node (Writer, Fact
Checker, Reviewer) and the eventual API response cite by ``citation_id``.

Unlike every other node written so far in Phase 6, this one has no
external dependency to inject -- no LLM call, no HTTP adapter, no vector
store. It is pure state transformation, so it is a plain module-level
``async def`` rather than a closure factory: the closure-factory pattern
(``build_planner_node``, ``build_retriever_node``, ``build_search_node``)
exists specifically to capture dependencies at graph-build time (see
``app.agents.state``'s design note), and a factory with nothing to capture
would itself be the kind of speculative abstraction this project's process
avoids.
"""

from __future__ import annotations

from typing import List, Optional

from app.agents.state import AgentName, ResearchGraphState, track_step
from app.core.exceptions import ResearcherError
from app.core.interfaces.search_provider import SearchResult
from app.core.interfaces.vector_store import RetrievedChunk
from app.core.logging import get_logger
from app.schemas.common import Citation, SourceType

logger = get_logger(__name__)

# Citation.excerpt's own docstring: "Short quoted excerpt ... shown in the
# frontend's source viewer." RAGSettings.chunk_size defaults to 1000
# characters -- a full chunk is the wrong shape to hand across as an
# "excerpt," so it's bounded here rather than passed through verbatim.
_MAX_EXCERPT_CHARS = 300


async def researcher_node(state: ResearchGraphState) -> dict:
    query = state["query"]
    retrieved_chunks = state.get("retrieved_chunks") or []
    search_results = state.get("search_results") or []

    try:
        async with track_step(AgentName.RESEARCHER, "consolidate") as rec:
            citations = _build_citations(retrieved_chunks, search_results)
            rec.summary = (
                f"Consolidated {len(citations)} citation(s) from "
                f"{len(retrieved_chunks)} chunk(s) and {len(search_results)} web result(s)"
            )
    except Exception as exc:
        # No external call happens in this node, so there is no known,
        # well-understood failure mode to degrade gracefully from (unlike
        # the Retriever's RetrievalError or the Search agent's
        # SearchProviderError) -- any exception here is a genuine bug, and
        # halting the run is the correct response, matching the Planner's
        # all-fatal pattern for the same reason.
        raise ResearcherError.wrap(
            exc, f"Researcher agent failed to consolidate evidence for query: {query!r}"
        ) from exc

    return {"citations": citations, "execution_steps": [rec.step]}


def _build_citations(retrieved_chunks: List[RetrievedChunk], search_results: List[SearchResult]) -> List[Citation]:
    """
    Build a flat, sequentially-numbered citation list: document citations
    first (sorted by score, most relevant first), then web citations (same
    ordering). Numbering starts at 1 to match ``Citation.citation_id``'s
    own docstring example (``"1"``, used as an inline marker like ``[1]``).

    Document citations carry ``chunk.document_id`` through onto
    ``Citation.document_id`` -- this is the only place in the pipeline
    that has it, and it's what ``ChatService`` (Phase 7) persists onto
    ``TurnCitation.document_id``, the column ``HistoryService``'s
    ``documents_considered`` reconstruction depends on. Web citations
    leave it ``None``; they have no backing ``Document`` row.
    """
    citations: List[Citation] = []
    next_id = 1

    for retrieved in sorted(retrieved_chunks, key=lambda r: r.score, reverse=True):
        chunk = retrieved.chunk
        citations.append(
            Citation(
                citation_id=str(next_id),
                source_type=SourceType.DOCUMENT,
                title=chunk.source_filename,
                url=None,
                source_filename=chunk.source_filename,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                excerpt=_truncate(chunk.text),
                score=_clamp_score(retrieved.score),
                published_date=None,
                accessed_at=chunk.uploaded_at,
            )
        )
        next_id += 1

    for result in sorted(search_results, key=_search_result_sort_key, reverse=True):
        citations.append(
            Citation(
                citation_id=str(next_id),
                source_type=SourceType.WEB,
                title=result.title,
                url=result.url,
                source_filename=None,
                page_number=None,
                section_title=None,
                excerpt=_truncate(result.snippet),
                score=_clamp_score(result.score),
                published_date=result.published_date,
                accessed_at=result.retrieved_at,
            )
        )
        next_id += 1

    return citations


def _search_result_sort_key(result: SearchResult) -> tuple:
    """Results with a score sort before scoreless ones; both group internally by score descending."""
    return (result.score is not None, result.score if result.score is not None else 0.0)


def _truncate(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    stripped = text.strip()
    if len(stripped) <= _MAX_EXCERPT_CHARS:
        return stripped
    return stripped[:_MAX_EXCERPT_CHARS].rstrip() + "..."


def _clamp_score(score: Optional[float]) -> Optional[float]:
    """
    ``Citation.score`` is bounded to ``[0.0, 1.0]``, but the upstream
    sources aren't: ``RetrievedChunk.score`` is an unconstrained float
    (``ChromaVectorStore``'s own comment notes cosine distance can push
    ``1 - distance`` slightly outside that range), and a search provider's
    reported relevance score has no contract guaranteeing the same bound.
    Clamping here avoids a ``Citation`` construction crash on an otherwise
    legitimate, borderline result.
    """
    if score is None:
        return None
    return max(0.0, min(1.0, score))
