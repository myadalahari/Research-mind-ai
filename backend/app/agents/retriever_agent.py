"""
The Retriever agent: a thin graph-node adapter over the already-complete
RAG retrieval pipeline (``app.rag.retriever.retrieve``).

Built as a closure factory, mirroring ``app.agents.planner`` -- dependencies
(``EmbeddingProvider``, ``VectorStore``, ``RAGSettings``) are captured once
at graph-build time, never smuggled through ``ResearchGraphState``.

This file's one real piece of logic (not just wiring) is resolving the
correct ``session_id``/``document_ids`` scope from ``RetrievalOptions``
before calling ``retrieve()`` -- see ``_resolve_scope``'s docstring for why
that's a genuine correctness concern, not boilerplate.
"""

from __future__ import annotations

from typing import Awaitable, Callable, List, Optional, Tuple

from app.agents.state import AgentName, ResearchGraphState, track_step
from app.core.config import RAGSettings
from app.core.exceptions import RetrievalError, RetrieverAgentError
from app.core.interfaces.embedding_provider import EmbeddingProvider
from app.core.interfaces.vector_store import RetrievedChunk, VectorStore
from app.core.logging import get_logger
from app.rag.retriever import retrieve
from app.schemas.chat import RetrievalOptions

logger = get_logger(__name__)


def build_retriever_node(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    settings: RAGSettings,
) -> Callable[[ResearchGraphState], Awaitable[dict]]:
    """
    Build the Retriever's LangGraph node function.

    Args:
        embedding_provider: Injected ``EmbeddingProvider`` -- must be the
            same instance used at ingestion time (see ``retrieve()``'s own
            docstring on why).
        vector_store: Injected ``VectorStore``.
        settings: Supplies the ``retrieval_top_k`` / ``retrieval_score_threshold``
            defaults when a request doesn't override them.

    Returns:
        An ``async def retriever_node(state) -> dict`` suitable for
        ``StateGraph.add_node``.
    """

    async def retriever_node(state: ResearchGraphState) -> dict:
        query = state["query"]
        retrieval_options = state.get("retrieval_options")
        effective_session_id, skip_entirely = _resolve_scope(state.get("session_id"), retrieval_options)
        document_ids = retrieval_options.document_ids if retrieval_options is not None else None
        top_k = retrieval_options.top_k if retrieval_options is not None else None
        score_threshold = retrieval_options.score_threshold if retrieval_options is not None else None

        try:
            async with track_step(AgentName.RETRIEVER, "retrieve") as rec:
                if skip_entirely:
                    rec.summary = (
                        "Session-scoped retrieval requested with no active session "
                        "(new conversation); returning no results rather than searching "
                        "across every session."
                    )
                    chunks: List[RetrievedChunk] = []
                else:
                    chunks = await retrieve(
                        query=query,
                        embedding_provider=embedding_provider,
                        vector_store=vector_store,
                        settings=settings,
                        top_k=top_k,
                        score_threshold=score_threshold,
                        session_id=effective_session_id,
                        document_ids=document_ids,
                    )
                    rec.summary = f"Retrieved {len(chunks)} chunk(s)"
        except RetrievalError:
            # A well-understood, already-defined failure mode of the RAG
            # pipeline -- degrade gracefully (empty results) rather than
            # kill a run that may still succeed via the Search agent. The
            # FAILED step (built by track_step before this exception
            # propagated out of the `async with` block) is still recorded.
            logger.warning("Retriever agent failed; continuing without document context", extra={"query": query})
            return {"retrieved_chunks": [], "execution_steps": [rec.step]}
        except Exception as exc:
            # Anything else is not a known RAG-pipeline failure mode -- a
            # genuine bug, not something to silently paper over.
            raise RetrieverAgentError.wrap(exc, f"Retriever agent failed unexpectedly for query: {query!r}") from exc

        return {"retrieved_chunks": chunks, "execution_steps": [rec.step]}

    return retriever_node


def _resolve_scope(
    session_id: Optional[str], retrieval_options: Optional[RetrievalOptions]
) -> Tuple[Optional[str], bool]:
    """
    Resolve the ``session_id`` to actually pass to ``retrieve()``.

    ``VectorStore.query``'s ``session_id`` filter is opt-in: passing
    ``None`` means "search across every session," not "no results."
    ``RetrievalOptions.session_scope`` defaults to ``True`` ("restrict to
    the current session"), but a brand-new conversation legitimately has
    ``session_id=None`` (``ChatRequest.session_id``'s own docstring: "Omit
    to start a new session"). Passing that ``None`` straight through while
    ``session_scope=True`` would silently search every *other* session's
    documents instead of correctly finding none -- a session that doesn't
    exist yet is guaranteed to have zero uploaded documents, so returning
    nothing is not just a safe default, it's the only correct answer.

    Returns:
        ``(effective_session_id, skip_entirely)``. When ``skip_entirely``
        is ``True``, the caller must not call ``retrieve()`` at all.
    """
    session_scope = retrieval_options.session_scope if retrieval_options is not None else True
    if not session_scope:
        # Deliberate, explicitly-requested "search across everything" mode.
        return None, False
    if session_id is None:
        return None, True
    return session_id, False
