"""
Retrieval orchestration: a query string -> ranked, citeable chunks.

Pipeline: ``embed_query`` -> ``VectorStore.query``. Framework-agnostic (no
FastAPI import) and persistence-agnostic (no SQLAlchemy/``Database``
import) -- this module's only job is turning a query into ranked chunks.
The Retriever agent (Phase 6) and ``ChatService`` are its callers.

Both sub-steps' failures are consolidated into a single ``RetrievalError``
rather than letting ``EmbeddingProviderError`` / ``VectorStoreError``
propagate directly. This is a deliberate difference from ``app.rag.ingest``,
which lets each stage's own exception type propagate distinctly: an
ingestion caller (``IngestionService``) genuinely needs to know *which*
stage failed to update ``Document.ingestion_status``/``error_code``
per-stage, whereas a retrieval caller (a chat turn in progress) only cares
that "retrieval, as one conceptual operation, failed" -- it reacts to that
fact the same way regardless of whether the embedding model or the vector
store was the proximate cause.
"""

from __future__ import annotations

from typing import List, Optional

from app.core.config import RAGSettings
from app.core.exceptions import EmbeddingProviderError, RetrievalError, VectorStoreError
from app.core.interfaces.embedding_provider import EmbeddingProvider
from app.core.interfaces.vector_store import RetrievedChunk, VectorStore
from app.core.logging import get_logger, measure_latency_ms

logger = get_logger(__name__)


async def retrieve(
    *,
    query: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    settings: RAGSettings,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    session_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
) -> List[RetrievedChunk]:
    """
    Retrieve the chunks most relevant to ``query``.

    Args:
        query: The user's question or search text. Rejected if blank.
        embedding_provider: Injected ``EmbeddingProvider`` -- must be the
            same one used at ingestion time, or scores are meaningless
            (different models produce geometrically incompatible vector
            spaces; enforcing this at the type level isn't possible since
            there is exactly one process-wide instance of each injected by
            ``app.core.dependencies`` today).
        vector_store: Injected ``VectorStore``.
        settings: Supplies the ``retrieval_top_k`` / ``retrieval_score_threshold``
            defaults when ``top_k`` / ``score_threshold`` aren't given.
        top_k: Overrides ``settings.retrieval_top_k`` for this call.
        score_threshold: Overrides ``settings.retrieval_score_threshold``
            for this call. Pass ``0.0`` explicitly to disable thresholding
            entirely for a single call (``None`` means "use the configured
            default," not "no threshold").
        session_id: Restrict retrieval to chunks from this research
            session.
        document_ids: Restrict retrieval to chunks from these specific
            documents (e.g. "summarize the uploaded papers").

    Returns:
        Chunks ordered by descending relevance score, each carrying full
        citation provenance (``document_id``, ``source_filename``,
        ``page_number``, ``section_title``) via ``RetrievedChunk.chunk``.
        Never raises on "no results" -- an empty list is a valid,
        meaningful answer ("nothing relevant was found"), not a failure.

    Raises:
        RetrievalError: ``query`` is blank (non-retryable -- retrying the
            same empty input cannot succeed), or embedding/querying failed
            (retryable -- the underlying cause is a transient
            ``EmbeddingProviderError`` or ``VectorStoreError``).
    """
    if not query.strip():
        raise RetrievalError("Cannot retrieve for a blank query.", retryable=False)

    effective_top_k = top_k if top_k is not None else settings.retrieval_top_k
    effective_score_threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold

    with measure_latency_ms() as elapsed:
        try:
            query_embedding = await embedding_provider.embed_query(query)
        except EmbeddingProviderError as exc:
            raise RetrievalError.wrap(exc, "Failed to embed the query for retrieval.") from exc

        try:
            results = await vector_store.query(
                query_embedding,
                top_k=effective_top_k,
                score_threshold=effective_score_threshold,
                session_id=session_id,
                document_ids=document_ids,
            )
        except VectorStoreError as exc:
            raise RetrievalError.wrap(exc, "Failed to query the vector store for retrieval.") from exc

    logger.debug(
        "Retrieved chunks",
        extra={
            "query_length": len(query),
            "returned": len(results),
            "top_k": effective_top_k,
            "score_threshold": effective_score_threshold,
            "session_id": session_id,
            "latency_ms": round(elapsed(), 2),
        },
    )
    return results
