"""
Ingestion orchestration: raw uploaded bytes -> indexed, retrievable chunks.

Pipeline: ``extract_segments`` -> ``chunk_segments`` -> ``embed_documents``
-> ``VectorStore.upsert``. Framework-agnostic (no FastAPI import anywhere)
and persistence-agnostic (no SQLAlchemy/``Database`` import anywhere) --
this module's only job is turning bytes into indexed vectors. The future
``IngestionService`` is what will call ``ingest_document``, then separately
update the owning ``Document`` row's status/stats columns inside its own
``database.session()`` unit of work. Mixing that SQL into this module would
force every caller (including, eventually, any offline reindexing script)
to carry a database dependency it doesn't need.

All-or-nothing: extraction, chunking, and embedding each raise their own
already-established exception type (``DocumentExtractionError`` /
``DocumentChunkingError`` / ``EmbeddingProviderError``) before anything is
written to the vector store, so a failure in any of those stages leaves no
side effect behind. ``VectorStore.upsert`` is the only stage with a side
effect; see ``ingest_document``'s docstring for how a failure there is
handled.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from pydantic import BaseModel, Field

from app.core.config import RAGSettings
from app.core.exceptions import VectorStoreError
from app.core.interfaces.embedding_provider import EmbeddingProvider
from app.core.interfaces.vector_store import VectorStore
from app.core.logging import get_logger, measure_latency_ms
from app.rag.chunking import chunk_segments
from app.rag.extraction import extract_segments

logger = get_logger(__name__)


class IngestionResult(BaseModel):
    """
    Observability/stats output of a single ``ingest_document`` run.

    Field names deliberately mirror ``schemas.upload.ProcessingStatistics``
    so the future ``IngestionService`` can populate that schema (and the
    matching ``Document`` ORM columns) from this result with a trivial
    field-for-field mapping, without this module importing the schema
    layer itself (see module docstring).
    """

    document_id: str
    chunk_count: int = Field(..., ge=0)
    extracted_text_length: int = Field(..., ge=0, description="Character count of the extracted text, pre-chunking.")
    average_chunk_size: float = Field(..., ge=0, description="Average chunk length in characters.")
    embedding_model: str
    embedding_dimension: int = Field(..., ge=1)
    extraction_time_ms: float = Field(..., ge=0)
    chunking_time_ms: float = Field(..., ge=0)
    embedding_time_ms: float = Field(..., ge=0)
    upsert_time_ms: float = Field(..., ge=0)
    processing_time_ms: float = Field(..., ge=0, description="Total wall-clock time, end to end.")


async def ingest_document(
    *,
    content: bytes,
    filename: str,
    document_id: str,
    session_id: Optional[str],
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    settings: RAGSettings,
) -> IngestionResult:
    """
    Extract, chunk, embed, and index a single uploaded document.

    Args:
        content: The raw uploaded file bytes.
        filename: The original filename (drives extractor dispatch and is
            stamped onto every resulting chunk for citation display).
        document_id: Identifier of the owning ``Document`` row. Every
            resulting chunk's ``chunk_id`` is derived from this
            (``f"{document_id}:{chunk_index}"``), so re-ingesting the same
            ``document_id`` (e.g. a retry) is an idempotent upsert rather
            than a duplicate insert.
        session_id: The research session this document belongs to.
        embedding_provider: Injected ``EmbeddingProvider``.
        vector_store: Injected ``VectorStore``.
        settings: Supplies ``chunk_size`` / ``chunk_overlap`` for chunking.

    Returns:
        Stats about the completed run.

    Raises:
        UnsupportedFileTypeError: ``filename``'s extension isn't supported.
        DocumentExtractionError: the file is corrupt or has no extractable
            text. Nothing is written to the vector store.
        DocumentChunkingError: chunking failed. Nothing is written to the
            vector store.
        EmbeddingProviderError: embedding failed. Nothing is written to
            the vector store.
        VectorStoreError: the upsert failed. This module makes a
            best-effort attempt to delete any partially-written chunks for
            ``document_id`` before re-raising -- a *compensating* action,
            not a true rollback, since ChromaDB has no multi-item
            transaction to roll back automatically. If the compensating
            delete itself also fails, that is logged at ``critical`` (this
            document may now be partially indexed and needs manual
            reconciliation) but the original ``VectorStoreError`` is still
            what's raised -- a cleanup failure must never masquerade as
            ingestion success, nor silently replace the real error.
    """
    with measure_latency_ms() as total_elapsed:
        with measure_latency_ms() as extraction_elapsed:
            # Parsing PDF/DOCX bytes is the one genuinely CPU-bound,
            # potentially-slow blocking call in this pipeline before
            # embedding -- offloaded for the same reason
            # SentenceTransformerEmbeddingProvider offloads model.encode().
            segments = await asyncio.to_thread(extract_segments, filename=filename, content=content)
        extraction_time_ms = extraction_elapsed()

        with measure_latency_ms() as chunking_elapsed:
            # Chunking is cheap, in-memory string splitting over text
            # that's already fully loaded -- not worth a thread-hop.
            chunks = chunk_segments(
                segments=segments,
                document_id=document_id,
                session_id=session_id,
                source_filename=filename,
                settings=settings,
            )
        chunking_time_ms = chunking_elapsed()

        with measure_latency_ms() as embedding_elapsed:
            embeddings: List[List[float]] = await embedding_provider.embed_documents([chunk.text for chunk in chunks])
        embedding_time_ms = embedding_elapsed()

        with measure_latency_ms() as upsert_elapsed:
            try:
                await vector_store.upsert(chunks, embeddings)
            except Exception as exc:
                logger.error(
                    "Vector store upsert failed during ingestion; attempting compensating delete",
                    extra={"document_id": document_id, "chunk_count": len(chunks)},
                )
                try:
                    await vector_store.delete(document_id)
                except Exception:
                    logger.critical(
                        "Compensating delete after failed upsert ALSO failed -- "
                        "document may be partially indexed and needs manual reconciliation",
                        extra={"document_id": document_id},
                    )
                if isinstance(exc, VectorStoreError):
                    raise
                raise VectorStoreError.wrap(exc, f"Failed to upsert chunks for document {document_id!r}.") from exc
        upsert_time_ms = upsert_elapsed()

    extracted_text_length = sum(len(segment.text) for segment in segments)
    average_chunk_size = sum(len(chunk.text) for chunk in chunks) / len(chunks)

    result = IngestionResult(
        document_id=document_id,
        chunk_count=len(chunks),
        extracted_text_length=extracted_text_length,
        average_chunk_size=average_chunk_size,
        embedding_model=embedding_provider.model_name,
        embedding_dimension=embedding_provider.dimension,
        extraction_time_ms=extraction_time_ms,
        chunking_time_ms=chunking_time_ms,
        embedding_time_ms=embedding_time_ms,
        upsert_time_ms=upsert_time_ms,
        processing_time_ms=total_elapsed(),
    )
    logger.info(
        "Document ingested",
        extra={
            "document_id": document_id,
            "chunk_count": result.chunk_count,
            "processing_time_ms": round(result.processing_time_ms, 2),
        },
    )
    return result
