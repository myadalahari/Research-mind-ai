"""
VectorStore interface.

The RAG pipeline (``app.rag``) and the Retriever agent depend on this
abstraction, not on the ChromaDB client directly. ``ChromaVectorStore`` is
the concrete implementation selected today, chosen because it is embedded
and zero-infrastructure (fits the one-command Docker Compose goal) while
still being a real, production-used vector database. The interface is
designed so a hosted store (Pinecone, Weaviate, Qdrant, pgvector) could
replace it later without touching ``app.rag.ingest`` or the Retriever agent.

Every chunk stored here carries enough provenance metadata to reconstruct
an exact citation (source document, page/section, chunk index) — citations
are a first-class requirement, not something bolted on after the fact, so
that contract is embedded directly in ``DocumentChunk``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.utils.time import utc_now


class DocumentChunk(BaseModel):
    """
    A single chunk of a source document, ready to be embedded and stored.

    Produced by ``app.rag.chunker`` during ingestion. ``chunk_id`` must be
    stable and unique within a document (e.g. ``f"{document_id}:{chunk_index}"``)
    so re-ingesting the same document is an idempotent upsert rather than a
    duplicate insert.
    """

    chunk_id: str = Field(..., description="Stable, unique identifier for this chunk.")
    document_id: str = Field(..., description="Identifier of the source document this chunk belongs to.")
    session_id: Optional[str] = Field(
        default=None, description="Research session this document was uploaded into, if session-scoped."
    )
    text: str = Field(..., description="The chunk's raw text content.")
    chunk_index: int = Field(..., description="Position of this chunk within its source document, 0-based.")
    source_filename: str = Field(..., description="Original filename of the source document.")
    page_number: Optional[int] = Field(
        default=None, description="Page number this chunk originated from, if the source format has pages."
    )
    section_title: Optional[str] = Field(
        default=None, description="Nearest section/heading this chunk falls under, if extractable."
    )
    uploaded_at: datetime = Field(default_factory=utc_now)
    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-agnostic metadata (e.g. content hash, extractor version).",
    )


class RetrievedChunk(BaseModel):
    """A ``DocumentChunk`` returned from a similarity query, with its score."""

    chunk: DocumentChunk
    score: float = Field(..., description="Similarity score for this chunk against the query, higher is more relevant.")


class VectorStore(ABC):
    """
    Abstract interface for the vector store backing document retrieval.

    Concrete implementations live under ``app.rag`` (e.g.
    ``ChromaVectorStore``). The Retriever agent and ``IngestionService``
    depend on this interface, injected via
    ``app.core.dependencies.get_vector_store``, and must never import
    ``chromadb`` (or any other vector database client) directly.
    """

    @abstractmethod
    async def upsert(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """
        Insert or update chunks and their precomputed embeddings.

        ``chunks`` and ``embeddings`` must be the same length and
        positionally aligned (``embeddings[i]`` is the embedding for
        ``chunks[i]``). Upserting a ``chunk_id`` that already exists must
        replace it, making re-ingestion of an updated document idempotent.

        Raises:
            VectorStoreError: on a storage-layer failure.
        """
        raise NotImplementedError

    @abstractmethod
    async def query(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        session_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Return the ``top_k`` chunks most similar to ``query_embedding``.

        Args:
            query_embedding: The embedded query vector.
            top_k: Maximum number of chunks to return.
            score_threshold: If set, drop results below this similarity
                score rather than padding out to ``top_k`` with weak
                matches — retrieval quality matters more than hitting a
                fixed count.
            session_id: If set, restrict the search to chunks uploaded
                within this research session.
            document_ids: If set, restrict the search to chunks from these
                specific documents (used by "summarize the uploaded
                papers" style requests that target specific uploads).

        Returns:
            Chunks ordered by descending similarity score.

        Raises:
            VectorStoreError: on a storage-layer failure.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, document_id: str) -> None:
        """
        Remove all chunks belonging to ``document_id`` from the store.

        Raises:
            VectorStoreError: on a storage-layer failure.
        """
        raise NotImplementedError

    @abstractmethod
    async def count(self, session_id: Optional[str] = None) -> int:
        """
        Return the number of chunks currently stored, optionally scoped to
        a single session. Used by ``GET /sources`` and by
        ``RAGSettings.max_documents_per_session`` enforcement in
        ``IngestionService``.
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Return ``True`` if the vector store is reachable and ready,
        ``False`` otherwise. Never raises — used by ``GET /health``.
        """
        raise NotImplementedError
