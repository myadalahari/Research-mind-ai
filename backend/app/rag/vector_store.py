"""
``ChromaVectorStore`` -- the concrete ``VectorStore`` selected for this
project.

See ``app.core.interfaces.vector_store`` for the interface contract and the
reasoning behind choosing ChromaDB (embedded, zero-infrastructure) over a
hosted vector database.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import VectorStoreSettings
from app.core.exceptions import VectorStoreError
from app.core.interfaces.vector_store import DocumentChunk, RetrievedChunk, VectorStore
from app.core.logging import get_logger, measure_latency_ms
from app.utils.time import utc_now

logger = get_logger(__name__)

# Chroma silently drops any metadata key whose value is ``None`` rather than
# storing it (verified empirically -- it is not documented behavior worth
# relying on blindly, but it is what the installed chromadb==1.5.9 does), so
# optional ``DocumentChunk`` fields can be written directly into the
# metadata dict without hand-rolled ``if value is not None`` branches; on
# the read side, absent keys are simply defaulted with ``.get(...)``.
_EXTRA_METADATA_KEY = "extra_metadata_json"


class ChromaVectorStore(VectorStore):
    """
    Vector store backed by an embedded, on-disk ChromaDB collection.

    One ``chromadb.PersistentClient`` and one collection are created at
    construction and reused for the process's lifetime, mirroring
    ``SentenceTransformerEmbeddingProvider``. Constructed once by
    ``app.core.dependencies`` at application startup.

    The installed ``chromadb`` client is synchronous; every method here
    offloads its blocking calls via ``asyncio.to_thread`` for the same
    reason ``SentenceTransformerEmbeddingProvider`` does -- to honor the
    interface's async contract without blocking the event loop.

    Embeddings are always supplied by the caller (``app.rag.ingest`` calls
    ``EmbeddingProvider`` first, then passes the vectors here), so the
    collection is created with ``embedding_function=None``: Chroma must
    never be allowed to compute its own embeddings from raw text, which
    would silently bypass the configured ``EmbeddingProvider`` and could
    even attempt a network call to download a default embedding model.
    """

    def __init__(self, settings: VectorStoreSettings) -> None:
        self._settings = settings
        try:
            self._client = chromadb.PersistentClient(
                path=settings.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=settings.telemetry_enabled),
            )
            self._collection = self._client.get_or_create_collection(
                name=settings.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": settings.distance_metric},
            )
        except Exception as exc:
            raise VectorStoreError.wrap(
                exc,
                f"Failed to open ChromaDB collection {settings.collection_name!r} " f"at {settings.persist_dir!r}.",
            ) from exc

        logger.info(
            "Vector store opened",
            extra={
                "persist_dir": settings.persist_dir,
                "collection_name": settings.collection_name,
                "distance_metric": settings.distance_metric,
            },
        )

    async def upsert(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"chunks and embeddings must be the same length "
                f"(got {len(chunks)} chunks and {len(embeddings)} embeddings)."
            )

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [_chunk_to_metadata(chunk) for chunk in chunks]

        with measure_latency_ms() as elapsed:
            try:
                await asyncio.to_thread(
                    self._collection.upsert,
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
            except Exception as exc:
                raise VectorStoreError.wrap(exc, f"Failed to upsert {len(chunks)} chunk(s).") from exc
        logger.debug(
            "Upserted chunks into vector store",
            extra={"chunk_count": len(chunks), "latency_ms": round(elapsed(), 2)},
        )

    async def query(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        session_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        where = _build_where(session_id=session_id, document_ids=document_ids)

        with measure_latency_ms() as elapsed:
            try:
                result = await asyncio.to_thread(
                    self._collection.query,
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where,
                    include=["metadatas", "documents", "distances"],
                )
            except Exception as exc:
                raise VectorStoreError.wrap(exc, "Failed to query the vector store.") from exc

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        retrieved: List[RetrievedChunk] = []
        # strict=True: these four lists are Chroma's own parallel arrays
        # for a single query result and are contractually the same
        # length. If they were ever mismatched (an internal Chroma bug,
        # not a case this project's own code can cause), silently
        # truncating to the shortest via a bare zip() would return
        # fewer/misaligned chunks without any signal that something was
        # wrong -- strict=True fails loudly instead, which is strictly
        # preferable for a correctness-sensitive retrieval path.
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances, strict=True):
            # Chroma returns a *distance*, not a similarity score. This
            # conversion (score = 1 - distance) is only correct for cosine
            # distance -- the configured (and only supported today)
            # ``distance_metric``. A future non-cosine metric would need
            # this conversion revisited; not solved speculatively now since
            # ``distance_metric`` has never been anything but "cosine" in
            # this project.
            score = 1.0 - distance
            if score_threshold is not None and score < score_threshold:
                continue
            retrieved.append(RetrievedChunk(chunk=_metadata_to_chunk(chunk_id, text, metadata), score=score))

        logger.debug(
            "Queried vector store",
            extra={
                "top_k": top_k,
                "returned": len(retrieved),
                "session_id": session_id,
                "latency_ms": round(elapsed(), 2),
            },
        )
        return retrieved

    async def delete(self, document_id: str) -> None:
        try:
            await asyncio.to_thread(self._collection.delete, where={"document_id": {"$eq": document_id}})
        except Exception as exc:
            raise VectorStoreError.wrap(exc, f"Failed to delete chunks for document {document_id!r}.") from exc
        logger.debug("Deleted chunks for document", extra={"document_id": document_id})

    async def count(self, session_id: Optional[str] = None) -> int:
        try:
            if session_id is None:
                return await asyncio.to_thread(self._collection.count)
            # Collection.count() takes no filter, so a session-scoped count
            # goes through get() with include=[] -- no embeddings/documents
            # are fetched, only the matching ids, keeping this cheap.
            result = await asyncio.to_thread(
                self._collection.get, where={"session_id": {"$eq": session_id}}, include=[]
            )
            return len(result["ids"])
        except Exception as exc:
            raise VectorStoreError.wrap(exc, "Failed to count vector store chunks.") from exc

    async def health_check(self) -> bool:
        """
        Verify the underlying Chroma client is reachable via a real
        heartbeat call, not just that the client object is non-``None``.
        Never raises.
        """
        try:
            await asyncio.to_thread(self._client.heartbeat)
            return True
        except Exception:
            logger.exception("Vector store health check failed")
            return False


def _build_where(*, session_id: Optional[str], document_ids: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """Build a Chroma ``where`` filter from the optional query scoping args."""
    conditions: List[Dict[str, Any]] = []
    if session_id is not None:
        conditions.append({"session_id": {"$eq": session_id}})
    if document_ids:
        conditions.append({"document_id": {"$in": document_ids}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _chunk_to_metadata(chunk: DocumentChunk) -> Dict[str, Any]:
    return {
        "document_id": chunk.document_id,
        "session_id": chunk.session_id,
        "chunk_index": chunk.chunk_index,
        "source_filename": chunk.source_filename,
        "page_number": chunk.page_number,
        "section_title": chunk.section_title,
        "uploaded_at": chunk.uploaded_at.isoformat(),
        _EXTRA_METADATA_KEY: json.dumps(chunk.extra_metadata) if chunk.extra_metadata else None,
    }


def _metadata_to_chunk(chunk_id: str, text: str, metadata: Dict[str, Any]) -> DocumentChunk:
    uploaded_at_raw = metadata.get("uploaded_at")
    uploaded_at: datetime = datetime.fromisoformat(uploaded_at_raw) if uploaded_at_raw else utc_now()

    extra_metadata_raw = metadata.get(_EXTRA_METADATA_KEY)
    extra_metadata: Dict[str, Any] = json.loads(extra_metadata_raw) if extra_metadata_raw else {}

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=metadata["document_id"],
        session_id=metadata.get("session_id"),
        text=text,
        chunk_index=metadata["chunk_index"],
        source_filename=metadata["source_filename"],
        page_number=metadata.get("page_number"),
        section_title=metadata.get("section_title"),
        uploaded_at=uploaded_at,
        extra_metadata=extra_metadata,
    )
