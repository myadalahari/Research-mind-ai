"""
EmbeddingProvider interface.

The RAG ingestion pipeline (``app.rag.ingest``) and the Retriever agent
depend on this abstraction, not on the ``sentence-transformers`` package
directly. ``SentenceTransformerEmbeddingProvider`` is the concrete
implementation selected today (local, no external API dependency, keeps
the whole pipeline self-hostable). The interface is designed so a hosted
embedding API (OpenAI embeddings, Cohere, Voyage) could replace it later
without touching ``app.rag.ingest``, ``app.rag.retriever``, or
``ChromaVectorStore``.

Methods are declared ``async`` even though local sentence-transformer
inference is CPU/GPU-bound rather than I/O-bound: this keeps the contract
uniform with every other interface in the system and lets a concrete
implementation offload the blocking call to a thread pool
(``asyncio.to_thread``) internally without ever changing the interface a
caller depends on — important because a future hosted-API adapter *would*
be genuinely I/O-bound and needs the same async signature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """
    Abstract interface for turning text into dense vector embeddings.

    Concrete implementations live under ``app.rag`` (e.g.
    ``SentenceTransformerEmbeddingProvider``). Injected via
    ``app.core.dependencies.get_embedding_provider``.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Identifier of the concrete embedding model in use (e.g.
        ``"all-MiniLM-L6-v2"``).

        Needed by ``app.rag.ingest.ingest_document`` to populate
        ``IngestionResult.embedding_model`` -- the same field, in turn,
        that the future ``IngestionService`` writes onto
        ``Document.embedding_model`` so a stored document's row records
        which model actually produced its vectors (relevant if the
        configured model ever changes between ingestion runs).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        The fixed length of vectors this provider produces.

        Exposed so ``ChromaVectorStore`` (and any other ``VectorStore``
        implementation) can validate or configure its collection
        dimensionality without hardcoding a model-specific constant.
        """
        raise NotImplementedError

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of document chunks for storage.

        Args:
            texts: Chunk texts to embed, in order.

        Returns:
            One embedding vector per input text, in the same order,
            each of length ``self.dimension``.

        Raises:
            EmbeddingProviderError: on an embedding-layer failure.
        """
        raise NotImplementedError

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string for similarity search.

        Kept separate from ``embed_documents`` because some embedding
        models (including several Sentence Transformers models) use
        different prefixes/instructions for queries versus documents to
        improve retrieval quality — a distinction the interface preserves
        even though today's default model treats them identically.

        Raises:
            EmbeddingProviderError: on an embedding-layer failure.
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Return ``True`` if the embedding model is loaded and ready to
        serve requests, ``False`` otherwise. Never raises — used by
        ``GET /health``.
        """
        raise NotImplementedError
