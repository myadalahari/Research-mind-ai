"""
``SentenceTransformerEmbeddingProvider`` -- the concrete ``EmbeddingProvider``
selected for this project.

See ``app.core.interfaces.embedding_provider`` for the interface contract
and the reasoning behind choosing Sentence Transformers (local,
self-hostable, no external API dependency) over a hosted embeddings API.
"""

from __future__ import annotations

import asyncio
from typing import List

from sentence_transformers import SentenceTransformer

from app.core.config import EmbeddingSettings
from app.core.exceptions import EmbeddingProviderError
from app.core.interfaces.embedding_provider import EmbeddingProvider
from app.core.logging import get_logger, measure_latency_ms

logger = get_logger(__name__)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider backed by a Sentence Transformers model.

    The model is loaded once at construction -- expensive (downloads/loads
    weights into memory) -- and reused for the process's lifetime.
    Constructed once by ``app.core.dependencies`` at application startup,
    mirroring every other adapter in this codebase (``Database``, and the
    LLM/search/vector-store adapters alongside it).

    ``SentenceTransformer.encode()`` is a blocking, CPU/GPU-bound call, not
    an I/O-bound one; every method here offloads it via
    ``asyncio.to_thread`` so the async interface contract (already
    established in ``core.interfaces.embedding_provider`` specifically to
    allow this) never blocks the event loop.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        try:
            self._model = SentenceTransformer(settings.model_name, device=settings.device)
        except Exception as exc:
            raise EmbeddingProviderError.wrap(
                exc,
                f"Failed to load embedding model {settings.model_name!r} on device {settings.device!r}.",
            ) from exc

        self._dimension: int = self._model.get_embedding_dimension()
        logger.info(
            "Embedding model loaded",
            extra={"model_name": settings.model_name, "device": settings.device, "dimension": self._dimension},
        )

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        with measure_latency_ms() as elapsed:
            try:
                vectors = await asyncio.to_thread(
                    self._model.encode,
                    texts,
                    batch_size=self._settings.batch_size,
                    normalize_embeddings=self._settings.normalize_embeddings,
                    show_progress_bar=False,
                )
            except Exception as exc:
                raise EmbeddingProviderError.wrap(exc, f"Failed to embed {len(texts)} document chunk(s).") from exc
        logger.debug("Embedded document chunks", extra={"chunk_count": len(texts), "latency_ms": round(elapsed(), 2)})
        return vectors.tolist()

    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.

        No query/document prefixing is applied: ``all-MiniLM-L6-v2`` is a
        symmetric sentence-embedding model with no distinct query
        instruction, unlike e.g. BGE/E5 models. This stays a separate
        method from ``embed_documents`` (rather than
        ``embed_documents([text])[0]``) purely to honor the interface's
        documented contract -- a future model that DOES need query-side
        prefixing changes only this method, not every caller.
        """
        try:
            vector = await asyncio.to_thread(
                self._model.encode,
                text,
                normalize_embeddings=self._settings.normalize_embeddings,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingProviderError.wrap(exc, "Failed to embed query text.") from exc
        return vector.tolist()

    async def health_check(self) -> bool:
        """
        Run a real, cheap inference call rather than just checking that
        the model object is non-``None`` -- consistent with how
        ``HealthService``'s other dependency checks verify actual
        reachability/functionality, not just object presence. Never
        raises.
        """
        try:
            vector = await self.embed_query("health check")
            return len(vector) == self._dimension
        except Exception:
            logger.exception("Embedding provider health check failed")
            return False
