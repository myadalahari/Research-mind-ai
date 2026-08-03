"""
RAG pipeline: document ingestion (extraction, chunking, embedding, vector
storage) and retrieval, plus the concrete ``EmbeddingProvider`` and
``VectorStore`` implementations selected for this project.

See ``app.core.interfaces.embedding_provider`` and
``app.core.interfaces.vector_store`` for why concrete implementations live
here rather than behind a separate top-level adapters package, and the
Architecture Decision Log for the Sentence Transformers / ChromaDB
technology choices.
"""

from __future__ import annotations
