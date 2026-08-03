"""
The Memory subsystem: loading a session's recent conversation history and
compacting older turns into a running summary when they no longer fit in
a bounded context window.

Distinct from RAG (``app.rag``): RAG retrieves relevant excerpts from
*uploaded documents*; Memory retrieves the *conversation's own prior
turns*. Both are session-scoped, but along different axes -- document
content versus turn history -- and neither subsystem depends on the
other.

``MemoryService`` (``app.services.memory_service``) is the sole
database-aware entry point into this subsystem, following the same
no-ABC-in-front-of-it precedent already established for
``HistoryService``: there is exactly one persistence technology in this
project, so a swappable-backend interface here would be an abstraction
with nothing to abstract over. The types in this package
(``app.memory.types``) and the pure compaction logic
(``app.memory.compaction``) have no database or FastAPI import anywhere,
mirroring ``app.rag.ingest``'s own framework-agnostic design.
"""

from __future__ import annotations
