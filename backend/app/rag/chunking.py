"""
Chunking: ``ExtractedSegment`` list -> ``DocumentChunk`` list, ready for
embedding and vector-store upsert.

Chunk boundaries never cross a segment boundary. A segment is exactly one
citeable unit (one PDF page, one DOCX/Markdown heading-delimited section --
see ``app.rag.extraction``), so splitting strictly within each segment
guarantees every resulting chunk maps to exactly one ``page_number`` /
``section_title`` unambiguously. The alternative -- packing words from
consecutive segments into a single chunk to hit ``chunk_size`` more
tightly -- would produce chunks whose citation is genuinely ambiguous
("this chunk spans pages 3 and 4, which do we cite?"), which is a worse
trade than occasionally emitting a chunk shorter than ``chunk_size`` (e.g.
a short page/section produces one small chunk on its own). Citation
accuracy was already established as the reason extraction is
segment-based; this module just carries that reasoning through.

Single, non-pluggable splitting strategy (word-based greedy packing with
character overlap) -- consistent with the project's "no speculative
abstractions" principle: there is exactly one chunking algorithm in this
project today, nothing about it is runtime-configurable beyond
``RAGSettings.chunk_size`` / ``chunk_overlap``, and no second strategy is
currently planned.
"""

from __future__ import annotations

from typing import List, Optional

from app.core.config import RAGSettings
from app.core.exceptions import DocumentChunkingError
from app.core.interfaces.vector_store import DocumentChunk
from app.core.logging import get_logger, measure_latency_ms
from app.rag.extraction import ExtractedSegment

logger = get_logger(__name__)


def chunk_segments(
    *,
    segments: List[ExtractedSegment],
    document_id: str,
    session_id: Optional[str],
    source_filename: str,
    settings: RAGSettings,
) -> List[DocumentChunk]:
    """
    Split extracted segments into ``DocumentChunk``s ready for embedding.

    Args:
        segments: Segments produced by ``app.rag.extraction.extract_segments``,
            in document order.
        document_id: Identifier of the source ``Document`` row -- becomes
            every resulting chunk's ``document_id`` and the ``chunk_id``
            prefix (``f"{document_id}:{chunk_index}"``), so re-chunking an
            updated document is an idempotent upsert (see
            ``VectorStore.upsert``'s contract).
        session_id: The research session this document belongs to, carried
            straight onto every chunk for session-scoped retrieval.
        source_filename: The original filename, carried onto every chunk
            for citation display.
        settings: Supplies ``chunk_size`` / ``chunk_overlap``. Injected
            rather than imported as a global singleton, consistent with
            every other adapter in this codebase.

    Returns:
        Chunks in document order, ``chunk_index`` continuous across the
        whole document (not reset per segment). Never empty.

    Raises:
        DocumentChunkingError: ``settings.chunk_size`` is non-positive (the
            splitting loop cannot make progress), ``segments`` is empty, or
            splitting produced no chunks.
    """
    if settings.chunk_size <= 0:
        raise DocumentChunkingError(f"RAG__CHUNK_SIZE must be positive, got {settings.chunk_size}.")
    if not segments:
        raise DocumentChunkingError(f"No segments to chunk for document {document_id!r}.")

    with measure_latency_ms() as elapsed:
        chunks: List[DocumentChunk] = []
        for segment in segments:
            for text in _split_text(segment.text, settings.chunk_size, settings.chunk_overlap):
                chunk_index = len(chunks)
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}:{chunk_index}",
                        document_id=document_id,
                        session_id=session_id,
                        text=text,
                        chunk_index=chunk_index,
                        source_filename=source_filename,
                        page_number=segment.page_number,
                        section_title=segment.section_title,
                        extra_metadata={"segment_index": segment.segment_index},
                    )
                )

    if not chunks:
        raise DocumentChunkingError(f"Chunking produced no chunks for document {document_id!r}.")

    logger.debug(
        "Chunked document",
        extra={
            "document_id": document_id,
            "segment_count": len(segments),
            "chunk_count": len(chunks),
            "latency_ms": round(elapsed(), 2),
        },
    )
    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Greedily pack whitespace-delimited words into chunks of at most
    ``chunk_size`` characters (a single word longer than ``chunk_size`` is
    still emitted whole, as its own chunk, rather than being cut mid-word
    or dropped), then step the next chunk's start backward by roughly
    ``chunk_overlap`` characters' worth of trailing words so consecutive
    chunks overlap.

    Guaranteed to make forward progress every iteration -- the overlap
    step never rewinds all the way back to the current chunk's own start,
    which combined with ``RAGSettings``'s ``chunk_overlap < chunk_size``
    invariant is what rules out an infinite loop.
    """
    words = text.split()
    if not words:
        return []

    n = len(words)
    chunks: List[str] = []
    start = 0

    while start < n:
        current_words: List[str] = []
        current_len = 0
        end = start
        while end < n:
            word = words[end]
            added_len = len(word) + (1 if current_words else 0)  # +1 for the joining space
            if current_words and current_len + added_len > chunk_size:
                break
            current_words.append(word)
            current_len += added_len
            end += 1

        if not current_words:
            # The single next word alone already exceeds chunk_size --
            # emit it whole rather than looping forever with zero progress.
            current_words = [words[end]]
            end += 1

        chunks.append(" ".join(current_words))

        if end >= n:
            break

        # Step back from `end` by ~chunk_overlap characters' worth of
        # words, but never past this chunk's own `start` -- that bound is
        # what guarantees the next iteration starts strictly after `start`.
        overlap_start = end
        overlap_len = 0
        while overlap_start > start and overlap_len < chunk_overlap:
            overlap_start -= 1
            overlap_len += len(words[overlap_start]) + 1

        start = overlap_start if overlap_start > start else end

    return chunks
