"""
Document text extraction: raw uploaded bytes -> ``ExtractedSegment`` list.

Segment-based (not "one big string") extraction is a deliberate design
choice: chunking (``app.rag.chunking``, the next file in this pipeline)
needs to know *where* text came from -- which PDF page, which DOCX/Markdown
section -- so every chunk it produces can carry accurate citation
provenance (``DocumentChunk.page_number`` / ``section_title``). Flattening
extraction into a single string up front would throw that provenance away
before chunking ever sees it.

Dispatch is by file extension (``.pdf``, ``.docx``, ``.txt``, ``.md``), not
a pluggable extractor-strategy interface: there is exactly one way to parse
each of these formats in this project, nothing about "which library reads
a PDF" is configurable at runtime, and there is no second implementation
ever planned to swap in. That is different from ``EmbeddingProvider`` /
``VectorStore`` (``core/interfaces``), which exist specifically because
those *do* have real, currently-relevant alternative implementations
(hosted embedding APIs, hosted vector databases). An ABC here would be an
abstraction with nothing to abstract over.

Deliberately NOT this module's job: enforcing ``RAGSettings.max_upload_size_mb``
or ``RAGSettings.allowed_upload_extensions``. Those are upload-time policy
checks on raw bytes/filename that the future ``IngestionService`` applies
*before* ever calling ``extract_segments`` -- the same "pre-check vs.
constraint" split already established for ``DocumentRepository.get_by_checksum``
(ADR-020). This module's own extension dispatch is the structural backstop
(``UnsupportedFileTypeError``) if that pre-check is ever missing or drifts
out of sync with what's actually implemented here.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import List, Optional

import docx
import pypdf
from pydantic import BaseModel, Field

from app.core.exceptions import DocumentExtractionError, UnsupportedFileTypeError
from app.core.logging import get_logger, measure_latency_ms

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt", ".md"})

# Matches a Markdown ATX heading line, e.g. "## Introduction" -> "Introduction".
# Only ATX-style (leading '#') headings are recognized; Setext-style
# ("Title\n=====") headings are treated as plain body text. Markdown
# documents in this project's target use case (research notes, README-style
# uploads) overwhelmingly use ATX headings, and misdetecting a Setext
# heading only means slightly coarser section boundaries, not lost content.
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


class ExtractedSegment(BaseModel):
    """
    A contiguous, citeable piece of a source document's extracted text.

    Local to the extraction/chunking stage of the RAG pipeline -- not a
    ``core.interfaces`` type, since it never crosses an interface boundary
    (``app.rag.chunking`` consumes it directly, in-process).
    """

    text: str = Field(..., description="The segment's extracted text, never blank/whitespace-only.")
    segment_index: int = Field(..., description="Position of this segment within the document, 0-based.")
    page_number: Optional[int] = Field(
        default=None, description="1-based source page number, for formats with pages (PDF only)."
    )
    section_title: Optional[str] = Field(
        default=None, description="Nearest heading this segment falls under, if the format has headings."
    )


def extract_segments(*, filename: str, content: bytes) -> List[ExtractedSegment]:
    """
    Extract citeable text segments from an uploaded document's raw bytes.

    Args:
        filename: The original filename (only its extension is used, to
            select the extractor -- the content itself is never assumed to
            live on disk).
        content: The raw file bytes.

    Returns:
        Extracted segments in document order, ``segment_index`` starting
        at 0. Never empty.

    Raises:
        UnsupportedFileTypeError: ``filename``'s extension isn't one this
            module knows how to parse.
        DocumentExtractionError: the file is corrupt/unreadable, or parsed
            cleanly but yielded no extractable text (e.g. a scanned,
            image-only PDF with no text layer -- OCR is out of scope for
            this project).
    """
    extension = Path(filename).suffix.lower()
    if extension not in _SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type {extension!r} for {filename!r}. "
            f"Supported types: {sorted(_SUPPORTED_EXTENSIONS)}."
        )

    with measure_latency_ms() as elapsed:
        if extension == ".pdf":
            segments = _extract_pdf(content, filename)
        elif extension == ".docx":
            segments = _extract_docx(content, filename)
        elif extension == ".md":
            segments = _extract_markdown(content, filename)
        else:
            segments = _extract_text(content, filename)

    if not segments:
        raise DocumentExtractionError(
            f"No extractable text found in {filename!r}. If this is a scanned or "
            f"image-only PDF, OCR is required and is not supported."
        )

    logger.debug(
        "Extracted document segments",
        # NOT "filename" -- Python's stdlib `logging` reserves that key as
        # one of LogRecord's own attributes (the source file of the
        # logging call itself); passing it via `extra` raises
        # `KeyError: Attempt to overwrite 'filename' in LogRecord` on
        # every real call to this function (found while fixing the
        # identical bug in app/api/routes/report.py).
        extra={
            "document_filename": filename,
            "extension": extension,
            "segment_count": len(segments),
            "latency_ms": round(elapsed(), 2),
        },
    )
    return segments


# =============================================================================
# Per-format extractors
# =============================================================================


def _extract_pdf(content: bytes, filename: str) -> List[ExtractedSegment]:
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            # An empty-password attempt covers PDFs that are "encrypted"
            # only in the sense of having owner-password restrictions with
            # no user password set (common from some export tools) --
            # pypdf's convention for that case. A PDF with a real user
            # password remains unreadable and falls through to the
            # DocumentExtractionError below via an empty page_texts list.
            reader.decrypt("")
        page_texts = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        raise DocumentExtractionError.wrap(exc, f"Failed to parse PDF {filename!r}.") from exc

    segments: List[ExtractedSegment] = []
    for page_number, text in enumerate(page_texts, start=1):
        stripped = text.strip()
        if not stripped:
            continue
        segments.append(
            ExtractedSegment(text=stripped, segment_index=len(segments), page_number=page_number, section_title=None)
        )
    return segments


def _extract_docx(content: bytes, filename: str) -> List[ExtractedSegment]:
    try:
        document = docx.Document(io.BytesIO(content))
    except Exception as exc:
        raise DocumentExtractionError.wrap(exc, f"Failed to parse DOCX {filename!r}.") from exc

    segments: List[ExtractedSegment] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        joined = "\n".join(current_lines).strip()
        if joined:
            segments.append(
                ExtractedSegment(
                    text=joined, segment_index=len(segments), page_number=None, section_title=current_title
                )
            )

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = paragraph.style.name if paragraph.style is not None else ""
        is_heading = style_name == "Title" or style_name.startswith("Heading")
        if is_heading:
            flush()
            current_title = text
            current_lines = []
        else:
            current_lines.append(text)
    flush()
    return segments


def _extract_markdown(content: bytes, filename: str) -> List[ExtractedSegment]:
    text = _decode_text(content, filename)

    segments: List[ExtractedSegment] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        joined = "\n".join(current_lines).strip()
        if joined:
            segments.append(
                ExtractedSegment(
                    text=joined, segment_index=len(segments), page_number=None, section_title=current_title
                )
            )

    for line in text.splitlines():
        heading_match = _MD_HEADING_RE.match(line)
        if heading_match:
            flush()
            current_title = heading_match.group(2)
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return segments


def _extract_text(content: bytes, filename: str) -> List[ExtractedSegment]:
    text = _decode_text(content, filename).strip()
    if not text:
        return []
    return [ExtractedSegment(text=text, segment_index=0, page_number=None, section_title=None)]


def _decode_text(content: bytes, filename: str) -> str:
    """
    Decode raw bytes as text.

    UTF-8 (the overwhelmingly common case for uploaded .txt/.md files) is
    tried first; Latin-1 is the fallback because it can decode *any* byte
    sequence (0x00-0xFF all map to valid code points), so it never itself
    raises -- it exists here purely to avoid rejecting a real but
    non-UTF-8-encoded text file outright, not because it's the "correct"
    encoding to assume.
    """
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        # latin-1 maps every byte value 0x00-0xFF to a valid code point, so
        # this branch cannot itself raise -- it exists to avoid rejecting a
        # real but non-UTF-8-encoded text file, not because it's assumed to
        # be the "correct" encoding.
        return content.decode("latin-1")
