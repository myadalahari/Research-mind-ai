"""
Shared filename-generation logic for ``ReportExporter`` implementations.

Extracted after ``MarkdownReportExporter`` and ``PDFReportExporter`` both
needed the identical slug/timestamp filename convention -- the same
extraction-on-second-real-consumer principle used for
``app.agents.memory_prompt.format_conversation_history`` (Phase 7):
implement the need once, and only pull it into a shared module once a
second, genuinely-identical consumer exists, rather than guessing at an
abstraction in advance.

``slugify()`` caps its output at ``_MAX_SLUG_BYTES`` UTF-8 bytes -- found
necessary while testing ``ReportService`` end-to-end against real disk
writes: ``ChatRequest.query`` allows up to 4000 characters, and an
all-word-character query used as a report title (no ``conversation_title``
to fall back to) slugified to hundreds of bytes with nothing to shorten
it, producing a filename past the ~255-byte limit most filesystems (ext4,
APFS, NTFS) enforce and raising a real ``OSError: File name too long`` on
write -- not a hypothetical, an actual failure reproduced during testing.
The cap truncates on a UTF-8 byte boundary (not a Python character count),
since a slug can contain multi-byte characters (e.g. CJK) preserved
verbatim per this function's own Unicode-preserving design -- truncating
by character count alone could still overflow a byte-based filesystem
limit.
"""

from __future__ import annotations

import re
from datetime import datetime

_SLUG_STRIP_RE = re.compile(r"[^\w]+", re.UNICODE)
_SLUG_COLLAPSE_RE = re.compile(r"-{2,}")

# Conservative cap on the slug portion of a filename, in UTF-8 bytes.
# Leaves ample room (a filename otherwise adds ~20 bytes for the
# `-YYYYMMDD-HHMMSS.ext` suffix) under the ~255-byte filename limit common
# to ext4/APFS/NTFS, without needing to know the exact suffix length here.
_MAX_SLUG_BYTES = 80


def slugify(title: str) -> str:
    """
    Turn ``title`` into a filesystem- and URL-safe slug for use in a
    downloaded filename.

    Preserves Unicode word characters (letters and digits in *any*
    script -- ``\\w`` under Python's default Unicode-aware ``re`` mode,
    not just ASCII) rather than transliterating or stripping non-Latin
    text. A title like "研究报告" slugifies to "研究报告" (readable, and still a
    valid filename on every filesystem this application targets), not to
    an empty string. Only whitespace and punctuation become hyphens.
    Falls back to the literal string ``"report"`` only when nothing
    word-like survives at all (e.g. a title that's entirely punctuation
    or emoji), so a filename is never empty. Truncated to
    ``_MAX_SLUG_BYTES`` UTF-8 bytes (see this module's own docstring) so a
    long title can never produce a filename too long for the filesystem.
    """
    lowered = title.strip().lower()
    hyphenated = _SLUG_STRIP_RE.sub("-", lowered)
    collapsed = _SLUG_COLLAPSE_RE.sub("-", hyphenated).strip("-")
    slug = collapsed or "report"
    return _truncate_utf8(slug, _MAX_SLUG_BYTES) or "report"


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """
    Truncate ``text`` to at most ``max_bytes`` UTF-8 bytes without
    splitting a multi-byte character, then strip any trailing hyphen left
    dangling by the cut.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    # errors="ignore" drops a multi-byte sequence truncated mid-character
    # at the cut point, rather than raising or leaving invalid bytes.
    return encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip("-")


def build_export_filename(title: str, generated_at: datetime, extension: str) -> str:
    """
    Build the downloadable filename for an exported report:
    ``{slug(title)}-{generated_at:%Y%m%d-%H%M%S}.{extension}``.

    Shared by every ``ReportExporter`` implementation so a report exported
    in multiple formats gets consistently-named files that differ only by
    extension.
    """
    return f"{slugify(title)}-{generated_at.strftime('%Y%m%d-%H%M%S')}.{extension}"
