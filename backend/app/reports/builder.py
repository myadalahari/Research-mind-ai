"""
Assembles a ``ReportDocument`` (``app.core.interfaces.report_exporter``)
from the Writer agent's REPORT-mode Markdown output.

Pure and framework-agnostic (see this package's own docstring) -- takes
plain strings/citations in, returns a ``ReportDocument`` out, no I/O, no
LLM calls, no database access. ``app.services.report_service.ReportService``
is the only caller, and owns everything this function doesn't: resolving
the report's title, running the underlying chat/research request, writing
files, and persisting a ``Report`` row.

Parsing strategy, matching exactly what ``app.agents.writer``'s own
``_REPORT_MODE_ADDENDUM`` asks the model to produce ("a brief executive
summary; one or more detailed findings sections (using '## ' headings)...;
and a concluding section"), not a general-purpose Markdown parser:

* Everything before the first ``## `` heading is the executive summary.
  Nothing in the Writer's prompt asks for that summary to itself be under
  a heading (only "findings sections" are told to use ``## ``), so no
  heading-stripping is needed for it -- except a single leading ``# ``
  (H1) line, which some models prepend as a document title despite never
  being asked to; that line is discarded here rather than left embedded
  in the summary prose, since the report's actual title is resolved by
  ``ReportService`` (from ``conversation_title`` or the query) and passed
  in explicitly, never inferred from the body text.
* Each ``## `` heading starts a new ``ReportSection``. No recursive
  ``### ``-level parsing into ``ReportSection.subsections`` -- the
  Writer's prompt never asks for sub-headings, so building that structure
  would be speculative; any deeper heading levels a model produces anyway
  simply stay as plain text inside their enclosing section's content.
* The LAST ``## `` block is always treated as the conclusion (its content
  becomes ``ReportDocument.conclusion``, a plain string -- its own heading
  text is discarded, matching that field having nowhere to store one).
  This is deterministic regardless of how the model titled that final
  heading ("Conclusion", "Summary", anything) -- matching the prompt's
  documented structure ("...; and a concluding section") literally: the
  last structural element is always the conclusion. If the model produces
  only one ``## `` block total, that block becomes the conclusion and
  ``sections`` is empty -- a degenerate but valid case, not an error.
* ``key_findings`` is always left empty. The Writer's prompt never asks
  for a findings bullet list distinct from the sections themselves;
  populating it would require either a second LLM call or a fragile
  heuristic extraction from prose, and nothing downstream depends on it
  yet -- not justified for this phase.
* GFM pipe-table blocks (`` | a | b | `` header, `` | --- | --- | ``
  delimiter, one or more `` | x | y | `` data rows) are detected inside
  each section's content (not the executive summary or conclusion, which
  are plain strings with no ``tables`` field to place them in) and
  extracted into structured ``ReportTable`` objects, removed from the
  prose ``content`` so they aren't rendered twice. This is intentionally
  bounded to the fully-piped form (leading and trailing ``|`` on every
  row) -- what a model asked to "use Markdown tables" almost always
  produces -- not a general GFM parser. A table-like block that doesn't
  match simply stays as plain text in ``content``; this is graceful
  degradation, never a parsing error.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.interfaces.report_exporter import Citation as ReportCitation
from app.core.interfaces.report_exporter import ReportDocument, ReportSection, ReportTable
from app.schemas.common import Citation as SchemaCitation

_H2_HEADING_RE = re.compile(r"(?m)^##[ \t]+(.+?)[ \t]*$")
_TABLE_DELIMITER_CELL_RE = re.compile(r"^:?-{2,}:?$")


def build_report_document(
    *,
    title: str,
    session_id: str,
    markdown: str,
    citations: Sequence[SchemaCitation] = (),
    metadata: Optional[Dict[str, Any]] = None,
) -> ReportDocument:
    """
    Build a ``ReportDocument`` from the Writer's REPORT-mode Markdown
    answer.

    Args:
        title: Already-resolved report title (``ReportService``'s
            responsibility -- see this module's own docstring for why the
            builder doesn't infer one from the body text).
        session_id: The research session this report belongs to.
        markdown: The Writer's ``draft_answer`` (via ``ChatResponse.answer``
            for a REPORT-mode request) -- a single Markdown string.
        citations: The sources backing the answer, in the API/agent-layer
            ``schemas.common.Citation`` shape -- narrowed onto this
            package's own, smaller ``report_exporter.Citation`` (renderer
            input, not an API contract) for each entry.
        metadata: Arbitrary extra metadata to attach verbatim (e.g. the
            originating query, an execution trace id) -- opaque to this
            function, whatever the caller wants recorded.

    Returns:
        A fully-populated ``ReportDocument``, ready for any
        ``ReportExporter``.
    """
    preamble, blocks = _split_into_h2_blocks(markdown)
    executive_summary = _strip_leading_h1(preamble)

    sections: List[ReportSection] = []
    conclusion = ""
    if blocks:
        *finding_blocks, (_, conclusion_content) = blocks
        sections = [_build_section(heading, content) for heading, content in finding_blocks]
        conclusion = conclusion_content.strip()
    elif not executive_summary:
        # No '## ' headings anywhere AND nothing left after stripping a
        # leading H1 -- fall back to the whole (unstripped) markdown
        # rather than producing an empty summary for a genuinely
        # non-empty answer.
        executive_summary = markdown.strip()

    return ReportDocument(
        title=title,
        session_id=session_id,
        executive_summary=executive_summary,
        key_findings=[],
        sections=sections,
        conclusion=conclusion,
        citations=[_to_report_citation(c) for c in citations],
        metadata=dict(metadata) if metadata else {},
    )


# =============================================================================
# Heading splitting
# =============================================================================


def _split_into_h2_blocks(markdown: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Split ``markdown`` on top-level (``## ``) headings.

    Returns ``(preamble, blocks)`` where ``preamble`` is everything before
    the first ``## `` heading (stripped), and ``blocks`` is an ordered
    list of ``(heading_text, content)`` pairs, each stripped. Returns
    ``(markdown.strip(), [])`` when no ``## `` heading exists at all.
    """
    matches = list(_H2_HEADING_RE.finditer(markdown))
    if not matches:
        return markdown.strip(), []

    preamble = markdown[: matches[0].start()].strip()
    blocks: List[Tuple[str, str]] = []
    for index, match in enumerate(matches):
        heading = match.group(1).strip()
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        blocks.append((heading, markdown[content_start:content_end].strip()))
    return preamble, blocks


def _strip_leading_h1(text: str) -> str:
    """Discard a single leading '# heading' line, if the text starts with one."""
    first_line, _, rest = text.partition("\n")
    stripped_first = first_line.strip()
    if stripped_first.startswith("# ") or stripped_first == "#":
        return rest.strip()
    return text


def _build_section(heading: str, raw_content: str) -> ReportSection:
    content, tables = _extract_tables(raw_content)
    return ReportSection(heading=heading, content=content, tables=tables, subsections=[])


# =============================================================================
# GFM pipe-table extraction
# =============================================================================


def _extract_tables(content: str) -> Tuple[str, List[ReportTable]]:
    """
    Pull fully-piped GFM table blocks out of ``content``, returning the
    remaining prose (tables removed) and the extracted ``ReportTable``\\ s,
    in the order they appeared.
    """
    lines = content.split("\n")
    tables: List[ReportTable] = []
    output_lines: List[str] = []

    index = 0
    while index < len(lines):
        if _is_table_row(lines[index]) and index + 1 < len(lines) and _is_table_delimiter_row(lines[index + 1]):
            header = _parse_row(lines[index])
            row_index = index + 2
            rows: List[List[str]] = []
            while row_index < len(lines) and _is_table_row(lines[row_index]):
                rows.append(_parse_row(lines[row_index]))
                row_index += 1
            tables.append(ReportTable(headers=header, rows=rows))
            index = row_index
            continue
        output_lines.append(lines[index])
        index += 1

    return "\n".join(output_lines).strip(), tables


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_table_delimiter_row(line: str) -> bool:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False
    cells = [cell.strip() for cell in stripped[1:-1].split("|")]
    return bool(cells) and all(_TABLE_DELIMITER_CELL_RE.match(cell) for cell in cells)


def _parse_row(line: str) -> List[str]:
    stripped = line.strip()[1:-1]  # drop the leading/trailing '|'
    return [cell.strip() for cell in stripped.split("|")]


# =============================================================================
# Citation mapping
# =============================================================================


def _to_report_citation(citation: SchemaCitation) -> ReportCitation:
    return ReportCitation(
        citation_id=citation.citation_id,
        source_type=citation.source_type.value,
        title=citation.title,
        url=citation.url,
        source_filename=citation.source_filename,
        page_number=citation.page_number,
        accessed_at=citation.accessed_at,
    )
