"""
``PDFReportExporter`` -- renders a ``ReportDocument`` into a single PDF
document, backed by ReportLab's Platypus layer.

The second concrete ``ReportExporter`` implementation (see
``markdown_exporter.py`` for the first, and that interface's own docstring
for why ``ReportService`` depends on the abstraction rather than either
concrete class directly). Like the Markdown exporter, this is pure
rendering with no filesystem access -- writing the returned bytes to disk
(if at all) is the caller's job.

Built on ``reportlab.platypus`` (``SimpleDocTemplate`` + a flat list of
flowables), not the low-level ``Canvas`` API -- flowables compose
directly from ``ReportDocument``'s structure (a paragraph per summary/
section, a table flowable per ``ReportTable``, ...) the same way
``markdown_exporter.py`` builds a list of Markdown lines, rather than this
file manually tracking a cursor position and page breaks by hand.

Rendering order mirrors ``markdown_exporter.py`` exactly (title,
generated-at/session metadata line, executive summary, key findings, each
section recursively including subsections and tables, conclusion,
sources) -- the two exporters intentionally share no rendering code
beyond ``app.reports.naming`` (filenames), since a Markdown line list and
a ReportLab flowable list have nothing else in common, but the *order and
content* they produce should stay in lockstep so a report looks
equivalent regardless of which format a user downloads.

Failure handling -- two tiers, corresponding to two genuinely different
failure classes:

* ``ReportTemplateError`` (proactive, before any layout is attempted):
  raised if a ``ReportTable``'s rows are ragged (a row's length doesn't
  match its header count). ``ReportDocument``/``ReportTable`` are plain
  Pydantic models a caller could construct by hand with mismatched rows
  even though ``app.reports.builder``'s own GFM extraction never produces
  that -- this is "content did not satisfy the structural requirements of
  the export template" precisely as ``ReportTemplateError``'s own
  docstring describes, and failing fast with a clear message is much more
  useful than whatever confusing internal error ReportLab's table layout
  code would raise on a ragged input.
* ``ReportRenderingError`` (reactive, wrapping ``doc.build()``): ReportLab
  can raise ``reportlab.platypus.doctemplate.LayoutError`` when a single
  flowable is too large to fit on any page at all (e.g. one unbroken
  table row wider than the page). This is the one ``try``/``except`` in
  the file -- every other line (style setup, string formatting, flowable
  construction) has no plausible failure mode and gets no defensive
  handling, matching ``markdown_exporter.py``'s reasoning.

No page-number footer in this first version -- deliberately deferred as a
follow-up enhancement rather than a core requirement (a correct PDF
matters more than page numbers), per explicit scope discussion before
this file was written. ``SimpleDocTemplate`` still paginates automatically
as flowables overflow a page; only the "Page X of Y" footer is omitted.
"""

from __future__ import annotations

from io import BytesIO
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.doctemplate import LayoutError

from app.core.exceptions import ReportRenderingError, ReportTemplateError
from app.core.interfaces.report_exporter import (
    Citation,
    ExportResult,
    ReportDocument,
    ReportSection,
    ReportTable,
)
from app.core.interfaces.report_exporter import ReportExporter
from app.reports.naming import build_export_filename

_MAX_HEADING_LEVEL = 4  # caps at Heading4; deeper nesting reuses Heading4 rather than shrinking further

_STYLESHEET = getSampleStyleSheet()
_TITLE_STYLE = ParagraphStyle(
    "RM_Title",
    parent=_STYLESHEET["Title"],
    fontSize=22,
    spaceAfter=4,
)
_META_STYLE = ParagraphStyle(
    "RM_Meta",
    parent=_STYLESHEET["Italic"],
    fontSize=9,
    textColor=colors.grey,
    spaceAfter=16,
)
_BODY_STYLE = ParagraphStyle(
    "RM_Body",
    parent=_STYLESHEET["BodyText"],
    fontSize=11,
    spaceAfter=10,
    leading=15,
)
_CAPTION_STYLE = ParagraphStyle(
    "RM_Caption",
    parent=_STYLESHEET["Italic"],
    fontSize=9,
    spaceAfter=4,
)
_CITATION_STYLE = ParagraphStyle(
    "RM_Citation",
    parent=_STYLESHEET["BodyText"],
    fontSize=9,
    spaceAfter=6,
    leading=12,
)
_TABLE_CELL_STYLE = ParagraphStyle(
    "RM_TableCell",
    parent=_STYLESHEET["BodyText"],
    fontSize=9,
    leading=12,
)
_TABLE_HEADER_CELL_STYLE = ParagraphStyle(
    "RM_TableHeaderCell",
    parent=_TABLE_CELL_STYLE,
    fontName="Helvetica-Bold",
    textColor=colors.white,
)

_HEADING_STYLE_NAMES = {2: "Heading2", 3: "Heading3", 4: "Heading4"}


class PDFReportExporter(ReportExporter):
    """Renders a ``ReportDocument`` as a single ``.pdf`` file via ReportLab."""

    @property
    def format_name(self) -> str:
        return "pdf"

    async def export(self, report: ReportDocument) -> ExportResult:
        _validate_tables(report)

        flowables: List = []
        flowables.append(Paragraph(_escape(report.title), _TITLE_STYLE))
        flowables.append(
            Paragraph(
                f"Generated {report.generated_at.isoformat()} &middot; Session {_escape(report.session_id)}",
                _META_STYLE,
            )
        )

        if report.executive_summary:
            flowables.append(Paragraph(_escape(report.executive_summary), _BODY_STYLE))

        flowables.extend(_render_key_findings(report.key_findings))

        for section in report.sections:
            flowables.extend(_render_section(section, level=2))

        flowables.append(Paragraph("Conclusion", _STYLESHEET["Heading2"]))
        flowables.append(Paragraph(_escape(report.conclusion), _BODY_STYLE))

        flowables.extend(_render_sources(report.citations))

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title=report.title,
        )
        try:
            doc.build(flowables)
        except LayoutError as exc:
            raise ReportRenderingError.wrap(
                exc,
                f"Failed to lay out PDF for report {report.title!r} (session {report.session_id!r}).",
            ) from exc

        filename = build_export_filename(report.title, report.generated_at, "pdf")
        return ExportResult(content=buffer.getvalue(), mime_type="application/pdf", filename=filename)


def _validate_tables(report: ReportDocument) -> None:
    """
    Fail fast, before any layout is attempted, if any table in ``report``
    is ragged (a row whose length doesn't match its header count) -- see
    this module's own docstring for why that's a template error rather
    than something to let ReportLab's table layout choke on.
    """
    for section in report.sections:
        _validate_section_tables(section)


def _validate_section_tables(section: ReportSection) -> None:
    for table in section.tables:
        expected = len(table.headers)
        for row_index, row in enumerate(table.rows):
            if len(row) != expected:
                raise ReportTemplateError(
                    f"Table in section {section.heading!r} has {expected} header column(s) but "
                    f"row {row_index} has {len(row)} cell(s)."
                )
    for subsection in section.subsections:
        _validate_section_tables(subsection)


def _render_key_findings(key_findings: List[str]) -> List:
    if not key_findings:
        return []
    flowables: List = [Paragraph("Key Findings", _STYLESHEET["Heading2"])]
    items = [ListItem(Paragraph(_escape(finding), _BODY_STYLE)) for finding in key_findings]
    flowables.append(ListFlowable(items, bulletType="bullet"))
    flowables.append(Spacer(1, 8))
    return flowables


def _render_section(section: ReportSection, *, level: int) -> List:
    style_name = _HEADING_STYLE_NAMES.get(min(level, _MAX_HEADING_LEVEL), "Heading4")
    flowables: List = [Paragraph(_escape(section.heading), _STYLESHEET[style_name])]
    if section.content:
        flowables.append(Paragraph(_escape(section.content), _BODY_STYLE))
    for table in section.tables:
        flowables.extend(_render_table(table))
    for subsection in section.subsections:
        flowables.extend(_render_section(subsection, level=level + 1))
    return flowables


def _render_table(table: ReportTable) -> List:
    flowables: List = []
    if table.caption:
        flowables.append(Paragraph(_escape(table.caption), _CAPTION_STYLE))

    header_row = [Paragraph(_escape(cell), _TABLE_HEADER_CELL_STYLE) for cell in table.headers]
    data_rows = [[Paragraph(_escape(cell), _TABLE_CELL_STYLE) for cell in row] for row in table.rows]
    grid_data = [header_row, *data_rows]

    column_count = len(table.headers)
    available_width = A4[0] - 4 * cm
    col_widths = [available_width / column_count] * column_count if column_count else None

    grid = Table(grid_data, colWidths=col_widths, repeatRows=1)
    grid.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    flowables.append(grid)
    flowables.append(Spacer(1, 10))
    return flowables


def _render_sources(citations: List[Citation]) -> List:
    if not citations:
        return []
    flowables: List = [Paragraph("Sources", _STYLESHEET["Heading2"])]
    items = [ListItem(Paragraph(_render_citation(citation), _CITATION_STYLE)) for citation in citations]
    flowables.append(ListFlowable(items, bulletType="1"))
    return flowables


def _render_citation(citation: Citation) -> str:
    title = _escape(citation.title)
    if citation.source_type == "document":
        detail = _escape(citation.source_filename or "unknown document")
        if citation.page_number is not None:
            detail = f"{detail}, p. {citation.page_number}"
        return f"<b>{title}</b> ({detail})"

    if citation.url:
        link = f'<link href="{_escape(citation.url)}"><u>{_escape(citation.url)}</u></link>'
    else:
        link = "unknown source"
    if citation.accessed_at is not None:
        link = f"{link}, accessed {citation.accessed_at.date().isoformat()}"
    return f"<b>{title}</b> ({link})"


def _escape(text: str) -> str:
    """
    Escape text destined for a ReportLab ``Paragraph``, which interprets a
    small XML-like markup subset (``<b>``, ``<link>``, ...) in its input.
    Any ``&``/``<``/``>`` coming from report content itself (executive
    summary, section content, citation titles, etc.) must be escaped
    first so it's rendered as literal text rather than being misread as
    markup -- this is applied to every piece of user/model-generated text
    before it's placed in a ``Paragraph``, and markup this module adds
    itself (e.g. ``<b>``/``<link>`` in ``_render_citation``) is always
    wrapped *around* already-escaped text, never the reverse.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
