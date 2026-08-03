"""
``MarkdownReportExporter`` -- renders a ``ReportDocument`` into a single
GFM-Markdown document.

The first concrete ``ReportExporter`` implementation (see that interface's
own docstring for why ``ReportService`` depends on the abstraction rather
than this class directly). Pure rendering: takes an already-assembled,
already-validated ``ReportDocument`` in, returns ``ExportResult`` bytes
out. No filesystem access -- writing the returned bytes to disk (if at
all) is the caller's job, which keeps this class trivially unit-testable
without a filesystem and free of any opinion about where reports get
stored (that's ``ReportSettings.output_dir`` / ``ReportService``'s
concern).

Rendering order mirrors ``ReportDocument``'s own field order: title,
a short generated-at/session metadata line, the executive summary as
plain prose (no heading, matching ``app.reports.builder``'s convention
that the summary isn't itself under a heading), key findings (only if
non-empty -- see ``_render_key_findings``), each section (recursively,
including subsections one heading level deeper each), the conclusion
under an explicit "## Conclusion" heading (the heading text
``app.reports.builder`` deliberately discards when parsing -- this is the
one place that heading text gets supplied back), and finally a numbered
sources list built from ``citations``.

No ``try``/``except`` anywhere in ``export()``. Rendering here is pure
string formatting over data that's already passed Pydantic validation --
there is no plausible failure mode (no template engine, no external I/O,
nothing that raises). ``ReportRenderingError`` exists in
``app.core.exceptions`` for exporters that genuinely can fail mid-render
(``app.reports.exporters.pdf_exporter``, where ReportLab layout can fail)
-- raising it here would be dead code with no real trigger.

Filename generation (slug + timestamp) lives in ``app.reports.naming``,
shared with ``PDFReportExporter`` -- extracted once a second exporter
needed the identical convention, rather than each exporter carrying its
own copy.
"""

from __future__ import annotations

from typing import List

from app.core.interfaces.report_exporter import (
    Citation,
    ExportResult,
    ReportDocument,
    ReportExporter,
    ReportSection,
    ReportTable,
)
from app.reports.naming import build_export_filename


class MarkdownReportExporter(ReportExporter):
    """Renders a ``ReportDocument`` as a single ``.md`` file."""

    @property
    def format_name(self) -> str:
        return "markdown"

    async def export(self, report: ReportDocument) -> ExportResult:
        lines: List[str] = [f"# {report.title}", ""]
        lines.append(f"*Generated {report.generated_at.isoformat()} · Session `{report.session_id}`*")
        lines.append("")

        if report.executive_summary:
            lines.append(report.executive_summary)
            lines.append("")

        lines.extend(_render_key_findings(report.key_findings))

        for section in report.sections:
            lines.extend(_render_section(section, level=2))

        lines.append("## Conclusion")
        lines.append("")
        lines.append(report.conclusion)
        lines.append("")

        lines.extend(_render_sources(report.citations))

        content = "\n".join(lines).strip() + "\n"
        filename = build_export_filename(report.title, report.generated_at, "md")

        return ExportResult(
            content=content.encode("utf-8"),
            mime_type="text/markdown",
            filename=filename,
        )


def _render_key_findings(key_findings: List[str]) -> List[str]:
    if not key_findings:
        return []
    lines = ["## Key Findings", ""]
    lines.extend(f"- {finding}" for finding in key_findings)
    lines.append("")
    return lines


def _render_section(section: ReportSection, *, level: int) -> List[str]:
    heading_marker = "#" * level
    lines = [f"{heading_marker} {section.heading}", ""]
    if section.content:
        lines.append(section.content)
        lines.append("")
    for table in section.tables:
        lines.extend(_render_table(table))
    for subsection in section.subsections:
        lines.extend(_render_section(subsection, level=level + 1))
    return lines


def _render_table(table: ReportTable) -> List[str]:
    lines = []
    if table.caption:
        lines.append(f"*{table.caption}*")
        lines.append("")
    lines.append(f"| {' | '.join(table.headers)} |")
    lines.append(f"| {' | '.join('---' for _ in table.headers)} |")
    for row in table.rows:
        lines.append(f"| {' | '.join(row)} |")
    lines.append("")
    return lines


def _render_sources(citations: List[Citation]) -> List[str]:
    if not citations:
        return []
    lines = ["## Sources", ""]
    for index, citation in enumerate(citations, start=1):
        lines.append(f"{index}. {_render_citation(citation)}")
    lines.append("")
    return lines


def _render_citation(citation: Citation) -> str:
    if citation.source_type == "document":
        detail = citation.source_filename or "unknown document"
        if citation.page_number is not None:
            detail = f"{detail}, p. {citation.page_number}"
        return f"**{citation.title}** ({detail})"
    detail = citation.url or "unknown source"
    if citation.accessed_at is not None:
        detail = f"{detail}, accessed {citation.accessed_at.date().isoformat()}"
    return f"**{citation.title}** ({detail})"
