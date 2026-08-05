"""
Renders a ``ReportGenerationResponse``.

Purely presentational, the same discipline as ``ui.chat_view``: this
module takes data in (a ``ReportGenerationResponse``, the query that
produced it, and any already-fetched download file) and draws Streamlit
widgets from it. It never makes an HTTP request and never reads or writes
``st.session_state`` -- ``pages/1_Report.py`` owns both. The one piece of
user interaction this module can produce (clicking "Prepare download") is
reported back to the caller as a plain boolean return value, the same
pattern ``chat_view.render_conversation`` uses for follow-up suggestions,
rather than this module reaching into ``api_client``/``session`` to fetch
the file itself.

Rendering branches entirely on ``response.status`` -- ``pages/1_Report.py``
never inspects it. ``QUEUED``/``GENERATING``/``EXPORTING`` are handled with
a plain "in progress" line even though the current backend
(``ReportService.generate_report()``) always resolves synchronously to
``COMPLETED`` or ``FAILED``: the status enum's own docstring documents this
as a forward-looking async-capable lifecycle, so this function treats
every status as reachable rather than assuming only the two values the
backend happens to return today.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import streamlit as st

from core.api_client import DownloadedFile
from core.models import ReportCitation, ReportGenerationResponse, ReportGenerationStatus, ReportSection, ReportTable

_TOP_LEVEL_HEADING_LEVEL = 3  # section headings start at "###"


def render_report(
    response: ReportGenerationResponse,
    *,
    query: Optional[str],
    cached_file: Optional[DownloadedFile],
) -> bool:
    """
    Render ``response`` and return ``True`` only if the user just clicked
    "Prepare download" this run -- ``False`` in every other case,
    including when a real ``st.download_button`` (not a click signal, a
    self-contained widget) is shown instead.
    """
    if query:
        st.caption(f"Report for: {query}")

    if response.status == ReportGenerationStatus.COMPLETED:
        return _render_completed(response, cached_file=cached_file)
    if response.status == ReportGenerationStatus.FAILED:
        _render_failed(response)
        return False

    st.info(f"Report is currently **{response.status.value}**...")
    return False


def _render_failed(response: ReportGenerationResponse) -> None:
    if response.error is None:
        # Unreachable given ReportGenerationResponse's own validator (error is
        # required when status == FAILED), but this function never assumes
        # that invariant held upstream rather than checking it here.
        st.error("Report generation failed.")
        return
    st.error(f"**{response.error.message}**\n\n`{response.error.error_code}`")


def _render_completed(response: ReportGenerationResponse, *, cached_file: Optional[DownloadedFile]) -> bool:
    report = response.report
    if report is None:
        # Unreachable given the same validator as above, guarded the same defensive way.
        st.error("The report finished generating but no content was returned.")
        return False

    st.header(report.title)
    st.markdown(report.executive_summary)

    if report.key_findings:
        st.subheader("Key findings")
        for finding in report.key_findings:
            st.markdown(f"- {finding}")

    for section in report.sections:
        _render_section(section, level=_TOP_LEVEL_HEADING_LEVEL)

    st.subheader("Conclusion")
    st.markdown(report.conclusion)

    if report.citations:
        _render_report_citations(report.citations)

    _render_report_stats(response)

    return _render_download_controls(response, cached_file=cached_file)


def _render_section(section: ReportSection, *, level: int) -> None:
    heading_marker = "#" * min(level, 6)
    st.markdown(f"{heading_marker} {section.heading}")
    st.markdown(section.content)

    for table in section.tables:
        _render_table(table)

    for subsection in section.subsections:
        _render_section(subsection, level=level + 1)


def _render_table(table: ReportTable) -> None:
    if table.caption:
        st.caption(table.caption)
    st.table(pd.DataFrame(table.rows, columns=table.headers))


def _render_report_citations(citations: List[ReportCitation]) -> None:
    with st.expander(f"Sources ({len(citations)})"):
        for citation in citations:
            title = f"[{citation.citation_id}] {citation.title}"
            if citation.url:
                st.markdown(f"**[{title}]({citation.url})**")
            else:
                st.markdown(f"**{title}**")

            source_line = _report_citation_source_line(citation)
            if source_line:
                st.caption(source_line)

            st.divider()


def _report_citation_source_line(citation: ReportCitation) -> Optional[str]:
    if citation.source_type == "document":
        parts = ["Document"]
        if citation.source_filename:
            parts.append(citation.source_filename)
        if citation.page_number is not None:
            parts.append(f"p. {citation.page_number}")
        return " · ".join(parts)
    if citation.source_type == "web":
        parts = ["Web"]
        if citation.accessed_at is not None:
            parts.append(f"accessed {citation.accessed_at.date().isoformat()}")
        return " · ".join(parts)
    return None


def _render_report_stats(response: ReportGenerationResponse) -> None:
    parts = [f"Format: {response.export_format.value.upper()}"]
    if response.file_size_bytes is not None:
        parts.append(f"Size: {_format_file_size(response.file_size_bytes)}")
    if response.generation_latency_ms is not None:
        parts.append(f"Generated in {response.generation_latency_ms / 1000:.1f}s")
    if response.token_usage is not None and response.token_usage.total_tokens is not None:
        parts.append(f"Tokens: {response.token_usage.total_tokens}")
    st.caption(" · ".join(parts))


def _format_file_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # unreachable in practice; keeps the function total


def _render_download_controls(response: ReportGenerationResponse, *, cached_file: Optional[DownloadedFile]) -> bool:
    if cached_file is not None:
        st.download_button(
            "Download report",
            data=cached_file.content,
            file_name=cached_file.filename,
            mime=cached_file.mime_type,
        )
        return False
    return st.button("Prepare download")
