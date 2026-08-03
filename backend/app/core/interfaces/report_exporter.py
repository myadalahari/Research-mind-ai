"""
ReportExporter interface.

``ReportService`` (Phase 8) depends on this abstraction, not on ReportLab
or the Markdown rendering logic directly. Two concrete implementations are
planned: ``MarkdownReportExporter`` and ``PDFReportExporter`` (the latter
backed by ReportLab). Both consume the same provider-agnostic
``ReportDocument`` domain model, so the Writer/Reviewer agents that
produce a report and the exporters that render it never need to agree on
anything beyond this shared contract — adding a third export format
(e.g. DOCX or HTML) later is a new ``ReportExporter`` implementation, not
a change to how reports are assembled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.utils.time import utc_now


class Citation(BaseModel):
    """A single reference cited within a report, resolvable back to its source."""

    citation_id: str = Field(..., description="Stable identifier used for inline citation markers, e.g. '[1]'.")
    source_type: str = Field(..., description="One of 'document' (uploaded file) or 'web' (search result).")
    title: str = Field(..., description="Title of the cited source.")
    url: Optional[str] = Field(default=None, description="URL of the source, for web citations.")
    source_filename: Optional[str] = Field(default=None, description="Filename of the source, for document citations.")
    page_number: Optional[int] = Field(default=None, description="Page number, for document citations.")
    accessed_at: Optional[datetime] = Field(
        default=None, description="When this source was retrieved, for web citations."
    )


class ReportTable(BaseModel):
    """A table embedded within a report section."""

    caption: Optional[str] = None
    headers: List[str]
    rows: List[List[str]]


class ReportSection(BaseModel):
    """A single section of detailed findings within a report."""

    heading: str
    content: str = Field(..., description="Section body, in Markdown-compatible text with inline citation markers.")
    tables: List[ReportTable] = Field(default_factory=list)
    subsections: List["ReportSection"] = Field(default_factory=list)


ReportSection.model_rebuild()


class ReportDocument(BaseModel):
    """
    Provider-agnostic representation of a completed research report.

    Assembled by ``app.reports.builder`` from the Writer/Reviewer agents'
    output in the LangGraph state, and consumed by every
    ``ReportExporter`` implementation. This is the single contract that
    keeps report *content* (agents, builder) decoupled from report
    *rendering* (Markdown/PDF exporters).
    """

    title: str
    session_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    executive_summary: str
    key_findings: List[str] = Field(default_factory=list)
    sections: List[ReportSection] = Field(default_factory=list)
    conclusion: str
    citations: List[Citation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra metadata (e.g. originating query, agent execution trace id).",
    )


class ExportResult(BaseModel):
    """The rendered output of a report export operation."""

    content: bytes
    mime_type: str
    filename: str

    model_config = {"arbitrary_types_allowed": True}


class ReportExporter(ABC):
    """
    Abstract interface for rendering a ``ReportDocument`` into a
    downloadable file format.

    Concrete implementations live under ``app.reports`` (e.g.
    ``MarkdownReportExporter``, ``PDFReportExporter``). Injected via
    ``app.core.dependencies.get_report_exporter(format=...)``.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Short identifier for this export format, e.g. 'markdown' or 'pdf'."""
        raise NotImplementedError

    @abstractmethod
    async def export(self, report: ReportDocument) -> ExportResult:
        """
        Render ``report`` into this exporter's output format.

        Raises:
            ReportExportError: on a rendering failure.
        """
        raise NotImplementedError
