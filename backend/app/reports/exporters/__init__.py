"""
Concrete ``ReportExporter`` (``app.core.interfaces.report_exporter``)
implementations -- one module per output format.

Each exporter takes an already-assembled ``ReportDocument`` (built by
``app.reports.builder``, never constructed here) and renders it into a
downloadable ``ExportResult``. Selecting a concrete exporter is
``app.core.dependencies.get_report_exporter(format=...)``'s job, not this
package's -- these classes are plain, independently-instantiable renderers
with no knowledge of dependency injection or HTTP.
"""
