"""
Report generation package (Phase 8) -- assembling and exporting
``ReportDocument``s (``app.core.interfaces.report_exporter``).

Framework- and database-agnostic, mirroring ``app.memory``'s own design:
this package has no FastAPI or SQLAlchemy import anywhere, and depends
only on interfaces (``ReportExporter``) or plain data (``ReportDocument``
and friends), never a concrete rendering library directly outside of a
concrete exporter's own module. ``app.services.report_service.ReportService``
(the DB-aware orchestrator) is the only caller that composes this
package's pure functions with persistence.
"""
