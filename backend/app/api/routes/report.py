"""
``POST /report`` and ``GET /report/{report_id}/download`` -- generating
and retrieving research reports.

As thin as ``app.api.routes.chat`` (see that file's own docstring for the
full rationale, repeated here only where it differs): every actual
decision already lives in ``ReportService`` --
session/research orchestration (via ``ChatService``), report assembly,
export, and persistence for ``POST /report``; resolving a completed
report's downloadable file for ``GET /report/{report_id}/download``. This
file's only job is HTTP-specific concerns -- parsing the request,
establishing correlation ids, measuring latency, wrapping/streaming the
result -- never business logic and never local error-to-status-code
mapping.

No ``try``/``except`` anywhere in this file, for the same reason
``chat.py`` has none: every exception ``ReportService`` raises is already
a ``ResearchMindError`` subclass, uniformly handled by the global
handlers in ``app.api.middleware.error_handler``. This matters especially
for the download route, which relies on two *different* exceptions
mapping to the same 404 status for two different reasons (see
``ReportService.get_download()``'s own docstring) -- duplicating that
distinction here would undo the point of centralizing it there.

Scope note, matching ``app.api.routes``'s own package docstring: mounting
this router onto the application instance is ``app.main``'s job, done as
its own separate step (the same sequencing ``chat_router`` went through),
not this file's.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from app.core.dependencies import get_report_service
from app.core.logging import get_logger, init_correlation_ids, measure_latency_ms
from app.schemas.common import DataResponse
from app.schemas.report import ReportGenerationRequest, ReportGenerationResponse
from app.services.report_service import ReportService

logger = get_logger(__name__)

router = APIRouter(tags=["report"])


@router.post(
    "/report",
    response_model=DataResponse[ReportGenerationResponse],
    summary="Generate a research report",
    description=(
        "Runs the same underlying research workflow as /chat with mode='report', then exports the "
        "result to the requested format (Markdown or PDF) and persists it for later download."
    ),
)
async def create_report(
    request: Request,
    body: ReportGenerationRequest,
    report_service: ReportService = Depends(get_report_service),
) -> DataResponse[ReportGenerationResponse]:
    init_correlation_ids(
        request_id=request.headers.get("x-request-id"),
        trace_id=request.headers.get("x-trace-id"),
    )
    logger.info(
        "Report generation requested",
        extra={"session_id": body.session_id, "export_format": body.export_format.value},
    )

    with measure_latency_ms() as elapsed:
        response = await report_service.generate_report(body)
    latency_ms = elapsed()

    logger.info(
        "Report generation request completed",
        extra={
            "report_id": response.report_id,
            "session_id": response.session_id,
            "status": response.status.value,
            "latency_ms": round(latency_ms, 2),
        },
    )
    return DataResponse.wrap(response, latency_ms=latency_ms)


@router.get(
    "/report/{report_id}/download",
    summary="Download a generated report file",
    description=("Streams the exported file (Markdown or PDF) for a previously generated, completed report."),
    response_class=FileResponse,
)
async def download_report(
    request: Request,
    report_id: str,
    report_service: ReportService = Depends(get_report_service),
) -> FileResponse:
    init_correlation_ids(
        request_id=request.headers.get("x-request-id"),
        trace_id=request.headers.get("x-trace-id"),
    )

    download = await report_service.get_download(report_id)

    logger.info(
        "Report download served",
        # NOT "filename" -- Python's stdlib `logging` reserves that key as
        # one of LogRecord's own attributes (the source file of the
        # logging call itself); passing it via `extra` raises
        # `KeyError: Attempt to overwrite 'filename' in LogRecord` at the
        # first real call, a failure that only appears once a genuine
        # logging backend processes the record (isolated calls in a test
        # harness that stub logging can miss it entirely).
        extra={"report_id": report_id, "download_filename": download.filename},
    )
    return FileResponse(
        path=download.file_path,
        media_type=download.mime_type,
        filename=download.filename,
    )
