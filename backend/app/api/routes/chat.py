"""
``POST /chat`` -- the single entry point for every conversational and
research workflow (``ChatRequest.mode`` selects lightweight chat, full
research, or report-oriented research; see ``app.schemas.chat``'s own
module docstring for why this is one endpoint rather than one per mode).

This router is deliberately thin: every actual decision already lives
below it. ``ChatService`` (``app.services.chat_service``) owns resolving
the session, loading memory, running the research graph, and persisting
the turn; this file's only job is the HTTP-specific concerns that
shouldn't leak into ``ChatService`` -- parsing the request, resolving
``include_trace`` from a query parameter, establishing correlation ids,
measuring request latency, and wrapping the result in the standard
``DataResponse`` envelope.

No ``try``/``except`` anywhere in this file. Every exception ``ChatService``
raises is already a ``ResearchMindError`` subclass (or, for a genuine bug,
an unanticipated exception) -- both cases are handled uniformly by the
global handlers registered in ``app.api.middleware.error_handler``, which
is where HTTP status code resolution and the error envelope shape are
decided exactly once for the whole application. Duplicating any of that
here (a local ``except SessionNotFoundError: raise HTTPException(404, ...)``)
would create a second place status codes get decided, exactly what that
module's own docstring says it exists to prevent.

Correlation ids are established at the top of the handler via
``init_correlation_ids`` -- the same fallback-to-inbound-headers pattern
``error_handler.py``'s own ``_ensure_correlation_ids`` already uses for the
error path, applied here for the success path. This project has no
``LoggingMiddleware`` yet to do this once for every route, so each route
establishes it directly for now; if/when a request-scoped logging
middleware is added, this becomes redundant-but-harmless (``init_correlation_ids``
is idempotent within a request) rather than something that needs to be
torn out.

Scope note: this file only defines the router. Mounting it onto a
``FastAPI`` app instance, registering the global exception handlers, and
running startup-lifecycle hooks (``Settings.ensure_runtime_directories()``,
``Database.create_all()``) are ``app.main``'s job -- not yet built, and out
of scope for this file (see ``app.api.routes``'s own package docstring).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from app.core.dependencies import get_chat_service
from app.core.logging import get_logger, init_correlation_ids, measure_latency_ms
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import DataResponse
from app.services.chat_service import ChatService

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=DataResponse[ChatResponse],
    summary="Send a chat/research request",
    description=(
        "Runs the requested workflow (chat/research/report) for a new or existing session "
        "and returns the generated response, citations, and metadata."
    ),
)
async def send_chat_message(
    request: Request,
    body: ChatRequest,
    include_trace: bool = Query(
        default=False,
        description=(
            "Include the full step-by-step agent execution trace in the response, for the "
            "frontend's Agent Execution Viewer. Omitted by default so lightweight chat "
            "responses aren't forced to pay the serialization cost -- see "
            "ChatResponse.execution_trace's own docstring."
        ),
    ),
    chat_service: ChatService = Depends(get_chat_service),
) -> DataResponse[ChatResponse]:
    init_correlation_ids(
        request_id=request.headers.get("x-request-id"),
        trace_id=request.headers.get("x-trace-id"),
    )
    logger.info(
        "Chat request received",
        extra={"mode": body.mode.value, "session_id": body.session_id, "include_trace": include_trace},
    )

    with measure_latency_ms() as elapsed:
        response = await chat_service.send_message(body, include_trace=include_trace)
    latency_ms = elapsed()

    logger.info(
        "Chat request completed",
        extra={
            "session_id": response.session_id,
            "message_id": response.message_id,
            "mode": response.mode.value,
            "latency_ms": round(latency_ms, 2),
        },
    )
    return DataResponse.wrap(response, latency_ms=latency_ms)
