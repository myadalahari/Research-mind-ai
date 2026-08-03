"""
Global FastAPI exception handling.

This module is the single place HTTP status codes are decided for the
entire application. No exception class in ``app.core.exceptions`` carries
its own status code by design (see that module's docstring) — every
failure, whether it's one of our own ``ResearchMindError`` subclasses, a
FastAPI/Pydantic request-validation failure, a Starlette ``HTTPException``,
or a completely unanticipated bug, is funneled through the handlers
registered here and rendered as one consistent JSON envelope::

    {
      "error": {
        "error_code": "RM-RAG-001",
        "message": "Vector store operation failed.",
        "retryable": true,
        "request_id": "3f2a1c9d4e5b4a1e9c7d2f8a1b6c9d0e",
        "trace_id": "9b7e0a11f3c24d5a8b1e6f0a2c3d4e5f",
        "details": {"document_id": "doc-42"}
      }
    }

Four things this module guarantees:

1. **Consistency** — every error response, regardless of source, has the
   same shape. Frontend and API clients write one parsing path, not one
   per exception family.
2. **Correlation** — ``request_id``/``trace_id`` are always present, both
   in the JSON body and as ``X-Request-ID``/``X-Trace-ID`` response
   headers, so a user-reported error can be traced straight to the matching
   structured log lines.
3. **Environment-aware detail exposure** — in production, unexpected
   (non-``ResearchMindError``) failures return a generic, safe message and
   omit ``details`` entirely, so internal exception text, stack traces, or
   provider payloads never leak to a client. In local/development/staging,
   the real exception message and diagnostic details are included to keep
   debugging fast. ``ResearchMindError.message`` values are authored by us
   (either a class ``default_message`` or a deliberately written message at
   the raise site) and are therefore considered safe to show in any
   environment — it's ``details`` (which may carry raw provider payloads,
   file paths, etc.) that gets the environment gate.
4. **Centralized logging** — every handled exception is logged exactly
   once, here, via the structured logging system (``app.core.logging``),
   at a level derived from the resolved HTTP status (5xx -> ERROR,
   4xx -> WARNING), with the exception's full ``to_log_dict()`` payload
   bound as structured context.

Wire this into the app once at startup::

    from app.api.middleware.error_handler import register_exception_handlers
    app = FastAPI(...)
    register_exception_handlers(app)
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, Optional, Tuple, Type

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import Environment, Settings, get_settings
from app.core.exceptions import (
    AgentError,
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    ConversationMemoryError,
    DatabaseError,
    DocumentChunkingError,
    DocumentExtractionError,
    EmbeddingProviderError,
    FileTooLargeError,
    LLMRateLimitError,
    LLMServiceError,
    MaxRetriesExceededError,
    RecordNotFoundError,
    RequestValidationError as ResearchMindRequestValidationError,
    ReportExportError,
    ResearchMindError,
    ResourceNotFoundError,
    RetrievalError,
    SearchProviderError,
    SearchQuotaExceededError,
    SessionNotFoundError,
    UnsupportedFileTypeError,
    VectorStoreError,
    RateLimitError as ResearchMindRateLimitError,
)
from app.core.logging import bind_context, get_logger, get_request_id, get_trace_id, init_correlation_ids
from app.schemas.common import ErrorBody, ErrorEnvelope

logger = get_logger(__name__)

# ErrorBody/ErrorEnvelope are defined once, canonically, in app.schemas.common
# (imported below) so the OpenAPI-documented error schema and the runtime
# response this module actually produces can never drift apart.


# =============================================================================
# Status code resolution — the ONLY place ResearchMindError subclasses are
# mapped to HTTP status codes. Resolution walks the exception's MRO and
# returns the first match, so mapping only the bases below is sufficient:
# any subclass not explicitly listed inherits its base's status through the
# MRO walk (e.g. PlannerError -> AgentError -> 500) without needing its own
# entry here.
# =============================================================================

_STATUS_CODE_MAP: Dict[Type[ResearchMindError], int] = {
    # --- 400 Bad Request: malformed client input -----------------------------
    UnsupportedFileTypeError: status.HTTP_400_BAD_REQUEST,
    FileTooLargeError: status.HTTP_400_BAD_REQUEST,
    # --- 401 / 403: auth (reserved for future use) ----------------------------
    AuthenticationError: status.HTTP_401_UNAUTHORIZED,
    AuthorizationError: status.HTTP_403_FORBIDDEN,
    # --- 404 Not Found ---------------------------------------------------------
    ResourceNotFoundError: status.HTTP_404_NOT_FOUND,
    SessionNotFoundError: status.HTTP_404_NOT_FOUND,
    RecordNotFoundError: status.HTTP_404_NOT_FOUND,
    # --- 422 Unprocessable Entity: well-formed request, unprocessable content --
    ResearchMindRequestValidationError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    DocumentExtractionError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    DocumentChunkingError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    # --- 429 Too Many Requests ---------------------------------------------------
    ResearchMindRateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
    SearchQuotaExceededError: status.HTTP_429_TOO_MANY_REQUESTS,
    LLMRateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
    # --- 500 Internal Server Error: our own logic/workflow failures ------------
    ResearchMindError: status.HTTP_500_INTERNAL_SERVER_ERROR,  # base fallback
    AgentError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ReportExportError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    MaxRetriesExceededError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    # --- 502 Bad Gateway: an upstream dependency (LLM/search/vector store) -----
    LLMServiceError: status.HTTP_502_BAD_GATEWAY,
    SearchProviderError: status.HTTP_502_BAD_GATEWAY,
    VectorStoreError: status.HTTP_502_BAD_GATEWAY,
    EmbeddingProviderError: status.HTTP_502_BAD_GATEWAY,
    RetrievalError: status.HTTP_502_BAD_GATEWAY,
    # --- 503 Service Unavailable: our own dependent infra is down/degraded -----
    DatabaseError: status.HTTP_503_SERVICE_UNAVAILABLE,
    ConversationMemoryError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def _resolve_status_code(exc: ResearchMindError) -> int:
    """
    Walk ``type(exc)``'s MRO and return the status code for the first class
    found in ``_STATUS_CODE_MAP``. Falls back to 500 if, somehow, nothing
    matches (unreachable in practice since ``ResearchMindError`` itself is
    always mapped, but kept explicit rather than relying on that silently).
    """
    for klass in type(exc).__mro__:
        if klass in _STATUS_CODE_MAP:
            return _STATUS_CODE_MAP[klass]
    return status.HTTP_500_INTERNAL_SERVER_ERROR


# =============================================================================
# Shared helpers
# =============================================================================


def _ensure_correlation_ids(request: Request) -> Tuple[str, str]:
    """
    Resolve request_id/trace_id for the current request.

    Prefers whatever is already bound in the logging context (normally set
    by ``LoggingMiddleware`` earlier in the middleware stack), falling back
    to inbound ``X-Request-ID``/``X-Trace-ID`` headers, and finally
    generating fresh ids. This makes the error handler correct even before
    ``LoggingMiddleware`` runs or in a request path that bypasses it.
    """
    return init_correlation_ids(
        request_id=get_request_id() or request.headers.get("x-request-id"),
        trace_id=get_trace_id() or request.headers.get("x-trace-id"),
    )


def _response_headers(request_id: str, trace_id: str) -> Dict[str, str]:
    return {"X-Request-ID": request_id, "X-Trace-ID": trace_id}


def _log_exception(exc: BaseException, *, status_code: int, log_payload: Dict[str, Any]) -> None:
    """
    Log a handled exception exactly once, at a level derived from the
    resolved HTTP status: server-side failures (5xx) are logged as errors
    (they represent something we should investigate); client-side failures
    (4xx) are logged as warnings (expected, caused by the request itself).
    """
    message = log_payload.pop("message", str(exc))
    with bind_context(**log_payload):
        if status_code >= 500:
            logger.error(message, exc_info=exc)
        else:
            logger.warning(message, exc_info=exc)


def _build_envelope(
    *,
    error_code: str,
    message: str,
    retryable: bool,
    request_id: str,
    trace_id: str,
    details: Optional[Dict[str, Any]],
    settings: Settings,
) -> Dict[str, Any]:
    """
    Construct the final JSON-serializable error envelope, applying the
    environment-aware ``details`` gate uniformly for every handler.
    """
    include_details = details is not None and settings.app.environment != Environment.PRODUCTION
    body = ErrorBody(
        error_code=error_code,
        message=message,
        retryable=retryable,
        request_id=request_id,
        trace_id=trace_id,
        details=details if include_details else None,
    )
    return ErrorEnvelope(error=body).model_dump(mode="json", exclude_none=True)


# =============================================================================
# Handlers
# =============================================================================


async def researchmind_error_handler(request: Request, exc: ResearchMindError) -> JSONResponse:
    """Handle any ``app.core.exceptions.ResearchMindError`` (and subclasses)."""
    settings = get_settings()
    request_id, trace_id = _ensure_correlation_ids(request)
    status_code = _resolve_status_code(exc)

    log_payload = exc.to_log_dict()
    log_payload["request_id"] = request_id
    log_payload["trace_id"] = trace_id
    log_payload["http_status"] = status_code
    _log_exception(exc, status_code=status_code, log_payload=dict(log_payload))

    envelope = _build_envelope(
        error_code=exc.error_code,
        message=exc.message,
        retryable=exc.retryable,
        request_id=request_id,
        trace_id=trace_id,
        details=exc.details or None,
        settings=settings,
    )
    return JSONResponse(status_code=status_code, content=envelope, headers=_response_headers(request_id, trace_id))


async def fastapi_validation_error_handler(request: Request, exc: FastAPIRequestValidationError) -> JSONResponse:
    """
    Handle FastAPI/Pydantic's own request validation failures (malformed
    JSON body, missing required fields, wrong types in path/query/body).

    Distinct from ``app.core.exceptions.RequestValidationError``, which is
    for application-level validation rules raised deliberately inside
    service code (e.g. a cross-field business rule) — this handler is for
    FastAPI's automatic request-schema validation, imported under an alias
    to avoid the name collision.
    """
    settings = get_settings()
    request_id, trace_id = _ensure_correlation_ids(request)
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

    errors = exc.errors()
    log_payload = {
        "error_code": "RM-API-002",
        "error_type": "FastAPIRequestValidationError",
        "retryable": False,
        "request_id": request_id,
        "trace_id": trace_id,
        "http_status": status_code,
        "validation_errors": errors,
    }
    _log_exception(exc, status_code=status_code, log_payload={**log_payload, "message": "Request validation failed"})

    envelope = _build_envelope(
        error_code="RM-API-002",
        message="Request validation failed.",
        retryable=False,
        request_id=request_id,
        trace_id=trace_id,
        details={"validation_errors": errors},
        settings=settings,
    )
    return JSONResponse(status_code=status_code, content=envelope, headers=_response_headers(request_id, trace_id))


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle ``starlette.exceptions.HTTPException`` — raised either
    explicitly (``raise HTTPException(404, "...")``) or internally by
    Starlette/FastAPI itself (404 route not found, 405 method not allowed).
    """
    settings = get_settings()
    request_id, trace_id = _ensure_correlation_ids(request)
    status_code = exc.status_code
    message = str(exc.detail) if exc.detail else "The request could not be completed."

    _log_exception(
        exc,
        status_code=status_code,
        log_payload={
            "error_code": f"RM-HTTP-{status_code}",
            "error_type": "HTTPException",
            "retryable": False,
            "request_id": request_id,
            "trace_id": trace_id,
            "http_status": status_code,
            "message": message,
        },
    )

    envelope = _build_envelope(
        error_code=f"RM-HTTP-{status_code}",
        message=message,
        retryable=False,
        request_id=request_id,
        trace_id=trace_id,
        details=None,
        settings=settings,
    )
    return JSONResponse(status_code=status_code, content=envelope, headers=_response_headers(request_id, trace_id))


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Last-resort safety net for any exception not covered by the handlers
    above — a genuine bug, an unwrapped third-party exception that leaked
    past a service boundary without being translated into a
    ``ResearchMindError``, etc.

    This handler is the sharpest edge of the environment-aware detail
    exposure requirement: in production it must never leak the real
    exception message or a traceback to the client, since either could
    contain internal paths, connection strings, or other sensitive detail.
    Outside production, the real message and a truncated traceback are
    included in ``details`` to keep local/staging debugging fast.
    """
    settings = get_settings()
    request_id, trace_id = _ensure_correlation_ids(request)
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _log_exception(
        exc,
        status_code=status_code,
        log_payload={
            "error_code": "RM-GEN-500",
            "error_type": type(exc).__name__,
            "retryable": False,
            "request_id": request_id,
            "trace_id": trace_id,
            "http_status": status_code,
            "message": f"Unhandled exception: {exc}",
        },
    )

    if settings.app.environment == Environment.PRODUCTION:
        message = "An unexpected internal error occurred. Please retry, or contact support with the request_id below."
        details: Optional[Dict[str, Any]] = None
    else:
        message = f"{type(exc).__name__}: {exc}"
        details = {"traceback": tb_text[-4000:]}

    envelope = _build_envelope(
        error_code="RM-GEN-500",
        message=message,
        retryable=False,
        request_id=request_id,
        trace_id=trace_id,
        details=details,
        settings=settings,
    )
    return JSONResponse(status_code=status_code, content=envelope, headers=_response_headers(request_id, trace_id))


# =============================================================================
# Registration
# =============================================================================


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register every exception handler on ``app``.

    Call once at application startup, immediately after constructing the
    ``FastAPI`` instance in ``app.main``. Order of registration does not
    affect dispatch — Starlette selects the most specific matching handler
    for a raised exception's type regardless of registration order — but
    handlers are listed here from most to least specific for readability.
    """
    app.add_exception_handler(ResearchMindError, researchmind_error_handler)
    app.add_exception_handler(FastAPIRequestValidationError, fastapi_validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
