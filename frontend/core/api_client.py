"""
HTTP client for the ResearchMind AI backend.

This is the **single place** in the frontend that performs HTTP requests,
parses response bodies, and validates them against the wire-contract models
in ``core.models``. Every UI component (``ui/chat_view.py``,
``ui/report_view.py``, and the page scripts in ``pages/``) calls a function
in this module and only ever handles already-validated ``core.models``
instances or an ``APIError`` -- never a ``requests.Response`` or raw JSON.
This mirrors the backend's own "routes are thin, one place owns HTTP
concerns" principle (see ``app.api.middleware.error_handler``'s docstring),
applied on the client side of the same boundary.

Three failure modes are distinguished, each its own ``APIError`` subclass,
so a caller (and ultimately the UI) can tell them apart and respond
appropriately rather than rendering one generic "something went wrong":

* ``APIConnectionError`` -- the request never got a response at all
  (connection refused, DNS failure, timeout). No backend-authored error
  exists yet, so this frontend synthesizes one. Always ``retryable=True``:
  a transient network hiccup is exactly the case where "try again" is
  reasonable advice.
* ``APIResponseError`` -- the backend responded with a non-2xx status and a
  well-formed ``ErrorEnvelope``. Every failure path in
  ``app.api.middleware.error_handler`` (our own exceptions, FastAPI's own
  request-validation errors, Starlette HTTPExceptions, even an unhandled
  bug) is funneled through that one handler and produces this same
  envelope shape, so this is the only error-parsing path this client needs.
* ``APIValidationError`` -- the backend responded with a 2xx status, but
  its body didn't match the expected ``core.models`` contract. This is not
  a "the request failed" case; it's a "the backend's contract has drifted
  from what this frontend was built against" case, and is surfaced
  distinctly so a developer sees a contract-mismatch signal rather than a
  misleading "network error" or a bare stack trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar
from urllib.parse import urljoin, urlsplit

import requests
from pydantic import BaseModel, ValidationError

from core.config import get_settings
from core.models import (
    ChatRequest,
    ChatResponse,
    DataEnvelope,
    ErrorEnvelope,
    ReportGenerationRequest,
    ReportGenerationResponse,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


# =============================================================================
# Errors
# =============================================================================


class APIError(Exception):
    """
    Base class for every error this client raises.

    Carries the same fields the backend's own ``ErrorBody`` does
    (``message``, ``error_code``, ``retryable``) plus optional correlation
    ids, so UI code can render a consistent "what went wrong" panel
    regardless of which subclass was actually raised.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        retryable: bool,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.retryable = retryable
        self.request_id = request_id
        self.trace_id = trace_id
        self.details = details
        self.status_code = status_code


class APIConnectionError(APIError):
    """The request never reached the backend, or never got a response back (network failure, timeout, DNS)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="RM-CLIENT-CONNECTION", retryable=True)


class APIResponseError(APIError):
    """The backend returned a non-2xx status with a well-formed ``ErrorEnvelope`` body."""


class APIValidationError(APIError):
    """A 2xx response body did not match the expected ``core.models`` contract."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="RM-CLIENT-CONTRACT", retryable=False)


# =============================================================================
# Report download result
# =============================================================================


@dataclass(frozen=True)
class DownloadedFile:
    """
    A fetched report file, ready to hand to ``st.download_button``.

    Not a ``core.models`` wire-contract type -- ``filename``/``mime_type``
    are derived from response *headers* (``Content-Disposition``/
    ``Content-Type``), not a JSON body, so there is no schema to validate
    against; this is simply the processed result of ``download_report()``.
    """

    content: bytes
    filename: str
    mime_type: str


_CONTENT_DISPOSITION_FILENAME_RE = re.compile(r'filename="?([^";]+)"?')


def _extract_filename(content_disposition: Optional[str], fallback: str) -> str:
    """
    Parse a filename out of a ``Content-Disposition`` header value, e.g.
    ``attachment; filename="quantum-computing-20260729-101503.pdf"``.

    Falls back to a caller-supplied name if the header is missing or
    doesn't contain a recognizable ``filename=`` directive -- the backend
    (``FileResponse(..., filename=download.filename)``) always sets this in
    practice, but a caller shouldn't crash if a future backend change (or
    an intermediary proxy) ever strips it.
    """
    if not content_disposition:
        return fallback
    match = _CONTENT_DISPOSITION_FILENAME_RE.search(content_disposition)
    return match.group(1) if match else fallback


# =============================================================================
# Low-level request/response plumbing
# =============================================================================


def _request(
    method: str,
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, str]] = None,
) -> requests.Response:
    """
    Issue one HTTP request against the configured backend and return the
    raw ``requests.Response`` -- callers are responsible for passing it
    through ``_raise_for_error`` before treating it as a success.

    Every transport-level failure (the request never producing an HTTP
    response at all) is normalized to ``APIConnectionError`` here, so
    nothing above this function ever needs to catch ``requests.exceptions.*``
    directly.
    """
    settings = get_settings()
    url = f"{settings.api_base_url}{path}"
    try:
        return requests.request(method, url, json=json_body, params=params, timeout=settings.request_timeout_seconds)
    except requests.exceptions.Timeout as exc:
        raise APIConnectionError(
            f"The request to {url} timed out after {settings.request_timeout_seconds:.0f}s. "
            "The backend may be slow or unreachable."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise APIConnectionError(f"Could not connect to the ResearchMind AI backend at {url}. Is it running?") from exc
    except requests.exceptions.RequestException as exc:
        raise APIConnectionError(f"Request to {url} failed: {exc}") from exc


def _raise_for_error(response: requests.Response) -> None:
    """
    Translate a non-2xx ``requests.Response`` into ``APIResponseError``.

    Every error path in the backend (``app.api.middleware.error_handler``)
    produces the same ``ErrorEnvelope`` shape regardless of what actually
    failed, so this is the one parsing path needed for every 4xx/5xx this
    client will ever see. The fallback branch (body isn't a valid
    ``ErrorEnvelope``) is defensive -- it should be unreachable against this
    project's own backend, but guards against a misconfigured
    ``RESEARCHMIND_API_BASE_URL`` pointing at something else entirely (a
    proxy's own error page, a different service) returning a differently
    shaped error body.
    """
    if response.ok:
        return
    try:
        envelope = ErrorEnvelope.model_validate(response.json())
    except (ValueError, ValidationError):
        raise APIResponseError(
            f"The backend returned HTTP {response.status_code} with an unrecognized error body.",
            error_code="RM-CLIENT-UNKNOWN",
            retryable=False,
            status_code=response.status_code,
            details={"raw_body": response.text[:2000]},
        ) from None
    body = envelope.error
    raise APIResponseError(
        body.message,
        error_code=body.error_code,
        retryable=body.retryable,
        request_id=body.request_id,
        trace_id=body.trace_id,
        details=body.details,
        status_code=response.status_code,
    )


def _parse_data(response: requests.Response, model: Type[ModelT]) -> ModelT:
    """
    Validate a successful response's body as ``DataEnvelope[model]`` and
    return its ``data`` payload.

    Any shape mismatch (missing field, wrong type, a field the backend
    dropped or renamed) raises ``APIValidationError`` here rather than
    letting an ``AttributeError``/``KeyError`` surface later, deep inside a
    UI component that assumed the field was there.
    """
    try:
        payload = response.json()
    except ValueError as exc:
        raise APIValidationError(f"The backend's response was not valid JSON: {exc}") from exc
    try:
        envelope = DataEnvelope[model].model_validate(payload)
    except ValidationError as exc:
        raise APIValidationError(
            f"The backend's response did not match the expected {model.__name__} contract: {exc}"
        ) from exc
    return envelope.data


# =============================================================================
# Public API
# =============================================================================


def send_chat_message(request: ChatRequest, *, include_trace: bool = False) -> ChatResponse:
    """
    Call ``POST /chat``.

    ``include_trace`` is a query parameter on the backend route (not part
    of ``ChatRequest``'s body), so it's threaded through as a separate
    keyword argument here rather than added to the ``ChatRequest`` model.
    """
    response = _request(
        "POST",
        "/chat",
        json_body=request.model_dump(mode="json", exclude_none=True),
        params={"include_trace": "true" if include_trace else "false"},
    )
    _raise_for_error(response)
    return _parse_data(response, ChatResponse)


def generate_report(request: ReportGenerationRequest) -> ReportGenerationResponse:
    """Call ``POST /report``."""
    response = _request("POST", "/report", json_body=request.model_dump(mode="json", exclude_none=True))
    _raise_for_error(response)
    return _parse_data(response, ReportGenerationResponse)


def download_report(
    download_url: str,
    *,
    fallback_filename: str,
    fallback_mime_type: str = "application/octet-stream",
) -> DownloadedFile:
    """
    Fetch a generated report's file bytes from ``download_url``
    (``ReportGenerationResponse.download_url``, e.g.
    ``/api/v1/report/rpt-9c4e5b4a/download``) and return them alongside a
    filename/MIME type derived from the response headers.

    ``download_url`` is resolved against the *origin* (scheme + host) of
    the configured backend, not concatenated onto ``api_base_url`` --
    the backend already returns a full absolute path including its own API
    prefix, so blindly appending it onto ``api_base_url`` (which also ends
    in that prefix) would double it up.

    ``fallback_filename``/``fallback_mime_type`` are used only if the
    response is missing a usable ``Content-Disposition``/``Content-Type``
    header -- callers (``ui/report_view.py``) have the originating
    ``ReportGenerationResponse.report_id``/``export_format`` on hand to
    build a sensible fallback name, which this generic client function
    does not.
    """
    settings = get_settings()
    origin = urlsplit(settings.api_base_url)
    url = (
        download_url
        if download_url.startswith(("http://", "https://"))
        else urljoin(f"{origin.scheme}://{origin.netloc}", download_url)
    )
    try:
        response = requests.get(url, timeout=settings.request_timeout_seconds)
    except requests.exceptions.Timeout as exc:
        raise APIConnectionError(
            f"The download request to {url} timed out after {settings.request_timeout_seconds:.0f}s."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise APIConnectionError(f"Could not connect to the ResearchMind AI backend at {url}.") from exc
    except requests.exceptions.RequestException as exc:
        raise APIConnectionError(f"Download request to {url} failed: {exc}") from exc

    _raise_for_error(response)

    filename = _extract_filename(response.headers.get("Content-Disposition"), fallback_filename)
    mime_type = response.headers.get("Content-Type") or fallback_mime_type
    return DownloadedFile(content=response.content, filename=filename, mime_type=mime_type)
