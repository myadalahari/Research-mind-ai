"""
Structured logging for ResearchMind AI.

This module provides the operational logging backbone used across the API
layer, services, and agents. It is deliberately kept separate from the
user-facing agent execution tracer (``app.utils.tracing``, built in Phase 6):
this module is for ops/debugging-facing logs (searchable, machine-parseable,
correlated by request/trace id), while the tracer persists a structured
execution record specifically for the frontend's Agent Execution Viewer.
They share the same correlation ids so a support engineer can jump from a
trace_id in the UI to the matching log lines, but they are not the same
mechanism.

Two concerns are handled here:

1. Correlation-id and structured-context propagation across the full
   request lifecycle, using ``contextvars`` so it works correctly with
   FastAPI's async request handling (unlike thread-locals, which break
   under concurrent async requests sharing a worker).

   * ``request_id`` — identifies a single HTTP request/response cycle.
     Set once per inbound request, typically by API middleware
     (``app.api.middleware.logging_middleware``, Phase 4).
   * ``trace_id`` — identifies a logical unit of work that may span
     multiple requests (e.g. an entire multi-turn research session, or
     every sub-call a single LangGraph run makes across agents). Pass an
     inbound trace_id (from a header, or the session id) to correlate logs
     across requests belonging to the same logical run; a new one is
     minted if none is supplied.

2. Structured field binding (``agent_name``, ``graph_node``, ``model_name``,
   ``tool_name``, ``latency_ms``, ``prompt_tokens``, ``completion_tokens``,
   ``total_tokens``) via ``bind_context``, so any code path — a LangGraph
   node, an LLMService call, a SearchProvider call — can attach the fields
   relevant to what it's doing without threading extra parameters through
   every function signature.

Output is configurable via ``LoggingSettings.format``:
  * ``json``    — one JSON object per line; intended for production, where
    logs are shipped to a log aggregator that indexes structured fields.
  * ``console`` — colorized, human-readable single-line output; intended
    for local development, where raw JSON is painful to read in a terminal.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from app.core.config import LogFormat, Settings, get_settings

# --------------------------------------------------------------------------- #
# Correlation-id & structured-context propagation
# --------------------------------------------------------------------------- #

_REQUEST_ID_CTX: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("researchmind_request_id", default=None)
_TRACE_ID_CTX: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("researchmind_trace_id", default=None)
_LOG_CONTEXT_CTX: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "researchmind_log_context", default=None
)
# default=None, not default={} -- a mutable default shared by every
# ContextVar.get() call that never .set() its own value is a known footgun
# (Ruff B039): every read site below already treats None as "no context
# bound yet" via `or {}`, so this trades a shared-mutable-object hazard
# for an explicit, harmless None check, without changing behavior for any
# caller today (bind_context() already always .set()s a brand-new dict,
# never mutates the default in place).

# Well-known structured fields surfaced by agents/services via bind_context().
# This list is documentation, not an allow-list: the formatters emit *any*
# field bound through bind_context(), known or not, so new observability
# dimensions can be added without touching this module.
STRUCTURED_FIELDS: Tuple[str, ...] = (
    "agent_name",
    "graph_node",
    "model_name",
    "tool_name",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def new_request_id() -> str:
    """Generate a new, unique request id."""
    return uuid.uuid4().hex


def new_trace_id() -> str:
    """Generate a new, unique trace id."""
    return uuid.uuid4().hex


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set the request id for the current execution context.

    Intended to be called once per inbound HTTP request (by
    ``LoggingMiddleware``). Returns the id that was set — either the one
    passed in (e.g. from an inbound ``X-Request-ID`` header) or a freshly
    generated one.
    """
    resolved = request_id or new_request_id()
    _REQUEST_ID_CTX.set(resolved)
    return resolved


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """
    Set the trace id for the current execution context.

    Pass an existing trace_id (e.g. the research session id) to correlate
    logs across multiple requests belonging to the same logical run;
    otherwise a new one is generated.
    """
    resolved = trace_id or new_trace_id()
    _TRACE_ID_CTX.set(resolved)
    return resolved


def get_request_id() -> Optional[str]:
    """Return the request id bound to the current execution context, if any."""
    return _REQUEST_ID_CTX.get()


def get_trace_id() -> Optional[str]:
    """Return the trace id bound to the current execution context, if any."""
    return _TRACE_ID_CTX.get()


def init_correlation_ids(request_id: Optional[str] = None, trace_id: Optional[str] = None) -> Tuple[str, str]:
    """
    Establish both correlation ids for the current execution context.

    Typical use is once per inbound request, e.g. inside
    ``LoggingMiddleware.dispatch``::

        request_id, trace_id = init_correlation_ids(
            request_id=request.headers.get("x-request-id"),
            trace_id=request.headers.get("x-trace-id") or session_id,
        )

    Returns the ``(request_id, trace_id)`` tuple that was set.
    """
    return set_request_id(request_id), set_trace_id(trace_id)


@contextmanager
def correlation_scope(request_id: Optional[str] = None, trace_id: Optional[str] = None) -> Iterator[Tuple[str, str]]:
    """
    Context manager variant of ``init_correlation_ids`` that restores the
    previous correlation ids on exit.

    Useful outside the request/response cycle — background jobs, scripts,
    tests — where there is no natural end-of-request point to rely on for
    cleanup and context should not leak into surrounding code.
    """
    rid = request_id or new_request_id()
    tid = trace_id or new_trace_id()
    rid_token = _REQUEST_ID_CTX.set(rid)
    tid_token = _TRACE_ID_CTX.set(tid)
    try:
        yield rid, tid
    finally:
        _REQUEST_ID_CTX.reset(rid_token)
        _TRACE_ID_CTX.reset(tid_token)


@contextmanager
def bind_context(**fields: Any) -> Iterator[None]:
    """
    Temporarily merge structured fields into the current logging context so
    every log statement executed within the ``with`` block automatically
    carries them — no need to pass ``extra=`` on every call.

    Any keyword is accepted; the well-known ones used across the codebase
    are listed in ``STRUCTURED_FIELDS`` (``agent_name``, ``graph_node``,
    ``model_name``, ``tool_name``, ``latency_ms``, ``prompt_tokens``,
    ``completion_tokens``, ``total_tokens``). ``None`` values are dropped
    so callers can pass optional fields unconditionally, e.g.
    ``bind_context(tool_name=tool_name, latency_ms=None)`` when latency
    isn't known yet.

    Usage::

        with bind_context(agent_name="planner", graph_node="plan"):
            logger.info("Generating research plan")
            ...
            with bind_context(model_name="qwen3", prompt_tokens=812):
                logger.info("LLM call completed", extra={"completion_tokens": 214})

    Nested calls merge with (and can override) the enclosing context, and
    the previous context is restored on exit via the contextvars token, so
    this is safe to use concurrently across async tasks.
    """
    current = _LOG_CONTEXT_CTX.get() or {}
    merged = {**current, **{key: value for key, value in fields.items() if value is not None}}
    token = _LOG_CONTEXT_CTX.set(merged)
    try:
        yield
    finally:
        _LOG_CONTEXT_CTX.reset(token)


@contextmanager
def measure_latency_ms() -> Iterator[Callable[[], float]]:
    """
    Context manager yielding a zero-argument callable returning elapsed
    milliseconds since the ``with`` block was entered.

    Typical use, pairing latency measurement with ``bind_context``::

        with measure_latency_ms() as elapsed:
            result = llm_service.generate(prompt)
        with bind_context(latency_ms=round(elapsed(), 2), model_name=model):
            logger.info("LLM generation completed")
    """
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000.0


class _ContextInjectingFilter(logging.Filter):
    """
    Logging filter that stamps every ``LogRecord`` with the current
    correlation ids and any structured fields bound via ``bind_context``.

    This runs for every log record regardless of formatter, so both
    ``JSONLogFormatter`` and ``ConsoleLogFormatter`` see the same enriched
    record and only differ in how they render it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        record.trace_id = get_trace_id()
        for key, value in (_LOG_CONTEXT_CTX.get() or {}).items():
            setattr(record, key, value)
        return True


# Attribute names that appear on every stock LogRecord; used by the
# formatters to distinguish "extra"/structured fields from record internals
# without maintaining a separate hand-written allow-list that could drift.
_STANDARD_LOG_RECORD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {
    "message",
    "asctime",
}
_CORRELATION_ATTRS = frozenset({"request_id", "trace_id"})


def _extract_structured_fields(record: logging.LogRecord) -> Dict[str, Any]:
    """Return the bind_context()-supplied fields attached to a log record."""
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _STANDARD_LOG_RECORD_ATTRS and key not in _CORRELATION_ATTRS
    }


# --------------------------------------------------------------------------- #
# Formatters
# --------------------------------------------------------------------------- #


class JSONLogFormatter(logging.Formatter):
    """
    Renders one JSON object per log line.

    Schema (fields present only when they have a value):
        timestamp, level, logger, message, module, function, line,
        request_id, trace_id, <any bind_context() fields>, exception, stack
    """

    def __init__(self, include_request_id: bool = True, include_trace_id: bool = True) -> None:
        super().__init__()
        self._include_request_id = include_request_id
        self._include_trace_id = include_trace_id

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        request_id = getattr(record, "request_id", None)
        trace_id = getattr(record, "trace_id", None)
        if self._include_request_id and request_id:
            payload["request_id"] = request_id
        if self._include_trace_id and trace_id:
            payload["trace_id"] = trace_id

        payload.update(_extract_structured_fields(record))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleLogFormatter(logging.Formatter):
    """
    Human-readable single-line formatter for local development.

    Example output::

        2026-07-29T10:15:03.512Z INFO     app.agents.planner   [req=3f2a1c9d trace=9b7e0a11] Generating research plan  (agent_name=planner, graph_node=plan)
    """

    _LEVEL_COLORS = {
        "DEBUG": "\x1b[36m",  # cyan
        "INFO": "\x1b[32m",  # green
        "WARNING": "\x1b[33m",  # yellow
        "ERROR": "\x1b[31m",  # red
        "CRITICAL": "\x1b[41m\x1b[97m",  # white on red
    }
    _RESET = "\x1b[0m"

    def __init__(self, use_color: Optional[bool] = None) -> None:
        super().__init__()
        self._use_color = sys.stdout.isatty() if use_color is None else use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        level = record.levelname
        if self._use_color:
            color = self._LEVEL_COLORS.get(level, "")
            level_display = f"{color}{level:<8}{self._RESET}"
        else:
            level_display = f"{level:<8}"

        correlation_bits = []
        request_id = getattr(record, "request_id", None)
        trace_id = getattr(record, "trace_id", None)
        if request_id:
            correlation_bits.append(f"req={request_id[:8]}")
        if trace_id:
            correlation_bits.append(f"trace={trace_id[:8]}")
        correlation = f"[{' '.join(correlation_bits)}] " if correlation_bits else ""

        extras = _extract_structured_fields(record)
        extras_display = ", ".join(f"{key}={value}" for key, value in sorted(extras.items()))

        line = f"{timestamp} {level_display} {record.name:<30} {correlation}{record.getMessage()}"
        if extras_display:
            line += f"  ({extras_display})"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            line += "\n" + self.formatStack(record.stack_info)
        return line


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

# Third-party loggers that are noisy at INFO/DEBUG and would otherwise drown
# out ResearchMind's own structured logs; capped at WARNING regardless of the
# configured app log level.
_NOISY_THIRD_PARTY_LOGGERS = (
    "httpx",
    "httpcore",
    "chromadb",
    "chromadb.telemetry",
    "sentence_transformers",
    "urllib3",
    "uvicorn.access",
)


def setup_logging(settings: Optional[Settings] = None) -> None:
    """
    Configure the root logger for the whole application process.

    Call this exactly once, at application startup (``app.main`` on FastAPI
    startup, and at the top of the Streamlit entrypoint / any standalone
    script or test session bootstrap). Idempotent: safe to call more than
    once (e.g. across repeated test runs) since it replaces rather than
    accumulates handlers.
    """
    resolved_settings = settings or get_settings()
    log_settings = resolved_settings.logging

    formatter: logging.Formatter
    if log_settings.format == LogFormat.JSON:
        formatter = JSONLogFormatter(
            include_request_id=log_settings.include_request_id,
            include_trace_id=log_settings.include_trace_id,
        )
    else:
        formatter = ConsoleLogFormatter()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(_ContextInjectingFilter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_settings.level)

    for noisy_logger_name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(noisy_logger_name).setLevel(
            max(logging.WARNING, getattr(logging, log_settings.level, logging.INFO))
        )

    # Route Python's warnings.warn() output through logging so it's captured
    # in the same structured/console stream instead of going straight to stderr.
    logging.captureWarnings(True)


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-scoped logger.

    Thin wrapper over ``logging.getLogger`` kept for a consistent import
    path (``from app.core.logging import get_logger``) and as the single
    seam to change logger construction behavior later if needed.
    """
    return logging.getLogger(name)
