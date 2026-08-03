"""
Custom exception hierarchy for ResearchMind AI.

Every exception in this module inherits from ``ResearchMindError`` and
carries four things uniformly:

  * ``error_code`` — a stable, machine-readable identifier in the form
    ``RM-<DOMAIN>-<NNN>`` (e.g. ``RM-LLM-002``), suitable for log
    aggregation queries, alerting rules, and client-side error handling.
    Codes are guaranteed unique at import time — see the self-check at the
    bottom of this module.
  * ``retryable`` — whether the failure is transient (a caller/orchestrator
    may reasonably retry) or permanent (retrying with the same input will
    fail the same way). The Coordinator agent and outbound HTTP clients use
    this to decide whether to retry a node/call or fail the run outright.
  * ``request_id`` / ``trace_id`` — automatically captured from the current
    logging context (``app.core.logging``) at construction time, so any
    exception raised anywhere in the call stack can be logged and
    correlated back to the request/run that caused it, without every
    ``raise`` site having to pass them explicitly.
  * ``details`` — a small structured dict of extra context (e.g. which
    document failed extraction, which provider timed out), merged straight
    into the structured log line via ``to_log_dict()``.

Deliberately **out of scope for this module**: HTTP status code mapping.
That mapping (e.g. ``ResourceNotFoundError`` -> 404, ``RequestValidationError``
-> 422, ``RateLimitError`` -> 429, most domain errors -> 502/503) lives in a
single place — the FastAPI exception handler built in
``app.api.middleware.error_handler`` (Phase 4) — so the HTTP-facing
decision of "what status code does a given failure produce" never has to
be duplicated or kept in sync across dozens of exception classes.

Exception chaining: raise sites should always use ``raise SomeError(...)
from original_exc`` when wrapping a lower-level failure (an ``httpx``
timeout, a ``chromadb`` error, a JSON decode error, ...). Python records
this as ``__cause__``, which ``to_log_dict()`` surfaces automatically —
nothing extra needs to be threaded through by hand. ``ResearchMindError.wrap()``
is a convenience for the common "catch a third-party exception, translate
it to our hierarchy, preserve the original" pattern.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional

from app.core.logging import get_request_id, get_trace_id

# =============================================================================
# Base
# =============================================================================


class ResearchMindError(Exception):
    """
    Root of the ResearchMind AI exception hierarchy.

    Not intended to be raised directly in normal code — raise the most
    specific applicable subclass so callers can distinguish failure modes
    and so ``error_code`` is meaningful in logs and alerts.
    """

    error_code: ClassVar[str] = "RM-GEN-000"
    default_message: ClassVar[str] = "An unexpected error occurred."
    retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        details: Optional[Dict[str, Any]] = None,
        retryable: Optional[bool] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)
        self.details: Dict[str, Any] = details or {}
        # Allow a per-instance override of the class-level default, e.g. an
        # LLMGenerationError that turned out to be caused by a malformed
        # (non-retryable) prompt rather than a transient provider hiccup.
        # mypy flags this: `retryable` is declared `ClassVar[bool]` on
        # every subclass specifically so `type(self).retryable` reads the
        # class-level default above, but assigning `self.retryable` here
        # is exactly the per-instance override this class is designed to
        # support -- the ClassVar annotation and the instance-attribute
        # write are two intentionally different things sharing one name,
        # not a typing mistake to "fix" by removing either capability.
        self.retryable: bool = type(self).retryable if retryable is None else retryable  # type: ignore[misc]
        # Auto-capture correlation ids from the current logging context so
        # every exception is traceable without raise sites doing it by hand.
        self.request_id: Optional[str] = request_id if request_id is not None else get_request_id()
        self.trace_id: Optional[str] = trace_id if trace_id is not None else get_trace_id()

    @classmethod
    def wrap(
        cls,
        exc: BaseException,
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> "ResearchMindError":
        """
        Translate a lower-level exception into this ResearchMind error type,
        preserving the original as the cause.

        Usage::

            try:
                response = await http_client.post(...)
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError.wrap(exc) from exc

        ``wrap`` is a construction convenience only — it does not set
        ``__cause__`` itself (Python's ``raise ... from ...`` syntax does
        that); always pair it with ``from exc`` at the raise site.
        """
        instance = cls(message or str(exc) or cls.default_message, **kwargs)
        return instance

    def to_log_dict(self) -> Dict[str, Any]:
        """
        Structured representation of this exception for logging.

        Intended to be passed as the ``extra``/``bind_context`` payload
        alongside a ``logger.exception(...)`` call, or merged into an API
        error response body by the central FastAPI exception handler.
        """
        payload: Dict[str, Any] = {
            "error_code": self.error_code,
            "error_type": type(self).__name__,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details:
            payload["details"] = self.details
        if self.request_id:
            payload["request_id"] = self.request_id
        if self.trace_id:
            payload["trace_id"] = self.trace_id
        cause = self.__cause__
        if cause is not None:
            payload["caused_by"] = f"{type(cause).__name__}: {cause}"
        return payload

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


# =============================================================================
# LLM (RM-LLM-xxx) — raised by LLMService implementations
# =============================================================================


class LLMServiceError(ResearchMindError):
    """Base class for all LLM backend failures."""

    error_code: ClassVar[str] = "RM-LLM-000"
    default_message: ClassVar[str] = "The LLM service encountered an error."
    retryable: ClassVar[bool] = False


class LLMGenerationError(LLMServiceError):
    """The LLM provider failed to produce a completion."""

    error_code: ClassVar[str] = "RM-LLM-001"
    default_message: ClassVar[str] = "LLM generation failed."
    retryable: ClassVar[bool] = True


class LLMTimeoutError(LLMServiceError):
    """The LLM provider did not respond within the configured timeout."""

    error_code: ClassVar[str] = "RM-LLM-002"
    default_message: ClassVar[str] = "LLM request timed out."
    retryable: ClassVar[bool] = True


class LLMProviderUnavailableError(LLMServiceError):
    """The configured LLM provider is unreachable (connection refused, DNS, etc.)."""

    error_code: ClassVar[str] = "RM-LLM-003"
    default_message: ClassVar[str] = "LLM provider is unavailable."
    retryable: ClassVar[bool] = True


class LLMOutputValidationError(LLMServiceError):
    """``generate_structured`` could not coerce the provider's output into the requested schema."""

    error_code: ClassVar[str] = "RM-LLM-004"
    default_message: ClassVar[str] = "LLM output did not match the expected schema."
    retryable: ClassVar[bool] = False


class LLMRateLimitError(LLMServiceError):
    """The LLM provider rejected the request due to rate limiting."""

    error_code: ClassVar[str] = "RM-LLM-005"
    default_message: ClassVar[str] = "LLM provider rate limit exceeded."
    retryable: ClassVar[bool] = True


# =============================================================================
# Search (RM-SEARCH-xxx) — raised by SearchProvider implementations
# =============================================================================


class SearchProviderError(ResearchMindError):
    """Base class for all web search backend failures."""

    error_code: ClassVar[str] = "RM-SEARCH-000"
    default_message: ClassVar[str] = "The web search provider encountered an error."
    retryable: ClassVar[bool] = False


class SearchTimeoutError(SearchProviderError):
    """The search provider did not respond within the configured timeout."""

    error_code: ClassVar[str] = "RM-SEARCH-001"
    default_message: ClassVar[str] = "Web search request timed out."
    retryable: ClassVar[bool] = True


class SearchProviderUnavailableError(SearchProviderError):
    """The configured search provider is unreachable."""

    error_code: ClassVar[str] = "RM-SEARCH-002"
    default_message: ClassVar[str] = "Web search provider is unavailable."
    retryable: ClassVar[bool] = True


class SearchQuotaExceededError(SearchProviderError):
    """The search provider rejected the request because the account's quota is exhausted."""

    error_code: ClassVar[str] = "RM-SEARCH-003"
    default_message: ClassVar[str] = "Web search provider quota exceeded."
    retryable: ClassVar[bool] = False


# =============================================================================
# RAG (RM-RAG-xxx) — raised by the ingestion pipeline, VectorStore, and
# EmbeddingProvider implementations
# =============================================================================


class RAGError(ResearchMindError):
    """Base class for all RAG pipeline failures (ingestion, embedding, retrieval)."""

    error_code: ClassVar[str] = "RM-RAG-000"
    default_message: ClassVar[str] = "The RAG pipeline encountered an error."
    retryable: ClassVar[bool] = False


class VectorStoreError(RAGError):
    """The vector store failed to complete an upsert/query/delete operation."""

    error_code: ClassVar[str] = "RM-RAG-001"
    default_message: ClassVar[str] = "Vector store operation failed."
    retryable: ClassVar[bool] = True


class EmbeddingProviderError(RAGError):
    """The embedding provider failed to embed one or more texts."""

    error_code: ClassVar[str] = "RM-RAG-002"
    default_message: ClassVar[str] = "Embedding generation failed."
    retryable: ClassVar[bool] = True


class DocumentExtractionError(RAGError):
    """Text extraction from an uploaded document failed (corrupt/unreadable file)."""

    error_code: ClassVar[str] = "RM-RAG-003"
    default_message: ClassVar[str] = "Failed to extract text from the uploaded document."
    retryable: ClassVar[bool] = False


class DocumentChunkingError(RAGError):
    """Chunking extracted document text failed."""

    error_code: ClassVar[str] = "RM-RAG-004"
    default_message: ClassVar[str] = "Failed to chunk the document."
    retryable: ClassVar[bool] = False


class UnsupportedFileTypeError(RAGError):
    """An uploaded file's type is not one of the supported formats (PDF, DOCX, TXT, MD)."""

    error_code: ClassVar[str] = "RM-RAG-005"
    default_message: ClassVar[str] = "Unsupported file type."
    retryable: ClassVar[bool] = False


class FileTooLargeError(RAGError):
    """An uploaded file exceeds ``RAGSettings.max_upload_size_mb``."""

    error_code: ClassVar[str] = "RM-RAG-006"
    default_message: ClassVar[str] = "Uploaded file exceeds the maximum allowed size."
    retryable: ClassVar[bool] = False


class RetrievalError(RAGError):
    """Top-K retrieval against the vector store failed."""

    error_code: ClassVar[str] = "RM-RAG-007"
    default_message: ClassVar[str] = "Document retrieval failed."
    retryable: ClassVar[bool] = True


# =============================================================================
# Report (RM-REPORT-xxx) — raised by ReportExporter implementations and the
# report builder
# =============================================================================


class ReportExportError(ResearchMindError):
    """Base class for all report generation/export failures."""

    error_code: ClassVar[str] = "RM-REPORT-000"
    default_message: ClassVar[str] = "Report export failed."
    retryable: ClassVar[bool] = False


class ReportRenderingError(ReportExportError):
    """The exporter failed while rendering the report content (e.g. ReportLab layout failure)."""

    error_code: ClassVar[str] = "RM-REPORT-001"
    default_message: ClassVar[str] = "Failed to render the report."
    retryable: ClassVar[bool] = False


class ReportTemplateError(ReportExportError):
    """The report content did not satisfy the structural requirements of the export template."""

    error_code: ClassVar[str] = "RM-REPORT-002"
    default_message: ClassVar[str] = "Report content is incompatible with the export template."
    retryable: ClassVar[bool] = False


# =============================================================================
# Agents (RM-AGENT-xxx) — raised by individual LangGraph nodes and the
# Coordinator's graph-execution logic
# =============================================================================


class AgentError(ResearchMindError):
    """Base class for all multi-agent workflow failures."""

    error_code: ClassVar[str] = "RM-AGENT-000"
    default_message: ClassVar[str] = "An agent encountered an error."
    retryable: ClassVar[bool] = False


class PlannerError(AgentError):
    """The Planner agent failed to produce a research plan."""

    error_code: ClassVar[str] = "RM-AGENT-001"
    default_message: ClassVar[str] = "Planner agent failed."


class ResearcherError(AgentError):
    """The Researcher agent failed to consolidate research sub-results."""

    error_code: ClassVar[str] = "RM-AGENT-002"
    default_message: ClassVar[str] = "Researcher agent failed."


class RetrieverAgentError(AgentError):
    """The Retriever agent failed to fetch context from the RAG pipeline."""

    error_code: ClassVar[str] = "RM-AGENT-003"
    default_message: ClassVar[str] = "Retriever agent failed."


class SearchAgentError(AgentError):
    """The Search agent failed to fetch context from the web search tool."""

    error_code: ClassVar[str] = "RM-AGENT-004"
    default_message: ClassVar[str] = "Search agent failed."


class FactCheckerError(AgentError):
    """The Fact Checker agent failed to verify claims against retrieved sources."""

    error_code: ClassVar[str] = "RM-AGENT-005"
    default_message: ClassVar[str] = "Fact Checker agent failed."


class WriterError(AgentError):
    """The Writer agent failed to synthesize the report draft."""

    error_code: ClassVar[str] = "RM-AGENT-006"
    default_message: ClassVar[str] = "Writer agent failed."


class ReviewerError(AgentError):
    """The Reviewer agent failed to evaluate the report draft."""

    error_code: ClassVar[str] = "RM-AGENT-007"
    default_message: ClassVar[str] = "Reviewer agent failed."


class CoordinatorError(AgentError):
    """The Coordinator failed to orchestrate the agent graph."""

    error_code: ClassVar[str] = "RM-AGENT-008"
    default_message: ClassVar[str] = "Coordinator failed to orchestrate the research workflow."


class MaxRetriesExceededError(AgentError):
    """The bounded Reviewer -> Writer revision loop hit ``AgentSettings.max_reviewer_retries``."""

    error_code: ClassVar[str] = "RM-AGENT-009"
    default_message: ClassVar[str] = "Maximum revision retries exceeded."
    retryable: ClassVar[bool] = False


class GraphExecutionError(AgentError):
    """The LangGraph run failed outside of any single node's own error handling."""

    error_code: ClassVar[str] = "RM-AGENT-010"
    default_message: ClassVar[str] = "Agent graph execution failed."


class AgentTimeoutError(AgentError):
    """A node exceeded ``AgentSettings.agent_timeout_seconds``."""

    error_code: ClassVar[str] = "RM-AGENT-011"
    default_message: ClassVar[str] = "Agent execution timed out."
    retryable: ClassVar[bool] = True


# =============================================================================
# Memory (RM-MEMORY-xxx) — raised by conversation memory / session management
# =============================================================================


class ConversationMemoryError(ResearchMindError):
    """
    Base class for conversation memory / session failures.

    Named ``ConversationMemoryError`` rather than ``MemoryError`` to avoid
    shadowing Python's built-in ``MemoryError`` (raised on actual
    out-of-memory conditions), which would be a confusing collision.
    """

    error_code: ClassVar[str] = "RM-MEMORY-000"
    default_message: ClassVar[str] = "Conversation memory operation failed."
    retryable: ClassVar[bool] = True


class SessionNotFoundError(ConversationMemoryError):
    """The referenced research session does not exist."""

    error_code: ClassVar[str] = "RM-MEMORY-001"
    default_message: ClassVar[str] = "Research session not found."
    retryable: ClassVar[bool] = False


class ChatHistoryLimitExceededError(ConversationMemoryError):
    """The session's chat history exceeds ``AgentSettings.max_chat_history``."""

    error_code: ClassVar[str] = "RM-MEMORY-002"
    default_message: ClassVar[str] = "Chat history limit exceeded for this session."
    retryable: ClassVar[bool] = False


# =============================================================================
# Database (RM-DB-xxx) — raised by repositories
# =============================================================================


class DatabaseError(ResearchMindError):
    """Base class for all relational-database failures."""

    error_code: ClassVar[str] = "RM-DB-000"
    default_message: ClassVar[str] = "A database error occurred."
    retryable: ClassVar[bool] = True


class RecordNotFoundError(DatabaseError):
    """A repository lookup by id found no matching row."""

    error_code: ClassVar[str] = "RM-DB-001"
    default_message: ClassVar[str] = "Record not found."
    retryable: ClassVar[bool] = False


class RecordConflictError(DatabaseError):
    """
    A write violated a uniqueness or other integrity constraint (e.g. a
    duplicate checksum within a session, a duplicate sequence number).

    Not retryable: unlike a transient connection failure, retrying the
    same write will fail again with the same conflict until the caller
    changes the data.
    """

    error_code: ClassVar[str] = "RM-DB-002"
    default_message: ClassVar[str] = "The write conflicts with an existing record."
    retryable: ClassVar[bool] = False


# =============================================================================
# API (RM-API-xxx) — raised at the service/router boundary; mapped to HTTP
# status codes exclusively by app.api.middleware.error_handler
# =============================================================================


class APIError(ResearchMindError):
    """Base class for request-facing failures raised at the API boundary."""

    error_code: ClassVar[str] = "RM-API-000"
    default_message: ClassVar[str] = "The request could not be processed."
    retryable: ClassVar[bool] = False


class ResourceNotFoundError(APIError):
    """A requested resource (document, session, report) does not exist."""

    error_code: ClassVar[str] = "RM-API-001"
    default_message: ClassVar[str] = "The requested resource was not found."
    retryable: ClassVar[bool] = False


class RequestValidationError(APIError):
    """
    A request failed application-level validation not already covered by
    Pydantic schema validation (e.g. a cross-field business rule).

    Named ``RequestValidationError`` rather than ``ValidationError`` to
    avoid colliding with ``pydantic.ValidationError``, which callers are
    likely to have imported in the same module.
    """

    error_code: ClassVar[str] = "RM-API-002"
    default_message: ClassVar[str] = "Request validation failed."
    retryable: ClassVar[bool] = False


class RateLimitError(APIError):
    """The caller exceeded an application-level rate limit."""

    error_code: ClassVar[str] = "RM-API-003"
    default_message: ClassVar[str] = "Rate limit exceeded."
    retryable: ClassVar[bool] = True


class ConfigurationError(APIError):
    """A request could not be served because of an invalid runtime configuration state."""

    error_code: ClassVar[str] = "RM-API-004"
    default_message: ClassVar[str] = "The service is misconfigured for this request."
    retryable: ClassVar[bool] = False


# =============================================================================
# Auth (RM-AUTH-xxx) — reserved for future authentication/authorization
# support (see core/dependencies.py and api/middleware/auth_placeholder.py).
# Not raised anywhere yet; defined now so the error-code space is reserved
# and downstream error handling/logging code can reference these types
# ahead of the actual auth implementation.
# =============================================================================


class AuthenticationError(APIError):
    """The request's credentials are missing or invalid. Reserved for future use."""

    error_code: ClassVar[str] = "RM-AUTH-001"
    default_message: ClassVar[str] = "Authentication failed."
    retryable: ClassVar[bool] = False


class AuthorizationError(APIError):
    """The caller is authenticated but not permitted to perform this action. Reserved for future use."""

    error_code: ClassVar[str] = "RM-AUTH-002"
    default_message: ClassVar[str] = "You are not authorized to perform this action."
    retryable: ClassVar[bool] = False


# =============================================================================
# Fail-fast self-check: guarantee every error_code in the hierarchy is unique.
# Runs once at import time so a copy-paste mistake (two classes reusing the
# same code) is caught immediately in CI/at process startup, not discovered
# later when two unrelated failures show up under the same code in a log
# dashboard.
# =============================================================================


def _collect_and_validate_error_codes(root: type) -> Dict[str, str]:
    codes: Dict[str, str] = {}

    def _walk(cls: type) -> None:
        for subclass in cls.__subclasses__():
            code = subclass.__dict__.get("error_code")
            if code is not None:
                if code in codes and codes[code] != subclass.__name__:
                    raise AssertionError(
                        f"Duplicate error_code {code!r} used by both "
                        f"{codes[code]!r} and {subclass.__name__!r} in "
                        "app.core.exceptions. Every exception class must "
                        "have a unique error_code."
                    )
                codes[code] = subclass.__name__
            _walk(subclass)

    _walk(root)
    return codes


ERROR_CODE_REGISTRY: Dict[str, str] = _collect_and_validate_error_codes(ResearchMindError)
