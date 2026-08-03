"""
``TavilySearchProvider`` -- the concrete ``SearchProvider`` selected for
this project.

See ``app.core.interfaces.search_provider`` for the interface contract and
the reasoning behind choosing Tavily (LLM/RAG-oriented result format) over
a general-purpose search API.

Built on the official ``tavily-python`` package's ``AsyncTavilyClient``
rather than hand-rolled HTTP/JSON, matching the same choice already made
for every other adapter in this codebase.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Awaitable, Callable, List, Optional, TypeVar
from urllib.parse import urlparse

import httpx
import tavily.errors as tavily_errors
from tavily import AsyncTavilyClient

from app.core.config import SearchSettings
from app.core.exceptions import (
    SearchProviderError,
    SearchProviderUnavailableError,
    SearchQuotaExceededError,
    SearchTimeoutError,
)
from app.core.interfaces.search_provider import SearchProvider, SearchResponse, SearchResult
from app.core.logging import get_logger, measure_latency_ms

logger = get_logger(__name__)

_DEFAULT_TAVILY_BASE_URL = "https://api.tavily.com"
_MAX_BACKOFF_SECONDS = 4.0

_T = TypeVar("_T")

# Credential/permission/malformed-request failures: retrying the identical
# request cannot succeed, so all of these map to the base SearchProviderError
# with retryable=False rather than one of the more specific subclasses.
_NON_RETRYABLE_TAVILY_ERRORS = (
    tavily_errors.InvalidAPIKeyError,
    tavily_errors.MissingAPIKeyError,
    tavily_errors.ForbiddenError,
    tavily_errors.BadRequestError,
    tavily_errors.KeylessUnsupportedEndpointError,
)


class TavilySearchProvider(SearchProvider):
    """
    Web search backend via the Tavily API.

    One ``AsyncTavilyClient`` is constructed once and reused for the
    process's lifetime, mirroring every other adapter in this codebase.
    Constructed once by ``app.core.dependencies`` at application startup.
    """

    def __init__(self, settings: SearchSettings) -> None:
        if settings.max_retries < 1:
            raise ValueError(f"SEARCH__MAX_RETRIES must be at least 1, got {settings.max_retries}.")
        self._settings = settings
        self._base_url = settings.base_url or _DEFAULT_TAVILY_BASE_URL
        # Tavily's SDK natively supports keyless mode (a real, documented,
        # rate-limited fallback) -- passing api_key=None straight through
        # when unconfigured rather than rejecting construction matches what
        # Settings' own enable_web_search/tavily_api_key validator already
        # encodes: an API key is only *required* in non-local environments.
        api_key = settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else None
        self._client = AsyncTavilyClient(api_key=api_key, api_base_url=self._base_url)
        logger.info(
            "Tavily search provider configured",
            extra={"base_url": self._base_url, "keyless": api_key is None},
        )

    async def search(
        self,
        query: str,
        *,
        max_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> SearchResponse:
        effective_max_results = max_results if max_results is not None else self._settings.max_results

        with measure_latency_ms() as elapsed:
            raw_response = await self._retry_transient(
                lambda: self._search_once(
                    query,
                    max_results=effective_max_results,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                )
            )

        results = [_to_search_result(raw) for raw in raw_response.get("results", [])]
        return SearchResponse(
            query=query,
            provider="tavily",
            results=results,
            latency_ms=elapsed(),
            answer=raw_response.get("answer"),
        )

    async def health_check(self) -> bool:
        """
        Confirm the Tavily API is network-reachable via a bare GET to its
        base URL -- deliberately *not* a real search call. Unlike Ollama's
        ``client.list()`` (free) or the embedding provider's already-
        resident-model inference (effectively free), a real Tavily search
        consumes a billed API credit. A ``/health`` endpoint that may be
        polled frequently should not burn through search quota just to
        prove reachability -- this trades "confirms the API key is valid"
        for "confirms the network path is reachable," the same tradeoff
        spirit as every other adapter's health check, just cost-driven
        here instead of latency-driven. Never raises.
        """
        try:
            async with httpx.AsyncClient(timeout=min(self._settings.timeout_seconds, 10)) as client:
                await client.get(self._base_url)
            return True
        except Exception:
            logger.exception("Tavily search provider health check failed")
            return False

    # =========================================================================
    # Internals
    # =========================================================================

    async def _search_once(
        self,
        query: str,
        *,
        max_results: int,
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]],
    ) -> dict:
        """One raw call to the Tavily search endpoint, translating exceptions. No retry here."""
        try:
            return await self._client.search(
                query=query,
                max_results=max_results,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                include_answer=True,
                timeout=self._settings.timeout_seconds,
            )
        except tavily_errors.TimeoutError as exc:
            raise SearchTimeoutError.wrap(exc, "Tavily search request timed out.") from exc
        except tavily_errors.UsageLimitExceededError as exc:
            # Covers both UsageLimitExceededError and its TavilyKeylessLimitError subclass.
            raise SearchQuotaExceededError.wrap(exc, f"Tavily quota exceeded: {exc}") from exc
        except _NON_RETRYABLE_TAVILY_ERRORS as exc:
            raise SearchProviderError.wrap(exc, f"Tavily rejected the request: {exc}", retryable=False) from exc
        except httpx.HTTPStatusError as exc:
            # Tavily's own SDK only translates 400/401/403/429/432/433 into
            # its named exceptions (verified against its source) and lets
            # every other non-2xx status -- notably 5xx -- surface as a raw
            # httpx.HTTPStatusError via response.raise_for_status().
            if exc.response.status_code >= 500:
                raise SearchProviderUnavailableError.wrap(
                    exc, f"Tavily server error (status {exc.response.status_code})."
                ) from exc
            raise SearchProviderError.wrap(
                exc, f"Tavily returned an unexpected error (status {exc.response.status_code}).", retryable=False
            ) from exc
        except (httpx.ConnectError, httpx.TransportError) as exc:
            # Tavily's SDK only catches httpx.TimeoutException internally
            # (verified against its source) -- a connection failure (no
            # server reachable at all) propagates as a raw httpx exception.
            raise SearchProviderUnavailableError.wrap(exc, "Could not reach the Tavily API.") from exc
        except Exception as exc:
            raise SearchProviderError.wrap(exc, f"Unexpected Tavily failure: {exc}") from exc

    async def _retry_transient(self, attempt_fn: Callable[[], Awaitable[_T]]) -> _T:
        """Retry ``attempt_fn`` while it keeps raising a retryable ``SearchProviderError``, bounded by ``max_retries``."""
        for attempt in range(1, self._settings.max_retries + 1):
            try:
                return await attempt_fn()
            except SearchProviderError as exc:
                if not exc.retryable or attempt == self._settings.max_retries:
                    raise
                logger.warning(
                    "Transient search failure, retrying",
                    extra={"error_code": exc.error_code, "attempt": attempt},
                )
                await asyncio.sleep(_backoff_delay(attempt))
        raise AssertionError("unreachable: __init__ guarantees max_retries >= 1")


def _to_search_result(raw: dict) -> SearchResult:
    url = raw.get("url", "")
    return SearchResult(
        title=raw.get("title") or url or "Untitled",
        url=url,
        snippet=raw.get("content") or "",
        content=raw.get("raw_content"),
        score=raw.get("score"),
        published_date=_parse_published_date(raw.get("published_date")),
        source_domain=urlparse(url).netloc or None,
    )


def _parse_published_date(value: Optional[str]) -> Optional[datetime]:
    """
    Best-effort parse of Tavily's ``published_date`` string.

    This is untrusted external data with no strictly guaranteed format --
    a malformed or unexpected date string should degrade to "no date"
    rather than crash citation-building for an otherwise-good result.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.warning("Could not parse Tavily published_date, dropping it", extra={"raw_value": value})
        return None


def _backoff_delay(attempt: int) -> float:
    """Capped exponential backoff: 0.5s, 1s, 2s, 4s, 4s, ..."""
    return min(0.5 * (2 ** (attempt - 1)), _MAX_BACKOFF_SECONDS)
