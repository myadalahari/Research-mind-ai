"""
SearchProvider interface.

The Search Agent depends on this abstraction, not on any specific web
search vendor's SDK or HTTP API. Tavily is the concrete implementation
selected today (``TavilySearchProvider``, chosen for its LLM/RAG-oriented
result format and clean LangChain integration), but the interface is
designed so SerpAPI, Brave Search, DuckDuckGo, or Bing can be dropped in
later as an alternate ``SearchProvider`` implementation with zero changes
to the Search Agent or the LangGraph node that calls it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from app.utils.time import utc_now


class SearchResult(BaseModel):
    """
    A single, provider-agnostic web search result.

    Every concrete ``SearchProvider`` normalizes its raw API response into
    a list of these, so the Search Agent, Fact Checker, and Writer never
    need to know which provider actually served a given result. This is
    also the shape citations are built from, so ``url`` and ``title`` are
    required — a search result the system can't cite back to its source is
    not usable in a report.
    """

    title: str = Field(..., description="Title of the source page.")
    url: str = Field(..., description="Canonical URL of the source page.")
    snippet: str = Field(..., description="Short excerpt/summary returned by the search provider.")
    content: Optional[str] = Field(
        default=None,
        description="Extended extracted content, if the provider supports full-content retrieval.",
    )
    score: Optional[float] = Field(default=None, description="Provider-reported relevance score, if available.")
    published_date: Optional[datetime] = Field(
        default=None, description="Publication date of the source, if the provider reports one."
    )
    source_domain: Optional[str] = Field(
        default=None, description="Domain the result was published on, e.g. 'nature.com'."
    )
    retrieved_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp this result was fetched by ResearchMind, used for citation freshness.",
    )


class SearchResponse(BaseModel):
    """The full result of a single search call, including provider metadata."""

    query: str = Field(..., description="The query string that was searched.")
    provider: str = Field(..., description="Name of the provider that served this search, e.g. 'tavily'.")
    results: List[SearchResult] = Field(default_factory=list)
    latency_ms: float = Field(..., description="Wall-clock time the search call took, in milliseconds.")
    answer: Optional[str] = Field(
        default=None,
        description=(
            "A provider-generated direct answer/summary for the query, if the provider "
            "supports it (Tavily does). Optional — callers must not assume it is present."
        ),
    )


class SearchProvider(ABC):
    """
    Abstract interface for a web search backend.

    Concrete implementations live under ``app.services.search`` (e.g.
    ``TavilySearchProvider``). The Search Agent depends on this interface,
    injected via ``app.core.dependencies.get_search_provider``, and must
    never import a provider SDK directly.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> SearchResponse:
        """
        Execute a web search and return normalized results.

        Args:
            query: The search query.
            max_results: Maximum number of results to return; falls back
                to the provider adapter's configured default when omitted.
            include_domains: Optional allow-list of domains to restrict
                results to, if the provider supports it.
            exclude_domains: Optional deny-list of domains to exclude, if
                the provider supports it.
            **kwargs: Provider-specific overrides; implementations must
                accept and ignore keys they don't understand.

        Returns:
            A populated ``SearchResponse``.

        Raises:
            SearchProviderError: on provider failure after configured
                retries are exhausted (see ``app.core.exceptions``).
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Return ``True`` if the search provider is reachable and configured
        with valid credentials, ``False`` otherwise. Never raises — used by
        ``GET /health`` and by the Coordinator to decide whether to route
        around the Search Agent when the feature is degraded rather than
        fully disabled via ``FeatureFlags.enable_web_search``.
        """
        raise NotImplementedError
