"""
LLMService interface.

Every code path in ResearchMind AI that needs an LLM completion — agents,
services, prompt-testing utilities — depends on this abstract interface,
never on a concrete provider SDK (``ollama``, ``openai``, ``anthropic``,
etc.) directly. The concrete adapter actually used at runtime is selected
by ``Settings.llm.provider`` and constructed once in
``app.core.dependencies``.

This is what makes the LLM backend swappable (Ollama today, OpenAI/
Anthropic/Azure OpenAI later) without touching a single agent or service:
swapping providers means adding a new adapter class and one branch in the
dependency-injection factory, not editing business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional, Type, TypeVar

from pydantic import BaseModel, Field

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMResponse(BaseModel):
    """
    Provider-agnostic result of a single LLM generation call.

    Every concrete ``LLMService`` implementation normalizes its provider's
    raw response into this shape, so callers (agents, services) can rely on
    a consistent contract — including token usage, which is required for
    the token-usage-tracking observability requirement — regardless of
    which provider actually served the request.
    """

    content: str = Field(..., description="The generated text.")
    model: str = Field(..., description="The concrete model identifier that served this request.")
    prompt_tokens: Optional[int] = Field(
        default=None, description="Tokens consumed by the prompt, if reported by the provider."
    )
    completion_tokens: Optional[int] = Field(
        default=None, description="Tokens generated in the completion, if reported by the provider."
    )
    total_tokens: Optional[int] = Field(
        default=None, description="prompt_tokens + completion_tokens, if reported by the provider."
    )
    latency_ms: float = Field(..., description="Wall-clock time the generation call took, in milliseconds.")
    finish_reason: Optional[str] = Field(
        default=None, description="Provider-reported reason generation stopped (e.g. 'stop', 'length')."
    )
    raw: Optional[dict] = Field(
        default=None,
        description=(
            "The provider's raw response payload, retained for debugging/audit. "
            "Not part of the stable contract — do not branch business logic on it."
        ),
    )


class LLMService(ABC):
    """
    Abstract interface for a large language model backend.

    Concrete implementations live under ``app.services.llm`` (e.g.
    ``OllamaLLMService``). Agents and services must type-hint against this
    interface, not a concrete class, and must be constructed with an
    ``LLMService`` injected via ``app.core.dependencies.get_llm_service`` —
    never by importing a provider SDK directly.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a single text completion.

        Args:
            prompt: The user/task prompt.
            system_prompt: Optional system-level instruction, if the
                underlying provider distinguishes system from user turns.
            temperature: Sampling temperature; falls back to the provider
                adapter's configured default when omitted.
            max_tokens: Maximum tokens to generate; falls back to the
                provider adapter's configured default when omitted.
            **kwargs: Provider-specific overrides. Implementations must
                accept and ignore keys they don't understand rather than
                raising, so callers can pass optional hints without
                coupling to a specific provider.

        Returns:
            A populated ``LLMResponse``.

        Raises:
            LLMServiceError: on provider failure after configured retries
                are exhausted (see ``app.core.exceptions``).
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: Type[SchemaT],
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> SchemaT:
        """
        Generate a completion constrained to a Pydantic schema.

        Used wherever an agent needs structured, machine-parseable output
        rather than free text — e.g. the Planner's research plan, the Fact
        Checker's per-claim verdicts. Implementations are responsible for
        whatever provider-specific mechanism achieves this (function
        calling, JSON mode, grammar-constrained decoding, or a
        parse-and-retry loop) and must return a validated instance of
        ``schema``, never raw text.

        Raises:
            LLMServiceError: on provider failure.
            LLMOutputValidationError: if the provider's output cannot be
                coerced into ``schema`` after configured retries.
        """
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a text completion incrementally.

        Returns an async iterator yielding content chunks as they are
        produced. Intended for latency-sensitive interactive paths (e.g. a
        future streaming chat endpoint); the core agent graph uses
        ``generate``/``generate_structured`` since it needs complete,
        structured output per node rather than a token stream.
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Return ``True`` if the underlying provider is reachable and ready
        to serve requests, ``False`` otherwise. Never raises — used by the
        ``GET /health`` endpoint, which must return promptly even when a
        dependency is down.
        """
        raise NotImplementedError
