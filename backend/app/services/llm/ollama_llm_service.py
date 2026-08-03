"""
``OllamaLLMService`` -- the concrete ``LLMService`` selected for this
project.

See ``app.core.interfaces.llm_service`` for the interface contract and the
reasoning behind choosing a local, self-hostable LLM backend (Ollama
serving Qwen3) over a hosted provider API.

Built on the official ``ollama`` Python package's ``AsyncClient`` rather
than hand-rolled HTTP/JSON, matching the "use the real, maintained client
library" choice already made for ``sentence-transformers`` / ``chromadb``
in Phase 5. Every method uses the chat endpoint (message-based), since
that's the one Ollama exposes structured-output support on
(``format=<json schema>``), which ``generate_structured`` depends on.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, AsyncIterator, Callable, Dict, List, Optional, Type, TypeVar

import httpx
import ollama
from ollama import AsyncClient
from pydantic import ValidationError

from app.core.config import LLMSettings
from app.core.exceptions import (
    LLMGenerationError,
    LLMOutputValidationError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)
from app.core.interfaces.llm_service import LLMResponse, LLMService, SchemaT
from app.core.logging import get_logger, measure_latency_ms

logger = get_logger(__name__)

_MAX_BACKOFF_SECONDS = 4.0

_T = TypeVar("_T")


class OllamaLLMService(LLMService):
    """
    Local LLM backend via a running Ollama server.

    One ``AsyncClient`` is constructed once and reused for the process's
    lifetime, mirroring every other adapter in this codebase. Constructed
    once by ``app.core.dependencies`` at application startup.
    """

    def __init__(self, settings: LLMSettings) -> None:
        if settings.max_retries < 1:
            # The retry loops below (`range(1, max_retries + 1)`) assume at
            # least one attempt is made; a misconfigured 0 would otherwise
            # silently return None from generate()/generate_structured()
            # instead of ever calling Ollama.
            raise ValueError(f"LLM__MAX_RETRIES must be at least 1, got {settings.max_retries}.")
        self._settings = settings
        self._ollama_settings = settings.ollama
        self._client = AsyncClient(host=self._ollama_settings.base_url, timeout=settings.timeout_seconds)
        logger.info(
            "Ollama LLM service configured",
            extra={"base_url": self._ollama_settings.base_url, "model": self._ollama_settings.model},
        )

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = self._build_messages(prompt, system_prompt)
        options = self._build_options(temperature, max_tokens)

        with measure_latency_ms() as elapsed:
            response = await self._retry_transient(lambda: self._chat(messages=messages, options=options))

        prompt_tokens = response.prompt_eval_count
        completion_tokens = response.eval_count
        total_tokens = (
            prompt_tokens + completion_tokens if prompt_tokens is not None and completion_tokens is not None else None
        )
        return LLMResponse(
            content=response.message.content or "",
            model=response.model or self._ollama_settings.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=elapsed(),
            finish_reason=response.done_reason,
            raw=response.model_dump(mode="json"),
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[SchemaT],
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> SchemaT:
        messages = self._build_messages(prompt, system_prompt)
        options = self._build_options(temperature, None)
        json_schema = schema.model_json_schema()

        for attempt in range(1, self._settings.max_retries + 1):
            try:
                response = await self._chat(messages=messages, options=options, format=json_schema)
            except LLMServiceError as exc:
                if not exc.retryable or attempt == self._settings.max_retries:
                    raise
                await asyncio.sleep(_backoff_delay(attempt))
                continue

            content = response.message.content or ""
            try:
                return schema.model_validate_json(content)
            except (ValidationError, json.JSONDecodeError) as exc:
                if attempt == self._settings.max_retries:
                    raise LLMOutputValidationError.wrap(
                        exc,
                        f"Ollama output did not match {schema.__name__} after {attempt} attempt(s): {content[:200]!r}",
                    ) from exc
                logger.warning(
                    "Structured output validation failed, retrying",
                    extra={"schema": schema.__name__, "attempt": attempt},
                )
                await asyncio.sleep(_backoff_delay(attempt))

        raise AssertionError("unreachable: __init__ guarantees max_retries >= 1")

    async def stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        messages = self._build_messages(prompt, system_prompt)
        options = self._build_options(temperature, max_tokens)
        try:
            response_stream = await self._client.chat(
                model=self._ollama_settings.model,
                messages=messages,
                options=options,
                stream=True,
                keep_alive=self._ollama_settings.keep_alive,
            )
            async for chunk in response_stream:
                content = chunk.message.content
                if content:
                    yield content
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError.wrap(exc, "Ollama stream timed out.") from exc
        except ConnectionError as exc:
            raise LLMProviderUnavailableError.wrap(exc, "Could not reach the Ollama server.") from exc
        except ollama.ResponseError as exc:
            raise LLMGenerationError.wrap(exc, f"Ollama returned an error while streaming: {exc.error}") from exc

    async def health_check(self) -> bool:
        """
        Verify the Ollama server is reachable via ``client.list()`` (lists
        locally available models) rather than a real generation call.
        Unlike ``SentenceTransformerEmbeddingProvider.health_check()``,
        which runs real inference because the model is already resident in
        process memory, a generation call here could trigger Ollama
        loading a multi-gigabyte model from disk -- far too expensive for
        a ``/health`` endpoint that must return promptly. Never raises.
        """
        try:
            await self._client.list()
            return True
        except Exception:
            logger.exception("Ollama LLM service health check failed")
            return False

    # =========================================================================
    # Internals
    # =========================================================================

    async def _chat(
        self,
        *,
        messages: List[Dict[str, str]],
        options: Dict[str, Any],
        format: Optional[Dict[str, Any]] = None,
    ) -> ollama.ChatResponse:
        """One raw call to the Ollama chat endpoint, translating exceptions. No retry here."""
        try:
            return await self._client.chat(
                model=self._ollama_settings.model,
                messages=messages,
                options=options,
                format=format,
                keep_alive=self._ollama_settings.keep_alive,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError.wrap(exc, "Ollama request timed out.") from exc
        except ConnectionError as exc:
            # ollama's AsyncClient raises the builtin ConnectionError (not an
            # httpx exception) when it cannot reach the server at all --
            # verified directly against an unreachable host.
            raise LLMProviderUnavailableError.wrap(exc, "Could not reach the Ollama server.") from exc
        except ollama.ResponseError as exc:
            if exc.status_code == 429:
                raise LLMRateLimitError.wrap(exc, "Ollama rate limit exceeded.") from exc
            # 5xx (or no status code available) is presumed transient;
            # other 4xx (e.g. unknown model) will fail identically on retry.
            retryable = exc.status_code >= 500 or exc.status_code == -1
            raise LLMGenerationError.wrap(exc, f"Ollama returned an error: {exc.error}", retryable=retryable) from exc
        except ollama.RequestError as exc:
            raise LLMGenerationError.wrap(exc, f"Invalid request to Ollama: {exc}", retryable=False) from exc

    async def _retry_transient(self, attempt_fn: Callable[[], Awaitable[_T]]) -> _T:
        """Retry ``attempt_fn`` while it keeps raising a retryable ``LLMServiceError``, bounded by ``max_retries``."""
        for attempt in range(1, self._settings.max_retries + 1):
            try:
                return await attempt_fn()
            except LLMServiceError as exc:
                if not exc.retryable or attempt == self._settings.max_retries:
                    raise
                logger.warning(
                    "Transient LLM failure, retrying",
                    extra={"error_code": exc.error_code, "attempt": attempt},
                )
                await asyncio.sleep(_backoff_delay(attempt))
        raise AssertionError("unreachable: __init__ guarantees max_retries >= 1")

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_options(self, temperature: Optional[float], max_tokens: Optional[int]) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "temperature": temperature if temperature is not None else self._ollama_settings.temperature,
            "num_ctx": self._settings.max_context_tokens,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        return options


def _backoff_delay(attempt: int) -> float:
    """Capped exponential backoff: 0.5s, 1s, 2s, 4s, 4s, ..."""
    return min(0.5 * (2 ** (attempt - 1)), _MAX_BACKOFF_SECONDS)
