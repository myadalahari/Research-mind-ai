"""
Request/response schemas for ``POST /chat``.

``/chat`` is intentionally the single entry point for every conversational
and research workflow — ``ChatRequest.mode`` selects the behavior
(lightweight chat, full multi-agent research, or report-oriented research)
rather than the API growing a new endpoint per workflow. ``POST /report``
(``schemas/report.py``, next) is a thin convenience wrapper that issues the
same underlying request with ``mode="report"`` and additionally persists/
exports the result — the two endpoints share this request/response
vocabulary rather than duplicating it.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated

from app.schemas.common import AgentExecutionTrace, Citation, TokenUsage
from app.utils.time import utc_now

# =============================================================================
# Request
# =============================================================================


class ChatMode(str, Enum):
    """
    Selects which workflow a ``/chat`` request runs.

    * ``chat`` — lightweight, conversational. May be served from memory/a
      single LLM call for simple follow-ups rather than the full
      Planner→...→Reviewer graph, when the Coordinator determines the
      full pipeline isn't warranted (e.g. "what did you mean by that?").
    * ``research`` — the full multi-agent pipeline (Planner, Researcher,
      Retriever, Search, Fact Checker, Writer, Reviewer, Coordinator)
      producing a grounded, cited answer. The default mode.
    * ``report`` — runs the same full pipeline as ``research`` but signals
      the Writer/Reviewer to produce report-structured output (executive
      summary, sections, tables, conclusion) suitable for export via
      ``POST /report``, rather than a conversational answer.
    """

    CHAT = "chat"
    RESEARCH = "research"
    REPORT = "report"


class RetrievalOptions(BaseModel):
    """
    Configurable retrieval behavior for a single ``/chat`` request.

    All fields are optional overrides — omitted fields fall back to
    ``RAGSettings`` defaults (``RAG__RETRIEVAL_TOP_K``,
    ``RAG__RETRIEVAL_SCORE_THRESHOLD``). Kept as its own model rather than
    flattening onto ``ChatRequest`` so retrieval configuration can grow
    (e.g. per-document-type weighting, hybrid search toggles) without
    widening the top-level request schema.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_ids": ["doc-42", "doc-43"],
                    "session_scope": True,
                    "top_k": 8,
                    "score_threshold": 0.35,
                    "include_web_search": True,
                }
            ]
        }
    )

    document_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Restrict retrieval to these specific uploaded documents, e.g. for "
            "'summarize the uploaded papers' style requests targeting a subset of uploads. "
            "Omit to search across all documents in scope."
        ),
    )
    session_scope: bool = Field(
        default=True,
        description="If true, restrict retrieval to documents uploaded within the current research session.",
    )
    top_k: Optional[Annotated[int, Field(ge=1, le=20)]] = Field(
        default=None, description="Override RAG__RETRIEVAL_TOP_K for this request."
    )
    score_threshold: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default=None, description="Override RAG__RETRIEVAL_SCORE_THRESHOLD for this request."
    )
    include_web_search: Optional[bool] = Field(
        default=None,
        description=(
            "Override whether the Search agent runs for this request. Omit to let the "
            "Planner decide based on the query and FEATURES__ENABLE_WEB_SEARCH."
        ),
    )


class ChatRequest(BaseModel):
    """Request body for ``POST /chat``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "Compare Llama 4 and Qwen3 for enterprise deployment",
                    "session_id": None,
                    "mode": "research",
                    "stream": False,
                    "retrieval_options": {"session_scope": True, "top_k": 6},
                    "conversation_title": None,
                }
            ]
        }
    )

    query: Annotated[str, Field(min_length=1, max_length=4000)] = Field(
        ..., description="The user's question or research request.", examples=["Research AI in Healthcare"]
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Existing research session to continue. Omit to start a new session.",
    )
    mode: ChatMode = Field(default=ChatMode.RESEARCH, description="Which workflow this request should run.")
    stream: bool = Field(
        default=False,
        description=(
            "Request an incrementally streamed response over Server-Sent Events instead of a single JSON body. "
            "Reserved for future use — LLMService.stream() exists at the interface level, but the /chat route "
            "does not yet implement SSE delivery; requesting stream=true currently has no effect on the response shape."
        ),
    )
    retrieval_options: Optional[RetrievalOptions] = Field(
        default=None, description="Configurable retrieval behavior for this request. Omit to use RAGSettings defaults."
    )
    conversation_title: Optional[Annotated[str, Field(max_length=200)]] = Field(
        default=None,
        description="Client-supplied title for a new session. Ignored if session_id refers to an existing session.",
    )

    @model_validator(mode="after")
    def _validate_mode_stream_combination(self) -> "ChatRequest":
        if self.stream and self.mode == ChatMode.REPORT:
            raise ValueError(
                "stream=true is not supported with mode='report'; reports are returned as a complete document."
            )
        return self


# =============================================================================
# Response
# =============================================================================


class ResearchMetadata(BaseModel):
    """
    Metadata about how a ``/chat`` response was produced.

    Surfaces exactly which providers/models served the request and how
    much retrieval/search work went into it, independent of the detailed
    step-by-step ``execution_trace`` — a client that only wants the
    high-level "what was used" summary doesn't need to pay the cost of
    parsing the full trace.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "llm_provider": "ollama",
                    "llm_model": "qwen3",
                    "rag_enabled": True,
                    "web_search_enabled": True,
                    "retrieved_chunk_count": 6,
                    "web_result_count": 4,
                    "documents_considered": ["doc-42"],
                    "total_latency_ms": 4820.5,
                    "token_usage": {"prompt_tokens": 3120, "completion_tokens": 812, "total_tokens": 3932},
                }
            ]
        }
    )

    llm_provider: str = Field(..., description="LLM provider that served this request, e.g. 'ollama'.")
    llm_model: str = Field(..., description="Concrete model identifier used, e.g. 'qwen3'.")
    rag_enabled: bool = Field(..., description="Whether document retrieval ran for this request.")
    web_search_enabled: bool = Field(..., description="Whether web search ran for this request.")
    retrieved_chunk_count: int = Field(default=0, ge=0, description="Number of document chunks retrieved.")
    web_result_count: int = Field(default=0, ge=0, description="Number of web search results retrieved.")
    documents_considered: List[str] = Field(
        default_factory=list, description="IDs of documents that contributed retrieved chunks to this response."
    )
    total_latency_ms: float = Field(..., ge=0, description="End-to-end wall-clock time for this request.")
    token_usage: TokenUsage = Field(..., description="Aggregate LLM token usage across the whole run.")


class ChatResponse(BaseModel):
    """Response payload for ``POST /chat`` (wrapped in ``DataResponse[ChatResponse]``)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "session_id": "sess-7f3a1c9d",
                    "message_id": "msg-4e5b4a1e",
                    "mode": "research",
                    "answer": "Qwen3 and Llama 4 differ primarily in...[1][2]",
                    "citations": [],
                    "conversation_title": "Llama 4 vs Qwen3 comparison",
                    "follow_up_suggestions": [
                        "How do their licensing terms compare?",
                        "Which performs better on long-context tasks?",
                    ],
                    "research_metadata": {},
                    "execution_trace": None,
                    "created_at": "2026-07-29T10:15:03Z",
                }
            ]
        }
    )

    session_id: str = Field(..., description="The research session this response belongs to (new or continued).")
    message_id: str = Field(..., description="Unique identifier for this response turn.")
    mode: ChatMode = Field(..., description="The workflow that actually produced this response.")
    answer: str = Field(..., description="The generated answer, with inline citation markers, e.g. '...[1]'.")
    citations: List[Citation] = Field(
        default_factory=list, description="Sources backing the answer's citation markers."
    )
    conversation_title: Optional[str] = Field(
        default=None, description="Resolved title for this session (client-supplied or auto-generated)."
    )
    follow_up_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Optional agent-suggested follow-up questions, to power a 'suggested next questions' UI affordance.",
    )
    research_metadata: ResearchMetadata = Field(..., description="Summary of providers/models/retrieval used.")
    execution_trace: Optional[AgentExecutionTrace] = Field(
        default=None,
        description=(
            "Full step-by-step agent execution trace, for the frontend's Agent Execution Viewer. "
            "Optional and typically omitted by default so lightweight chat responses aren't forced to pay "
            "the serialization cost of the full trace; populated when the caller requests it "
            "(e.g. via an 'include_trace' query parameter on the route)."
        ),
    )
    created_at: datetime = Field(default_factory=utc_now, description="When this response was generated.")
