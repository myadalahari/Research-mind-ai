"""
Request/response schemas for ``POST /upload``.

The file itself travels as multipart form data (handled by FastAPI's
``UploadFile``, not a JSON body), so there is no ``UploadRequest`` model
here in the usual sense — ``UploadMetadataForm`` documents the
accompanying form fields the router accepts alongside the file.

Two state machines are modeled separately and deliberately kept apart:

* ``UploadStatus`` — the file *transfer* lifecycle (did the bytes arrive,
  pass validation, get stored). This can fail for reasons that have
  nothing to do with RAG (wrong extension, file too large, storage I/O
  error) and typically resolves within the request/response cycle itself.
* ``IngestionStatus`` — the RAG *processing* lifecycle (extract, clean,
  chunk, embed, index) that runs after the file is safely stored, may take
  longer than the HTTP request, and fails for entirely different reasons
  (corrupt PDF content, embedding model failure).

Collapsing these into one status enum would make it impossible to tell
"the upload itself failed" from "the file is stored but indexing is still
in progress" — a distinction the frontend's upload panel needs to render
correctly (e.g. showing a spinner for the latter, a hard error for the
former).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated

from app.schemas.common import ErrorBody
from app.utils.time import utc_now

# =============================================================================
# Status enums
# =============================================================================


class UploadStatus(str, Enum):
    """Lifecycle of the raw file transfer/storage, independent of RAG processing."""

    PENDING = "pending"
    RECEIVED = "received"
    VALIDATED = "validated"
    STORED = "stored"
    REJECTED = "rejected"
    FAILED = "failed"


class IngestionStatus(str, Enum):
    """Lifecycle of the RAG ingestion pipeline for a stored document."""

    NOT_STARTED = "not_started"
    QUEUED = "queued"
    EXTRACTING = "extracting"
    CLEANING = "cleaning"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Reusable document metadata
#
# Deliberately self-contained (no upload-specific fields) so it can be
# reused as-is by schemas/sources.py's GET /sources response later,
# rather than that endpoint redefining an equivalent shape.
# =============================================================================


class DocumentMetadata(BaseModel):
    """File-level metadata for an uploaded document, independent of ingestion state."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "filename": "qwen3_technical_report.pdf",
                    "file_extension": ".pdf",
                    "mime_type": "application/pdf",
                    "size_bytes": 2_458_112,
                    "checksum_sha256": "9f2c1e...a41b",
                    "uploaded_at": "2026-07-29T10:15:03Z",
                    "session_id": "sess-7f3a1c9d",
                }
            ]
        }
    )

    filename: str = Field(..., description="Original filename as uploaded by the client.")
    file_extension: str = Field(..., description="Lowercased file extension including the dot, e.g. '.pdf'.")
    mime_type: str = Field(..., description="Detected MIME type of the uploaded file.")
    size_bytes: Annotated[int, Field(ge=0)] = Field(..., description="File size in bytes.")
    checksum_sha256: str = Field(
        ..., description="SHA-256 hex digest of the file content, used for deduplication and integrity checks."
    )
    uploaded_at: datetime = Field(default_factory=utc_now, description="When the file was received.")
    session_id: str = Field(..., description="Research session this document was uploaded into.")


class ProcessingStatistics(BaseModel):
    """
    Observability metrics for a completed (or in-progress) ingestion run.

    Populated incrementally as ingestion advances; fields for stages not
    yet reached remain ``None`` rather than being zero-filled, so a client
    can distinguish "this stage hasn't run yet" from "this stage took 0ms".
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "processing_time_ms": 4210.5,
                    "extraction_time_ms": 820.0,
                    "chunking_time_ms": 140.5,
                    "embedding_time_ms": 3250.0,
                    "extracted_text_length": 48213,
                    "chunk_count": 62,
                    "average_chunk_size": 777.6,
                    "embedding_model": "all-MiniLM-L6-v2",
                    "embedding_dimension": 384,
                }
            ]
        }
    )

    processing_time_ms: Optional[float] = Field(
        default=None, ge=0, description="Total wall-clock time for the ingestion pipeline, end to end."
    )
    extraction_time_ms: Optional[float] = Field(
        default=None, ge=0, description="Time spent extracting text from the source file."
    )
    chunking_time_ms: Optional[float] = Field(default=None, ge=0, description="Time spent chunking extracted text.")
    embedding_time_ms: Optional[float] = Field(
        default=None, ge=0, description="Time spent generating chunk embeddings."
    )
    extracted_text_length: Optional[int] = Field(
        default=None, ge=0, description="Character count of the cleaned, extracted text."
    )
    chunk_count: Optional[int] = Field(default=None, ge=0, description="Number of chunks produced.")
    average_chunk_size: Optional[float] = Field(default=None, ge=0, description="Average chunk length in characters.")
    embedding_model: Optional[str] = Field(default=None, description="Embedding model used, e.g. 'all-MiniLM-L6-v2'.")
    embedding_dimension: Optional[int] = Field(
        default=None, ge=1, description="Dimensionality of the generated embeddings."
    )


# =============================================================================
# Request (accompanying form fields; the file itself is multipart)
# =============================================================================


class UploadMetadataForm(BaseModel):
    """
    Form fields accompanying the uploaded file in a ``POST /upload``
    multipart request. This model documents that contract and provides
    validation; the router does not accept it directly as a single
    parameter, though.

    Implementation note for the Phase 4 router: FastAPI 0.141's combined
    ``UploadFile`` + Pydantic-model-``Form()`` parameter support has a
    verified bug where the model parameter resolves to ``None``/a 422
    "field required" error when mixed with a ``File()`` parameter in the
    same route (reproducible independent of anything in this schema).
    The reliable, version-safe pattern is to declare each field as its own
    ``Form(default=...)`` parameter in the router signature and construct
    ``UploadMetadataForm(session_id=..., document_title=..., tags=...)``
    from them inside the handler — still gets this model's validation,
    just not as a single injected parameter.
    """

    session_id: Optional[str] = Field(
        default=None, description="Existing research session to attach this document to. Omit to start a new session."
    )
    document_title: Optional[Annotated[str, Field(max_length=200)]] = Field(
        default=None, description="Optional client-supplied display title, distinct from the original filename."
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Optional freeform tags for later filtering in GET /sources."
    )


# =============================================================================
# Response
# =============================================================================


class UploadResponse(BaseModel):
    """Response payload for a single uploaded document (wrapped in ``DataResponse[UploadResponse]``)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_id": "doc-42",
                    "session_id": "sess-7f3a1c9d",
                    "upload_status": "stored",
                    "ingestion_status": "completed",
                    "indexed": True,
                    "metadata": {},
                    "processing_stats": {},
                    "error": None,
                    "created_at": "2026-07-29T10:15:03Z",
                    "updated_at": "2026-07-29T10:15:07Z",
                }
            ]
        }
    )

    document_id: str = Field(
        ..., description="Unique identifier for this document, used by RetrievalOptions.document_ids."
    )
    session_id: str = Field(..., description="Research session this document belongs to.")
    upload_status: UploadStatus = Field(..., description="Status of the raw file transfer/storage.")
    ingestion_status: IngestionStatus = Field(
        ..., description="Status of the RAG ingestion pipeline for this document."
    )
    indexed: bool = Field(
        ...,
        description="Whether this document's chunks are currently searchable via retrieval. True only once ingestion_status='completed'.",
    )
    metadata: DocumentMetadata = Field(..., description="File-level metadata.")
    processing_stats: Optional[ProcessingStatistics] = Field(
        default=None,
        description="Ingestion observability metrics. Populated once ingestion reaches at least the chunking stage.",
    )
    error: Optional[ErrorBody] = Field(
        default=None,
        description="Populated when upload_status or ingestion_status is 'failed', using the same error shape as API error responses.",
    )
    created_at: datetime = Field(default_factory=utc_now, description="When this document record was created.")
    updated_at: datetime = Field(default_factory=utc_now, description="When this document record was last updated.")

    @model_validator(mode="after")
    def _validate_status_consistency(self) -> "UploadResponse":
        failed = self.upload_status == UploadStatus.FAILED or self.ingestion_status == IngestionStatus.FAILED
        if failed and self.error is None:
            raise ValueError("error must be set when upload_status or ingestion_status is 'failed'.")
        if self.indexed and self.ingestion_status != IngestionStatus.COMPLETED:
            raise ValueError("indexed=true requires ingestion_status='completed'.")
        if self.ingestion_status == IngestionStatus.COMPLETED and self.processing_stats is None:
            raise ValueError("processing_stats must be populated once ingestion_status='completed'.")
        return self


class BatchUploadResponse(BaseModel):
    """
    Response payload for a (future) multi-file upload request.

    Not yet wired to a route — ``POST /upload`` currently accepts one file
    per request and returns ``DataResponse[UploadResponse]`` directly — but
    defined now so accepting multiple files later is an additive change
    (a new route returning ``DataResponse[BatchUploadResponse]``) rather
    than a redesign of the single-file response shape.
    """

    results: List[UploadResponse] = Field(..., description="Per-file results, in the order the files were received.")
    total_count: int = Field(..., ge=0, description="Total number of files submitted in this batch.")
    accepted_count: int = Field(..., ge=0, description="Number of files that were stored/queued successfully.")
    rejected_count: int = Field(..., ge=0, description="Number of files that were rejected or failed outright.")

    @classmethod
    def create(cls, results: List[UploadResponse]) -> "BatchUploadResponse":
        """Build a ``BatchUploadResponse`` from individual results, computing the summary counts."""
        rejected = sum(1 for r in results if r.upload_status in (UploadStatus.REJECTED, UploadStatus.FAILED))
        return cls(
            results=results,
            total_count=len(results),
            accepted_count=len(results) - rejected,
            rejected_count=rejected,
        )
