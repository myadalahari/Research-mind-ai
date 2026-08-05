# Architecture Decision Log — ResearchMind AI

Each entry: the decision, why it was chosen, alternatives considered, and
tradeoffs accepted. Ordered chronologically as the project was built.
This log is written to double as interview preparation material — each
entry should stand on its own as a defensible, explainable choice.

---

## ADR-001: Every external dependency sits behind an interface (`core/interfaces/*.py`)

**Decision:** `LLMService`, `SearchProvider`, `VectorStore`, `EmbeddingProvider`, and
`ReportExporter` are all ABCs. Business logic (services, agents) depends only on
these interfaces; concrete implementations (Ollama, Tavily, ChromaDB, ReportLab) are
injected via `core/dependencies.py`.

**Why:** The project must support swapping providers (Ollama → OpenAI/Anthropic/Azure,
Tavily → SerpAPI/Brave, local Chroma → hosted vector DB) without touching business
logic, and must be testable without live network/model calls.

**Alternatives considered:** Direct coupling to concrete SDKs (simpler initially, but
locks the codebase to one vendor and makes unit testing require real API calls or
heavy mocking of vendor SDKs instead of a small interface).

**Tradeoffs:** More files and more indirection up front; every new capability needs an
interface method added deliberately rather than "just calling the SDK." Accepted
because this is the core differentiator between a tutorial project and a
production-shaped one.

---

## ADR-002: LLM provider selection via `LLM_PROVIDER` + nested Pydantic settings

**Decision:** `LLMSettings` holds one sub-settings block per provider
(`OllamaSettings`, `OpenAISettings`, `AnthropicSettings`, `AzureOpenAISettings`) plus a
`provider: LLMProvider` enum selecting which one `core.dependencies.get_llm_service()`
constructs.

**Why:** Keeps all provider credentials/config centrally validated (cross-field
validators reject e.g. `LLM_PROVIDER=openai` with no API key set) while keeping each
provider's config namespaced and independently extensible.

**Alternatives considered:** A single flat settings block with optional fields for
every provider (messier, no clear ownership of which fields apply when); separate
`.env` files per provider (operationally awkward for a single-process app).

**Tradeoffs:** Settings classes are more verbose. Accepted for the validation and
clarity benefits.

---

## ADR-003: Structured logging with async-safe correlation IDs via `contextvars`

**Decision:** `request_id`/`trace_id` are threaded through `contextvars`
(`correlation_scope()`), not passed as explicit function parameters, and every log
record is enriched with structured fields (`agent_name`, `graph_node`, `model_name`,
`latency_ms`, token counts, etc.) via `bind_context()`.

**Why:** A multi-agent LangGraph pipeline has deep call stacks (Coordinator → Planner
→ Researcher → tools); threading correlation IDs through every function signature
would pollute every interface. `contextvars` (not thread-locals) is required because
the app is async and multiple requests interleave on the same event loop.

**Alternatives considered:** Passing `request_id` explicitly through every function
call (correct but invasive); global mutable state (not async-safe, would leak IDs
across concurrent requests).

**Tradeoffs:** Slightly "spooky action at a distance" — a log call's context isn't
visible at the call site. Mitigated by keeping `bind_context()`/`correlation_scope()`
as the only two entry points, both well-documented.

---

## ADR-004: Custom exception hierarchy with machine-readable codes and centralized HTTP mapping

**Decision:** `ResearchMindError` root, with per-domain branches (`LLMServiceError`,
`AgentError` with one subclass per agent, `RAGError`, `ReportExportError`,
`ConversationMemoryError`, `DatabaseError`, `APIError`). Every exception carries a
unique `error_code` (e.g. `RM-LLM-001`, validated for uniqueness at import time via
`_collect_and_validate_error_codes()`), a `retryable: bool`, and auto-captures
`request_id`/`trace_id` from the logging context. HTTP status mapping lives entirely
in `api/middleware/error_handler.py`, not on the exception classes themselves.

**Why:** Machine-readable codes let clients/monitoring distinguish failure modes
programmatically instead of string-matching messages. Keeping HTTP status mapping out
of the exception hierarchy preserves the framework-agnostic boundary — the same
exceptions could back a CLI or gRPC service without dragging in HTTP semantics.
`retryable` lets calling code (e.g. agent retry loops) make automatic decisions
without knowing the specific exception type.

**Alternatives considered:** FastAPI's built-in `HTTPException` used directly
throughout business logic (couples business logic to HTTP, and gives no
machine-readable code or retryability signal); string-based error codes without a
uniqueness check (would silently collide over time).

**Tradeoffs:** More boilerplate per exception type. Accepted — this is exactly the
kind of "senior engineer" polish the project is meant to demonstrate, and the
import-time uniqueness check was verified to actually catch a deliberately introduced
collision during testing.

---

## ADR-005: `schemas/*` (API DTOs) and `models/*` (persistence) and `core/interfaces/*`
(domain contracts) are deliberately independent, structurally-similar types

**Decision:** e.g. `schemas.chat.ChatMode`, `models.enums.ChatMode`, and no
persistence-layer import of anything from `schemas/`. Similarly `Citation` exists
independently in `core.interfaces.report_exporter` (domain) and would exist again in
`schemas.common` (DTO) rather than being shared.

**Why:** Clean Architecture's dependency rule: inner layers (persistence, domain)
must not depend on outer layers (API). If `models/enums.py` imported
`schemas.chat.ChatMode`, a change to the public API's enum values (an outer-layer,
frequently-changing surface) could silently break persisted data semantics
(inner-layer, stability-critical). The two families are allowed to diverge
independently.

**Alternatives considered:** Share one enum/model definition across all three layers
(less code, but couples layers that should be independently versionable — the
textbook mistake this pattern exists to prevent).

**Tradeoffs:** Duplication of structurally identical types. Accepted deliberately, and
verified with automated tests proving both value-parity (today) and class-independence
(structurally, not just nominally).

---

## ADR-006: SQLAlchemy 2.0 async ORM + `aiosqlite`, no Alembic (yet)

**Decision:** `create_async_engine`/`AsyncSession`/`async_sessionmaker`, SQLite via
`aiosqlite` by default (Postgres via `asyncpg` is a URL change only). Schema managed
via `Base.metadata.create_all()`, not Alembic migrations.

**Why:** The rest of the stack is async-everywhere (FastAPI, every `core.interfaces`
ABC); a sync ORM session would force sync/async boundary juggling throughout the
service layer. Alembic is deliberately deferred: this is a single-developer,
SQLite-by-default project with no concurrent-schema-change scenario to justify a
migration framework yet — noted explicitly rather than silently omitted, since a
shared Postgres deployment with multiple engineers would make Alembic the right call.

**Alternatives considered:** Sync SQLAlchemy with a thread pool executor (works, but
adds a sync/async seam everywhere it's called from async route handlers); Alembic from
day one (correct long-term, premature for the current single-environment scope).

**Tradeoffs:** Schema changes during development require re-running `create_all()`
against a fresh DB (no migration history) — acceptable pre-production, called out here
so it isn't mistaken for an oversight.

---

## ADR-007: Repository Pattern for persistence access

**Decision:** Services depend on repository interfaces (`ConversationRepository`,
`DocumentRepository`, `ReportRepository`), not on SQLAlchemy sessions/queries
directly. Aggregate computations needed by the API (e.g.
`ConversationSession.turn_count`/`document_count`/`total_token_usage`) are computed by
the repository via a single aggregate query per page, not N+1 per-row queries.

**Why:** Keeps the service layer (already required to be framework-agnostic)
independent of the ORM too — a future swap of persistence technology touches only the
repository implementations. The N+1 avoidance is a concrete performance requirement,
not just a style preference: a history page listing 20 sessions must not issue 20+
extra queries to compute each session's summary.

**Alternatives considered:** Services querying SQLAlchemy directly (simpler, but
couples every service to the ORM and makes the "framework-agnostic services" rule
impossible to keep); computing aggregates in Python after fetching all rows (avoids
N+1 but pulls unbounded data into memory instead of letting the database aggregate).

**Tradeoffs:** An extra abstraction layer and more careful query-writing up front.
Accepted for testability and the explicit performance requirement.

---

## ADR-008: Documents are session-scoped, not a cross-session library

**Decision:** `Document.session_id` is a required, non-nullable foreign key; a document
uploaded in one session cannot be referenced from another.

**Why:** A research session is a bounded, self-contained investigation — the RAG
retrieval scope (`RetrievalOptions.session_scope`) and the citation/report model both
assume "documents relevant to this investigation," not a shared library where
unrelated sessions' documents could leak into retrieval results. Session-scoping also
makes the cascade-delete story clean: deleting a session's data (including its
uploaded documents) is a complete, unambiguous operation with no orphan-reference risk
into other sessions.

**Alternatives considered:** A many-to-many `session_documents` join table allowing
document reuse across sessions (more flexible, but reintroduces exactly the shared-
library retrieval-scope ambiguity described above, and complicates deletion: deleting
a document now requires checking whether other sessions still reference it before
physically removing its ChromaDB vectors).

**Tradeoffs:** Re-uploading the same file into a second session duplicates storage and
re-runs embedding. Accepted as the right tradeoff for a research-assistant domain
where "these documents belong to this investigation" is the more natural and safer
default; document reuse could be added later as an explicit opt-in feature without
breaking this default.

---

## ADR-009: Physical deletes, not soft deletes

**Decision:** `DELETE` operations physically remove rows (`ON DELETE CASCADE`/`SET
NULL` throughout), no `deleted_at`/`is_deleted` columns anywhere.

**Why:** Soft deletes exist to serve two needs this project doesn't have: audit/compliance
trails required by regulation, and "undo" UX for accidental deletion in a
multi-user product. This is a single-user portfolio project with no regulatory
requirement and no product surface for restoring a deleted session. Soft deletes would
also force every single query in every repository to remember to filter
`WHERE deleted_at IS NULL` — a well-known source of real production bugs (data
"deleted" by the user still showing up somewhere a filter was forgotten) — for a
benefit this project doesn't need.

**Alternatives considered:** Soft deletes with a global query filter (SQLAlchemy
event/`with_loader_criteria`) to avoid the forgotten-filter risk — genuinely viable,
but adds meaningful complexity to defend against a class of bug that a single-user
app without compliance requirements doesn't need to defend against at all. Documented
here as the honest right answer if this became a multi-tenant product.

**Tradeoffs:** No recovery path for an accidental delete. Acceptable for the current
scope; flagged as a concrete "if this became multi-tenant, do X" item for interview
discussion.

---

## ADR-010: `ResearchSession` is the aggregate root; everything else cascades from it

**Decision:** `ConversationTurn`, `TurnCitation`, `AgentExecutionStep`, `Document`, and
`Report` all have `ON DELETE CASCADE` foreign keys to `sessions.id` (directly or
transitively). None of these entities has a lifecycle independent of its owning
session.

**Why:** Matches the actual domain: nothing about a turn, citation, execution step,
document, or report is meaningful once its session is gone. Modeling this as cascading
ownership (rather than independent top-level entities with optional session
references) means "delete a session" is a single, correct, unambiguous operation
enforced by the database itself — not something the application layer has to
orchestrate correctly across five tables every time.

**Alternatives considered:** Application-level cascade (service layer manually deletes
children before parent) — strictly more error-prone (a new child table added later
could be forgotten) and moves an invariant that the database can guarantee into code
that has to remember to guarantee it.

**Tradeoffs:** None significant — this is the textbook-correct modeling for a strict
ownership hierarchy. The one exception, `TurnCitation.document_id`, is intentionally
`ON DELETE SET NULL` rather than `CASCADE` (see ADR-011) because a citation is *not*
owned by the document it happened to reference.

---

## ADR-011: `TurnCitation.document_id` is `ON DELETE SET NULL`, not `CASCADE`

**Decision:** Deleting a `Document` nulls out any `TurnCitation.document_id` that
referenced it, rather than deleting the citation.

**Why:** A citation is a historical record of what was cited in a specific turn's
answer at generation time. `TurnCitation` deliberately denormalizes the display fields
it needs (`title`, `excerpt`, `page_number`, etc.) precisely so it can keep rendering
correctly even after its source document is deleted — the citation describes something
that *happened*, and that history shouldn't be erased just because the underlying file
was later removed from the session.

**Alternatives considered:** `CASCADE` (would silently corrupt a turn's answer history
by deleting citations out from under it — a past AI response would start rendering
with missing footnotes); leaving `document_id` un-nullable and blocking document
deletion entirely while citations reference it (overly restrictive — a user should be
able to remove a document from a session without being blocked by history it already
generated).

**Tradeoffs:** A `TurnCitation` can end up with a null `document_id` while still
displaying full citation details — verified explicitly via a live test (deleting the
document, confirming `document_id` is null but `title` etc. survive).

---

## ADR-012: `ResearchMetadata`/`ProcessingStatistics` flattened onto their owning row, not separate 1:1 tables

**Decision:** `ConversationTurn` has `llm_provider`, `prompt_tokens`, etc. directly as
columns (not a separate `turn_research_metadata` table); `Document` has
`processing_time_ms`, `chunk_count`, etc. directly as columns; `Report` has
`generation_latency_ms`, `prompt_tokens`, etc. directly as columns.

**Why:** Each of these is a strict 1:1 relationship, always present (or nullable-until-
populated) alongside its parent, with no independent lifecycle, and never queried or
joined against on its own. A separate table would require a join on every single read
of a turn/document/report for no benefit — pure normalization theater for data that
never varies independently of its parent row.

**Alternatives considered:** Separate 1:1 child tables (textbook-normalized, but adds
a mandatory join to the hottest read paths in the app for zero real benefit); storing
this data as JSON instead of typed columns (loses the ability to type-check/index
individual fields like `total_tokens` for future analytics queries).

**Tradeoffs:** Wide rows on `conversation_turns`/`documents`/`reports`. Accepted — this
is a deliberate, justified denormalization, not an oversight; each such decision is
documented at the point it's made (see also ADR-013 for where JSON *was* the right
call instead).

---

## ADR-013: `Report` content (`sections`, `citations`, `key_findings`) stored as JSON, not normalized tables

**Decision:** `Report.sections`/`citations`/`key_findings`/`report_metadata` are JSON
columns holding an exact serialization of
`core.interfaces.report_exporter.ReportDocument`'s nested structure (including
recursive `ReportSection.subsections` and embedded `ReportTable`s), rather than a
`report_sections`/`report_tables`/`report_citations` relational schema.

**Why:** Unlike ADR-012's flat 1:1 fields, this data is a deep, recursive tree with no
query requirement to reach *into* it — nothing in the product needs "find all report
sections titled X" or "list every table across all reports." A report is always
written once (by the Writer/Reviewer agents via `app.reports.builder`) and read as a
complete document (by an exporter or the API). Relational modeling here would mean a
self-referential `parent_section_id` + ordering column, a child table for tables, and
a child table for citations — real schema complexity purchased for a query pattern
that doesn't exist. JSON instead preserves `ReportDocument`'s exact shape 1:1, so a
stored row round-trips losslessly back through the same Pydantic model the
Writer/Reviewer/Exporter agents already share as their contract — verified with a live
test constructing a real `ReportDocument` with nested subsections/tables/citations,
storing it, and rebuilding an equivalent `ReportDocument` from the row.

**Alternatives considered:** Full relational normalization (rejected per above);
storing the entire `ReportDocument` as one opaque JSON blob column instead of separate
`sections`/`citations`/`key_findings`/`report_metadata` columns (would lose the
ability to index/query `status` and other flat fields, and conflates content that
genuinely has different update patterns, e.g. `report_metadata` vs. `sections`).

**Tradeoffs:** No database-level referential integrity between a citation embedded in
`Report.citations` JSON and the `Document`/web source it describes (unlike
`TurnCitation`, which does have a real FK). Accepted: report citations are a rendered,
point-in-time snapshot of a finished document, analogous to `TurnCitation`'s own
denormalization rationale (ADR-011) but taken one step further since the report as a
whole is already a frozen artifact once generated.

---

## ADR-014: `models/__init__.py` explicitly imports every model module

**Decision:** `app/models/__init__.py` imports `ResearchSession`, `ConversationTurn`,
`TurnCitation`, `AgentExecutionStep`, `Document`, and `Report` and re-exports them via
`__all__`, so `import app.models` alone guarantees every table is registered on
`Base.metadata`.

**Why:** Caught during live testing of `models/report.py`: `Database.create_all()`
only creates tables for model classes whose module has actually been imported
somewhere in the process. Without a central registry import, whether the `documents`/
`reports` tables exist after `create_all()` would depend on incidental import order
elsewhere in the app — e.g. whichever router happens to import `conversation.py` but
not `report.py` first. This was reproduced directly: `create_all()` raised
`NoReferencedTableError` until `document.py` was explicitly imported, even though
`document.py` itself had no bug. Fixed at the root with a single registry import
point, matching the pattern the `database/session.py` docstring already called for but
didn't yet enforce.

**Alternatives considered:** Relying on `app.core.dependencies` or the FastAPI app
entrypoint to import every model module individually (works today, but is exactly the
kind of implicit ordering dependency that silently breaks the first time someone adds
a new model file and forgets to also add the import somewhere else).

**Tradeoffs:** None — this is a strict correctness fix with no downside.

---

## ADR-015: Repositories accept an injected `AsyncSession`; `session.flush()`, never `commit()`

**Decision:** `BaseRepository.__init__(self, session: AsyncSession)` takes an
already-open session from the caller. Write methods call `session.flush()` to surface
constraint violations and make writes visible within the unit of work, but never
`session.commit()`/`rollback()` on success — those remain owned exclusively by
`Database.session()`.

**Why:** Lets a service compose multiple repositories into one atomic transaction
(e.g. a future `ChatService` inserting a `ConversationTurn` together with its
`TurnCitation`/`AgentExecutionStep` rows in a single commit). If each repository
opened and committed its own session, that composition would be impossible without
the service reaching around the repository abstraction to manage transactions itself
— defeating the point of the abstraction.

**Alternatives considered:** Each repository owning a `Database` reference and opening
its own session per call (simpler call sites, `repo = FooRepository(db)`, no need to
pass a session around) — rejected because it makes cross-repository atomicity
impossible, which the service layer genuinely needs.

**Tradeoffs:** Every service method that touches the database needs an explicit
`async with database.session() as session:` block and to construct repositories
against it, rather than a repository being a simple standalone singleton. Accepted —
this is the correct cost for real transactional composability.

---

## ADR-016: SQLAlchemy exceptions are rolled back and translated at the repository boundary

**Decision:** Every repository method wraps SQLAlchemy calls in `try`/`except
SQLAlchemyError`, and on failure calls `await self._session.rollback()` *before*
raising the translated domain exception (`RecordConflictError` for
`IntegrityError`, `DatabaseError` for anything else), chained via `from exc`.

**Why (and what this caught):** Live testing surfaced a real bug in the first version
of this code, which translated the exception but skipped the rollback. Once a flush
fails, SQLAlchemy marks the `Session` unusable — the *next* operation on it, including
the `commit()` that `Database.session()` issues on normal `async with` exit, raises
`PendingRollbackError` and masks the original, more meaningful error. A service that
legitimately catches `RecordConflictError` (e.g. to return "document already exists"
to the client) would have crashed with a confusing second exception the moment its
`async with database.session():` block tried to exit cleanly. Rolling back inside the
repository, before raising, is what makes catching a translated exception actually
safe.

**Alternatives considered:** Using SQLAlchemy `SAVEPOINT`s (`session.begin_nested()`)
around each write so a single failed operation could be rolled back without
invalidating the rest of the unit of work's uncommitted changes — a more sophisticated
option that would let, e.g., 9 successful inserts survive an 10th's constraint
violation in the same transaction. Rejected for now as unnecessary complexity for this
project's actual call patterns (repository calls that need this level of partial
resilience haven't materialized), but noted here as the correct upgrade path if a
future service needs it.

**Tradeoffs:** A failed repository call discards *all* uncommitted work earlier in the
same unit of work, not just the failing operation — ordinary SQL transaction semantics
(one failed statement aborts the transaction), made explicit here so services
composing multiple repository calls understand that any `DatabaseError` invalidates
the whole surrounding `database.session()` block, not just the call that raised it.

---

## ADR-017: Session-list aggregates use independent correlated scalar subqueries, not a JOIN + GROUP BY

**Decision:** `ConversationRepository.list_summaries()`/`get_summary()` compute
`turn_count`, `document_count`, `report_count`, `total_token_usage`, and `preview` for
each session via separate correlated scalar subqueries (`SELECT COUNT(*) FROM
conversation_turns WHERE session_id = sessions.id`, etc.), selected as labeled columns
alongside `ResearchSession` in one query — not by joining `conversation_turns`,
`documents`, and `reports` to `sessions` and grouping.

**Why:** A join across three independent one-to-many child tables multiplies rows
before any `GROUP BY` can run — a session with 3 turns and 2 documents produces 6
joined rows, so `COUNT(turns.id)` over that join overcounts by a factor of
`document_count` (and vice versa) unless every count is wrapped in `COUNT(DISTINCT
...)`, which itself gets expensive and error-prone to get right across more than two
joined children. Independent correlated subqueries sidestep the fan-out entirely: each
aggregate is computed against only its own table, correlated back to the outer
`sessions.id`, still in exactly one round-trip per page (the requirement this design
exists to satisfy in the first place). Verified with a live test specifically
constructed to catch this: a session with 3 turns + 1 document + 1 report alongside a
second, empty session, confirming neither aggregate leaks or inflates across either
session or across each other.

**Alternatives considered:** `JOIN` + `GROUP BY` with `COUNT(DISTINCT child.id)` for
every child table (works, but becomes fragile and hard to read as more child tables
are added, and DISTINCT-count performance degrades faster than independent subqueries
on large tables); N+1 (one query for sessions, then one aggregate query per session) —
exactly what this design was explicitly required to avoid.

**Tradeoffs:** Multiple subqueries per row is more SQL text than a single join, and
each subquery is a separate index lookup rather than one merged join plan — acceptable
here since each subquery hits an indexed foreign key (`ix_turns_session_id`,
`ix_documents_session_id`, `ix_reports_session_id`) and page sizes are small
(pagination-bounded), so the cost stays proportional to `page_size × child_tables`,
not `total_rows`.

---

## ADR-018: `document_count`/`report_count` computed in `ConversationRepository`, not their own repositories

**Decision:** `ConversationRepository.list_summaries()` reads `Document` and `Report`
row counts directly, even though those models live in `models/document.py` and
`models/report.py` respectively, which this repository doesn't otherwise own.

**Why:** `ResearchSession` is the aggregate root of both (ADR-010); these are
read-only `COUNT`s, not writes or business logic; and computing them in a separate
`DocumentRepository`/`ReportRepository` call per page would reintroduce exactly the
N+1 problem ADR-017 exists to avoid. The line drawn here is specific: cheap aggregate
counts against the aggregate root's own children belong with the aggregate root's
repository; fetching a *full* related row (e.g. `ConversationSession.latest_report`'s
title/format/status, which needs actual `Report` content) is different in kind and
stays `ReportRepository`'s responsibility once that file exists — `HistoryService`
will compose the two.

**Alternatives considered:** Strict one-repository-per-model-file boundaries with
`HistoryService` issuing three separate count queries per session (clean boundaries,
reintroduces N+1); denormalized `turn_count`/`document_count`/`report_count` counter
columns on `sessions` updated on every child write (rejected in the original database
design discussion — counter-drift risk, explicitly called out in
`models/conversation.py`'s module docstring).

**Tradeoffs:** `ConversationRepository` has a compile-time dependency on `Document` and
`Report` models it doesn't otherwise touch. Acceptable — these are read-only `SELECT`s
against sibling tables owned by the same aggregate root, not a write coupling, and the
alternative (N+1) is a genuine production performance bug on any session list with
real traffic.

---

## ADR-019: `UTCDateTime`, a custom column type, replaces `DateTime(timezone=True)` everywhere

**Decision:** Added `app/database/types.py`'s `UTCDateTime` (a `TypeDecorator` around
`DateTime(timezone=True)`) and replaced every `DateTime(timezone=True)` column across
`models/conversation.py`, `models/document.py`, and `models/report.py` with it.

**Why (and what this caught):** Live-tested while building `ConversationRepository`'s
`touch_session()` test, which compared a freshly loaded `updated_at` against a
newly-assigned one and hit `TypeError: can't compare offset-naive and offset-aware
datetimes`. Root-caused directly: SQLite has no native timezone-aware storage, so
SQLAlchemy's sqlite dialect silently returns a *naive* `datetime` on a fresh query even
though a timezone-aware one was written — confirmed by inserting a row and comparing
`sess.created_at.tzinfo` (aware, from the identity map) against the same row reloaded
via a new `SELECT` (`tzinfo=None`). This is the same underlying bug class as the
earlier `datetime.utcnow()` fix (ADR — see the "naive `datetime.utcnow()` bug" fixed
via `utils/time.py`), except worse: it doesn't depend on which function wrote the
timestamp, it depends on whether the object happens to still be in SQLAlchemy's
identity map or was reloaded fresh — meaning it would appear and disappear
unpredictably depending on unrelated code elsewhere in the same request. `UTCDateTime`
rejects naive datetimes on write (forcing every write path through
`utils.time.utc_now()`) and re-attaches UTC tzinfo on read, so every timestamp read
from any repository is guaranteed aware regardless of dialect or identity-map state —
verified directly by reproducing the original failure, applying the fix, and
confirming a fresh `SELECT`'s `created_at` now has `tzinfo=UTC` and compares equal to
the pre-fix value.

**Alternatives considered:** Only using `DateTime(timezone=True)` and having every
caller manually re-attach `timezone.utc` after each query (correct in principle, but
relies on every single call site across every future service remembering to do it —
exactly the kind of scattered discipline that produced the original bug); switching
straight to Postgres (whose driver preserves tzinfo correctly) to sidestep the issue —
rejected because SQLite-by-default for dev/test is an established, deliberate decision
(ADR-006), and the fix needed to work there too, not just in a hypothetical future
Postgres deployment.

**Tradeoffs:** One more indirection layer (a `TypeDecorator`) between the model
definitions and the raw column type. Accepted — this is a strict correctness fix with
no behavioral downside, and it has the added benefit of making timestamp handling
dialect-independent, so the same guarantee holds whether the project runs on SQLite
today or Postgres later.

---

## ADR-020: `get_by_checksum()` is a UX pre-check, not the source of truth for deduplication

**Decision:** `DocumentRepository.get_by_checksum(session_id, checksum_sha256)` performs
a plain read scoped to `session_id`, used to give a friendly "you already uploaded
this" response. The database's `(session_id, checksum_sha256)` unique constraint
(already declared in `models/document.py`) remains the actual enforcement — a
duplicate write still raises `RecordConflictError` regardless of whether the caller
checked first.

**Why:** A check-then-insert pattern is inherently racy (two concurrent uploads of the
same file could both pass the check before either write completes) — relying on it
alone would be a real correctness bug. But requiring every caller to catch
`RecordConflictError` as their *only* way to detect a duplicate turns an expected,
common outcome ("this file already exists") into exception-based control flow for
what should be a normal response path. Providing both is deliberate: the read-check
handles the common case cheaply and readably; the constraint (and its exception
translation, already built in `base_repository.py`) is what actually guarantees
correctness under concurrency. Verified with a live test that also confirms scope: the
same checksum in a *different* session is correctly not a match (`get_by_checksum`
returns `None`), and inserting it there succeeds without conflict — proving
`get_by_checksum` mirrors the constraint's actual scope (ADR-008's session-scoped
dedup), not a broader, incorrect global check.

**Alternatives considered:** Relying solely on catching `RecordConflictError` (correct
but poor UX/ergonomics for the common case — every caller would need to know that a
"duplicate file" business outcome surfaces as a caught database exception);
`SELECT ... FOR UPDATE`-style locking before insert (unnecessary complexity for a
single-user, single-process application with no real concurrent-upload scenario).

**Tradeoffs:** None significant — this is standard "optimistic check, pessimistic
guarantee" design, and the constraint means an incorrect or stale pre-check can never
lead to actual data corruption.

---

## ADR-021: No status-transition helper methods on `DocumentRepository`

**Decision:** `DocumentRepository` provides only CRUD (inherited) and query methods
(`get_by_checksum`, `list_by_session`, `count_by_session`) — no
`mark_ingestion_completed(...)`-style helper that would set `ingestion_status`,
`indexed`, and the processing-stats fields together in one call, even though
`schemas.upload.UploadResponse` already documents the exact invariants such a helper
would need to enforce (`indexed=True` requires `ingestion_status='completed'`;
`processing_stats` required once `ingestion_status='completed'`).

**Why:** Those invariants are business rules belonging to the multi-stage ingestion
pipeline (extract → clean → chunk → embed → index) that `IngestionService` (Phase 5)
will own — and that pipeline doesn't exist yet. Writing a helper now would mean
guessing at IngestionService's actual update pattern (does it update all fields
atomically at the end, or incrementally per stage for progress reporting?) before the
service that needs it has been designed, which is exactly the kind of speculative code
the project's "no placeholders, no TODOs" rule exists to prevent. Direct ORM attribute
mutation plus the already-inherited `BaseRepository.flush()` gives `IngestionService`
everything it needs today without this file constraining a design that hasn't
happened yet.

**Alternatives considered:** Adding the helper now with a best-guess shape (rejected —
premature, and likely to be wrong once the real ingestion pipeline is built, requiring
rework here anyway).

**Tradeoffs:** `IngestionService` will need to remember to set the related fields
consistently by hand rather than through a single enforced method. Accepted as the
honest tradeoff of not building ahead of a not-yet-designed consumer; revisit once
Phase 5 exists and the real update pattern is known.

---

## ADR-022: `TurnCitation.source_filename` added as a field distinct from `title`

**Decision:** Added a nullable `source_filename` column to `TurnCitation`
(`models/conversation.py`), populated independently from the existing `title` column.

**Why (and what this caught):** Found while building `HistoryService`'s ORM-to-schema
mapping: `schemas.common.Citation` (and its own docstring example) treats `title` and
`source_filename` as distinct fields, but `TurnCitation` only stored `title` — and
`Document.title` is explicitly documented elsewhere as "an optional, client-supplied
display title, distinct from the original filename." Mapping `source_filename =
citation.title` would silently show the wrong value for any document with a custom
title (e.g. a paper titled "Attention Is All You Need" uploaded as
`a1b2c3-download.pdf` would show its title where the filename belonged, or vice
versa) — an active correctness bug in the citation display, not just a missing
convenience field. Fixed at the model layer rather than papering over it in
`HistoryService`'s mapping function, consistent with the same "denormalize what a
citation needs to remain accurate after its source changes" reasoning already
governing every other field on `TurnCitation` (ADR-011).

**Alternatives considered:** Deriving `source_filename` from `citation.title` at
read time (rejected — actively wrong whenever a document has a distinct display
title, not just incomplete); joining back to `Document.filename` at read time instead
of denormalizing (rejected — defeats the entire purpose of `TurnCitation`'s
denormalization, ADR-011: a citation must stay fully displayable after its source
`Document` is deleted, and `document_id` is nullable specifically for that case).

**Tradeoffs:** None significant — this is a straightforward correctness fix, caught
before any real citations existed (verified via a live test proving `title` and
`source_filename` are stored and read back independently), and the full
`conversation_repository.py`/`document_repository.py`/`report_repository.py` test
suites were re-run afterward to confirm no regressions from the shared model change.

---

## ADR-023: `ChromaVectorStore` collection created with `embedding_function=None`

**Decision:** `ChromaDB`'s `get_or_create_collection(...)` is called with
`embedding_function=None` explicitly, rather than leaving Chroma's default
(`DefaultEmbeddingFunction`) in place.

**Why:** By default, a Chroma collection will silently compute its own embeddings
from raw text whenever `documents=` is passed to `upsert`/`query` without
accompanying `embeddings=` — using whatever `embedding_function` the collection was
created with. This project always computes embeddings explicitly through the
injected `EmbeddingProvider` (`SentenceTransformerEmbeddingProvider`) *before*
calling `VectorStore.upsert`/`query`, specifically so the embedding model is a single
swappable, testable, explicitly-injected dependency (see the `EmbeddingProvider`
interface docstring). Leaving Chroma's default embedding function in place would mean
that any future call site that accidentally omitted `embeddings=` wouldn't fail loudly
— it would silently start using a *different*, un-configured embedding model
(Chroma's own default, `all-MiniLM-L6-v2` via ONNX by default, which also requires a
network fetch the first time it runs), producing vectors that are geometrically
incompatible with every already-stored vector in the same collection. Setting
`embedding_function=None` makes that failure mode impossible: Chroma raises
immediately if a caller ever omits `embeddings=`, instead of quietly corrupting the
collection's vector space.

**Alternatives considered:** Leaving the default embedding function in place and
relying on code review/discipline to always pass `embeddings=` explicitly (rejected —
this is exactly the class of silent, hard-to-diagnose bug that should be made
structurally impossible rather than merely documented against).

**Tradeoffs:** None significant — `ChromaVectorStore` never relies on Chroma's
text-only `query_texts=`/`documents=`-without-`embeddings=` code paths, so removing
that capability costs nothing here. Verified via a live test suite (chunk
upsert/query/delete/count roundtrips, session- and document-scoped filtering,
idempotent re-upsert, cross-instance persistence at the same `persist_dir`, and
dimension-mismatch/invalid-path failures correctly wrapped as `VectorStoreError`).

---

## ADR-024: `ingest_document`'s all-or-nothing guarantee via compensating delete, and `EmbeddingProvider.model_name`

**Decision (two related changes made together while building `app/rag/ingest.py`):**

1. `ingest_document` (extract → chunk → embed → upsert) treats extraction, chunking,
   and embedding failures as inherently all-or-nothing (each raises before anything is
   written), and additionally wraps the one stage with a real side effect —
   `VectorStore.upsert` — so that if it fails, `ingest_document` makes a best-effort
   compensating `vector_store.delete(document_id)` before re-raising the original
   error. If the compensating delete itself also fails, that is logged at `critical`
   (manual reconciliation needed) but the original error is still what propagates —
   a cleanup failure must never masquerade as success or replace the real error.
2. Added an abstract `model_name` property to `EmbeddingProvider`
   (`core/interfaces/embedding_provider.py`), implemented in
   `SentenceTransformerEmbeddingProvider` as `self._settings.model_name`.

**Why:** ChromaDB has no multi-item transaction to roll back automatically, so "don't
leave a document partially indexed" can only be approximated with a compensating
action, not a true rollback — this is the standard saga/compensating-transaction
pattern for a pipeline with no shared transactional resource. The `model_name`
addition was needed because `IngestionResult.embedding_model` (mirroring
`schemas.upload.ProcessingStatistics.embedding_model`, which the future
`IngestionService` writes onto `Document.embedding_model`) has no other way to learn
which model produced a given document's vectors — `dimension` alone doesn't identify
the model, and `Document.embedding_model` exists specifically so a stored document's
row records this even if the configured model changes between ingestion runs.

**Alternatives considered:** Leaving `EmbeddingProvider` without `model_name` and
having `ingest_document` accept the model name as a separate injected string
parameter (rejected — that duplicates a fact the provider itself already owns, and
risks the caller passing a name that doesn't match what actually produced the
vectors); doing nothing on upsert failure and letting the caller reconcile
(rejected — silently leaves a same-`document_id` upsert retry to accidentally merge
with stale, no-longer-current chunks rather than starting clean).

**Tradeoffs:** The compensating delete is not atomic with the failed upsert — in the
narrow window where ChromaDB partially wrote before failing, the delete still cleans
up correctly (it removes *all* chunks for `document_id`, not just the ones from the
failed call), but if the delete call itself fails, the document is left in a
genuinely inconsistent state requiring manual reconciliation; this is called out
explicitly in the docstring and logged at `critical` rather than hidden. Verified via
a live test suite: full PDF/DOCX ingestion end-to-end against a real offline
`SentenceTransformer` and a real `ChromaVectorStore`, idempotent re-ingestion,
unsupported-file-type short-circuit, and both all-or-nothing failure paths (a
`FailingEmbeddingProvider` proving zero vector-store writes on embedding failure, and
a `FailingUpsertVectorStore` proving the compensating delete is actually invoked with
the correct `document_id`). The full `embedding_provider.py` regression suite was
re-run after the interface change — all scenarios, including the new `model_name`
check, passed.

---

## ADR-025: `app/rag/retriever.py` consolidates embedding/query failures into a single `RetrievalError`

**Decision:** `retrieve()` catches both `EmbeddingProviderError` (from `embed_query`)
and `VectorStoreError` (from `VectorStore.query`) and re-raises both as
`RetrievalError` (RM-RAG-007, retryable), with the original exception chained as
`__cause__`. This is the opposite choice from `app.rag.ingest`, which deliberately
lets `DocumentExtractionError` / `DocumentChunkingError` / `EmbeddingProviderError`
propagate as their own distinct types.

**Why:** The two pipelines have different callers with different needs.
`IngestionService` (ingest's caller) needs to know *which* stage failed, because it
persists that as `Document.ingestion_status` / `Document.error_code` for the user to
see ("extraction failed" vs. "embedding failed" are different, actionable messages on
an upload). `retrieve()`'s caller is a chat turn in progress (the Retriever agent /
`ChatService`) — it doesn't display per-stage failure detail to the end user, it just
needs to know "retrieval, as one conceptual operation, failed" so it can decide
whether to retry, fall back to a non-RAG answer, or surface a generic error. Exposing
two different exception types there would only add branching the caller doesn't
need, and the original exception is still fully recoverable via `__cause__` for
logging/debugging.

Also decided in this file: a blank/whitespace-only query raises `RetrievalError` with
`retryable=False` set explicitly (overriding the class default of `True`) — retrying
the exact same empty input can never succeed, so marking it retryable would be
actively misleading to a caller/orchestrator deciding whether to retry.

**Alternatives considered:** Letting `EmbeddingProviderError` / `VectorStoreError`
propagate directly from `retrieve()`, matching `ingest`'s pattern (rejected — would
force the Retriever agent to know about and handle two unrelated exception
hierarchies for what is, from its perspective, one operation).

**Tradeoffs:** A caller that *does* want to distinguish "the embedding model is down"
from "the vector store is down" (e.g. for a more specific health-degraded message)
has to inspect `__cause__` rather than catch a distinct type — accepted, since no
current caller needs that distinction, and the information isn't lost, just one level
deeper. Verified via a live test suite: end-to-end retrieval against a real offline
`SentenceTransformer` and real `ChromaVectorStore` (including genuine semantic
ranking — an ML-topic query correctly ranks an ML document above an unrelated ocean
document), `top_k`/`score_threshold`/`session_id`/`document_ids` defaults and
per-call overrides, determinism across repeated calls, blank-query rejection, and
both failure paths wrapped correctly with chained causes.

---

## ADR-026: LangGraph adopted with a `TypedDict` state schema; `track_step` as the shared per-node execution-tracing pattern

**Decision:** Phase 6's multi-agent workflow is built on `langgraph` (pinned
`langgraph==1.2.10` in `requirements.txt`, transitive `langchain-core` left
unpinned per this project's existing transitive-dependency policy). The
graph's shared state (`app.agents.state.ResearchGraphState`) is a
`TypedDict`, not a Pydantic model, even though this project otherwise
validates data with Pydantic everywhere. Every node's per-run execution
step is built via a shared `track_step` async context manager, which times
the node, always produces an `AgentExecutionStep` (COMPLETED or FAILED),
and never itself swallows an exception -- the choice of whether a given
node failure is fatal (propagate, halting the graph) or recoverable
(caught by the node itself, which reads the already-built FAILED step and
returns degraded state) is left entirely to each node.

**Why:** LangGraph merges each node's *partial* return dict onto the
running state via its own reducer machinery every step -- verified
directly (both `TypedDict` and Pydantic `BaseModel` schemas were tried
against the installed version), and `TypedDict` fits that merge-a-partial
model far more naturally than a Pydantic model, which validates a
*complete* object on construction. Every value stored inside the
`TypedDict` is still a real Pydantic type (existing ones reused directly --
`RetrievedChunk`, `SearchResult`, `Citation`, `ChatMode`,
`RetrievalOptions` -- extending the same reuse-over-reinvention precedent
already set for `DocumentChunk`/`RetrievedChunk` in Phase 5; new ones —
`ResearchPlan`, `VerifiedClaim`, `ReviewFeedback` — defined fresh where no
existing type fit), so runtime validation still happens everywhere it
matters, just not on the outer state container itself.

`track_step` exists because 7 more node files are about to be written
across 7 separate turns of this project's process, and every one of them
needs identical timing/step-building boilerplate around its own work —
building it once now, rather than letting each node file reinvent it,
directly prevents the kind of drift this project has repeatedly guarded
against (see the `AgentName` enum in the same file, for the same reason).
Making it never swallow exceptions, and leaving the fatal-vs-recoverable
call to each node, follows this project's established exception-based
error handling (errors are Python exceptions, not fields threaded through
data) while still producing the FAILED-step data the Agent Execution
Viewer needs for genuinely recoverable failures (e.g. web search being
down shouldn't necessarily kill a run that still has RAG results) — a
single utility that tried to also decide "is this fatal?" on the node's
behalf would need to know something it structurally can't: whether the
rest of that specific node's work can proceed without the failed call's
result.

Also decided in this file, as an explicit scope boundary rather than an
oversight: conversation history / prior turns are deliberately absent from
`ResearchGraphState`. `ChatMode.CHAT`'s own docstring ties lightweight chat
to memory, and Memory is this project's own distinct, not-yet-designed
later phase (Phase 7) — adding multi-turn context to this state now would
mean guessing at a design that phase hasn't reached yet, the same
reasoning already applied in ADR-021 (`DocumentRepository` status-transition
helpers deferred to the not-yet-built `IngestionService`).

**Alternatives considered:** A Pydantic `BaseModel` state schema (rejected —
fights LangGraph's partial-update merge model rather than working with it,
for no benefit since node-internal values are already Pydantic-typed);
letting each node build its own ad hoc step-timing code (rejected — 7
near-identical copies is exactly the drift this project's discipline
exists to prevent); `track_step` itself deciding fatal-vs-recoverable via
a parameter (rejected — that information genuinely lives with the node,
not a generic timing utility, and forcing it into a parameter would just
relocate the same per-node judgment call one file away without simplifying
anything).

**Tradeoffs:** A node that lets an exception propagate as genuinely fatal
loses that step's FAILED record for this run, since LangGraph's `ainvoke`
aborts without returning partial state on an unhandled exception —
accepted as a known, documented limitation rather than solved with
partial-state capture machinery (e.g. LangGraph's checkpointing) that
nothing in this project needs yet; revisit if/when a future phase actually
needs mid-run failure observability for runs that ultimately error out.
Verified via a live test suite: `create_initial_state`'s defaults,
`ResearchPlan`/`VerifiedClaim`/`ReviewFeedback` validation,
`track_step`'s success path (including a node setting `model_name`/
`tool_name`/`token_usage`), its failure path against both a
`ResearchMindError` subclass (confirming `error_code`/`retryable` are
carried through correctly) and a generic exception (confirming the
`RM-GEN-000` fallback), the graceful-degradation pattern (catching around
the block, reading `rec.step`, continuing), and — critically — a real
`langgraph.graph.StateGraph` built with parallel fan-out nodes both
writing `execution_steps` in the same superstep, proving the
`Annotated[List[AgentExecutionStep], operator.add]` reducer works inside
actual LangGraph execution, not just via a standalone `operator.add` call.
The full `backend/requirements.txt` was also re-verified installing
cleanly into a fresh, fully isolated venv with `langgraph` added.

---

## ADR-027: `OllamaLLMService` -- retry policy, `health_check` design, and offline protocol-level testing

**Decision:** `OllamaLLMService` wraps the official `ollama` Python package. Three
related design choices:

1. **Retry policy differs by method.** `generate` retries only on transient failures
   (`LLMServiceError.retryable=True`), bounded by `LLMSettings.max_retries`, with
   capped exponential backoff. `generate_structured` retries on *both* transient
   failures and schema-validation failures (a non-conforming JSON response gets one
   more attempt), bounded by the same `max_retries`. `stream` never retries.
2. **`health_check` calls `client.list()`** (list locally available models) instead of
   running a real generation, unlike `SentenceTransformerEmbeddingProvider.health_check()`,
   which does run real inference.
3. **Tested against a real local HTTP server speaking Ollama's actual wire protocol**
   (verified field-for-field against the installed `ollama` package's own response
   types), rather than mocking the adapter's internals -- the same environment
   constraint as Phase 5's blocked `huggingface.co` access (only PyPI-style
   registries are reachable from this sandbox; `ollama.com` and `github.com` were
   both verified unreachable), solved with the same philosophy but a different
   technique, since this adapter's own code does no ML computation to fall back to
   running "for real" -- it's pure HTTP/JSON, so protocol-level fidelity (a real
   socket, real request/response bytes) is the correct place to be faithful, the way
   computation-level fidelity (a real, if untrained, model) was the correct place in
   Phase 5.

**Why:** A caller that gets a validation failure from `generate_structured` doesn't
know whether the *model* momentarily deviated from the JSON constraint (worth
retrying) or the constraint itself is unsatisfiable (retrying won't help) --
treating it as retryable-with-a-bound is the more useful default, since the interface
docstring already documents `generate_structured` as expected to retry internally
before surfacing `LLMOutputValidationError`. `stream` doesn't retry because a caller
may already have rendered partial content to a user by the time a mid-stream failure
occurs; transparently retrying would either duplicate or silently discard what's
already been delivered -- the caller is better positioned to decide what to do with a
partial stream than this adapter is. `health_check` avoids a real generation call
because Ollama model loading (potentially multi-gigabyte, from disk) is orders of
magnitude more expensive than an already-resident Sentence Transformers forward pass
-- the same "run real work to prove readiness" principle applied to a case where that
would violate `GET /health`'s own requirement to respond promptly.

**Alternatives considered:** A single unified retry loop shared by `generate` and
`generate_structured` (rejected -- their retry *triggers* are genuinely different:
one only cares about transport/provider failures, the other also cares about output
shape, and forcing them through one code path would either over-retry `generate` on
conditions it can't act on or under-retry `generate_structured` on validation);
mocking `ollama.AsyncClient` directly with a test double (rejected -- would only
prove the adapter calls the client the way the test expects it to, not that it
correctly handles the actual bytes a real Ollama server sends, including its
newline-delimited streaming format and its exact error-response shapes).

**Tradeoffs:** The fake server is a hand-maintained approximation of Ollama's
protocol, not the real server -- if a future Ollama release changes response field
names or streaming framing, this test suite won't catch that until a real server is
available to test against (deferred to Phase 10/11, when a real Ollama instance may
be reachable). Verified via a live test suite (17 scenarios) against the real fake
server: full `LLMResponse` field mapping, system-prompt/temperature/max_tokens
request construction, successful and failing `generate_structured` (including a
genuinely flaky case that fails once then succeeds, proving the retry loop recovers
mid-run, not just on the first or last attempt), 5xx (retryable) vs. 4xx
(non-retryable) vs. 429 (rate limit) error mapping with attempt counts confirming the
actual retry/no-retry behavior, real timeout via a genuinely slow server, real
connection-refused via an unreachable port, `health_check` true/false, real
incremental `stream()` chunks that reassemble correctly, `stream()` error
propagation, and the `__init__` guard against a misconfigured `max_retries < 1`. The
full `requirements.txt` (now including `langgraph`, `ollama`, `httpx`) was re-verified
installing cleanly into a fresh, fully isolated venv.

---

## ADR-028: `TavilySearchProvider` -- keyless-mode support, cost-driven `health_check`, and two `SearchSettings` additions

**Decision:** `TavilySearchProvider` wraps `tavily.AsyncTavilyClient`. Three related
choices:

1. Two fields added to the already-approved `SearchSettings`:
   `timeout_seconds: int = 30` and `base_url: Optional[str] = None` (mirroring
   `OllamaSettings`'s existing identically-named fields). Neither existed before --
   `SearchTimeoutError` was already a defined exception type with nothing
   configuring the timeout it's named for.
2. When `SearchSettings.tavily_api_key` is unset, `TavilySearchProvider` passes
   `api_key=None` straight through to `AsyncTavilyClient` rather than refusing to
   construct. Traced directly in the `tavily-python` source: the SDK has a real,
   documented keyless mode (rate-limited, `search`/`extract` only) for exactly this
   case, not an error condition.
3. `health_check` performs a bare, unauthenticated `GET` to the configured base URL
   -- never a real `search()` call.

**Why:** The `base_url` addition serves double duty: it's what makes this file's own
test suite possible (pointing the real, unmodified provider at a local fake server,
the same protocol-level testing technique already used for `OllamaLLMService`), and
it's a legitimate production knob for a self-hosted/enterprise-proxied Tavily
endpoint, which the SDK itself already supports via its own `api_base_url`
constructor parameter -- this file just exposes it through config rather than adding
capability the SDK didn't already have. The keyless-mode pass-through isn't a new
policy invented here: `Settings`'s own top-level validator already encodes "an API
key is only *required* in non-local environments" -- refusing to construct
`TavilySearchProvider` without a key would silently contradict a decision this
project already made at the config layer.

`health_check` avoiding a real search call is a genuinely different constraint from
every prior adapter's health check, worth distinguishing explicitly: Ollama's
`health_check` (ADR-027) optimizes for *latency* (avoid triggering an expensive model
load), and the embedding provider's real-inference health check is cheap because the
model is already resident in memory. Here the constraint is *cost* -- a real Tavily
search consumes a billed API credit per call, and a `/health` endpoint can be polled
far more often than a human would ever manually check status (liveness probes,
uptime monitors). Trading "confirms the API key is valid" for "confirms the network
path is reachable" is the honest tradeoff available without either paying per health
check or inventing a fake verification that provides no real signal.

**Alternatives considered:** Requiring an API key at construction, rejecting keyless
mode entirely (rejected -- contradicts `Settings`'s own already-decided policy, for
no benefit, since the SDK-level keyless support is real and functional, not a hack);
a full `search()` call as the health check, matching every other adapter's pattern
(rejected -- the cost model here is fundamentally different, per-call billing rather
than latency, and copying the pattern without accounting for that difference would
be a real production cost regression, not a stylistic inconsistency).

**Tradeoffs:** `health_check` returning `True` doesn't guarantee the configured API
key is actually valid -- only that the Tavily API is network-reachable. A key that's
present but revoked/invalid would show healthy right up until the first real search
fails with `InvalidAPIKeyError`. Accepted as the correct tradeoff given the cost
constraint; a stricter check would need a genuinely free "validate this key" endpoint,
which Tavily's API does not currently expose. Verified via a live test suite (18
scenarios) against a real local HTTP server speaking Tavily's actual wire protocol
(response shape, status-code-to-exception mapping, and header conventions all
verified directly against the `tavily-python` SDK's own source, the same technique
established in ADR-027): full `SearchResponse`/`SearchResult` field mapping including
graceful handling of a missing `published_date`, `max_results`/`include_domains`/
`exclude_domains` request construction, keyless construction with no `Authorization`
header sent, status-code-driven exception mapping (429/401/403/400 non-retryable,
5xx retryable with confirmed retry-attempt counts, an unmapped status also
non-retryable), real timeout and real connection-refused, `health_check` true/false
including confirmation it never calls `/search` (no quota consumed), and the
`max_retries` guard. `requirements.txt` (now including `tavily-python`) was
re-verified installing cleanly into a fresh, fully isolated venv.

---

## ADR-029: `retriever_agent.py` short-circuits retrieval for session-scoped requests with no session yet, rather than passing `session_id=None` through to `VectorStore.query`

**Status:** Accepted

**Context:** `app.agents.retriever_agent` is a thin graph-node wrapper around
`app.rag.retriever.retrieve` (ADR-025), following the same closure-factory
dependency-injection pattern as `app.agents.planner` (ADR-026). While wiring
`RetrievalOptions.session_scope` (default `True` -- "restrict to the current
session," matching ADR-008's session-scoped document model) through to
`retrieve()`, a genuine correctness gap surfaced: `VectorStore.query`'s
`session_id` filter is opt-in by design -- passing `session_id=None` means
"search across every session's documents," not "return no results." A brand-new
conversation legitimately has `session_id=None` (`ChatRequest.session_id`'s own
docstring: "Omit to start a new session"), so a request with the default
`session_scope=True` and no session yet would, without this fix, silently fall
through to an unscoped, cross-session search -- returning another user's or
another conversation's documents to a request that asked to be scoped to "this
session" and, per ADR-008, has zero documents of its own by definition.

**Decision:** `retriever_agent._resolve_scope(session_id, retrieval_options)`
distinguishes three cases before ever calling `retrieve()`:
1. `session_scope=False` (explicit opt-out) -- pass `session_id=None` through
   deliberately; this is the one legitimate case for an unscoped search.
2. `session_scope=True` (default) with a real `session_id` -- pass it through
   normally.
3. `session_scope=True` (default) with `session_id=None` -- skip calling
   `retrieve()` entirely and return an empty chunk list. A session that doesn't
   exist yet is guaranteed to have no uploaded documents, so this is not just a
   safe default, it is the only correct answer, and it avoids ever handing
   `VectorStore.query` a combination of arguments whose meaning ("search
   everything") contradicts what the caller actually asked for ("search my
   session").

**Why:** This is the same session-isolation principle ADR-008 already
established at the document-storage level, extended to the retrieval-request
level where a `None` sentinel's dual meaning (opt-in filter, not "empty"
result) creates a sharp edge that a naive pass-through would fall into
silently -- no exception, no error log, just quietly wrong (cross-session)
results. Recorded as its own ADR rather than folded into ADR-026's already-
documented closure/node pattern because the *correctness/isolation*
consequence is new and non-obvious, not a reapplication of an existing pattern.

**Alternatives considered:** Passing `session_id` straight through and relying
on `retrieval_options.session_scope`'s default to rarely combine with
`session_id=None` in practice (rejected -- "rarely" is exactly the profile of a
latent data-isolation bug: every first turn of every new conversation hits this
path, not an edge case); making `VectorStore.query` itself reject `session_id=
None` (rejected -- would break the legitimate, already-used "search everything"
mode that `session_scope=False` and admin/cross-session tooling rely on;
the ambiguity belongs to the caller to resolve, not the store to forbid).

**Tradeoffs:** None of substance -- the short-circuit is strictly more correct
than the alternative and costs nothing extra (it avoids an embedding call and a
vector-store round trip in exactly the case where both would be wasted work
anyway, since the correct result is always empty). Verified via 9 live scenarios
against a real `ChromaVectorStore` with real ingested documents across two
sessions and a real (offline, tiny-weights) `SentenceTransformer` embedding
provider: session-scoped retrieval correctly isolated to one session; all four
`(session_id, session_scope)` combinations of `_resolve_scope` unit-checked;
the specific new-session/`session_scope=True` case confirmed empty with no
cross-session search performed; `session_scope=False` confirmed to
deliberately search across sessions; `document_ids`/`top_k` overrides honored;
`RetrievalError` degrading gracefully (empty chunks, `FAILED` step recorded,
no exception propagated) versus a genuinely unexpected exception being wrapped
as a fatal `RetrieverAgentError`; and a real compiled `langgraph.graph.StateGraph`
integration test.

---

## ADR-030: `graph.py`'s routing design -- conditional-edge fan-out/fan-in for optional Retriever/Search branches, and the Reviewer→Writer bound enforced by a routing function that raises `MaxRetriesExceededError` directly

**Status:** Accepted

**Context:** `app.agents.state`'s own module docstring sketched the intended graph
shape back when it was first written (ADR-026) -- Coordinator entry/exit,
Planner, a parallel Retriever/Search stage, Researcher, Writer, Fact Checker,
Reviewer, and "the one cycle in this graph" back to Writer -- but that was a
shape description, not an implementation decision. Building `app.agents.graph`
required resolving three genuinely open questions that no prior Phase 6 file
decided:

1. How does a *dynamically optional* parallel fan-out (the Planner decides,
   per request, whether zero, one, or both of Retriever/Search should run) fan
   back into a single downstream node (Researcher) without either node
   double-running or the graph hanging on a branch that was never scheduled?
2. Where does `AgentSettings.max_reviewer_retries` get enforced, and by what
   mechanism does the graph actually stop the Reviewer↔Writer cycle and raise
   `MaxRetriesExceededError`?
3. Who is responsible for the `ChatMode.CHAT` bypass decision, given that
   `coordinator.py` (by its own explicit design) contains no routing logic at
   all?

**Decision:**
1. `retriever`/`search` connect to `researcher` via plain, static `add_edge`
   calls, not a synchronized join. LangGraph's Pregel-style scheduling makes a
   node eligible for the next superstep when *any* of its declared
   predecessors updated state in the current superstep, not only when *all*
   declared predecessors did -- so `researcher` runs exactly once whether the
   Planner's conditional-edge routing function (`_route_after_planner`,
   returning `[]`, `["retriever"]`, `["search"]`, or
   `["retriever", "search"]`) scheduled zero, one, or both of them. This was
   verified empirically (not assumed) across all four combinations in this
   file's own test suite, since it was judged the highest-risk assumption in
   the whole file.
2. `AgentSettings.max_reviewer_retries` is read only by a routing function
   (`_route_after_reviewer`, closed over `agent_settings`) attached via
   `add_conditional_edges` after the Reviewer node -- never by `reviewer_node`
   itself (per that file's own explicit design) and never duplicated into any
   other node. On rejection with retries remaining, it returns `"writer"`,
   re-entering the existing `writer -> fact_checker -> reviewer` edges
   unchanged -- no extra edges needed for the loop body itself. On rejection
   with retries exhausted, it raises `MaxRetriesExceededError` directly from
   the routing function rather than routing to a dedicated node that raises
   it: there is no meaningful unit of work (no `track_step`-worthy step) for
   "the bound was exceeded," so a real node would exist purely as raise-error
   ceremony.
3. `_route_entry`, attached to `START`, is the *only* place `state["mode"] ==
   ChatMode.CHAT` is checked anywhere in the Phase 6 codebase -- matching the
   scope boundary `coordinator.py` already committed to explicitly.

**Why:** All three decisions follow the same principle already established
across every other Phase 6 file: routing/control-flow decisions belong to
`graph.py`'s conditional-edge functions, and node functions only ever produce
data. `reviewer.py` explicitly deferred `max_reviewer_retries` enforcement
here for exactly this reason; this ADR is where that deferred decision is
actually made. The off-by-one behavior was worked out explicitly rather than
guessed: with `max_reviewer_retries = N`, a rejection produces the
`revision_count` sequence `1, 2, ..., N, N+1` (incremented by `reviewer_node`,
per its own design), and the condition `revision_count <= N` permits exactly
`N` loop-backs (`N+1` total Writer invocations) before raising on the
`(N+1)`th rejection -- verified numerically with `N=1` in this file's test
suite (exactly 2 Reviewer calls, then a raise).

**Alternatives considered:** A single "gather" node between the Planner and
Researcher that itself decides whether to have already run Retriever/Search
inline, rather than LangGraph-level conditional fan-out (rejected -- would
re-implement scheduling LangGraph already provides, and would make Retriever/
Search un-parallelizable when both are needed, contradicting the Retriever's/
Search's own agent design, which assumes true parallel execution). Enforcing
`max_reviewer_retries` inside `reviewer_node` itself (rejected -- already
explicitly ruled out in `reviewer.py`'s own docstring, for the node/routing
separation reason above). Routing to a dedicated `max_retries_exceeded` node
that raises the error (rejected -- no real work happens there, so it would
exist only to satisfy "nodes do work, routing functions decide," a rule that
doesn't actually apply to raising a graph-level bound violation).

**Tradeoffs:** None of substance. Verified via 10 live end-to-end scenarios
using real adapters at every boundary (a real local HTTP server per
Ollama/Tavily wire protocol driving the real, unmodified
`OllamaLLMService`/`TavilySearchProvider`; a real on-disk `ChromaVectorStore`
with a real document actually ingested; a real offline tiny-weights
`SentenceTransformer`; the real compiled `StateGraph` via `.ainvoke()`):
`ChatMode.CHAT` bypassing the full pipeline entirely (only `coordinator_chat`
runs, zero Tavily/RAG calls); the full pipeline with both Retriever and
Search producing exactly 8 execution steps with `researcher` running exactly
once; retrieval-only and search-only plans each producing exactly 7 steps
with the skipped branch never invoked and `researcher` still running exactly
once (no hang); neither-source plans jumping straight from Planner to
Researcher (6 steps); a reject-then-approve revision cycle correctly
re-running `writer`/`fact_checker`/`reviewer` exactly twice each with
`revision_count == 1`; `max_reviewer_retries = 1` exhausting correctly with
no off-by-one (exactly 2 Reviewer calls before the raise); and three
independent graceful-degradation scenarios (Retriever failure, Search
failure, Fact Checker failure) each still producing a valid, non-empty
`final_answer` by the time the graph reaches `coordinator_finalize`.

---

## ADR-031: `app.memory.compaction` folds only newly aged-out turns into the running summary, rather than re-summarizing a session's full older-turn history each request

**Status:** Accepted

**Context:** Phase 7's Memory subsystem needs to keep a long session's conversation
history bounded before it's injected into a prompt: a fixed-size "recent turns"
window kept verbatim, plus a summary of everything older. The naive design --
re-summarize every turn older than the recent window from scratch, every request,
once a session crosses the threshold -- has two real problems, not hypothetical
ones. First, it requires fetching a session's entire, unboundedly-growing turn
history from the database on every request past the threshold, which is exactly
the kind of unbounded-query cost this project has consistently avoided elsewhere
(e.g. the bounded Reviewer/Writer revision loop, ADR-030's off-by-one-safe retry
bound). Second, it makes persisting a summary between requests pointless: if the
summary is always regenerated from the raw turns anyway, there is no reason to
store it at all.

**Decision:** `AgentSettings.max_chat_history` defines a fixed-size sliding
window over a session's turns. Once a session has exceeded that size, exactly one
turn ages out of the window per new turn added, forever -- so compaction only
ever needs to fold in the turns that *newly* aged out since the summary was last
updated (almost always zero or one per request), not the full older-turn bucket.
This splits responsibility cleanly: `app.services.memory_service.MemoryService`
(the database-aware layer) owns figuring out *which* turns newly aged out --
bounded bookkeeping against the persisted turn history and the summary's own
watermark -- while `app.memory.compaction.build_memory_context` stays a pure,
DB-agnostic function that takes `recent_turns` (passthrough), `newly_aged_out_turns`
(the small delta), and `existing_summary`, and produces the updated
`MemoryContext`. When `newly_aged_out_turns` is empty (the common case for most
requests within a session's lifetime), no LLM call happens at all.

Compaction failure is deliberately non-fatal: a genuine `LLMServiceError` during
summarization degrades to keeping the existing summary unchanged (never lost,
never replaced with nothing) rather than raising -- compaction is
request-lifecycle housekeeping, not a step in the user's own request, matching
the graceful-degradation reasoning already applied to every non-essential
sub-call across Phase 6 (the Fact Checker's/Reviewer's own LLM calls,
`coordinator.py`'s follow-up suggestions). Each aged-out turn's answer text is
truncated (500 chars, same bounded-excerpt technique as `researcher.py`'s
`_truncate`) before being folded into the summarization prompt, as a safety net
against unbounded prompt growth on a session's first, potentially large,
compaction batch.

**Why:** This keeps both the per-request database read and the per-request LLM
cost proportional to how much *new* history exists since the last check, not to
total session length -- the same "bounded, not unbounded" discipline already
established for the revision loop, applied here to memory instead of agent
retries. It also makes the persisted `ResearchSession.conversation_summary`
column meaningful: it is genuinely incremental state being extended, not a
cache of something trivially recomputable from scratch every time.

**Alternatives considered:** Always re-summarize the full older-turn bucket from
scratch (rejected -- unbounded per-request DB read and LLM cost as sessions grow,
and makes persisting a summary pointless, as explained above). Treating
compaction as a background/scheduled job rather than inline per-request (rejected
-- this project has no worker/queue infrastructure anywhere, and building one
speculatively for a single use case would be exactly the premature-abstraction
this project's process consistently avoids; inline compaction of a small delta is
already cheap enough not to need one). Giving `MemoryTurn` a `sequence_number`
field so `compaction.py` itself could determine "newly aged out" (rejected --
already decided against per the prior file's review: `MemoryTurn`/`MemoryContext`
stay minimal DTOs independent of persistence identifiers; the aging-out
bookkeeping belongs entirely to `MemoryService`, which already has the real
persisted turn identifiers to work with, not to the pure compaction function).

**Tradeoffs:** None of substance for the common case. A session's very first
compaction (crossing the threshold for the first time) still processes however
many turns fell outside the window in that one batch, bounded only by the
per-turn truncation safety net -- accepted, since this happens exactly once per
session's lifetime, not on an ongoing basis. Verified via 7 live scenarios
against a real local fake-Ollama server driving the real `OllamaLLMService`: no
aged-out turns skipping the LLM call entirely and passing the existing summary
through unchanged; a first-ever compaction with no prior summary; an incremental
compaction correctly folding new turns in alongside prior summary text; multiple
aged-out turns batched into one call; long answer text truncated in the
summarization prompt; and two LLM-failure scenarios (with and without an
existing summary) both degrading gracefully without losing state or raising.

---

## ADR-032: Black/Ruff/mypy configured to this codebase's actual conventions, not each tool's defaults; git repository initialized at the Phase 8 checkpoint

**Decision:** Before starting Phase 9, added `pyproject.toml` configuring Black
(`line-length = 120`), Ruff (a conservative `select = ["E", "F", "B", "W"]` rule
set, with `B008` ignored project-wide), and mypy (not `--strict`; targeted
per-module `disable_error_code` overrides for third-party stub friction), plus a
separate `requirements-dev.txt` pinning the exact tool versions installed
(`black==26.5.1`, `ruff==0.16.1`, `mypy==2.3.0`). Applied Black across the whole
tree and fixed every genuine Ruff/mypy finding. Also initialized this project's
git repository for the first time at this checkpoint (`git init`, one commit
covering Phases 1–8, tagged `phase-8-complete`) — the project had no version
control at all before this point; every prior phase's "approval" was this
conversation's own review process, not a commit history.

**Why line-length=120, not Black's default 88:** Measuring the existing codebase
first (326 lines already over 100 columns, 41 over 120, out of Phases 1–8's ~80
files) showed that adopting Black's default would force a disruptive full-tree
reformat driven by the tool's own opinion rather than this codebase's actual,
already-reviewed style (prose-heavy docstrings and descriptive names routinely
run past 88/100 columns by design — see this log's own stated goal of every
entry standing on its own as interview-ready explanation, which favors readable
prose over artificially wrapped lines). 120 accommodates the existing style
without needing to relax it further.

**Why a conservative Ruff rule set, not Ruff's full opinionated defaults:**
`select = ["E", "F", "B", "W"]` catches real-bug-shaped issues (unused imports/
names, undefined names, common bugbear patterns like mutable `ContextVar`
defaults or `zip()` without `strict=`) without also surfacing import-sorting,
docstring-convention, or cyclomatic-complexity findings across already-reviewed
code that would be pure churn with no correctness value. `B008` ("no function
call in argument defaults") is ignored project-wide because it's a real
anti-pattern in general but a false positive against FastAPI's own documented,
idiomatic `Depends(...)`-in-signature-default dependency-injection mechanism,
used throughout `app/api/routes/*`.

**Why mypy is not `--strict`:** This codebase's primary correctness signal
throughout Phases 1–8 was real functional testing (real DB, real files, real
HTTP requests, no mocks of the code under test) per file, not a mypy-strict
discipline from the start. Enabling `--strict` now would surface a large,
one-time backlog of pre-existing annotation gaps unrelated to any real bug,
rather than catching new mistakes going forward. The chosen config
(`disallow_untyped_defs = false`, `check_untyped_defs = true`, `warn_return_any
= true`, `no_implicit_optional = true`) still catches genuine type errors —
wrong argument types, missing returns, incompatible overrides — without
demanding every function already have exhaustive annotations.

**What the first mypy/Ruff pass actually found (34 mypy + 7 Ruff findings, all
individually triaged rather than blanket-suppressed):** two genuine, if narrow,
issues fixed directly — a `ContextVar` mutable-default footgun in
`app.core.logging` (`default={}` shared across every reader that never
`.set()`s its own value; changed to `default=None` with `or {}` at each read
site) and a documented `assert` added in `app.services.memory_service` to
narrow `Optional[str]` to `str` at the one call site where
`CompactionOutcome.summary_changed=True` provably implies a real summary string,
by construction in `app.memory.compaction` — the invariant existed before this
ADR, just wasn't expressed to the type checker. Two genuine unused imports
removed. One defensive hardening applied on inspection: `zip(...,
strict=True)` in `app.rag.vector_store`'s Chroma-result unpacking, so a
(never-observed, but not type-system-impossible) length mismatch between
Chroma's four parallel result lists fails loudly instead of silently
truncating. Everything else — `ClassVar`/per-instance-override tension in
`ResearchMindError.retryable`, SQLAlchemy cross-file forward-reference
`relationship()` declarations (already flagged with `# noqa: F821` before this
ADR), and the bulk of the mypy findings (LangGraph's `add_node`/`ainvoke`
overload imprecision for `TypedDict` state schemas, ChromaDB's
stricter-or-looser-than-runtime stubs, Starlette's generic
`add_exception_handler` signature, untyped-`Any`-returning calls into
`tavily-python`/`ollama`/`sentence-transformers`) — is documented,
targeted third-party stub friction, suppressed with an explanation of exactly
why each is not a real defect, rather than either left as unexplained noise or
"fixed" by reshaping already-tested working code around another library's
imprecise type stubs.

**Two real, unrelated production bugs found and fixed along the way, discovered
specifically because verification ran through the real, fully-wired
application (`app.main.create_app()`, real `TestClient` lifespan) rather than
an isolated per-file check:** `logger.info(..., extra={"filename": ...})` in
both `app/api/routes/report.py` (this phase) and `app/rag/extraction.py` (an
already-approved Phase 3 file) raised `KeyError: Attempt to overwrite
'filename' in LogRecord` on every real call — Python's stdlib `logging`
reserves `filename` as one of `LogRecord`'s own attributes, a collision that
isolated tests using stubbed logging never exercised. Fixed by renaming both
call sites' `extra` keys (`download_filename`, `document_filename`); grepped
the rest of the codebase for every other reserved `LogRecord` attribute name
across every `extra={...}` call site and found no further collisions.

**Alternatives considered:** Deferring all tooling setup to Phase 11
(Testing)/Phase 12 (final polish) instead of now — rejected because linting and
type-checking catch a different, cheaper class of issue (unused imports, type
mismatches, footguns) than the functional tests this project already relies on,
and running them once now, before eight more phases of code accumulate on top,
keeps the fix cost low. Full `--strict` mypy adoption immediately — rejected
per the reasoning above (large unrelated backlog, no proportional bug-catching
value for this codebase's actual risk profile). Reformatting to Black's default
88-column width — rejected as unnecessary churn against an already-reviewed,
intentionally prose-heavy style.

**Tradeoffs accepted:** The mypy config's per-module `disable_error_code`
overrides mean a *new*, real type error introduced later in one of those eight
overridden modules in one of the four disabled error-code categories
(`no-any-return`, `arg-type`, `call-overload`, `index`, `dict-item`) would not
be caught until those overrides are eventually narrowed or removed — accepted
because the alternative (leaving them fully strict) would currently be pure
false-positive noise in those specific modules, and the overrides are scoped to
exactly the modules and error codes that produced the noise, not applied
globally.

---

## ADR-033: Phase 9 frontend architecture — independent contract models, a single HTTP boundary, centralized session-state, and a two-step download flow

**Context.** Phase 9 added the Streamlit frontend (`frontend/`): `app.py` (Chat),
`pages/1_Report.py` (Report), `core/{config,models,api_client}.py`,
`state/session.py`, and `ui/{chat_view,report_view,common}.py`. This entry
records the decisions that shape that whole layer, made incrementally across
the phase's file-by-file review process rather than as one upfront design.

**Decision 1 — the frontend owns its own wire-contract models; it does not
import `app.schemas.*`.** `core/models.py` defines its own `ChatRequest`/
`ChatResponse`/`ReportGenerationRequest`/`ReportGenerationResponse`/etc.,
independent Pydantic classes that happen to describe the same JSON shapes the
backend's schemas describe, deliberately not the same Python classes. This
mirrors the cross-layer enum duplication already established in Phase 3–8
(`ReportExportFormat` existing independently in `core.config`/`models.enums`/
`schemas.report`) applied at the boundary between two separately deployable
services rather than between two layers of one service: a frontend importing
`app.schemas` would require the backend package to be installed and
importable just to run the UI, defeating independent deployability (relevant
now that Phase 10 will put backend and frontend in separate Docker images).
Two consequences worth naming explicitly: the frontend's models mirror only
the fields each page actually consumes (e.g. `Citation` omits
`document_id`, `RetrievalOptions` omits `document_ids`/`session_scope` —
both tied to document upload, out of this phase's scope per the Phase 9
scoping discussion below), and the backend's *two* different citation shapes
on the wire (`schemas.common.Citation` for `/chat`, the narrower
`core.interfaces.report_exporter.Citation` embedded in `ReportDocument` for
`/report`) got two distinct frontend models (`Citation`, `ReportCitation`)
rather than one loosened one, so each stays an accurate description of what
that specific endpoint actually sends.

**Decision 2 — `core/api_client.py` is the only place that performs HTTP
requests, parses response bodies, or handles `requests` exceptions.** Every
UI component and page calls a function here and only ever receives an
already-validated `core.models` instance or an `APIError`. Three
`APIError` subclasses distinguish *why* a call failed:
`APIConnectionError` (transport-level — connection refused, timeout; always
`retryable=True`), `APIResponseError` (the backend responded with a non-2xx
status and a well-formed `ErrorEnvelope` — every failure path in the
backend's `error_handler.py`, including FastAPI's own validation errors,
funnels through that one envelope shape, so this is the only error-parsing
path this client needs), and `APIValidationError` (a 2xx response whose body
didn't match the expected contract — a signal that the backend's contract
has drifted from what this frontend was built against, not a "request
failed" case). Every successful response is validated through a generic
`DataEnvelope[T]` wrapper before any UI code touches it, so a malformed or
unexpectedly-shaped payload fails fast as a clear validation error rather
than surfacing later as an `AttributeError` deep inside a rendering
function.

**Decision 3 — `state/session.py` is the only module that touches
`st.session_state` directly.** Three independent state groups (active chat
session, last generated report, UI preferences), each with its own clear
function so resetting one never side-effects another. Every getter lazily
initializes its own key on first access rather than requiring an
`init_session_state()` call to have already run — necessary because
Streamlit's multipage model lets a user land directly on
`pages/1_Report.py` via URL without `app.py` ever executing in that
session. `get_retrieval_options()` was originally private to `app.py` and
moved here once `pages/1_Report.py` needed the identical packaging logic —
the same "implement twice, extract on the second real need" rule already
used for `app.reports.naming` in Phase 8.

**Decision 4 — `ui/common.py` exists for the same extraction reason, kept
deliberately minimal.** `render_api_error()` was first written directly
inside `app.py` (the "don't extract with only one consumer" call made
explicitly at the time); once `pages/1_Report.py` needed identical
rendering, it moved to this new, single-function module rather than either
duplicating it or growing a speculative general-purpose `ui/utils.py`.

**Decision 5 — presentational components report user interaction back via
return values, never by acting on it themselves.** `chat_view.
render_conversation()` returns the text of a clicked follow-up suggestion;
`report_view.render_report()` returns whether "Prepare download" was just
clicked. Neither function makes an HTTP call or touches session state — the
owning page (`app.py`/`1_Report.py`) is what acts on the signal. This is the
same pattern `st.chat_input` itself uses (return the value, let the caller
decide), applied consistently to every piece of interactive UI this project
added on top of Streamlit's own widgets.

**Decision 6 — report downloads are a genuine two-step flow, not a
simplification.** `st.download_button` requires file bytes to already be
present before it's drawn; there is no "click triggers a fetch" version of
it. So downloading a report is necessarily two separate user-visible
actions: a plain button that, only when clicked, fetches the file via
`api_client.download_report()` and caches the result in
`state.session` (never fetched automatically on report generation), followed
by the real `download_button` once those bytes exist. `filename`/`mime_type`
are parsed from the response's `Content-Disposition`/`Content-Type` headers
rather than reconstructed on the frontend, so the backend's existing
`FileResponse(..., filename=download.filename)` contract stays the single
source of truth for both.

**Decision 7 — Phase 9 scope excludes `/upload`, `/history`, and a
health-check route.** The backend survey at the start of this phase found
`app/schemas/upload.py` fully designed with no `UploadService`/route at
all, and `HistoryService`/`HealthService` both fully implemented and already
wired into `core/dependencies.py` (`get_history_service`/`get_health_service`)
with no route file for either — a real, asymmetric gap (`/upload` needs a
new service designed from scratch; `/history`/`/health` need only a thin
route file each). Presented to the user as an explicit scoping choice;
"Chat + Report only" was chosen, deferring all three. This frontend was
therefore built against exactly `/chat` and `/report` as they exist today,
with no speculative UI for document upload, session history browsing, or a
health dashboard gated on backend routes that don't exist yet.

**Stabilization pass (this entry's own trigger).** `frontend/pyproject.toml`
added, mirroring the backend's Black/Ruff/mypy configuration exactly
(`line-length=120`, the same conservative Ruff `select`, non-strict mypy) —
same versions (`black==26.5.1`, `ruff==0.16.1`, `mypy==2.3.0`) pinned in a new
`frontend/requirements-dev.txt`, so both halves of the project are checked by
one consistent toolchain. First pass found:

- **One real structural bug:** `frontend/__init__.py`, created during initial
  scaffolding, was dead weight nothing ever imported (confirmed via a
  repository-wide grep — every internal import is the bare `core.`/`state.`/
  `ui.` form, exactly matching `frontend/`'s actual role as the `streamlit
  run` entry directory, the same way `backend/` has no top-level
  `__init__.py` above `app/`). It also actively broke mypy ("Source file
  found twice under different module names") by making mypy's directory walk
  treat the same files as both `core.config` and `frontend.core.config`.
  Removed.
- **One real defensive-hardening finding**, the same category as Phase 8's
  `vector_store.py` fix: `zip(columns, suggestions)` in
  `chat_view._render_follow_up_suggestions` changed to `zip(..., strict=True)`
  — the two are always the same length by construction (`columns` is built as
  exactly `len(suggestions)` columns), so this documents that invariant and
  fails loudly rather than silently truncating if it's ever violated later.
- **Two genuine missing-stub gaps**, both resolved by installing the real stub
  packages (`types-requests`, `pandas-stubs`) rather than suppressing —
  preferred over `ignore_missing_imports` wherever an accurate stub actually
  exists, the same preference already stated in the backend's own mypy
  overrides.
- **One genuine mypy-vs-runtime-generics limitation**, suppressed narrowly and
  documented rather than worked around: `core/api_client.py`'s
  `_parse_data(response, model: Type[ModelT])` subscripts
  `DataEnvelope[model]` using `model` as a runtime value (the caller passes
  either `ChatResponse` or `ReportGenerationResponse`), which mypy cannot
  statically resolve — Pydantic v2 fully supports this at runtime, exercised
  by every real end-to-end test in this phase against a genuinely running
  backend.
- Two files (`app.py`, `core/api_client.py`) needed Black's `line-length=120`
  reformat, applied directly (purely mechanical) and re-verified via the full
  real end-to-end regression suite afterward, the same "apply then verify via
  real functional tests, don't just trust the formatter" approach used in
  Phase 8.

**Test suite decision, matching the existing project plan.** No permanent
pytest-based suite was added for the frontend in this phase, mirroring the
Phase 8 stabilization decision to defer the backend's own permanent suite to
Phase 11 — this ADR log's own closing line already named Phase 11 as
"testing" for exactly this reason, and building one test infrastructure
half now and the other half later would fragment that decision rather than
honor it. Every file in this phase was still verified with real, non-mocked
functional tests as it was built (`streamlit.testing.v1.AppTest` driving
real widget interactions against a genuinely running `uvicorn` instance of
the backend, with only the LLM-dependent `ChatService`/`ReportService`
substituted via FastAPI's `dependency_overrides` and duck-typed fakes,
never a mocking framework) — those test scripts simply aren't yet persisted
as a committed suite, the same situation the backend was already in at the
end of Phase 8.

**Alternatives considered.** Sharing one Pydantic model set between backend
and frontend via a shared internal package — rejected for the independent-
deployability reasoning in Decision 1. Auto-fetching a report's download
bytes as soon as generation completes — rejected because it would fetch
data the user hasn't asked for yet and hold it in `st.session_state`
indefinitely; the explicit two-step flow in Decision 6 was chosen instead.
Building `/upload`/`/history`/`/health` routes now, ahead of the frontend
that would use them — rejected in favor of the narrower "Chat + Report
only" scope explicitly chosen in Decision 7.

**Tradeoffs accepted.** The frontend's models will need manual updates
whenever the backend's corresponding schema changes, rather than picking up
the change automatically via a shared import — accepted as the direct cost
of Decision 1's independent-deployability benefit; `APIValidationError`
exists specifically so that drift fails loudly and immediately rather than
silently.

---

*(This log will continue to grow as Phases 10–12 are implemented — Docker,
testing, and the final README will each surface their own decisions.)*
