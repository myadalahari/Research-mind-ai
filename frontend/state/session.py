"""
Centralized ``st.session_state`` access for the ResearchMind AI frontend.

This is the **only** module in the frontend that reads or writes
``st.session_state`` directly. Every page (``app.py``,
``pages/1_Report.py``) and every UI component (``ui/chat_view.py``,
``ui/report_view.py``) goes through the small, typed functions below --
never a raw ``st.session_state["some_key"]`` lookup. This keeps the storage
layout (key names, defaults, what gets cleared together) a concern of this
one file, so it can change without touching every place that reads state.

Three independent state groups are stored, each with its own clear
function so resetting one never has a side effect on another:

* **Active chat session** -- ``session_id`` and the running list of
  ``ChatTurn``s. This is the *only* copy of the conversation anywhere in
  the frontend; there is no ``/history`` endpoint in this phase (see the
  Phase 9 scoping discussion), so this is primary state, not a cache of
  something fetchable elsewhere.
* **Last generated report** -- the last ``ReportGenerationResponse``, the
  query that produced it, and (separately, so it can be cleared
  independently) the last downloaded file. ``st.download_button`` needs
  file bytes available before a click, not fetched on click, so a
  downloaded file is deliberately cached here -- but only the *current*
  report's file, never persisted longer than that. Every path that
  replaces or clears the current report (``set_last_report``,
  ``clear_last_report``, ``reset_all``) drops the cached file immediately,
  so a large PDF's bytes are never held in ``st.session_state`` once
  they're no longer the download for the report currently on screen.
* **UI preferences** -- the execution-trace visibility toggle and
  retrieval overrides (``top_k``, ``include_web_search``). These are
  choices the user makes about how to *use* the app, not data returned by
  it, and are kept separate from both state groups above for that reason.

Every getter lazily initializes its own key via ``_get()`` on first access,
rather than requiring some ``init_session_state()`` call to have already
run. This matters concretely for a Streamlit multipage app: a user can
land directly on ``pages/1_Report.py`` via a bookmarked URL without
``app.py`` ever having executed in that browser session, and every
function here must still behave correctly in exactly that case.

This module has no opinion on layout -- no ``st.button``/``st.checkbox``/
``st.sidebar`` calls anywhere here, only ``st.session_state`` reads and
writes -- so it is reusable, unchanged, from any page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypeVar

import streamlit as st

from core.api_client import DownloadedFile
from core.models import ChatResponse, ReportGenerationResponse, RetrievalOptions

T = TypeVar("T")

# =============================================================================
# Storage keys (private -- callers never see or use these directly)
# =============================================================================

_KEY_SESSION_ID = "rm_session_id"
_KEY_TURNS = "rm_turns"
_KEY_LAST_REPORT = "rm_last_report"
_KEY_LAST_REPORT_QUERY = "rm_last_report_query"
_KEY_LAST_REPORT_FILE = "rm_last_report_file"
_KEY_INCLUDE_TRACE = "rm_include_trace"
_KEY_RETRIEVAL_TOP_K = "rm_retrieval_top_k"
_KEY_RETRIEVAL_INCLUDE_WEB_SEARCH = "rm_retrieval_include_web_search"


@dataclass(frozen=True)
class ChatTurn:
    """One query/answer exchange as rendered in the frontend's conversation view."""

    query: str
    response: ChatResponse


# =============================================================================
# Internal helpers
# =============================================================================


def _get(key: str, default_factory: Callable[[], T]) -> T:
    """
    Return ``st.session_state[key]``, lazily initializing it to
    ``default_factory()`` on first access.

    The single point every public getter routes through, so "missing key"
    is never a case any caller (inside or outside this module) has to
    think about.
    """
    if key not in st.session_state:
        st.session_state[key] = default_factory()
    return st.session_state[key]  # type: ignore[no-any-return]


def _set(key: str, value: Any) -> None:
    st.session_state[key] = value


# =============================================================================
# Active chat session
# =============================================================================


def get_active_session_id() -> Optional[str]:
    """The backend-assigned session id for the current conversation, or ``None`` before the first response."""
    return _get(_KEY_SESSION_ID, lambda: None)


def get_turns() -> List[ChatTurn]:
    """
    The current conversation's turns, oldest first.

    Returns a shallow copy -- mutating the returned list (append, sort,
    clear) never affects stored state; ``append_turn()`` is the only way
    to add to it.
    """
    return list(_get(_KEY_TURNS, list))


def append_turn(query: str, response: ChatResponse) -> None:
    """
    Record one query/answer exchange and adopt ``response.session_id`` as
    the active session.

    The only place ``session_id`` is ever updated: a successful
    ``ChatResponse`` is always the authoritative source for which session
    a turn belongs to (a new conversation's first response is what
    *assigns* the session id in the first place), so there is exactly one
    code path that performs this update rather than every caller having
    to remember to also call ``set_active_session_id`` alongside
    appending a turn.
    """
    turns = _get(_KEY_TURNS, list)
    turns.append(ChatTurn(query=query, response=response))
    _set(_KEY_TURNS, turns)
    _set(_KEY_SESSION_ID, response.session_id)


def clear_chat() -> None:
    """Start a new conversation: clear the active session id and all turns. Leaves report state and preferences untouched."""
    _set(_KEY_SESSION_ID, None)
    _set(_KEY_TURNS, [])


# =============================================================================
# Last generated report
# =============================================================================


def get_last_report() -> Optional[ReportGenerationResponse]:
    """The most recently generated report's response, or ``None`` if none has been generated yet this session."""
    return _get(_KEY_LAST_REPORT, lambda: None)


def get_last_report_query() -> Optional[str]:
    """The query that produced ``get_last_report()``, for re-display alongside it."""
    return _get(_KEY_LAST_REPORT_QUERY, lambda: None)


def set_last_report(query: str, response: ReportGenerationResponse) -> None:
    """
    Record a newly generated report.

    Always drops any previously cached downloaded file first: a file
    cached against an earlier report is never valid for this one, and
    leaving it in place even briefly would risk a UI component serving a
    stale download alongside a fresh report.
    """
    _set(_KEY_LAST_REPORT_FILE, None)
    _set(_KEY_LAST_REPORT_QUERY, query)
    _set(_KEY_LAST_REPORT, response)


def get_last_report_file() -> Optional[DownloadedFile]:
    """
    The last report's downloaded file (bytes + filename + mime type), if
    it has been fetched via ``core.api_client.download_report`` yet this
    session -- ``None`` until a UI component fetches and caches it.

    Deliberately not fetched automatically when a report is generated:
    generating a report and downloading it are separate user actions, and
    holding a large PDF's bytes in ``st.session_state`` is only justified
    once the user has actually asked to download it.
    """
    return _get(_KEY_LAST_REPORT_FILE, lambda: None)


def set_last_report_file(file: DownloadedFile) -> None:
    """Cache the current report's downloaded file so a rerun (e.g. from an unrelated widget interaction) doesn't re-fetch it."""
    _set(_KEY_LAST_REPORT_FILE, file)


def clear_last_report() -> None:
    """Clear the last report, its query, and its cached file together -- these three always travel as one unit."""
    _set(_KEY_LAST_REPORT, None)
    _set(_KEY_LAST_REPORT_QUERY, None)
    _set(_KEY_LAST_REPORT_FILE, None)


# =============================================================================
# UI preferences
# =============================================================================


def get_include_trace() -> bool:
    """Whether the "show agent execution trace" toggle is on. Defaults to off (matches ChatRequest's own default framing)."""
    return _get(_KEY_INCLUDE_TRACE, lambda: False)


def set_include_trace(value: bool) -> None:
    _set(_KEY_INCLUDE_TRACE, value)


def get_retrieval_top_k() -> Optional[int]:
    """User override for retrieval top_k, or ``None`` to let the backend use its own configured default."""
    return _get(_KEY_RETRIEVAL_TOP_K, lambda: None)


def set_retrieval_top_k(value: Optional[int]) -> None:
    _set(_KEY_RETRIEVAL_TOP_K, value)


def get_include_web_search() -> Optional[bool]:
    """User override for whether web search runs, or ``None`` to let the Planner decide."""
    return _get(_KEY_RETRIEVAL_INCLUDE_WEB_SEARCH, lambda: None)


def set_include_web_search(value: Optional[bool]) -> None:
    _set(_KEY_RETRIEVAL_INCLUDE_WEB_SEARCH, value)


def get_retrieval_options() -> Optional[RetrievalOptions]:
    """
    Package the stored retrieval preferences into a ``RetrievalOptions``
    ready to attach to a ``ChatRequest``/``ReportGenerationRequest``, or
    ``None`` if neither preference has been set.

    Originally private to ``app.py``; moved here once ``pages/1_Report.py``
    needed the identical logic -- both a chat message and a report request
    should draw retrieval overrides from the same one set of preferences
    rather than each page keeping its own copy of how to build the object,
    and this module already owns the two values being packaged.
    """
    top_k = get_retrieval_top_k()
    include_web_search = get_include_web_search()
    if top_k is None and include_web_search is None:
        return None
    return RetrievalOptions(top_k=top_k, include_web_search=include_web_search)


# =============================================================================
# Full reset
# =============================================================================


def reset_all() -> None:
    """Clear every piece of state this module owns: chat, report, and preferences alike."""
    clear_chat()
    clear_last_report()
    _set(_KEY_INCLUDE_TRACE, False)
    _set(_KEY_RETRIEVAL_TOP_K, None)
    _set(_KEY_RETRIEVAL_INCLUDE_WEB_SEARCH, None)
