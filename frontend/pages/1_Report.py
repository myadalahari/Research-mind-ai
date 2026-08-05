"""
ResearchMind AI -- Streamlit frontend Report page.

Structured identically to ``app.py``: a thin orchestration layer that owns
layout and workflow sequencing only. Every actual decision lives elsewhere
-- HTTP calls go through ``core.api_client``, all state lives in
``state.session``, and rendering a ``ReportGenerationResponse`` (including
branching on ``status``) is entirely ``ui.report_view.render_report``'s job.
This file never inspects ``response.status`` itself.

The report-generation flow is a single path regardless of how the page got
there (a fresh submission or an unrelated rerun such as clicking "Clear
report" first): collect input -> call ``generate_report()`` -> store the
response via ``session.set_last_report()`` -> render through
``report_view``. A successful HTTP call is stored unconditionally, even
when the report itself came back with ``status="failed"`` -- that is a
normal, successful response carrying a business-level failure (the
backend's own documented two-tier failure handling from Phase 8), not an
``APIError``, and ``report_view`` is what decides how to present it.

Downloading is necessarily two separate user-visible steps, not one,
because ``st.download_button`` requires file bytes to already be present
before it's drawn -- there is no "click triggers a fetch" version of it.
``report_view.render_report()`` reports back (via its boolean return
value, the same pattern ``chat_view``'s follow-up suggestions use) only
when the user has just clicked "Prepare download"; this file is the only
place that acts on that signal by actually calling
``api_client.download_report()`` and caching the result via
``session.set_last_report_file()``. Nothing here ever fetches a file the
user hasn't explicitly asked for.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from core.api_client import APIError, download_report, generate_report
from core.models import ReportExportFormat, ReportGenerationRequest, ReportGenerationResponse
from state import session
from ui.common import render_api_error
from ui.report_view import render_report

logger = logging.getLogger("researchmind_frontend")

_FALLBACK_MIME_TYPES = {
    ReportExportFormat.PDF: "application/pdf",
    ReportExportFormat.MARKDOWN: "text/markdown",
}


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("ResearchMind AI")
        active_session_id = session.get_active_session_id()
        st.caption(f"Session: `{active_session_id}`" if active_session_id else "No active chat session yet.")

        if session.get_last_report() is not None and st.button("Clear report", use_container_width=True):
            session.clear_last_report()
            st.rerun()


def _render_input_form() -> Optional[ReportGenerationRequest]:
    """
    Render the report-request form and, only on submission, return the
    ``ReportGenerationRequest`` to send -- ``None`` otherwise.

    A ``st.form`` bundles every field behind a single submit action so
    filling in the query/title doesn't trigger a rerun per keystroke.
    """
    active_session_id = session.get_active_session_id()

    with st.form("report_request_form"):
        query = st.text_area(
            "Research topic or question",
            placeholder="Create a report on Quantum Computing",
            help="The topic the report should address.",
        )
        export_format_label = st.radio("Export format", options=["PDF", "Markdown"], horizontal=True)
        conversation_title = st.text_input(
            "Report title (optional)",
            help="Leave blank to let the backend generate one from the topic.",
        )

        continue_session = False
        if active_session_id is not None:
            continue_session = st.checkbox(
                f"Continue current chat session (`{active_session_id}`)",
                value=True,
                help="Generate this report from your current conversation instead of starting a new session.",
            )

        submitted = st.form_submit_button("Generate report")

    if not submitted:
        return None
    if not query.strip():
        st.warning("Please enter a research topic or question.")
        return None

    export_format = ReportExportFormat.PDF if export_format_label == "PDF" else ReportExportFormat.MARKDOWN
    return ReportGenerationRequest(
        query=query,
        # Only ever reads the active session id to decide whether to attach it to this
        # request -- never mutates chat session state itself, that stays app.py's concern.
        session_id=active_session_id if continue_session else None,
        retrieval_options=session.get_retrieval_options(),
        export_format=export_format,
        conversation_title=conversation_title or None,
    )


def _generate_report(request: ReportGenerationRequest) -> None:
    try:
        with st.spinner("Generating report... this can take a while."):
            response = generate_report(request)
    except APIError as exc:
        render_api_error(exc)
        return
    except Exception:
        # Last-resort safety net only: log it, show one generic message, stop.
        logger.exception("Unexpected error while generating a report")
        st.error("Something went wrong while generating the report. Please try again.")
        return

    # Stored unconditionally: an HTTP-successful response whose report
    # status is "failed" is still a successful call to record -- see this
    # module's own docstring. set_last_report() already drops any
    # previously cached download file, so a report generated a second time
    # can never be downloaded alongside a stale file from an earlier one.
    session.set_last_report(request.query, response)
    st.rerun()


def _prepare_download(report: ReportGenerationResponse) -> None:
    """
    Fetch the completed report's file and cache it, only ever called in
    direct response to the user clicking "Prepare download" (signaled by
    ``report_view.render_report()``'s return value).
    """
    assert report.download_url is not None  # guaranteed by the backend contract whenever status == COMPLETED

    fallback_filename = f"{report.report_id}.{report.export_format.value}"
    fallback_mime_type = _FALLBACK_MIME_TYPES[report.export_format]

    try:
        with st.spinner("Preparing download..."):
            file = download_report(
                report.download_url, fallback_filename=fallback_filename, fallback_mime_type=fallback_mime_type
            )
    except APIError as exc:
        # The existing report stays on screen exactly as it was; nothing
        # partial or invalid is ever cached, and no download button appears.
        render_api_error(exc)
        return
    except Exception:
        logger.exception("Unexpected error while preparing a report download")
        st.error("Something went wrong while preparing the download. Please try again.")
        return

    session.set_last_report_file(file)
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="ResearchMind AI - Report", page_icon=":test_tube:", layout="wide")
    _render_sidebar()

    st.title("Generate a Research Report")
    st.caption("Runs the same research pipeline as Chat, then exports the result as a downloadable report.")

    request = _render_input_form()
    if request is not None:
        _generate_report(request)

    last_report = session.get_last_report()
    if last_report is not None:
        download_requested = render_report(
            last_report, query=session.get_last_report_query(), cached_file=session.get_last_report_file()
        )
        if download_requested:
            _prepare_download(last_report)


main()
