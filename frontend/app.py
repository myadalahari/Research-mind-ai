"""
ResearchMind AI -- Streamlit frontend entry point (Chat / Research page).

A thin composition layer, structured the same way ``app.main`` is on the
backend: this file owns page layout and wiring only. Every actual decision
lives elsewhere -- HTTP calls go through ``core.api_client``, all state
lives in ``state.session``, and the conversation itself is rendered by
``ui.chat_view.render_conversation``. This file's own job is coordinating
those three: read the sidebar's current choices, hand a new query to
``api_client``, record the result via ``session``, and let Streamlit's
rerun mechanism redraw the page through ``chat_view``'s one rendering path.

Error handling is deliberately layered exactly two ways:

* ``api_client.APIError`` (and its three subclasses -- connection failure,
  a real backend error, or a contract mismatch) is always caught and
  rendered as a friendly message plus the error's ``error_code``, never a
  raw exception or a traceback. This is the expected, designed-for failure
  path -- every way ``api_client.py`` can fail is represented by this one
  exception hierarchy.
* A bare ``except Exception`` around the request call is a last-resort
  safety net for a genuine bug -- something ``api_client.py`` didn't
  anticipate. It does exactly three things (log it, show one generic
  message, stop) and never attempts special-case recovery, mirroring the
  backend's own ``unhandled_exception_handler``'s role as the final,
  deliberately minimal line of defense.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import streamlit as st

from core.api_client import APIError, send_chat_message
from core.models import ChatMode, ChatRequest, RetrievalOptions
from state import session
from ui.chat_view import render_conversation
from ui.common import render_api_error

logger = logging.getLogger("researchmind_frontend")

_MODE_CHOICES = {
    "Research (full multi-agent pipeline)": ChatMode.RESEARCH,
    "Chat (lightweight, conversational)": ChatMode.CHAT,
}
_WEB_SEARCH_CHOICES = ["Auto", "On", "Off"]
_WEB_SEARCH_TO_VALUE = {"Auto": None, "On": True, "Off": False}
_DEFAULT_TOP_K = 6


def _web_search_choice_label(value: Optional[bool]) -> str:
    """Map the stored ``include_web_search`` preference back to its radio label."""
    for label, mapped in _WEB_SEARCH_TO_VALUE.items():
        if mapped is value:
            return label
    return "Auto"  # unreachable in practice; a safe default if the stored value is ever unexpected


def _render_sidebar() -> Tuple[ChatMode, Optional[RetrievalOptions]]:
    """
    Render the sidebar and return the workflow mode and retrieval options
    the next submitted query should use.

    The execution-trace toggle and both retrieval controls are always
    seeded from, and written straight back to, ``state.session`` -- this
    function never lets one of those three settings live only in a
    Streamlit widget's own internal state.
    """
    with st.sidebar:
        st.header("ResearchMind AI")

        active_session_id = session.get_active_session_id()
        st.caption(
            f"Session: `{active_session_id}`" if active_session_id else "New session -- ask a question to begin."
        )

        if st.button("New chat", use_container_width=True):
            session.clear_chat()
            st.rerun()

        st.divider()

        mode_label = st.radio(
            "Workflow",
            options=list(_MODE_CHOICES.keys()),
            help="Research runs the full multi-agent pipeline; Chat is a lighter-weight conversational mode.",
        )
        mode = _MODE_CHOICES[mode_label]

        include_trace = st.checkbox(
            "Show agent execution trace",
            value=session.get_include_trace(),
            help="Include the step-by-step agent execution trace with each response.",
        )
        session.set_include_trace(include_trace)

        with st.expander("Retrieval settings"):
            override_top_k = st.checkbox("Override result count", value=session.get_retrieval_top_k() is not None)
            if override_top_k:
                top_k = st.slider(
                    "Top K", min_value=1, max_value=20, value=session.get_retrieval_top_k() or _DEFAULT_TOP_K
                )
                session.set_retrieval_top_k(top_k)
            else:
                session.set_retrieval_top_k(None)

            web_search_label = st.radio(
                "Web search",
                options=_WEB_SEARCH_CHOICES,
                index=_WEB_SEARCH_CHOICES.index(_web_search_choice_label(session.get_include_web_search())),
                horizontal=True,
            )
            session.set_include_web_search(_WEB_SEARCH_TO_VALUE[web_search_label])

        retrieval_options = session.get_retrieval_options()

    return mode, retrieval_options


def _handle_new_query(query: str, *, mode: ChatMode, retrieval_options: Optional[RetrievalOptions]) -> None:
    """
    Send a new query and, on success, record it via ``state.session`` and
    rerun -- the new turn then reaches the screen through
    ``chat_view.render_conversation``'s normal path, not a separate
    one-off render here.
    """
    request = ChatRequest(
        query=query,
        session_id=session.get_active_session_id(),
        mode=mode,
        retrieval_options=retrieval_options,
    )
    try:
        with st.spinner("Researching..." if mode == ChatMode.RESEARCH else "Thinking..."):
            response = send_chat_message(request, include_trace=session.get_include_trace())
    except APIError as exc:
        render_api_error(exc)
        return
    except Exception:
        # Last-resort safety net only: log it, show one generic message, stop.
        # Never attempt recovery or special-case handling here -- every
        # failure mode api_client.py anticipates is already an APIError.
        logger.exception("Unexpected error while sending a chat message")
        st.error("Something went wrong while processing your request. Please try again.")
        return

    session.append_turn(query, response)
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="ResearchMind AI", page_icon=":test_tube:", layout="wide")

    mode, retrieval_options = _render_sidebar()

    st.title("ResearchMind AI")
    st.caption("Enterprise multi-agent research assistant")

    clicked_suggestion = render_conversation(session.get_turns())
    typed_query = st.chat_input("Ask a research question...")
    query = clicked_suggestion or typed_query
    if query:
        _handle_new_query(query, mode=mode, retrieval_options=retrieval_options)


main()
