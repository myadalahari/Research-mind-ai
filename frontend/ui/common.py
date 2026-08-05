"""
Shared presentation helpers used by more than one page.

Deliberately kept small -- this module exists to hold genuinely shared
rendering logic, not to become a catch-all utilities module. Add to it only
when a second real page/component needs something a first one already has
(the same "implement twice, extract on the second need" rule this project
has followed elsewhere, e.g. ``app.reports.naming`` on the backend).
``render_api_error`` was first written directly inside ``app.py`` and only
moved here once ``pages/1_Report.py`` needed the identical behavior.
"""

from __future__ import annotations

import streamlit as st

from core.api_client import APIError


def render_api_error(error: APIError) -> None:
    """Render an ``APIError`` as a friendly message plus its machine-readable code -- never the raw exception."""
    message = f"**{error.message}**\n\n`{error.error_code}`"
    if error.retryable:
        message += "\n\nThis may be a temporary issue -- you can try again."
    st.error(message)
