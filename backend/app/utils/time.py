"""
Single source of truth for "now" across the codebase.

Every timestamp default in this project (Pydantic ``default_factory``
fields across ``schemas/*`` and ``core/interfaces/*``, and any future
service-layer code) uses ``utc_now()`` from this module rather than
``datetime.utcnow()`` directly.

Why this matters: ``datetime.utcnow()`` returns a *naive* datetime (no
``tzinfo``). Naive and timezone-aware datetimes cannot be compared or
subtracted — mixing them raises ``TypeError: can't subtract offset-naive
and offset-aware datetimes`` at runtime, not at type-check time. Every API
example in this codebase's docstrings shows UTC timestamps with a ``Z``
suffix (e.g. ``"2026-07-29T10:15:03Z"``), and any correctly-written caller
constructing datetimes explicitly should reasonably use
``datetime.now(timezone.utc)`` (the modern, non-deprecated equivalent) —
so a naive default anywhere in this codebase is a latent crash waiting for
exactly that kind of caller. ``utc_now()`` always returns a timezone-aware
UTC datetime, so arithmetic and comparisons against it are always safe
regardless of how the other operand was constructed, as long as it's also
aware.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC ``datetime``."""
    return datetime.now(timezone.utc)
