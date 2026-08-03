"""
Custom SQLAlchemy column types shared across ``app.models``.

Currently home to a single type, ``UTCDateTime``, which exists to close a
real bug found during live testing of ``ConversationRepository``: plain
``DateTime(timezone=True)`` writes a timezone-aware value correctly, but
SQLite (unlike Postgres) has no native timezone-aware storage, so
SQLAlchemy's sqlite dialect silently returns a *naive* ``datetime`` on
read. Confirmed directly -- a freshly inserted ``ResearchSession`` read
back from the identity map still has ``tzinfo=UTC``, but the same row
reloaded via a new query has ``tzinfo=None``, even though both represent
the identical instant. Any later arithmetic or comparison against
``app.utils.time.utc_now()`` (always aware) -- e.g. "how long since this
session was last active" -- would intermittently hit the exact
``TypeError: can't compare offset-naive and offset-aware datetimes`` bug
already fixed once at the Pydantic layer, except here it would depend on
whether the object came from the identity map or a fresh query, making it
substantially harder to reproduce and diagnose.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import DateTime
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator[datetime]):
    """
    A timezone-aware UTC ``datetime`` that round-trips correctly on every
    dialect, including SQLite.

    Normalizes both directions:

    * On write, a naive ``datetime`` is rejected outright -- every
      timestamp in this codebase must originate from
      ``app.utils.time.utc_now()``, which is always aware -- and an aware
      value is converted to UTC before storage.
    * On read, UTC tzinfo is re-attached to whatever the underlying DBAPI
      returned (a no-op if the driver already preserved it, as Postgres's
      ``TIMESTAMP WITH TIME ZONE`` does via asyncpg).

    Using this type everywhere (instead of relying on
    ``DateTime(timezone=True)`` directly) keeps timestamp behavior
    identical across SQLite (dev/test) and Postgres (a future production
    deployment) rather than depending on per-dialect quirks that would
    only surface as a bug in whichever environment wasn't being tested.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect: Dialect) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                "UTCDateTime received a naive datetime; every timestamp in this "
                "codebase must be timezone-aware (use app.utils.time.utc_now())."
            )
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: Optional[Any], dialect: Dialect) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
