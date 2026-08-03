"""
Database engine, async session factory, and declarative base.

``Base`` must exist before any ORM model (``app.models.*``) can be
defined — every model class in this project subclasses it — so this
module has no dependency on ``app.models`` itself; models depend on it,
never the other way around.

Built on SQLAlchemy 2.0's async engine (``sqlite+aiosqlite://`` by
default; a future Postgres deployment would use ``postgresql+asyncpg://``)
to match the async-everywhere pattern already established by every
``core.interfaces`` ABC.

Table creation here uses ``Base.metadata.create_all`` rather than Alembic
migrations — a deliberate scope decision for this project: SQLite/dev
usage doesn't need a migration framework, and introducing Alembic without
a real multi-environment deployment pipeline to justify it would be
process overhead the project doesn't currently need. If this were deployed
against a shared Postgres instance with other engineers making concurrent
schema changes, Alembic would become the right call — noted here rather
than silently deferred.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base every ORM model in ``app.models`` subclasses."""


def _enable_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    """
    SQLite does not enforce foreign key constraints by default — ``PRAGMA
    foreign_keys`` is off unless set explicitly per connection. Without
    this, every ``ON DELETE CASCADE``/``ON DELETE SET NULL`` declared on
    ``app.models`` relationships would silently do nothing at the database
    level for anything that isn't routed through SQLAlchemy's own
    ORM-level relationship cascades (e.g. a raw bulk DELETE). This is a
    SQLite-specific gap — PostgreSQL enforces foreign keys by default — so
    the listener only attaches when the engine's dialect is actually
    SQLite.
    """
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Database:
    """
    Owns the async engine and session factory for the application's
    lifetime.

    Constructed once at application startup (see ``core.dependencies``,
    built later) and shared across the process — mirroring how
    ``LLMService``/``VectorStore``/etc. adapters are constructed once
    rather than per-request. Also directly satisfies the
    ``DatabaseHealthChecker`` protocol (``app.services.health_service``)
    via its own ``health_check()`` method, so it can be injected into
    ``HealthService`` without a separate wrapper class.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._engine: AsyncEngine = create_async_engine(
            self._settings.database.url,
            echo=self._settings.database.echo,
            pool_pre_ping=self._settings.database.pool_pre_ping,
        )
        _enable_sqlite_foreign_keys(self._engine)
        self._session_factory = async_sessionmaker(bind=self._engine, expire_on_commit=False, autoflush=False)

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """
        Yield an ``AsyncSession`` scoped to a single unit of work,
        committing on success and rolling back on any exception.

        Repositories and services use this rather than constructing
        sessions directly, so transaction boundaries are consistent
        everywhere: ``async with database.session() as session: ...``.
        """
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def create_all(self) -> None:
        """
        Create every table registered on ``Base.metadata``.

        Call once at application startup, after every ``app.models``
        module has been imported (import order matters here: a model
        class only registers itself on ``Base.metadata`` once its module
        has been imported at least once).
        """
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        logger.info("Database schema ensured (create_all)")

    async def dispose(self) -> None:
        """Close the engine's connection pool. Call once at application shutdown."""
        await self._engine.dispose()

    async def health_check(self) -> bool:
        """
        Return ``True`` if the database is reachable, ``False`` otherwise.
        Never raises — satisfies ``app.services.health_service.DatabaseHealthChecker``.
        """
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            logger.exception("Database health check failed")
            return False
