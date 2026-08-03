"""
Generic CRUD base for the repository pattern.

See the database layer design discussion (and the Architecture Decision
Log) for the full rationale. Summary of what's encoded here:

* A repository is constructed with an already-open ``AsyncSession`` — it
  never opens or commits its own transaction. The caller (a service, via
  ``app.database.session.Database.session()``) owns the unit-of-work
  boundary, so multiple repositories can be composed into one atomic
  operation (e.g. inserting a turn together with its citations and
  execution steps).
* Write methods call ``session.flush()``, never ``session.commit()`` —
  flush surfaces constraint violations and makes writes visible to
  subsequent reads within the same unit of work, without ending the
  transaction the caller owns.
* Every SQLAlchemy exception is translated into a
  ``app.core.exceptions.DatabaseError`` (or the more specific
  ``RecordConflictError`` for integrity violations) before it leaves this
  module, chained with ``from exc``. This is what keeps the service layer
  genuinely framework/ORM-agnostic: a service catching a repository
  failure never needs to import anything from ``sqlalchemy``.
* Generic over ``ModelType`` (bound to ``app.database.session.Base``),
  with the concrete model set as a subclass attribute
  (``class FooRepository(BaseRepository[Foo]): model = Foo``), not passed
  to the constructor — keeps concrete repositories declarative and lets
  type checkers infer return types per subclass.
"""

from __future__ import annotations

from typing import Any, Generic, List, NoReturn, Optional, Sequence, Type, TypeVar

from sqlalchemy import func, select
from sqlalchemy.engine import Result
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import ColumnExpressionArgument, Executable
from sqlalchemy.sql.elements import UnaryExpression

from app.core.exceptions import DatabaseError, RecordConflictError, RecordNotFoundError
from app.core.logging import get_logger
from app.database.session import Base

ModelType = TypeVar("ModelType", bound=Base)

logger = get_logger(__name__)


class BaseRepository(Generic[ModelType]):
    """
    Generic CRUD operations shared by every concrete repository.

    Concrete repositories subclass this and set the ``model`` class
    attribute; domain-specific query methods (e.g. "list sessions with
    aggregate turn/document counts") live on the subclass, not here — this
    base class deliberately covers only the CRUD operations that are
    identical across every entity.
    """

    model: Type[ModelType]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -------------------------------------------------------------------
    # Reads
    # -------------------------------------------------------------------

    async def get_by_id(self, entity_id: str) -> Optional[ModelType]:
        """Return the row with this primary key, or ``None`` if it doesn't exist."""
        try:
            return await self._session.get(self.model, entity_id)
        except SQLAlchemyError as exc:
            await self._handle_error(exc, f"Failed to fetch {self.model.__name__} id={entity_id!r}.")

    async def require_by_id(self, entity_id: str) -> ModelType:
        """
        Return the row with this primary key, or raise ``RecordNotFoundError``.

        Centralizes the common "fetch or 404" pattern so every service
        doesn't reimplement its own ``if entity is None: raise ...``.
        """
        entity = await self.get_by_id(entity_id)
        if entity is None:
            raise RecordNotFoundError(
                f"{self.model.__name__} with id={entity_id!r} was not found.",
                details={"model": self.model.__name__, "entity_id": entity_id},
            )
        return entity

    async def list(
        self,
        *filters: ColumnExpressionArgument[bool],
        order_by: Optional[Sequence[UnaryExpression[Any]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ModelType]:
        """
        Return rows matching ``filters`` (each a SQLAlchemy column
        expression, e.g. ``Document.session_id == session_id``), optionally
        ordered and paginated.
        """
        stmt = select(self.model)
        if filters:
            stmt = stmt.where(*filters)
        if order_by:
            stmt = stmt.order_by(*order_by)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self._execute(stmt, f"Failed to list {self.model.__name__} rows.")
        return list(result.scalars().all())

    async def count(self, *filters: ColumnExpressionArgument[bool]) -> int:
        """Return the number of rows matching ``filters``."""
        stmt = select(func.count()).select_from(self.model)
        if filters:
            stmt = stmt.where(*filters)
        result = await self._execute(stmt, f"Failed to count {self.model.__name__} rows.")
        return int(result.scalar_one())

    async def exists(self, *filters: ColumnExpressionArgument[bool]) -> bool:
        """Return whether any row matches ``filters``."""
        return await self.count(*filters) > 0

    # -------------------------------------------------------------------
    # Writes
    # -------------------------------------------------------------------

    async def add(self, entity: ModelType) -> ModelType:
        """Stage ``entity`` for insertion and flush it to the database."""
        self._session.add(entity)
        await self._flush(f"Failed to create {self.model.__name__}.")
        return entity

    async def add_many(self, entities: Sequence[ModelType]) -> Sequence[ModelType]:
        """Stage multiple entities for insertion and flush them together."""
        self._session.add_all(entities)
        await self._flush(f"Failed to create {self.model.__name__} rows.")
        return entities

    async def delete(self, entity: ModelType) -> None:
        """Delete ``entity`` and flush the deletion."""
        await self._session.delete(entity)
        await self._flush(f"Failed to delete {self.model.__name__} id={getattr(entity, 'id', None)!r}.")

    async def delete_by_id(self, entity_id: str) -> bool:
        """Delete the row with this primary key. Returns ``False`` if it didn't exist."""
        entity = await self.get_by_id(entity_id)
        if entity is None:
            return False
        await self.delete(entity)
        return True

    async def flush(self) -> None:
        """
        Explicitly flush pending changes without ending the transaction.

        Exposed for subclasses/services that mutate an already-tracked
        entity in place (e.g. ``document.ingestion_status = ...``) rather
        than going through ``add()``, and want those changes visible to a
        subsequent read within the same unit of work.
        """
        await self._flush(f"Failed to flush pending {self.model.__name__} changes.")

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    async def _execute(self, stmt: Executable, error_message: str) -> Result[Any]:
        """
        Run an arbitrary SQLAlchemy statement with the same
        log-rollback-translate error handling as every CRUD method above.

        Protected rather than private (single underscore) so subclasses
        can use it for domain-specific queries (aggregate subqueries,
        custom joins, ...) without re-implementing the
        ``try/except SQLAlchemyError`` boilerplate — this is exactly the
        query path ``list()``/``count()`` themselves use.
        """
        try:
            return await self._session.execute(stmt)
        except SQLAlchemyError as exc:
            await self._handle_error(exc, error_message)

    async def _flush(self, error_message: str) -> None:
        try:
            await self._session.flush()
        except SQLAlchemyError as exc:
            await self._handle_error(exc, error_message)

    async def _handle_error(self, exc: SQLAlchemyError, message: str) -> NoReturn:
        """
        Log, roll back, and re-raise a SQLAlchemy failure as the
        appropriate ``DatabaseError`` subclass. Always raises.

        The rollback is not optional. Once any statement in a session's
        transaction fails (e.g. a flush raises ``IntegrityError``),
        SQLAlchemy marks that ``Session`` unusable — any further operation,
        *including the commit that ``Database.session()`` issues on normal
        exit*, raises ``PendingRollbackError`` until ``rollback()`` is
        called. Rolling back here is what lets a caller catch a translated
        exception (e.g. ``RecordConflictError``) and have the surrounding
        ``async with database.session():`` block exit cleanly, rather than
        having the real error masked by a second, confusing failure.

        This does mean a failed operation discards every uncommitted
        change made earlier in the same unit of work, not just the one
        that failed -- that mirrors ordinary SQL transaction semantics (one
        failed statement aborts the transaction) and is called out here
        deliberately: services composing multiple repository calls in one
        ``database.session()`` block should treat any ``DatabaseError`` as
        having invalidated the whole transaction.
        """
        logger.error(
            "Repository operation failed: %s",
            message,
            exc_info=exc,
            extra={"model": self.model.__name__, "sqlalchemy_error_type": type(exc).__name__},
        )
        await self._session.rollback()
        if isinstance(exc, IntegrityError):
            raise RecordConflictError.wrap(exc, message) from exc
        raise DatabaseError.wrap(exc, message) from exc
