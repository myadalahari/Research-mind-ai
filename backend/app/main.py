"""
Application composition root: builds the ``FastAPI`` app, wires every
cross-cutting concern (logging, CORS, exception handling, routing), and
owns process startup/shutdown lifecycle.

Every other module written so far has pointed here by name rather than
performing these steps itself:

* ``app.core.logging.setup_logging`` -- "Call this exactly once, at
  application startup (``app.main`` on FastAPI startup, ...)".
* ``app.core.config.Settings.ensure_runtime_directories`` -- "Called once
  at application startup (see ``app.main``)".
* ``app.database.session.Database.create_all`` -- documented as an
  explicit, opt-in call, never automatic.
* ``app.api.middleware.error_handler.register_exception_handlers`` --
  "Call once at application startup, immediately after constructing the
  ``FastAPI`` instance in ``app.main``".
* ``app.api.routes``'s own package docstring -- mounting routers onto the
  application instance is this file's job, not theirs.

This file is where all of those forward references finally resolve. It
contains no business logic of its own -- only composition.

Every direct call this module makes into ``app.core.dependencies`` goes
through the module reference (``dependencies.get_settings()``,
``dependencies.get_database()``), never ``from app.core.dependencies
import get_settings``. This is deliberate, not a style preference: a
plain ``from ... import`` would bind its own separate name in this
module's namespace, and every prior test suite in this project overrides
the DI graph for tests by monkeypatching the *module attribute*
(``dependencies.get_settings = lambda: fake_settings``) -- a technique
that only reaches call sites resolving the name through
``app.core.dependencies``'s own namespace at call time, which a
``from...import`` binding here would silently opt out of. Routes
(``app.api.routes.chat``) don't have this concern: ``Depends(get_chat_service)``
is overridden by FastAPI's ``dependency_overrides``, which substitutes by
object identity, not by re-resolving a module attribute.

``chat_router``, ``report_router``, and ``health_router`` are wired below.
``app/api/routes``'s own docstring already establishes the pattern
(``app.include_router(...)``) for whatever routers come next. ``health_router``
was added in Phase 10, specifically to give Docker's container healthcheck
something more meaningful than bare process liveness to check -- ``upload``
and ``history`` remain unbuilt (see the Phase 9 scoping discussion), still
deliberately left for their own, separate steps. Every router here was
mounted as its own plumbing step only after its route file was built and
tested in isolation -- this file never grows a new router until there's a
working, tested one to add.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware.error_handler import register_exception_handlers
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.report import router as report_router
from app.core import dependencies
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Startup: ensure on-disk directories exist, then ensure the database
    schema exists. Shutdown: dispose the database engine's connection
    pool. Both halves run exactly once per process, around the app
    serving any requests -- matching ``Database.create_all()``/``dispose()``'s
    own "call once at startup/shutdown" contract.
    """
    settings = dependencies.get_settings()
    settings.ensure_runtime_directories()

    database = dependencies.get_database()
    await database.create_all()

    logger.info(
        "ResearchMind AI startup complete",
        extra={"environment": settings.app.environment.value, "version": settings.app.version},
    )
    yield

    await database.dispose()
    logger.info("ResearchMind AI shutdown complete")


def create_app() -> FastAPI:
    """
    Build and fully wire the ``FastAPI`` application.

    A factory (not just a module-level ``app = FastAPI(...)``) so tests
    can construct a fresh, independently-configured app instance rather
    than importing and mutating one process-wide singleton -- the same
    reasoning ``app.core.dependencies``' cached-singleton design already
    applies one layer down.
    """
    settings = dependencies.get_settings()
    # First thing, before anything else below might log -- so route
    # registration, middleware setup, etc. are all captured with the
    # correctly configured formatter/level from the very first line.
    setup_logging(settings)

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        debug=settings.app.debug,
        lifespan=_lifespan,
    )

    register_exception_handlers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router, prefix=settings.app.api_prefix)
    app.include_router(report_router, prefix=settings.app.api_prefix)
    app.include_router(health_router, prefix=settings.app.api_prefix)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    _settings = dependencies.get_settings()
    uvicorn.run(app, host=_settings.app.api_host, port=_settings.app.api_port)
