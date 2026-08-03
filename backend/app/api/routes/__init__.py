"""API route modules -- one file per resource, each exposing a plain ``fastapi.APIRouter``.

Mounting these routers onto the application instance (alongside
``app.api.middleware.error_handler.register_exception_handlers``,
``Settings.ensure_runtime_directories()``, and ``Database.create_all()``)
is ``app.main``'s job, not this package's -- consistent with every other
startup-lifecycle concern already documented as belonging there.
"""
