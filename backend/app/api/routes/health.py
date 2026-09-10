"""
``GET /health`` -- liveness/readiness and per-dependency health.

As thin as ``app.api.routes.chat``/``report`` (see those files' own
docstrings for the full rationale): every actual decision already lives in
``HealthService`` -- which dependencies to check, how to run them
concurrently, and how to roll up an overall ``status``/``ready``. This
file's only job is HTTP-specific: parsing the request (there is none),
wrapping the result in the standard ``DataResponse`` envelope, and mapping
``HealthCheckResponse.ready`` onto an HTTP status code so that generic
infrastructure tooling -- a Docker ``HEALTHCHECK``, a load balancer, a
future container-orchestration readiness probe -- can act on this endpoint
using nothing more than the response status, without needing to parse the
body at all.

No ``try``/``except`` anywhere in this file, for the same reason
``chat.py``/``report.py`` have none: ``HealthService.check_health()`` is
documented to never raise (every per-dependency check already catches its
own exceptions and reports ``unhealthy`` instead) -- and the one unhandled
exception this route theoretically could still see is exactly the kind
``app.api.middleware.error_handler``'s last-resort handler exists for.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from app.core.dependencies import get_health_service
from app.schemas.common import DataResponse
from app.schemas.health import HealthCheckResponse
from app.services.health_service import HealthService

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=DataResponse[HealthCheckResponse],
    summary="Liveness/readiness and per-dependency health",
    description=(
        "Reports overall service health plus the individual status of every external dependency "
        "(LLM, vector store, search provider, database). Returns HTTP 503 instead of 200 when "
        "ready=false, so infrastructure health checks can act on the status code alone."
    ),
)
async def get_health(
    response: Response,
    health_service: HealthService = Depends(get_health_service),
) -> DataResponse[HealthCheckResponse]:
    health = await health_service.check_health()

    # The only HTTP-specific decision this route makes: reflect readiness in
    # the status code itself. FastAPI defaults to 200 for a route with no
    # explicit status_code, so only the not-ready case needs to be set.
    if not health.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return DataResponse.wrap(health)
