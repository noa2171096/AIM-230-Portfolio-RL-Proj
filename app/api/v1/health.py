"""
Adapted VisualVault

Health Check Endpoints

This module demonstrates:
- Simple and detailed health check patterns
- Pydantic response models with examples
- Dependency checking (DB, Redis, Storage)
- Proper HTTP status codes for health states

Health checks are critical for:
- Container orchestration (K8s liveness/readiness probes)
- Load balancer health checks
- Monitoring and alerting systems
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from fastapi.responses import PlainTextResponse

from app.config import Settings, get_settings
from app.database import check_db_connection

router = APIRouter()


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Some non-critical services down
    UNHEALTHY = "unhealthy"  # Critical services down


class ComponentHealth(BaseModel):
    """Health status of an individual component."""

    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "latency_ms": 1.5,
                    "message": None,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Simple health check response."""

    status: HealthStatus
    timestamp: datetime
    version: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "version": "0.1.0",
                }
            ]
        }
    }


class DetailedHealthResponse(BaseModel):
    """Detailed health check with component status."""

    status: HealthStatus
    timestamp: datetime
    version: str
    environment: str
    components: dict[str, ComponentHealth] = Field(
        description="Health status of each system component"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "version": "0.1.0",
                    "environment": "development",
                    "components": {
                        "database": {"status": "healthy", "latency_ms": 2.1},
                        "redis": {"status": "healthy", "latency_ms": 0.5},
                        "storage": {"status": "healthy", "latency_ms": 0.1},
                    },
                }
            ]
        }
    }


@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic Health Check",
    description="Quick health check for load balancers and basic monitoring.",
)
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """
    Basic health check endpoint.

    Returns a simple status indicating if the API is responding.
    Use this for:
    - Kubernetes liveness probes
    - Load balancer health checks
    - Quick "is it up?" checks
    """
    return HealthResponse(
        status=HealthStatus.HEALTHY,
        timestamp=datetime.now(timezone.utc),
        version=settings.app_version,
    )


@router.get(
    "/ready",
    response_model=DetailedHealthResponse,
    summary="Readiness Check",
    description="Detailed health check including all dependencies.",
)
async def readiness_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> DetailedHealthResponse:
    """
    Detailed readiness check endpoint.

    Checks all system dependencies and returns their status.
    Use this for:
    - Kubernetes readiness probes
    - Detailed system monitoring
    - Debugging connectivity issues

    A service is "ready" when all critical dependencies are available.
    """
    components: dict[str, ComponentHealth] = {}
    overall_status = HealthStatus.HEALTHY

    # Check database connection
    db_health = await check_database_health()
    components["database"] = db_health
    if db_health.status != HealthStatus.HEALTHY:
        overall_status = HealthStatus.DEGRADED

    # Check Redis connection
    redis_health = await check_redis_health()
    components["redis"] = redis_health
    if redis_health.status != HealthStatus.HEALTHY:
        overall_status = HealthStatus.DEGRADED

    # Check storage accessibility
    storage_status = await check_storage_health(settings)
    components["storage"] = storage_status
    if storage_status.status != HealthStatus.HEALTHY:
        overall_status = HealthStatus.DEGRADED

    # Check Celery workers
    worker_health = await check_worker_health()
    components["workers"] = worker_health
    if worker_health.status == HealthStatus.UNHEALTHY:
        overall_status = HealthStatus.DEGRADED

    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version=settings.app_version,
        environment=settings.environment,
        components=components,
    )


async def check_database_health() -> ComponentHealth:
    """
    Check if database is reachable.

    This verifies:
    - Connection pool is initialized
    - Database accepts queries
    """
    import time

    start = time.perf_counter()

    try:
        is_connected = await check_db_connection()
        latency = (time.perf_counter() - start) * 1000

        if is_connected:
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 2),
            )
        else:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message="Database connection failed",
            )

    except Exception as e:
        return ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message=f"Database check failed: {str(e)}",
        )


async def check_storage_health(settings: Settings) -> ComponentHealth:
    """Check if uploads directory exists and is writable."""
    import time
    from pathlib import Path

    start = time.perf_counter()

    try:
        uploads_path = Path("uploads")
        uploads_path.mkdir(exist_ok=True)

        # Check writable
        test_file = uploads_path / ".health_check"
        test_file.touch()
        test_file.unlink()

        latency = (time.perf_counter() - start) * 1000

        return ComponentHealth(
            status     = HealthStatus.HEALTHY,
            latency_ms = round(latency, 2),
        )

    except PermissionError:
        return ComponentHealth(
            status  = HealthStatus.UNHEALTHY,
            message = "Uploads directory is not writable",
        )
    except Exception as e:
        return ComponentHealth(
            status  = HealthStatus.UNHEALTHY,
            message = f"Storage check failed: {str(e)}",
        )


async def check_redis_health() -> ComponentHealth:
    """
    Check if Redis is reachable.
    """
    import time

    start = time.perf_counter()

    try:
        from app.services.cache import get_cache_service

        cache = get_cache_service()
        await cache.redis.ping()
        latency = (time.perf_counter() - start) * 1000

        return ComponentHealth(
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
        )

    except RuntimeError:
        # Cache not initialized
        return ComponentHealth(
            status=HealthStatus.DEGRADED,
            message="Cache service not initialized",
        )
    except Exception as e:
        return ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message=f"Redis check failed: {str(e)}",
        )


async def check_worker_health() -> ComponentHealth:
    """
    Check if Celery workers are available.
    """
    import time

    start = time.perf_counter()

    try:
        from app.workers.celery_app import celery_app

        # Ping workers with short timeout
        result = celery_app.control.ping(timeout=1.0)
        latency = (time.perf_counter() - start) * 1000

        if result:
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 2),
                message=f"{len(result)} worker(s) online",
            )
        else:
            return ComponentHealth(
                status=HealthStatus.DEGRADED,
                message="No workers responded",
            )

    except Exception as e:
        return ComponentHealth(
            status=HealthStatus.DEGRADED,
            message=f"Worker check failed: {str(e)}",
        )


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus Metrics",
    description="Application metrics in Prometheus format.",
)
async def get_metrics() -> str:
    """
    Export metrics in Prometheus text format.

    Use this endpoint for Prometheus scraping.
    """
    from app.middleware.metrics import get_metrics_collector

    collector = get_metrics_collector()
    return collector.get_prometheus_format()