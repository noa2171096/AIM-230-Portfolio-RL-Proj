"""
Adapted from Visual Vault

Application Metrics

This module provides metrics collection for monitoring:
- Request counts and latencies
- Error rates
- Active users
- Processing queue stats

Metrics can be exposed via /metrics endpoint for Prometheus scraping.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass
class RequestMetrics:
    """Metrics for a single endpoint."""
    total_requests: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    status_codes: dict = field(default_factory=lambda: defaultdict(int))
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests


class MetricsCollector:
    """
    Collects and aggregates application metrics.

    Thread-safe for use across async handlers.
    """

    def __init__(self):
        self._endpoints: dict[str, RequestMetrics] = defaultdict(RequestMetrics)
        self._start_time = datetime.now(timezone.utc)
        self._total_requests = 0
        self._active_requests = 0

    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        """Record metrics for a completed request."""
        key = f"{method}:{path}"
        metrics = self._endpoints[key]

        metrics.total_requests += 1
        metrics.total_latency_ms += latency_ms
        metrics.status_codes[status_code] += 1
        metrics.min_latency_ms = min(metrics.min_latency_ms, latency_ms)
        metrics.max_latency_ms = max(metrics.max_latency_ms, latency_ms)

        if status_code >= 400:
            metrics.total_errors += 1

        self._total_requests += 1

    def start_request(self) -> None:
        """Mark a request as started."""
        self._active_requests += 1

    def end_request(self) -> None:
        """Mark a request as ended."""
        self._active_requests -= 1

    def get_metrics(self) -> dict:
        """Get all collected metrics."""
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        endpoints_data = {}
        for key, metrics in self._endpoints.items():
            endpoints_data[key] = {
                "total_requests": metrics.total_requests,
                "total_errors": metrics.total_errors,
                "error_rate": round(metrics.error_rate, 4),
                "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                "min_latency_ms": round(metrics.min_latency_ms, 2) if metrics.min_latency_ms != float("inf") else 0,
                "max_latency_ms": round(metrics.max_latency_ms, 2),
                "status_codes": dict(metrics.status_codes),
            }

        return {
            "uptime_seconds": round(uptime, 2),
            "total_requests": self._total_requests,
            "active_requests": self._active_requests,
            "requests_per_second": round(self._total_requests / uptime, 2) if uptime > 0 else 0,
            "endpoints": endpoints_data,
        }

    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        metrics = self.get_metrics()

        # Uptime
        lines.append(f"# HELP app_uptime_seconds Application uptime in seconds")
        lines.append(f"# TYPE app_uptime_seconds gauge")
        lines.append(f'app_uptime_seconds {metrics["uptime_seconds"]}')

        # Total requests
        lines.append(f"# HELP app_requests_total Total number of requests")
        lines.append(f"# TYPE app_requests_total counter")
        lines.append(f'app_requests_total {metrics["total_requests"]}')

        # Active requests
        lines.append(f"# HELP app_requests_active Number of active requests")
        lines.append(f"# TYPE app_requests_active gauge")
        lines.append(f'app_requests_active {metrics["active_requests"]}')

        # Per-endpoint metrics
        lines.append(f"# HELP app_endpoint_requests_total Requests per endpoint")
        lines.append(f"# TYPE app_endpoint_requests_total counter")
        for endpoint, data in metrics["endpoints"].items():
            method, path = endpoint.split(":", 1)
            lines.append(
                f'app_endpoint_requests_total{{method="{method}",path="{path}"}} {data["total_requests"]}'
            )

        lines.append(f"# HELP app_endpoint_latency_avg_ms Average latency per endpoint")
        lines.append(f"# TYPE app_endpoint_latency_avg_ms gauge")
        for endpoint, data in metrics["endpoints"].items():
            method, path = endpoint.split(":", 1)
            lines.append(
                f'app_endpoint_latency_avg_ms{{method="{method}",path="{path}"}} {data["avg_latency_ms"]}'
            )

        lines.append(f"# HELP app_endpoint_errors_total Errors per endpoint")
        lines.append(f"# TYPE app_endpoint_errors_total counter")
        for endpoint, data in metrics["endpoints"].items():
            method, path = endpoint.split(":", 1)
            lines.append(
                f'app_endpoint_errors_total{{method="{method}",path="{path}"}} {data["total_errors"]}'
            )

        return "\n".join(lines) + "\n"


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting request metrics.

    Records:
    - Request count per endpoint
    - Latency distribution
    - Status code distribution
    - Error rates
    """

    def __init__(self, app, collector: MetricsCollector | None = None):
        super().__init__(app)
        self.collector = collector or get_metrics_collector()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        self.collector.start_request()
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Normalize path (replace IDs with placeholders)
            normalized_path = self._normalize_path(request.url.path)

            self.collector.record_request(
                method=request.method,
                path=normalized_path,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )

            return response

        finally:
            self.collector.end_request()

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path by replacing numeric IDs with placeholders.

        Example: /api/v1/assets/123 -> /api/v1/assets/{id}
        """
        parts = path.split("/")
        normalized = []
        for part in parts:
            if part.isdigit():
                normalized.append("{id}")
            else:
                normalized.append(part)
        return "/".join(normalized)