"""
Adopted from Visual Vault

Rate Limiting Middleware

This module provides rate limiting using SlowAPI with Redis backend.
Rate limits can be configured per-endpoint and per-user tier.

Features:
- IP-based rate limiting for anonymous requests
- User-based rate limiting for authenticated requests
- API key tier-based limits
- Configurable limits per endpoint
"""

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings


def get_rate_limit_key(request: Request) -> str:
    """
    Generate a rate limit key based on authentication.

    Priority:
    1. API key (with tier-based limits)
    2. JWT user ID
    3. IP address (for anonymous)
    """
    # Check for API key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        # Use API key prefix as identifier
        return f"apikey:{api_key[:11]}"

    # Check for JWT token
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        # Extract user ID from token (simplified - in production decode the token)
        token = auth_header[7:]
        return f"token:{token[:20]}"

    # Fall back to IP address
    return f"ip:{get_remote_address(request)}"


def get_user_rate_limit_key(request: Request) -> str:
    """
    Get rate limit key specifically for authenticated endpoints.
    Returns user identifier or raises if not authenticated.
    """
    # This will be used with endpoints that require auth
    # The actual user ID comes from the dependency injection
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key[:11]}"

    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return f"token:{auth_header[7:27]}"

    return get_remote_address(request)


# Create the limiter instance
settings = get_settings()

limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=[f"{settings.auth.rate_limit_per_minute}/minute"],
    storage_uri=settings.redis.url,
    strategy="fixed-window",  # or "moving-window"
)


# Rate limit tiers
RATE_LIMIT_TIERS = {
    "anonymous": "30/minute",
    "standard": "60/minute",
    "premium": "300/minute",
    "unlimited": "10000/minute",  # Effectively unlimited
}


def get_tier_limit(tier: str) -> str:
    """Get rate limit string for a tier."""
    return RATE_LIMIT_TIERS.get(tier, RATE_LIMIT_TIERS["standard"])


# Endpoint-specific limits
ENDPOINT_LIMITS = {
    "/api/v1/assets/upload": "10/minute",  # Uploads are expensive
    "/api/v1/search/text": "30/minute",  # ML inference
    "/api/v1/search/image": "10/minute",  # Heavy processing
    "/api/v1/auth/login": "5/minute",  # Prevent brute force
    "/api/v1/auth/register": "3/minute",  # Prevent spam
}


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Custom handler for rate limit exceeded errors.

    Returns a JSON response with retry information.
    """
    from fastapi.responses import JSONResponse

    # Parse the limit info
    retry_after = getattr(exc, "retry_after", 60)

    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": f"Rate limit exceeded. Try again in {retry_after} seconds.",
            "retry_after": retry_after,
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": str(exc.detail) if hasattr(exc, "detail") else "unknown",
        },
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds rate limit headers to all responses.

    Headers added:
    - X-RateLimit-Limit: Maximum requests allowed
    - X-RateLimit-Remaining: Requests remaining in window
    - X-RateLimit-Reset: Unix timestamp when limit resets
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add rate limit headers (if available from limiter state)
        # This is informational - actual enforcement happens via decorators
        if hasattr(request.state, "view_rate_limit"):
            limit_info = request.state.view_rate_limit
            response.headers["X-RateLimit-Limit"] = str(limit_info.get("limit", ""))
            response.headers["X-RateLimit-Remaining"] = str(limit_info.get("remaining", ""))
            response.headers["X-RateLimit-Reset"] = str(limit_info.get("reset", ""))

        return response