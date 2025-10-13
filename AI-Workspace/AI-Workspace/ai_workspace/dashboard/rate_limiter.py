"""Rate limiting configuration for API endpoints.

Provides Flask-Limiter integration for protecting API endpoints from abuse.
"""

from typing import Optional

from flask import Flask, request

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    LIMITER_AVAILABLE = True
except ImportError:
    LIMITER_AVAILABLE = False
    Limiter = None  # type: ignore


def get_client_identifier() -> str:
    """Get client identifier for rate limiting.

    Uses IP address as the primary identifier. In production, this could be
    enhanced to use API keys or authenticated user IDs.

    Returns:
        Client identifier string
    """
    # Try to get real IP from headers (if behind proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    # Fall back to remote address
    return get_remote_address()


def create_limiter(app: Flask) -> Optional[Limiter]:
    """Create and configure rate limiter for the Flask application.

    Args:
        app: Flask application instance

    Returns:
        Configured Limiter instance or None if flask-limiter not available
    """
    if not LIMITER_AVAILABLE:
        app.logger.warning(
            "flask-limiter not installed. Rate limiting disabled. "
            "Install with: pip install flask-limiter"
        )
        return None

    limiter = Limiter(
        app=app,
        key_func=get_client_identifier,
        default_limits=["100 per minute", "1000 per hour"],
        storage_uri="memory://",  # In-memory storage for development
        strategy="fixed-window",
        headers_enabled=True,  # Add rate limit headers to responses
    )

    return limiter


def configure_rate_limits(limiter: Optional[Limiter]) -> None:
    """Configure specific rate limits for different endpoint groups.

    Args:
        limiter: Limiter instance (may be None if not available)
    """
    if not limiter:
        return

    # Metrics endpoints - stricter limits (more expensive operations)
    limiter.limit("30 per minute")(
        limiter.app.view_functions.get("metrics.get_summary")
    )
    limiter.limit("30 per minute")(
        limiter.app.view_functions.get("metrics.get_reuse_metrics")
    )
    limiter.limit("30 per minute")(
        limiter.app.view_functions.get("metrics.get_quality_metrics")
    )
    limiter.limit("30 per minute")(
        limiter.app.view_functions.get("metrics.get_trends")
    )

    # Health endpoint - very permissive
    limiter.limit("300 per minute")(
        limiter.app.view_functions.get("health")
    )


# Rate limit configuration constants
DEFAULT_RATE_LIMIT = "100 per minute"
STRICT_RATE_LIMIT = "30 per minute"
HEALTH_RATE_LIMIT = "300 per minute"
HOURLY_RATE_LIMIT = "1000 per hour"

# Rate limit exempt paths (for internal use)
EXEMPT_PATHS = [
    "/static/",
    "/favicon.ico",
]


def is_exempt_from_rate_limit(path: str) -> bool:
    """Check if a path is exempt from rate limiting.

    Args:
        path: Request path to check

    Returns:
        True if path is exempt from rate limiting
    """
    return any(path.startswith(exempt_path) for exempt_path in EXEMPT_PATHS)


class RateLimitConfig:
    """Rate limit configuration for different endpoint categories.

    Attributes:
        health: Rate limit for health check endpoints
        metrics: Rate limit for metrics endpoints
        gates: Rate limit for gates endpoints
        reuse: Rate limit for reuse endpoints
        stack: Rate limit for stack endpoints
        default: Default rate limit for unlabeled endpoints
    """

    health = HEALTH_RATE_LIMIT
    metrics = STRICT_RATE_LIMIT
    gates = DEFAULT_RATE_LIMIT
    reuse = DEFAULT_RATE_LIMIT
    stack = DEFAULT_RATE_LIMIT
    default = DEFAULT_RATE_LIMIT
