"""Dashboard page routes (Async Version).

Multi-page navigation system for AI Workspace Dashboard with async rendering.
All routes serve HTML templates using the same base layout.
"""

from typing import Any, Dict, Tuple

from quart import Blueprint, render_template, session

# Create blueprint for dashboard pages
pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
async def index() -> str:
    """Dashboard home page (async).

    Returns:
        Rendered dashboard template with overview metrics
    """
    # Track current page in session for navigation state
    session["current_page"] = "dashboard"
    return await render_template("dashboard.html")


@pages_bp.route("/metrics")
async def metrics() -> str:
    """Metrics analysis page (async).

    Displays detailed quality metrics, compliance scores,
    and trend analysis across all 5 core principles.

    Returns:
        Rendered metrics template
    """
    session["current_page"] = "metrics"
    return await render_template("metrics.html")


@pages_bp.route("/gates")
async def gates() -> str:
    """Phase gates tracking page (async).

    Shows current phase gate status, completion criteria,
    and historical gate progression.

    Returns:
        Rendered gates template
    """
    session["current_page"] = "gates"
    return await render_template("gates.html")


@pages_bp.route("/reuse")
async def reuse() -> str:
    """Reuse analysis page (async).

    Displays reuse percentage trends, check-before-create
    enforcement status, and multi-language code analysis.

    Returns:
        Rendered reuse template
    """
    session["current_page"] = "reuse"
    return await render_template("reuse.html")


@pages_bp.route("/agents")
async def agents() -> str:
    """AI agents explorer page (async).

    Browse and search through 156 specialized agents,
    view recommendations, and track agent usage.

    Returns:
        Rendered agents template
    """
    session["current_page"] = "agents"
    return await render_template("agents.html")


@pages_bp.errorhandler(404)
async def page_not_found(error: Any) -> Tuple[str, int]:
    """Handle 404 errors for page routes (async).

    Args:
        error: Error object

    Returns:
        Rendered 404 error page with 404 status code
    """
    return await render_template("errors/404.html"), 404


@pages_bp.errorhandler(500)
async def internal_server_error(error: Any) -> Tuple[str, int]:
    """Handle 500 errors for page routes (async).

    Args:
        error: Error object

    Returns:
        Rendered 500 error page with 500 status code
    """
    return await render_template("errors/500.html"), 500


def get_current_page() -> str:
    """Get current page from session.

    Utility function for navigation state management.

    Returns:
        Current page name (default: 'dashboard')
    """
    return session.get("current_page", "dashboard")
