"""AI Workspace Dashboard Application (Async Version).

Optional visualization layer for AI Workspace with async I/O.
Requires: pip install ai-workspace[dashboard]
"""

from pathlib import Path
from typing import Any, Dict

try:
    from quart import Quart, jsonify
    from quart_cors import cors
except ImportError as e:
    raise ImportError(
        "Dashboard dependencies not installed. "
        "Install with: pip install ai-workspace[dashboard]"
    ) from e

from .api import gates_bp, metrics_bp, reuse_bp, search_bp, stack_bp, filters_bp, cache_bp
from .error_handlers import register_error_handlers
from .rate_limiter import create_limiter
from .routes import pages_bp

# Create Quart app (async-compatible Flask replacement)
app = Quart(__name__)
app.config["SECRET_KEY"] = "ai-workspace-dashboard"
app.config["JSON_SORT_KEYS"] = False  # Preserve JSON key order

# Enable CORS
app = cors(app, allow_origin="*")

# Register error handlers
register_error_handlers(app)

# Create and configure rate limiter
limiter = create_limiter(app)

# Register page routes blueprint (multi-page navigation)
app.register_blueprint(pages_bp)

# Register API blueprints
app.register_blueprint(metrics_bp)
app.register_blueprint(gates_bp)
app.register_blueprint(reuse_bp)
app.register_blueprint(search_bp)
app.register_blueprint(stack_bp)
app.register_blueprint(filters_bp)
app.register_blueprint(cache_bp)  # Cache management endpoints


@app.route("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint (async).

    Returns:
        JSON response with health status
    """
    return jsonify({"status": "ok", "dashboard": "running", "version": "1.2.0"})


async def run_async(host: str = "127.0.0.1", port: int = 8900, debug: bool = False) -> None:
    """Run async dashboard server.

    Args:
        host: Host address (default: 127.0.0.1)
        port: Port number (default: 8900)
        debug: Debug mode (default: False)
    """
    # Import and start file monitor (if available)
    try:
        from .services.monitor import FileMonitor

        # Note: Monitor may need async adaptation
        monitor = FileMonitor(None)  # SocketIO integration pending
        # monitor.start()
    except Exception as e:
        print(f"Warning: File monitoring not available: {e}")

    # Run Quart server (async)
    await app.run_task(host=host, port=port, debug=debug)


def run(host: str = "127.0.0.1", port: int = 8900, debug: bool = False) -> None:
    """Run dashboard server (sync wrapper for backward compatibility).

    Args:
        host: Host address (default: 127.0.0.1)
        port: Port number (default: 8900)
        debug: Debug mode (default: False)
    """
    import asyncio

    # Run async server
    asyncio.run(run_async(host, port, debug))


if __name__ == "__main__":
    run()
