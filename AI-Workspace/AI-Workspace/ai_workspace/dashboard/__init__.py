"""AI Workspace Dashboard - Optional Visualization Layer.

This module provides a web-based dashboard for monitoring and visualizing
AI Workspace metrics, phase gates, and reuse analysis.

Installation:
    pip install ai-workspace[dashboard]

Usage:
    ai-workspace dashboard --port 8900

Note:
    This is an optional module. The core AI Workspace system works
    independently without dashboard installed.
"""

from typing import Any, Optional

__all__ = ["app", "run", "DASHBOARD_AVAILABLE"]

DASHBOARD_AVAILABLE = True
app: Optional[Any] = None

try:
    from .app import app, run
except ImportError as e:
    # Dashboard dependencies not installed
    DASHBOARD_AVAILABLE = False
    app = None

    def run(*args: Any, **kwargs: Any) -> None:
        """Placeholder run function when dashboard not available."""
        raise ImportError(
            "Dashboard dependencies not installed. "
            "Install with: pip install ai-workspace[dashboard]"
        )
