"""Dashboard REST API endpoints (Async Version).

Read-only API for accessing AI Workspace data with async I/O.
All endpoints are GET-only - no modifications to core workspace files.
"""

from typing import List

__all__: List[str] = [
    "metrics_bp",
    "gates_bp",
    "reuse_bp",
    "search_bp",
    "stack_bp",
    "filters_bp",
    "cache_bp",
]

# Import async blueprints
from .metrics import metrics_bp
from .gates import gates_bp
from .reuse_async import reuse_bp  # Use async version
from .search import search_bp
from .stack import stack_bp
from .filters import filters_bp
from .cache import cache_bp
