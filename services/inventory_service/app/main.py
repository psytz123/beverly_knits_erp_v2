"""
FastAPI entrypoint for the inventory microservice.

Provides a thin wrapper that wires routers and shared dependencies.
Concrete business logic will be implemented during the strangler rollout.
"""

from __future__ import annotations

from fastapi import FastAPI

from .api.routers import register_routes
from ..config.settings import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Inventory Service",
        version="0.1.0",
        description="Inventory analytics microservice for Beverly Knits ERP",
    )

    register_routes(app)

    if settings.enable_health_route:
        @app.get("/health", tags=["system"])
        async def health_check() -> dict[str, str]:
            """Basic service health endpoint."""
            return {"status": "ok", "service": "inventory", "version": app.version}

    return app


app = create_app()
