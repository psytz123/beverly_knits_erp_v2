"""Route registration for the inventory service."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI

from .dependencies import get_inventory_service
from ..domain.dto import InventoryItemResponse, InventorySummaryResponse


def register_routes(app: FastAPI) -> None:
    """Attach routers to the FastAPI app."""
    router = APIRouter(prefix="/inventory", tags=["inventory"])

    @router.get("/items", response_model=list[InventoryItemResponse])
    async def list_items(service= get_inventory_service()) -> list[InventoryItemResponse]:  # type: ignore[return-value]
        """
        List inventory items.

        Implementation deferred to strangler extraction phases.
        """
        return await service.list_items()

    @router.get("/items/{item_id}", response_model=InventoryItemResponse)
    async def get_item(item_id: str, service=get_inventory_service()) -> InventoryItemResponse:  # type: ignore[return-value]
        """Retrieve a single inventory item."""
        return await service.get_item(item_id)

    @router.get("/summary", response_model=InventorySummaryResponse)
    async def summary(service=get_inventory_service()) -> InventorySummaryResponse:  # type: ignore[return-value]
        """Return aggregated inventory summary metrics."""
        return await service.summary()

    @router.post("/recalculate", status_code=202)
    async def recalculate(service=get_inventory_service()) -> dict[str, str]:
        """Trigger asynchronous recomputation of plan balances."""
        await service.trigger_recalculation()
        return {"status": "accepted"}

    app.include_router(router)
