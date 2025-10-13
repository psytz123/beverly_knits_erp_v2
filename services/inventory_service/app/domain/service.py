"""Domain orchestration for inventory operations."""

from __future__ import annotations

from typing import List

from .dto import InventoryItemResponse, InventorySummaryResponse


class InventoryDomainService:
    """Placeholder domain service for strangler rollout."""

    async def list_items(self) -> List[InventoryItemResponse]:
        """TODO: Implement once data extraction is complete."""
        raise NotImplementedError("Inventory list_items not yet implemented")

    async def get_item(self, item_id: str) -> InventoryItemResponse:
        """TODO: Implement retrieval logic."""
        raise NotImplementedError("Inventory get_item not yet implemented")

    async def summary(self) -> InventorySummaryResponse:
        """TODO: Implement summary aggregation logic."""
        raise NotImplementedError("Inventory summary not yet implemented")

    async def trigger_recalculation(self) -> None:
        """TODO: Kick off async recalculation pipeline."""
        raise NotImplementedError("Inventory trigger_recalculation not yet implemented")
