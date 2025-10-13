"""Data transfer objects for inventory service contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InventoryItemResponse(BaseModel):
    """Represents inventory metrics for a single yarn item."""

    yarn_id: str = Field(..., description="Unique identifier for the yarn item")
    description: str = Field(..., description="Human-friendly name")
    supplier: str | None = Field(None, description="Supplier name if available")
    theoretical_balance: float = Field(..., description="Calculated theoretical balance in pounds")
    planning_balance: float = Field(..., description="Planning balance including orders")
    allocated: float = Field(..., description="Allocated quantity (negative indicates allocation)")
    on_order: float = Field(..., description="Quantity currently on order")
    risk_level: str = Field(..., description="Risk classification (CRITICAL/HIGH/MEDIUM/LOW)")


class InventorySummaryResponse(BaseModel):
    """Aggregated inventory metrics across the dataset."""

    total_items: int = Field(..., description="Total number of inventory items")
    critical_count: int = Field(..., description="Number of critical risk items")
    high_count: int = Field(..., description="Number of high risk items")
    medium_count: int = Field(..., description="Number of medium risk items")
    low_count: int = Field(..., description="Number of low risk items")
