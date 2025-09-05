"""Yarn domain entity - core business model for yarn inventory."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Yarn:
    """Core yarn domain entity representing yarn inventory."""
    
    yarn_id: str
    description: str
    theoretical_balance: float
    allocated: float  # Already stored as negative in source data
    on_order: float
    min_stock_level: float = 0.0
    lead_time_days: int = 14
    last_updated: Optional[datetime] = None
    unit_of_measure: str = "lbs"
    color: Optional[str] = None
    yarn_type: Optional[str] = None
    supplier: Optional[str] = None
    cost_per_unit: float = 0.0
    
    @property
    def planning_balance(self) -> float:
        """
        Calculate planning balance.
        Formula: Theoretical + Allocated + On Order
        Note: Allocated is already negative in the data
        """
        return self.theoretical_balance + self.allocated + self.on_order
    
    @property
    def available_balance(self) -> float:
        """Calculate available balance (theoretical + allocated)."""
        return self.theoretical_balance + self.allocated
    
    def has_shortage(self) -> bool:
        """Check if yarn has a shortage based on min stock level."""
        return self.planning_balance < self.min_stock_level
    
    def get_shortage_amount(self) -> float:
        """Calculate shortage amount if exists."""
        if self.has_shortage():
            return self.min_stock_level - self.planning_balance
        return 0.0
    
    def days_of_stock(self, daily_usage: float) -> float:
        """Calculate days of stock remaining based on usage rate."""
        if daily_usage <= 0:
            return float('inf')
        return max(0, self.planning_balance / daily_usage)
    
    def needs_reorder(self, daily_usage: float) -> bool:
        """Check if yarn needs to be reordered based on lead time."""
        days_remaining = self.days_of_stock(daily_usage)
        return days_remaining <= self.lead_time_days
    
    def calculate_reorder_quantity(self, daily_usage: float, target_days: int = 90) -> float:
        """Calculate optimal reorder quantity."""
        target_stock = daily_usage * target_days
        current_with_orders = self.planning_balance
        reorder_qty = max(0, target_stock - current_with_orders)
        return reorder_qty
    
    def to_dict(self) -> dict:
        """Convert entity to dictionary for serialization."""
        return {
            'yarn_id': self.yarn_id,
            'description': self.description,
            'theoretical_balance': self.theoretical_balance,
            'allocated': self.allocated,
            'on_order': self.on_order,
            'planning_balance': self.planning_balance,
            'available_balance': self.available_balance,
            'min_stock_level': self.min_stock_level,
            'lead_time_days': self.lead_time_days,
            'has_shortage': self.has_shortage(),
            'shortage_amount': self.get_shortage_amount(),
            'unit_of_measure': self.unit_of_measure,
            'color': self.color,
            'yarn_type': self.yarn_type,
            'supplier': self.supplier,
            'cost_per_unit': self.cost_per_unit,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Yarn':
        """Create entity from dictionary."""
        last_updated = data.get('last_updated')
        if last_updated and isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        
        return cls(
            yarn_id=data['yarn_id'],
            description=data['description'],
            theoretical_balance=float(data.get('theoretical_balance', 0)),
            allocated=float(data.get('allocated', 0)),
            on_order=float(data.get('on_order', 0)),
            min_stock_level=float(data.get('min_stock_level', 0)),
            lead_time_days=int(data.get('lead_time_days', 14)),
            last_updated=last_updated,
            unit_of_measure=data.get('unit_of_measure', 'lbs'),
            color=data.get('color'),
            yarn_type=data.get('yarn_type'),
            supplier=data.get('supplier'),
            cost_per_unit=float(data.get('cost_per_unit', 0))
        )
    
    def __str__(self) -> str:
        """String representation of yarn entity."""
        return f"Yarn({self.yarn_id}: {self.description}, Balance: {self.planning_balance:.2f} {self.unit_of_measure})"