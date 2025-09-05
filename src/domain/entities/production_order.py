"""Production order domain entity."""

from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class OrderStatus(Enum):
    """Production order status enumeration."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"


class OrderPriority(Enum):
    """Production order priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class ProductionOrder:
    """Domain entity representing a production order."""
    
    order_id: str
    style_id: str
    customer_name: str
    quantity_ordered: float
    quantity_produced: float = 0.0
    scheduled_date: datetime = None
    due_date: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    priority: OrderPriority = OrderPriority.NORMAL
    machine_id: Optional[str] = None
    work_center: Optional[str] = None
    yarn_requirements: Dict[str, float] = None
    production_time_hours: float = 0.0
    setup_time_hours: float = 0.0
    notes: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    created_by: Optional[str] = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.yarn_requirements is None:
            self.yarn_requirements = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    @property
    def completion_percentage(self) -> float:
        """Calculate order completion percentage."""
        if self.quantity_ordered <= 0:
            return 0.0
        return min(100.0, (self.quantity_produced / self.quantity_ordered) * 100)
    
    @property
    def remaining_quantity(self) -> float:
        """Calculate remaining quantity to produce."""
        return max(0, self.quantity_ordered - self.quantity_produced)
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete."""
        return self.status == OrderStatus.COMPLETED or self.remaining_quantity == 0
    
    @property
    def is_overdue(self) -> bool:
        """Check if order is overdue."""
        if not self.due_date:
            return False
        return datetime.utcnow() > self.due_date and not self.is_complete
    
    @property
    def days_until_due(self) -> int:
        """Calculate days until due date."""
        if not self.due_date:
            return 999
        delta = self.due_date - datetime.utcnow()
        return delta.days
    
    @property
    def total_production_time(self) -> float:
        """Calculate total production time including setup."""
        return self.setup_time_hours + self.production_time_hours
    
    def can_be_scheduled(self) -> bool:
        """Check if order can be scheduled for production."""
        return self.status in [OrderStatus.PENDING, OrderStatus.ON_HOLD]
    
    def assign_to_machine(self, machine_id: str, work_center: str):
        """Assign order to a machine."""
        self.machine_id = machine_id
        self.work_center = work_center
        self.status = OrderStatus.ASSIGNED
        self.updated_at = datetime.utcnow()
    
    def start_production(self):
        """Mark order as in progress."""
        if self.status != OrderStatus.ASSIGNED:
            raise ValueError("Order must be assigned before starting production")
        self.status = OrderStatus.IN_PROGRESS
        self.updated_at = datetime.utcnow()
    
    def update_progress(self, quantity_produced: float):
        """Update production progress."""
        self.quantity_produced = min(quantity_produced, self.quantity_ordered)
        if self.quantity_produced >= self.quantity_ordered:
            self.status = OrderStatus.COMPLETED
        self.updated_at = datetime.utcnow()
    
    def add_yarn_requirement(self, yarn_id: str, quantity: float):
        """Add or update yarn requirement."""
        self.yarn_requirements[yarn_id] = quantity
    
    def get_total_yarn_weight(self) -> float:
        """Calculate total yarn weight required."""
        return sum(self.yarn_requirements.values())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'order_id': self.order_id,
            'style_id': self.style_id,
            'customer_name': self.customer_name,
            'quantity_ordered': self.quantity_ordered,
            'quantity_produced': self.quantity_produced,
            'completion_percentage': self.completion_percentage,
            'remaining_quantity': self.remaining_quantity,
            'scheduled_date': self.scheduled_date.isoformat() if self.scheduled_date else None,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'status': self.status.value,
            'priority': self.priority.value,
            'machine_id': self.machine_id,
            'work_center': self.work_center,
            'yarn_requirements': self.yarn_requirements,
            'production_time_hours': self.production_time_hours,
            'setup_time_hours': self.setup_time_hours,
            'total_production_time': self.total_production_time,
            'is_overdue': self.is_overdue,
            'days_until_due': self.days_until_due,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by': self.created_by
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProductionOrder':
        """Create from dictionary."""
        # Parse dates
        scheduled_date = data.get('scheduled_date')
        if scheduled_date and isinstance(scheduled_date, str):
            scheduled_date = datetime.fromisoformat(scheduled_date)
        
        due_date = data.get('due_date')
        if due_date and isinstance(due_date, str):
            due_date = datetime.fromisoformat(due_date)
        
        created_at = data.get('created_at')
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        updated_at = data.get('updated_at')
        if updated_at and isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        # Parse enums
        status = data.get('status', 'pending')
        if isinstance(status, str):
            status = OrderStatus(status)
        
        priority = data.get('priority', 3)
        if isinstance(priority, (int, str)):
            priority = OrderPriority(int(priority))
        
        return cls(
            order_id=data['order_id'],
            style_id=data['style_id'],
            customer_name=data.get('customer_name', ''),
            quantity_ordered=float(data['quantity_ordered']),
            quantity_produced=float(data.get('quantity_produced', 0)),
            scheduled_date=scheduled_date,
            due_date=due_date,
            status=status,
            priority=priority,
            machine_id=data.get('machine_id'),
            work_center=data.get('work_center'),
            yarn_requirements=data.get('yarn_requirements', {}),
            production_time_hours=float(data.get('production_time_hours', 0)),
            setup_time_hours=float(data.get('setup_time_hours', 0)),
            notes=data.get('notes'),
            created_at=created_at,
            updated_at=updated_at,
            created_by=data.get('created_by')
        )