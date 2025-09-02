#!/usr/bin/env python3
"""
Knit Order Generator Module for Beverly Knits ERP
Generates knit production orders based on demand and inventory requirements
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import json
from pathlib import Path
import uuid


class Priority(Enum):
    """Production order priority levels"""
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    
    def __str__(self):
        return self.name


class KnitOrderGenerator:
    """
    Generates knit production orders based on:
    - Net requirements
    - Demand dates
    - Priority levels
    - Machine capacity
    - Yarn availability
    """
    
    def __init__(self):
        """Initialize the knit order generator"""
        self.default_lead_time_days = 14
        self.min_order_quantity = 100  # Minimum order quantity in lbs
        self.max_order_quantity = 10000  # Maximum order quantity in lbs
        self.batch_size = 500  # Standard batch size in lbs
        self.order_counter = self._load_counter()
        
    def _load_counter(self) -> int:
        """Load the order counter from file or initialize"""
        counter_file = Path("data/knit_order_counter.json")
        if counter_file.exists():
            try:
                with open(counter_file, 'r') as f:
                    data = json.load(f)
                    return data.get('counter', 1000)
            except:
                pass
        return 1000
    
    def _save_counter(self):
        """Save the current order counter"""
        counter_file = Path("data/knit_order_counter.json")
        try:
            counter_file.parent.mkdir(parents=True, exist_ok=True)
            with open(counter_file, 'w') as f:
                json.dump({'counter': self.order_counter, 'updated': datetime.now().isoformat()}, f)
        except:
            pass
    
    def _generate_order_id(self) -> str:
        """Generate a unique order ID"""
        self.order_counter += 1
        self._save_counter()
        timestamp = datetime.now().strftime('%Y%m%d')
        return f"KO-{timestamp}-{self.order_counter:05d}"
    
    def generate_knit_orders(self, 
                           net_requirements: Dict[str, float],
                           demand_dates: Optional[Dict[str, str]] = None,
                           priorities: Optional[Dict[str, Priority]] = None,
                           constraints: Optional[Dict] = None) -> List[Dict]:
        """
        Generate knit orders based on net requirements
        
        Args:
            net_requirements: Dictionary of style -> quantity required
            demand_dates: Optional dictionary of style -> required date
            priorities: Optional dictionary of style -> Priority enum
            constraints: Optional production constraints
            
        Returns:
            List of knit order dictionaries
        """
        orders = []
        
        if not net_requirements:
            return orders
        
        # Process each style requirement
        for style, required_quantity in net_requirements.items():
            if required_quantity <= 0:
                continue
            
            # Determine priority
            if priorities and style in priorities:
                priority = priorities[style]
            else:
                priority = self._determine_priority(required_quantity, demand_dates, style)
            
            # Determine demand date
            if demand_dates and style in demand_dates:
                demand_date = self._parse_date(demand_dates[style])
            else:
                demand_date = self._calculate_demand_date(priority)
            
            # Calculate lead time based on priority
            lead_time = self._calculate_lead_time(priority)
            
            # Determine production start date
            production_start = demand_date - timedelta(days=lead_time)
            
            # Split large orders into batches if needed
            if required_quantity > self.max_order_quantity:
                # Create multiple orders
                num_orders = int(np.ceil(required_quantity / self.batch_size))
                qty_per_order = required_quantity / num_orders
                
                for i in range(num_orders):
                    batch_order = self._create_order(
                        style=style,
                        quantity=min(qty_per_order, required_quantity - (i * qty_per_order)),
                        priority=priority,
                        demand_date=demand_date,
                        production_start=production_start + timedelta(days=i),
                        batch_number=i + 1,
                        total_batches=num_orders
                    )
                    orders.append(batch_order)
            else:
                # Create single order
                order = self._create_order(
                    style=style,
                    quantity=max(self.min_order_quantity, required_quantity),
                    priority=priority,
                    demand_date=demand_date,
                    production_start=production_start
                )
                orders.append(order)
        
        # Apply constraints if provided
        if constraints:
            orders = self._apply_constraints(orders, constraints)
        
        # Sort orders by priority and production start date
        orders = self._prioritize_orders(orders)
        
        return orders
    
    def _determine_priority(self, quantity: float, demand_dates: Optional[Dict], style: str) -> Priority:
        """Determine priority based on quantity and dates"""
        # High quantity orders get higher priority
        if quantity > 5000:
            return Priority.HIGH
        elif quantity > 1000:
            return Priority.NORMAL
        
        # Check if demand date is soon
        if demand_dates and style in demand_dates:
            try:
                demand_date = self._parse_date(demand_dates[style])
                days_until_demand = (demand_date - datetime.now()).days
                if days_until_demand < 7:
                    return Priority.URGENT
                elif days_until_demand < 14:
                    return Priority.HIGH
            except:
                pass
        
        return Priority.NORMAL
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime"""
        if isinstance(date_str, datetime):
            return date_str
        
        # Try common date formats
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        # Default to future date if parsing fails
        return datetime.now() + timedelta(days=30)
    
    def _calculate_demand_date(self, priority: Priority) -> datetime:
        """Calculate demand date based on priority"""
        base_date = datetime.now()
        
        if priority == Priority.URGENT:
            return base_date + timedelta(days=7)
        elif priority == Priority.HIGH:
            return base_date + timedelta(days=14)
        elif priority == Priority.NORMAL:
            return base_date + timedelta(days=30)
        else:  # LOW
            return base_date + timedelta(days=45)
    
    def _calculate_lead_time(self, priority: Priority) -> int:
        """Calculate production lead time based on priority"""
        if priority == Priority.URGENT:
            return 3
        elif priority == Priority.HIGH:
            return 7
        elif priority == Priority.NORMAL:
            return 14
        else:  # LOW
            return 21
    
    def _create_order(self, 
                     style: str,
                     quantity: float,
                     priority: Priority,
                     demand_date: datetime,
                     production_start: datetime,
                     batch_number: int = 1,
                     total_batches: int = 1) -> Dict[str, Any]:
        """Create a knit order dictionary"""
        order_id = self._generate_order_id()
        
        order = {
            'order_id': order_id,
            'style': style,
            'quantity_lbs': round(quantity, 2),
            'priority': str(priority),
            'priority_level': priority.value,
            'demand_date': demand_date.strftime('%Y-%m-%d'),
            'production_start_date': production_start.strftime('%Y-%m-%d'),
            'lead_time_days': (demand_date - production_start).days,
            'batch_number': batch_number,
            'total_batches': total_batches,
            'status': 'PLANNED',
            'created_at': datetime.now().isoformat(),
            'created_by': 'KnitOrderGenerator',
            'estimated_completion': (production_start + timedelta(days=(demand_date - production_start).days * 0.8)).strftime('%Y-%m-%d'),
            'machine_assignment': None,
            'yarn_allocated': False,
            'production_notes': self._generate_production_notes(style, quantity, priority)
        }
        
        # Add batch info if part of a batch
        if total_batches > 1:
            order['batch_id'] = f"{order_id}-BATCH"
            order['production_notes'] += f" | Batch {batch_number} of {total_batches}"
        
        return order
    
    def _generate_production_notes(self, style: str, quantity: float, priority: Priority) -> str:
        """Generate production notes for the order"""
        notes = []
        
        if priority == Priority.URGENT:
            notes.append("URGENT: Expedite production")
        elif priority == Priority.HIGH:
            notes.append("HIGH PRIORITY: Schedule immediately")
        
        if quantity > 5000:
            notes.append("Large order - consider parallel machines")
        elif quantity < 200:
            notes.append("Small order - combine with similar styles if possible")
        
        # Add style-specific notes
        if 'COTTON' in style.upper():
            notes.append("Cotton blend - check moisture levels")
        elif 'WOOL' in style.upper():
            notes.append("Wool content - temperature control required")
        
        return " | ".join(notes) if notes else "Standard production"
    
    def _apply_constraints(self, orders: List[Dict], constraints: Dict) -> List[Dict]:
        """Apply production constraints to orders"""
        # Apply machine capacity constraints
        if 'max_daily_capacity' in constraints:
            max_daily = constraints['max_daily_capacity']
            daily_totals = {}
            
            for order in orders:
                date = order['production_start_date']
                if date not in daily_totals:
                    daily_totals[date] = 0
                
                if daily_totals[date] + order['quantity_lbs'] > max_daily:
                    # Push to next day
                    new_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
                    order['production_start_date'] = new_date.strftime('%Y-%m-%d')
                    order['production_notes'] += " | Rescheduled due to capacity"
                else:
                    daily_totals[date] += order['quantity_lbs']
        
        # Apply yarn availability constraints
        if 'yarn_availability' in constraints:
            for order in orders:
                style = order['style']
                if style in constraints['yarn_availability']:
                    if constraints['yarn_availability'][style] < order['quantity_lbs']:
                        order['status'] = 'PENDING_MATERIALS'
                        order['production_notes'] += " | Awaiting yarn procurement"
        
        return orders
    
    def _prioritize_orders(self, orders: List[Dict]) -> List[Dict]:
        """Sort orders by priority and date"""
        return sorted(orders, key=lambda x: (
            x['priority_level'],
            datetime.strptime(x['production_start_date'], '%Y-%m-%d')
        ))
    
    def generate_order_summary(self, orders: List[Dict]) -> Dict:
        """Generate summary statistics for orders"""
        if not orders:
            return {
                'total_orders': 0,
                'total_quantity': 0,
                'priority_breakdown': {},
                'status_breakdown': {},
                'timeline': {}
            }
        
        df = pd.DataFrame(orders)
        
        summary = {
            'total_orders': len(orders),
            'total_quantity_lbs': round(df['quantity_lbs'].sum(), 2),
            'average_quantity_lbs': round(df['quantity_lbs'].mean(), 2),
            'priority_breakdown': df['priority'].value_counts().to_dict(),
            'status_breakdown': df['status'].value_counts().to_dict(),
            'styles': df['style'].nunique(),
            'earliest_start': df['production_start_date'].min(),
            'latest_demand': df['demand_date'].max(),
            'urgent_orders': len(df[df['priority'] == 'URGENT']),
            'batched_orders': len(df[df['total_batches'] > 1])
        }
        
        # Add timeline breakdown
        timeline = {}
        for _, order in df.iterrows():
            week = pd.to_datetime(order['production_start_date']).strftime('%Y-W%V')
            if week not in timeline:
                timeline[week] = {'orders': 0, 'quantity': 0}
            timeline[week]['orders'] += 1
            timeline[week]['quantity'] += order['quantity_lbs']
        
        summary['weekly_timeline'] = timeline
        
        return summary
    
    def export_orders(self, orders: List[Dict], format: str = 'json', path: Optional[Path] = None) -> Optional[str]:
        """Export orders to file"""
        if not orders:
            return None
        
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"knit_orders_{timestamp}.{format}"
            path = Path(f"data/exports/{filename}")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(path, 'w') as f:
                json.dump(orders, f, indent=2, default=str)
        elif format == 'csv':
            df = pd.DataFrame(orders)
            df.to_csv(path, index=False)
        elif format == 'excel':
            df = pd.DataFrame(orders)
            df.to_excel(path, index=False, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(path)
    
    def validate_orders(self, orders: List[Dict]) -> Dict[str, List]:
        """Validate generated orders for completeness and consistency"""
        errors = []
        warnings = []
        
        for order in orders:
            # Check required fields
            required_fields = ['order_id', 'style', 'quantity_lbs', 'priority', 'demand_date']
            for field in required_fields:
                if field not in order or order[field] is None:
                    errors.append(f"Order {order.get('order_id', 'unknown')}: Missing {field}")
            
            # Check quantity
            if 'quantity_lbs' in order:
                if order['quantity_lbs'] <= 0:
                    errors.append(f"Order {order['order_id']}: Invalid quantity {order['quantity_lbs']}")
                elif order['quantity_lbs'] < self.min_order_quantity:
                    warnings.append(f"Order {order['order_id']}: Quantity below minimum ({order['quantity_lbs']} < {self.min_order_quantity})")
            
            # Check dates
            if 'production_start_date' in order and 'demand_date' in order:
                try:
                    start = datetime.strptime(order['production_start_date'], '%Y-%m-%d')
                    demand = datetime.strptime(order['demand_date'], '%Y-%m-%d')
                    if start > demand:
                        errors.append(f"Order {order['order_id']}: Production starts after demand date")
                except:
                    errors.append(f"Order {order['order_id']}: Invalid date format")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'total_orders': len(orders),
            'orders_with_errors': len(set(e.split(':')[0] for e in errors)),
            'orders_with_warnings': len(set(w.split(':')[0] for w in warnings))
        }


# Helper functions for quick usage
def generate_orders_from_requirements(net_requirements: Dict[str, float], 
                                     priority: str = 'NORMAL') -> List[Dict]:
    """Quick function to generate orders from net requirements"""
    generator = KnitOrderGenerator()
    
    # Convert string priority to Priority enum
    priority_map = {
        'URGENT': Priority.URGENT,
        'HIGH': Priority.HIGH,
        'NORMAL': Priority.NORMAL,
        'LOW': Priority.LOW
    }
    
    priorities = {style: priority_map.get(priority, Priority.NORMAL) 
                 for style in net_requirements.keys()}
    
    return generator.generate_knit_orders(net_requirements, priorities=priorities)


def generate_urgent_orders(net_requirements: Dict[str, float]) -> List[Dict]:
    """Generate urgent orders for immediate production"""
    return generate_orders_from_requirements(net_requirements, 'URGENT')