"""
Fixed Production Planning API - Uses real knit orders with actual quantities
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from src.utils.json_sanitizer import sanitize_for_json, safe_float, safe_int

def get_production_planning_fixed(data_loader) -> Dict[str, Any]:
    """
    Generate production planning data with real order quantities
    """
    try:
        if not data_loader or not hasattr(data_loader, 'data'):
            return {"status": "error", "message": "Data loader not available"}
        
        # Get real data
        knit_orders = data_loader.data.get('knit_orders')
        
        if knit_orders is None or knit_orders.empty:
            return {"status": "error", "message": "No knit orders data available"}
        
        production_schedule = []
        current_date = datetime.now()
        
        # Process knit orders into production schedule
        for idx, order in knit_orders.iterrows():
            style = order.get('Style #', order.get('Style', 'Unknown'))
            order_number = order.get('Order #', f'KO-{idx+1:04d}')
            quantity_ordered = safe_float(order.get('Qty Ordered (lbs)', 0))
            balance = safe_float(order.get('Balance (lbs)', 0))
            shipped = safe_float(order.get('Shipped (lbs)', 0))
            g00_stage = safe_float(order.get('G00 (lbs)', 0)) if pd.notna(order.get('G00 (lbs)')) else 0
            
            # Parse dates
            start_date = pd.to_datetime(order.get('Start Date'), errors='coerce')
            quoted_date = pd.to_datetime(order.get('Quoted Date'), errors='coerce')
            
            # Skip completed orders
            if balance <= 0:
                continue
            
            # Determine priority based on due date and quantity
            if pd.notna(quoted_date):
                days_until_due = (quoted_date - current_date).days
            elif pd.notna(start_date):
                days_until_due = (start_date + timedelta(days=30) - current_date).days
            else:
                days_until_due = 14
            
            if days_until_due < 7:
                priority = "Critical"
            elif days_until_due < 14:
                priority = "High"
            elif days_until_due < 30:
                priority = "Medium"
            else:
                priority = "Normal"
            
            # Calculate production details
            production_item = {
                'style': style,
                'order_id': order_number,
                'customer': 'Beverly Knits',  # Default, update if customer data available
                'quantity': quantity_ordered,
                'planned_quantity': balance,  # Amount still to produce
                'completed_quantity': shipped,
                'in_progress_quantity': g00_stage,
                'priority': priority,
                'start_date': start_date.isoformat() if pd.notna(start_date) else None,
                'due_date': quoted_date.isoformat() if pd.notna(quoted_date) else None,
                'status': 'In Progress' if g00_stage > 0 else 'Scheduled',
                'completion_percentage': round((shipped / quantity_ordered * 100), 1) if quantity_ordered > 0 else 0,
                'days_until_due': days_until_due,
                'production_stage': 'Greige' if g00_stage > 0 else 'Knitting'
            }
            
            production_schedule.append(production_item)
        
        # Sort by priority and due date
        priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Normal": 3}
        production_schedule.sort(key=lambda x: (priority_order.get(x['priority'], 4), x['days_until_due']))
        
        # Take top 10 for display
        production_schedule = production_schedule[:10]
        
        # Calculate capacity analysis
        total_planned = sum(item['planned_quantity'] for item in production_schedule)
        daily_capacity = 10000  # Example daily capacity in lbs
        days_required = total_planned / daily_capacity if daily_capacity > 0 else 0
        
        capacity_analysis = {
            'daily_capacity_lbs': daily_capacity,
            'total_planned_lbs': total_planned,
            'days_required': round(days_required, 1),
            'utilization_percentage': min(100, (total_planned / (daily_capacity * 30)) * 100),  # 30-day utilization
            'available_capacity_lbs': max(0, daily_capacity * 30 - total_planned),
            'overtime_required': days_required > 30
        }
        
        # Identify bottlenecks
        bottlenecks = []
        if total_planned > daily_capacity * 7:
            bottlenecks.append({
                'type': 'Capacity',
                'description': f'Weekly capacity exceeded by {total_planned - daily_capacity * 7:.0f} lbs',
                'severity': 'High'
            })
        
        # Calculate summary
        summary = {
            'scheduled_orders': len(production_schedule),
            'total_production_lbs': total_planned,
            'capacity_utilization': f"{capacity_analysis['utilization_percentage']:.1f}%",
            'bottleneck_count': len(bottlenecks),
            'critical_orders': len([p for p in production_schedule if p['priority'] == 'Critical']),
            'high_priority_orders': len([p for p in production_schedule if p['priority'] == 'High'])
        }
        
        result = {
            'status': 'success',
            'production_schedule': production_schedule,
            'capacity_analysis': capacity_analysis,
            'bottlenecks': bottlenecks,
            'summary': summary,
            'planning_horizon': '30 days',
            'active_orders': len(production_schedule),
            'total_orders': len(knit_orders),
            'timestamp': datetime.now().isoformat()
        }
        
        # Sanitize the entire result to remove NaN values
        return sanitize_for_json(result)
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'production_schedule': [],
            'capacity_analysis': {},
            'bottlenecks': [],
            'summary': {}
        }