"""
Capacity Planning Service
Extracted from beverly_comprehensive_erp.py (lines 2724-2815)
PRESERVED EXACTLY - Advanced capacity planning with finite capacity scheduling
"""

import numpy as np
from datetime import datetime


class CapacityPlanningEngine:
    """Advanced capacity planning with finite capacity scheduling and bottleneck analysis"""
    
    def __init__(self):
        self.production_lines = {}
        self.capacity_constraints = {}
        self.shift_patterns = {
            'day': {'hours': 8, 'efficiency': 0.95},
            'night': {'hours': 8, 'efficiency': 0.90},
            'weekend': {'hours': 12, 'efficiency': 0.85}
        }
        self.resource_pools = {}
        
    def calculate_finite_capacity_requirements(self, production_plan, time_horizon_days=30):
        """Calculate capacity requirements with finite capacity constraints"""
        capacity_requirements = {}
        
        for product, quantity in production_plan.items():
            # Calculate machine hours needed
            machine_hours = quantity * 0.5  # Placeholder - should come from routing data
            labor_hours = quantity * 0.3
            
            capacity_requirements[product] = {
                'machine_hours': machine_hours,
                'labor_hours': labor_hours,
                'total_days': machine_hours / (self.shift_patterns['day']['hours'] * 
                                              self.shift_patterns['day']['efficiency'])
            }
            
        return capacity_requirements
    
    def identify_capacity_bottlenecks(self, capacity_utilization):
        """Identify and analyze production bottlenecks using Theory of Constraints"""
        bottlenecks = []
        
        for resource, utilization in capacity_utilization.items():
            if utilization > 0.85:  # Bottleneck threshold
                bottlenecks.append({
                    'resource': resource,
                    'utilization': utilization,
                    'severity': 'critical' if utilization > 0.95 else 'warning',
                    'impact': self._calculate_bottleneck_impact(resource, utilization)
                })
                
        return sorted(bottlenecks, key=lambda x: x['utilization'], reverse=True)
    
    def optimize_capacity_allocation(self, demand_forecast, capacity_constraints):
        """Optimize capacity allocation across production lines using linear programming"""
        allocation_plan = {}
        available_capacity = self._get_available_capacity()
        
        # Simple allocation algorithm - should use scipy.optimize for real optimization
        for product, demand in demand_forecast.items():
            allocated = min(demand, available_capacity.get(product, 0))
            allocation_plan[product] = {
                'allocated': allocated,
                'deficit': max(0, demand - allocated),
                'utilization': allocated / available_capacity.get(product, 1) if available_capacity.get(product, 0) > 0 else 0
            }
            
        return allocation_plan
    
    def _calculate_bottleneck_impact(self, resource, utilization):
        """Calculate the production impact of a bottleneck"""
        return {
            'throughput_loss': (utilization - 0.85) * 100,  # Percentage loss
            'queue_time': utilization * 2,  # Hours of queue time
            'priority': 'high' if utilization > 0.95 else 'medium'
        }
    
    def _get_available_capacity(self):
        """Get available production capacity based on machine allocation"""
        # Machine capacity in lbs per day (based on historical data)
        machine_capacities = {
            45: 150,   # Machine 45: 150 lbs/day
            88: 200,   # Machine 88: 200 lbs/day
            127: 250,  # Machine 127: 250 lbs/day
            147: 180,  # Machine 147: 180 lbs/day
            'default': 100  # Default capacity for unassigned machines
        }
        
        # Calculate total available capacity
        total_capacity_per_day = sum(cap for machine, cap in machine_capacities.items() if machine != 'default')
        
        # Return capacity by product category (simplified)
        return {
            'knit_lightweight': total_capacity_per_day * 0.3,
            'knit_medium': total_capacity_per_day * 0.5,
            'knit_heavy': total_capacity_per_day * 0.2,
            'total_daily': total_capacity_per_day
        }