#!/usr/bin/env python3
"""
Beverly Knits ERP - Capacity Planning Service
Extracted from beverly_comprehensive_erp.py (lines 2052-2146)
Advanced capacity planning with finite capacity scheduling and bottleneck analysis
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CapacityConfig:
    """Configuration for capacity planning service"""
    bottleneck_threshold: float = 0.85  # Utilization above this is a bottleneck
    critical_threshold: float = 0.95    # Utilization above this is critical
    default_machine_efficiency: float = 0.95
    default_labor_efficiency: float = 0.90


class CapacityPlanningService:
    """
    Advanced capacity planning with finite capacity scheduling and bottleneck analysis
    Extracted from monolith for modular architecture
    """
    
    def __init__(self, config: Optional[CapacityConfig] = None):
        """
        Initialize capacity planning service
        
        Args:
            config: Optional configuration for capacity planning
        """
        self.config = config or CapacityConfig()
        
        # Production resources
        self.production_lines = {}
        self.capacity_constraints = {}
        self.resource_pools = {}
        
        # Shift patterns with efficiency factors
        self.shift_patterns = {
            'day': {'hours': 8, 'efficiency': 0.95},
            'night': {'hours': 8, 'efficiency': 0.90},
            'weekend': {'hours': 12, 'efficiency': 0.85}
        }
        
        # Machine capacities (lbs per day based on historical data)
        self.machine_capacities = {
            45: 150,   # Machine 45: 150 lbs/day
            88: 200,   # Machine 88: 200 lbs/day
            127: 250,  # Machine 127: 250 lbs/day
            147: 180,  # Machine 147: 180 lbs/day
            'default': 100  # Default capacity for unassigned machines
        }
        
        logger.info(f"CapacityPlanningService initialized with bottleneck threshold: {self.config.bottleneck_threshold}")
    
    def calculate_finite_capacity_requirements(self, 
                                              production_plan: Dict[str, float], 
                                              time_horizon_days: int = 30) -> Dict[str, Any]:
        """
        Calculate capacity requirements with finite capacity constraints
        
        Args:
            production_plan: Dictionary of product -> quantity
            time_horizon_days: Planning horizon in days
            
        Returns:
            Dictionary with capacity requirements per product
        """
        capacity_requirements = {}
        
        for product, quantity in production_plan.items():
            # Get production rates based on product characteristics
            # In production, these would come from routing/BOM data
            machine_rate = self._get_machine_rate(product)
            labor_rate = self._get_labor_rate(product)
            setup_time = self._get_setup_time(product)
            
            # Calculate total hours needed including amortized setup time
            batch_size = max(100, quantity)  # Minimum batch of 100 units
            machine_hours = (quantity * machine_rate) + (setup_time * quantity / batch_size)
            labor_hours = (quantity * labor_rate) + (setup_time * 0.5 * quantity / batch_size)
            
            # Calculate days needed based on shift efficiency
            total_days = machine_hours / (
                self.shift_patterns['day']['hours'] * 
                self.shift_patterns['day']['efficiency']
            )
            
            capacity_requirements[product] = {
                'machine_hours': machine_hours,
                'labor_hours': labor_hours,
                'total_days': total_days,
                'feasible_within_horizon': total_days <= time_horizon_days,
                'capacity_utilization': min(1.0, total_days / time_horizon_days)
            }
        
        logger.debug(f"Calculated capacity requirements for {len(production_plan)} products")
        return capacity_requirements
    
    def identify_capacity_bottlenecks(self, 
                                     capacity_utilization: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Identify and analyze production bottlenecks using Theory of Constraints
        
        Args:
            capacity_utilization: Dictionary of resource -> utilization percentage
            
        Returns:
            List of bottlenecks sorted by severity
        """
        bottlenecks = []
        
        for resource, utilization in capacity_utilization.items():
            if utilization > self.config.bottleneck_threshold:
                severity = 'critical' if utilization > self.config.critical_threshold else 'warning'
                
                bottlenecks.append({
                    'resource': resource,
                    'utilization': utilization,
                    'severity': severity,
                    'impact': self._calculate_bottleneck_impact(resource, utilization),
                    'recommendation': self._get_bottleneck_recommendation(utilization)
                })
        
        # Sort by utilization (highest first)
        bottlenecks.sort(key=lambda x: x['utilization'], reverse=True)
        
        if bottlenecks:
            logger.warning(f"Identified {len(bottlenecks)} bottlenecks, "
                         f"{sum(1 for b in bottlenecks if b['severity'] == 'critical')} critical")
        
        return bottlenecks
    
    def optimize_capacity_allocation(self, 
                                    demand_forecast: Dict[str, float], 
                                    capacity_constraints: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Optimize capacity allocation across production lines using linear programming
        
        Args:
            demand_forecast: Dictionary of product -> forecasted demand
            capacity_constraints: Optional specific capacity constraints
            
        Returns:
            Optimized allocation plan
        """
        allocation_plan = {}
        available_capacity = self._get_available_capacity()
        
        # Apply any specific constraints
        if capacity_constraints:
            for product, max_capacity in capacity_constraints.items():
                if product in available_capacity:
                    available_capacity[product] = min(available_capacity[product], max_capacity)
        
        # Simple allocation algorithm
        # TODO: Use scipy.optimize for real linear programming optimization
        total_deficit = 0
        total_allocated = 0
        
        for product, demand in demand_forecast.items():
            # Get available capacity for this product category
            product_capacity = self._get_product_capacity(product, available_capacity)
            
            # Allocate based on available capacity
            allocated = min(demand, product_capacity)
            deficit = max(0, demand - allocated)
            
            allocation_plan[product] = {
                'demand': demand,
                'allocated': allocated,
                'deficit': deficit,
                'utilization': allocated / product_capacity if product_capacity > 0 else 0,
                'fulfillment_rate': allocated / demand if demand > 0 else 1.0
            }
            
            total_deficit += deficit
            total_allocated += allocated
        
        # Add summary metrics
        allocation_plan['_summary'] = {
            'total_demand': sum(demand_forecast.values()),
            'total_allocated': total_allocated,
            'total_deficit': total_deficit,
            'overall_fulfillment_rate': total_allocated / sum(demand_forecast.values()) 
                                       if demand_forecast else 0
        }
        
        logger.info(f"Optimized allocation for {len(demand_forecast)} products, "
                   f"fulfillment rate: {allocation_plan['_summary']['overall_fulfillment_rate']:.1%}")
        
        return allocation_plan
    
    def get_capacity_metrics(self) -> Dict[str, Any]:
        """
        Get current capacity metrics and KPIs
        
        Returns:
            Dictionary with capacity metrics
        """
        total_capacity = sum(cap for machine, cap in self.machine_capacities.items() 
                           if machine != 'default')
        
        return {
            'total_daily_capacity_lbs': total_capacity,
            'machine_count': len(self.machine_capacities) - 1,  # Exclude 'default'
            'shift_patterns': self.shift_patterns,
            'efficiency_metrics': {
                'day_shift': self.shift_patterns['day']['efficiency'],
                'night_shift': self.shift_patterns['night']['efficiency'],
                'weekend_shift': self.shift_patterns['weekend']['efficiency']
            },
            'capacity_by_category': self._get_available_capacity()
        }
    
    def _calculate_bottleneck_impact(self, resource: str, utilization: float) -> Dict[str, Any]:
        """
        Calculate the production impact of a bottleneck
        
        Args:
            resource: Resource identifier
            utilization: Utilization percentage
            
        Returns:
            Impact metrics
        """
        # Calculate impact based on utilization above threshold
        excess_utilization = max(0, utilization - self.config.bottleneck_threshold)
        
        return {
            'throughput_loss': excess_utilization * 100,  # Percentage loss
            'queue_time': utilization * 2,  # Hours of queue time (simplified)
            'priority': 'high' if utilization > self.config.critical_threshold else 'medium',
            'overtime_hours_needed': excess_utilization * 8  # Approximate overtime needed
        }
    
    def _get_bottleneck_recommendation(self, utilization: float) -> str:
        """
        Get recommendation for handling bottleneck
        
        Args:
            utilization: Utilization percentage
            
        Returns:
            Recommendation string
        """
        if utilization > 1.0:
            return "CRITICAL: Add capacity immediately or reduce demand"
        elif utilization > self.config.critical_threshold:
            return "Schedule overtime shifts or redistribute load"
        elif utilization > self.config.bottleneck_threshold:
            return "Monitor closely and prepare contingency plans"
        else:
            return "Within acceptable limits"
    
    def _get_available_capacity(self) -> Dict[str, float]:
        """
        Get available production capacity based on machine allocation
        
        Returns:
            Dictionary with capacity by product category
        """
        # Calculate total available capacity
        total_capacity_per_day = sum(cap for machine, cap in self.machine_capacities.items() 
                                    if machine != 'default')
        
        # Return capacity by product category (simplified allocation)
        return {
            'knit_lightweight': total_capacity_per_day * 0.3,
            'knit_medium': total_capacity_per_day * 0.5,
            'knit_heavy': total_capacity_per_day * 0.2,
            'total_daily': total_capacity_per_day
        }
    
    def _get_product_capacity(self, product: str, available_capacity: Dict[str, float]) -> float:
        """
        Get capacity available for a specific product
        
        Args:
            product: Product identifier
            available_capacity: Available capacity dictionary
            
        Returns:
            Available capacity for the product
        """
        # Map product to category (simplified logic)
        if 'light' in product.lower():
            return available_capacity.get('knit_lightweight', 0)
        elif 'heavy' in product.lower():
            return available_capacity.get('knit_heavy', 0)
        else:
            return available_capacity.get('knit_medium', 0)
    
    def _get_machine_rate(self, product: str) -> float:
        """
        Get machine hours per unit for a product
        
        Args:
            product: Product identifier
            
        Returns:
            Machine hours per unit
        """
        # Determine product type from name/characteristics
        product_lower = product.lower()
        
        if 'knit' in product_lower or 'yarn' in product_lower:
            return 0.75  # 45 minutes for knitting operations
        elif 'dye' in product_lower or 'color' in product_lower:
            return 1.0   # 1 hour for dyeing operations
        elif 'finish' in product_lower or 'final' in product_lower:
            return 0.33  # 20 minutes for finishing
        else:
            return 0.5   # Default 30 minutes
    
    def _get_labor_rate(self, product: str) -> float:
        """
        Get labor hours per unit for a product
        
        Args:
            product: Product identifier
            
        Returns:
            Labor hours per unit
        """
        # Labor rates are typically lower than machine rates
        machine_rate = self._get_machine_rate(product)
        
        # Labor is usually 50-60% of machine time in textile operations
        if 'auto' in product.lower() or 'automated' in product.lower():
            return machine_rate * 0.3  # Less labor for automated processes
        else:
            return machine_rate * 0.6  # Standard labor ratio
    
    def _get_setup_time(self, product: str) -> float:
        """
        Get setup time in hours for a product batch
        
        Args:
            product: Product identifier
            
        Returns:
            Setup time in hours
        """
        product_lower = product.lower()
        
        if 'dye' in product_lower or 'color' in product_lower:
            return 3.0  # Longer setup for color changes
        elif 'custom' in product_lower or 'special' in product_lower:
            return 2.5  # Custom products need more setup
        elif 'standard' in product_lower or 'basic' in product_lower:
            return 1.0  # Standard products have minimal setup
        else:
            return 2.0  # Default setup time


def test_capacity_planning_service():
    """Test the capacity planning service"""
    print("=" * 80)
    print("Testing CapacityPlanningService")
    print("=" * 80)
    
    # Create service with custom config
    config = CapacityConfig(
        bottleneck_threshold=0.80,
        critical_threshold=0.90
    )
    service = CapacityPlanningService(config)
    
    # Test 1: Calculate capacity requirements
    print("\n1. Testing Capacity Requirements Calculation:")
    production_plan = {
        'FABRIC_LIGHT_001': 500,
        'FABRIC_MEDIUM_002': 800,
        'FABRIC_HEAVY_003': 300
    }
    
    requirements = service.calculate_finite_capacity_requirements(production_plan, 30)
    for product, req in requirements.items():
        print(f"  {product}:")
        print(f"    Machine Hours: {req['machine_hours']:.1f}")
        print(f"    Days Needed: {req['total_days']:.1f}")
        print(f"    Feasible in 30 days: {req['feasible_within_horizon']}")
    
    # Test 2: Identify bottlenecks
    print("\n2. Testing Bottleneck Identification:")
    capacity_utilization = {
        'Machine_45': 0.75,
        'Machine_88': 0.88,
        'Machine_127': 0.96,
        'Machine_147': 0.82
    }
    
    bottlenecks = service.identify_capacity_bottlenecks(capacity_utilization)
    if bottlenecks:
        print(f"  Found {len(bottlenecks)} bottlenecks:")
        for bottleneck in bottlenecks:
            print(f"    {bottleneck['resource']}: {bottleneck['utilization']:.1%} "
                  f"({bottleneck['severity']})")
            print(f"      Recommendation: {bottleneck['recommendation']}")
    else:
        print("  No bottlenecks identified")
    
    # Test 3: Optimize allocation
    print("\n3. Testing Capacity Allocation Optimization:")
    demand_forecast = {
        'knit_lightweight_A': 400,
        'knit_medium_B': 600,
        'knit_heavy_C': 200
    }
    
    allocation = service.optimize_capacity_allocation(demand_forecast)
    for product, alloc in allocation.items():
        if product != '_summary':
            print(f"  {product}:")
            print(f"    Demand: {alloc['demand']:.0f}")
            print(f"    Allocated: {alloc['allocated']:.0f}")
            print(f"    Deficit: {alloc['deficit']:.0f}")
            print(f"    Fulfillment: {alloc['fulfillment_rate']:.1%}")
    
    print(f"\n  Overall Fulfillment Rate: {allocation['_summary']['overall_fulfillment_rate']:.1%}")
    
    # Test 4: Get capacity metrics
    print("\n4. Capacity Metrics:")
    metrics = service.get_capacity_metrics()
    print(f"  Total Daily Capacity: {metrics['total_daily_capacity_lbs']} lbs")
    print(f"  Machine Count: {metrics['machine_count']}")
    print(f"  Capacity by Category:")
    for category, capacity in metrics['capacity_by_category'].items():
        if category != 'total_daily':
            print(f"    {category}: {capacity:.0f} lbs/day")
    
    print("\n" + "=" * 80)
    print("✅ CapacityPlanningService test complete")


if __name__ == "__main__":
    test_capacity_planning_service()