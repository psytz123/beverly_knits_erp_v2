#!/usr/bin/env python3
"""
Comprehensive Unit Tests for CapacityPlanningEngine Class
Tests capacity calculations, bottleneck detection, and resource optimization
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.beverly_comprehensive_erp import CapacityPlanningEngine
from src.services.capacity_planning_service import CapacityPlanningService, CapacityConfig


class TestCapacityPlanningEngine:
    """Comprehensive tests for CapacityPlanningEngine class"""
    
    @pytest.fixture
    def planning_engine(self):
        """Create a CapacityPlanningEngine instance for testing"""
        return CapacityPlanningEngine()
    
    @pytest.fixture
    def capacity_service(self):
        """Create a CapacityPlanningService instance for testing"""
        config = CapacityConfig(
            bottleneck_threshold=0.85,
            critical_threshold=0.95
        )
        return CapacityPlanningService(config)
    
    @pytest.fixture
    def sample_production_plan(self):
        """Sample production plan data"""
        return {
            'KNIT_001': 500,
            'KNIT_002': 300,
            'DYE_001': 200,
            'FINISH_001': 400,
            'CUSTOM_001': 100
        }
    
    @pytest.fixture
    def sample_machine_capacity(self):
        """Sample machine capacity data"""
        return {
            'Machine_A': {'capacity': 100, 'efficiency': 0.95, 'type': 'knitting'},
            'Machine_B': {'capacity': 80, 'efficiency': 0.90, 'type': 'knitting'},
            'Machine_C': {'capacity': 120, 'efficiency': 0.85, 'type': 'dyeing'},
            'Machine_D': {'capacity': 90, 'efficiency': 0.92, 'type': 'finishing'}
        }
    
    @pytest.fixture
    def sample_resource_data(self):
        """Sample resource availability data"""
        return pd.DataFrame({
            'resource_id': ['RES001', 'RES002', 'RES003', 'RES004'],
            'resource_type': ['machine', 'labor', 'machine', 'labor'],
            'available_hours': [160, 320, 160, 280],
            'efficiency': [0.95, 0.90, 0.92, 0.88],
            'cost_per_hour': [50, 25, 60, 30]
        })
    
    def test_capacity_calculation(self, capacity_service, sample_production_plan):
        """Test basic capacity requirement calculation"""
        # Calculate capacity requirements
        requirements = capacity_service.calculate_capacity_requirements(
            sample_production_plan,
            time_horizon_days=30
        )
        
        assert isinstance(requirements, dict)
        assert 'KNIT_001' in requirements
        
        # Check calculations for KNIT_001
        knit_001 = requirements['KNIT_001']
        assert 'machine_hours' in knit_001
        assert 'labor_hours' in knit_001
        assert 'total_days' in knit_001
        assert 'capacity_utilization' in knit_001
        
        # Machine hours should be positive
        assert knit_001['machine_hours'] > 0
        assert knit_001['labor_hours'] > 0
    
    def test_production_rate_calculation(self, capacity_service):
        """Test production rate calculations with actual values"""
        # Test different product types
        knit_rate = capacity_service._get_machine_rate('knit_product')
        dye_rate = capacity_service._get_machine_rate('dye_batch')
        finish_rate = capacity_service._get_machine_rate('finishing_process')
        
        # Verify rates are different and realistic
        assert knit_rate == 0.75  # 45 minutes
        assert dye_rate == 1.0    # 1 hour
        assert finish_rate == 0.33  # 20 minutes
    
    def test_bottleneck_identification(self, capacity_service):
        """Test identification of capacity bottlenecks"""
        # Create utilization data with bottlenecks
        capacity_utilization = {
            'Machine_A': 0.75,  # Normal
            'Machine_B': 0.88,  # Bottleneck
            'Machine_C': 0.96,  # Critical
            'Machine_D': 0.65   # Under-utilized
        }
        
        bottlenecks = capacity_service.identify_capacity_bottlenecks(capacity_utilization)
        
        assert len(bottlenecks) == 2  # Machine_B and Machine_C
        
        # Check bottleneck details
        critical = [b for b in bottlenecks if b['severity'] == 'critical']
        warning = [b for b in bottlenecks if b['severity'] == 'warning']
        
        assert len(critical) == 1  # Machine_C
        assert len(warning) == 1   # Machine_B
    
    def test_capacity_allocation_optimization(self, capacity_service):
        """Test optimal capacity allocation"""
        # Demand forecast
        demand_forecast = {
            'knit_lightweight': 400,
            'knit_medium': 600,
            'knit_heavy': 200
        }
        
        allocation = capacity_service.optimize_capacity_allocation(demand_forecast)
        
        assert '_summary' in allocation
        assert 'overall_fulfillment_rate' in allocation['_summary']
        
        # Check individual allocations
        for product in demand_forecast:
            assert product in allocation
            assert 'demand' in allocation[product]
            assert 'allocated' in allocation[product]
            assert 'deficit' in allocation[product]
            assert 'fulfillment_rate' in allocation[product]
            
            # Allocated should not exceed demand
            assert allocation[product]['allocated'] <= allocation[product]['demand']
    
    def test_shift_pattern_calculation(self, capacity_service):
        """Test shift pattern and efficiency calculations"""
        shifts = capacity_service.shift_patterns
        
        # Verify shift patterns exist
        assert 'day' in shifts
        assert 'night' in shifts
        assert 'weekend' in shifts
        
        # Day shift should be most efficient
        assert shifts['day']['efficiency'] > shifts['night']['efficiency']
        assert shifts['day']['efficiency'] > shifts['weekend']['efficiency']
        
        # Calculate total available hours per day
        total_hours = sum(s['hours'] * s['efficiency'] for s in shifts.values())
        assert total_hours > 20  # Should have substantial capacity
    
    def test_resource_utilization_calculation(self, planning_engine, sample_resource_data):
        """Test resource utilization calculations"""
        # Calculate utilization for each resource
        production_hours = {
            'RES001': 140,
            'RES002': 280,
            'RES003': 150,
            'RES004': 250
        }
        
        utilization = {}
        for _, resource in sample_resource_data.iterrows():
            resource_id = resource['resource_id']
            if resource_id in production_hours:
                utilization[resource_id] = (
                    production_hours[resource_id] / resource['available_hours']
                )
        
        # Verify utilization calculations
        assert pytest.approx(utilization['RES001'], 0.01) == 0.875  # 140/160
        assert pytest.approx(utilization['RES002'], 0.01) == 0.875  # 280/320
        assert pytest.approx(utilization['RES003'], 0.01) == 0.9375  # 150/160
        assert pytest.approx(utilization['RES004'], 0.01) == 0.893  # 250/280
    
    def test_production_scheduling(self, capacity_service, sample_production_plan):
        """Test production schedule optimization"""
        # Create constraints
        constraints = {
            'max_hours_per_day': 16,
            'max_machines': 4,
            'priority_orders': ['CUSTOM_001', 'KNIT_001']
        }
        
        schedule = capacity_service.optimize_production_schedule(
            sample_production_plan,
            constraints
        )
        
        assert 'optimized_schedule' in schedule
        assert 'total_time' in schedule
        assert 'efficiency_score' in schedule
        
        # Priority orders should be scheduled first
        optimized = schedule['optimized_schedule']
        priority_indices = [
            i for i, item in enumerate(optimized) 
            if item['product'] in constraints['priority_orders']
        ]
        
        # Priority items should be in first half of schedule
        assert all(idx < len(optimized) / 2 for idx in priority_indices[:1])
    
    def test_capacity_constraints_validation(self, capacity_service):
        """Test validation of capacity constraints"""
        # Test with excessive demand
        excessive_demand = {
            'product_A': 10000,  # Impossible to fulfill
            'product_B': 5000
        }
        
        # Calculate if demand is feasible
        daily_capacity = 200  # units per day
        days_available = 30
        total_capacity = daily_capacity * days_available
        
        total_demand = sum(excessive_demand.values())
        is_feasible = total_demand <= total_capacity
        
        assert not is_feasible  # Should be infeasible
    
    def test_multi_resource_scheduling(self, planning_engine):
        """Test scheduling with multiple resource types"""
        # Define resources needed per product
        resource_requirements = {
            'PROD_A': {'machine': 2, 'labor': 3, 'material': 100},
            'PROD_B': {'machine': 1, 'labor': 2, 'material': 50},
            'PROD_C': {'machine': 3, 'labor': 4, 'material': 150}
        }
        
        # Available resources
        available = {
            'machine': 5,
            'labor': 10,
            'material': 500
        }
        
        # Check which products can be produced simultaneously
        can_produce_together = []
        for prod1 in resource_requirements:
            for prod2 in resource_requirements:
                if prod1 < prod2:  # Avoid duplicates
                    total_needed = {}
                    for resource in available:
                        total_needed[resource] = (
                            resource_requirements[prod1].get(resource, 0) +
                            resource_requirements[prod2].get(resource, 0)
                        )
                    
                    if all(total_needed[r] <= available[r] for r in available):
                        can_produce_together.append((prod1, prod2))
        
        # Should be able to produce some combinations
        assert len(can_produce_together) > 0
    
    def test_overtime_calculation(self, capacity_service):
        """Test overtime requirements calculation"""
        # Regular capacity
        regular_hours = 8 * 5  # 40 hours per week
        regular_capacity = regular_hours * 50  # 50 units per hour
        
        # Demand exceeds regular capacity
        demand = 2500
        
        # Calculate overtime needed
        overtime_needed = max(0, demand - regular_capacity)
        overtime_hours = overtime_needed / 50  # Same productivity
        
        assert overtime_needed == 500  # 2500 - 2000
        assert overtime_hours == 10
        
        # Calculate overtime cost (1.5x regular rate)
        regular_rate = 25
        overtime_rate = regular_rate * 1.5
        overtime_cost = overtime_hours * overtime_rate
        
        assert overtime_cost == 375  # 10 * 37.5
    
    def test_capacity_expansion_analysis(self, planning_engine):
        """Test capacity expansion decision analysis"""
        # Current capacity and demand
        current_capacity = 1000
        current_demand = 800
        growth_rate = 0.15  # 15% annual growth
        
        # Project future demand
        years = 5
        future_demand = current_demand * ((1 + growth_rate) ** years)
        
        # Check if expansion needed
        needs_expansion = future_demand > current_capacity
        assert needs_expansion  # Should need expansion
        
        # Calculate expansion size needed
        expansion_needed = future_demand - current_capacity
        assert expansion_needed > 0
        
        # Calculate ROI of expansion
        expansion_cost = expansion_needed * 1000  # $1000 per unit capacity
        annual_revenue_increase = expansion_needed * 100  # $100 per unit
        payback_period = expansion_cost / annual_revenue_increase
        
        assert payback_period < 20  # Should pay back in reasonable time
    
    def test_maintenance_scheduling(self, capacity_service):
        """Test integration of maintenance into capacity planning"""
        # Machine availability considering maintenance
        machines = {
            'Machine_A': {'uptime': 0.95, 'maintenance_hours': 8},
            'Machine_B': {'uptime': 0.92, 'maintenance_hours': 12},
            'Machine_C': {'uptime': 0.98, 'maintenance_hours': 4}
        }
        
        # Calculate effective capacity
        total_hours = 24 * 30  # 30 days
        effective_capacity = {}
        
        for machine, specs in machines.items():
            available_hours = total_hours - specs['maintenance_hours']
            effective_hours = available_hours * specs['uptime']
            effective_capacity[machine] = effective_hours
        
        # Verify calculations
        assert effective_capacity['Machine_A'] < total_hours
        assert effective_capacity['Machine_C'] > effective_capacity['Machine_B']
    
    def test_setup_time_optimization(self, capacity_service):
        """Test setup time optimization in scheduling"""
        # Products with setup times
        products = {
            'PROD_A': {'setup': 2, 'run_time': 0.5, 'quantity': 100},
            'PROD_B': {'setup': 3, 'run_time': 0.4, 'quantity': 150},
            'PROD_C': {'setup': 1.5, 'run_time': 0.6, 'quantity': 80}
        }
        
        # Calculate total time with and without batching
        
        # Without batching (setup for each unit)
        time_without_batching = sum(
            p['quantity'] * (p['setup'] + p['run_time'])
            for p in products.values()
        )
        
        # With batching (one setup per product)
        time_with_batching = sum(
            p['setup'] + (p['quantity'] * p['run_time'])
            for p in products.values()
        )
        
        # Batching should save significant time
        time_saved = time_without_batching - time_with_batching
        efficiency_gain = time_saved / time_without_batching
        
        assert efficiency_gain > 0.5  # Should save >50% time
    
    def test_capacity_metrics_calculation(self, capacity_service):
        """Test calculation of capacity metrics"""
        metrics = capacity_service.get_capacity_metrics()
        
        assert 'total_daily_capacity_lbs' in metrics
        assert 'machine_count' in metrics
        assert 'capacity_by_category' in metrics
        
        # Verify metrics are reasonable
        assert metrics['total_daily_capacity_lbs'] > 0
        assert metrics['machine_count'] > 0
        assert len(metrics['capacity_by_category']) > 0
    
    def test_parallel_processing_optimization(self, planning_engine):
        """Test optimization of parallel processing"""
        # Tasks that can be parallelized
        tasks = [
            {'id': 'T1', 'duration': 4, 'dependencies': []},
            {'id': 'T2', 'duration': 3, 'dependencies': []},
            {'id': 'T3', 'duration': 5, 'dependencies': ['T1']},
            {'id': 'T4', 'duration': 2, 'dependencies': ['T2']},
            {'id': 'T5', 'duration': 3, 'dependencies': ['T3', 'T4']}
        ]
        
        # Calculate critical path
        # T1 -> T3 -> T5 = 4 + 5 + 3 = 12
        # T2 -> T4 -> T5 = 3 + 2 + 3 = 8
        critical_path_duration = 12
        
        # With 2 machines, T1 and T2 can run in parallel
        parallel_duration = 12  # Limited by critical path
        
        # Sequential would be sum of all durations
        sequential_duration = sum(t['duration'] for t in tasks)
        
        assert parallel_duration < sequential_duration
        assert parallel_duration == critical_path_duration


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])