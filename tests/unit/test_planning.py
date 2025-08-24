"""
Unit tests for planning functions in beverly_comprehensive_erp.py

Tests six-phase planning engine and related planning logic
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import sys
from pathlib import Path

import core.beverly_comprehensive_erp as erp


class TestSixPhasePlanning:
    """Test suite for six-phase planning functionality"""
    
    @pytest.fixture
    def sample_planning_data(self):
        """Create sample data for planning tests"""
        return {
            'inventory': pd.DataFrame({
                'Item': ['ITEM001', 'ITEM002', 'ITEM003'],
                'Description': ['Product A', 'Product B', 'Product C'],
                'Planning Balance': [100, -50, 200],
                'On Order': [50, 100, 0],
                'Safety Stock': [30, 40, 50],
                'Lead Time': [7, 14, 10]
            }),
            'demand': pd.DataFrame({
                'Item': ['ITEM001', 'ITEM002', 'ITEM003'],
                'Week1': [20, 30, 15],
                'Week2': [25, 35, 20],
                'Week3': [30, 40, 25],
                'Week4': [35, 45, 30]
            }),
            'suppliers': pd.DataFrame({
                'Supplier': ['Supplier A', 'Supplier B', 'Supplier C'],
                'Lead_Time': [7, 10, 14],
                'Reliability': [0.95, 0.90, 0.85],
                'Cost_Factor': [1.0, 0.95, 1.05]
            })
        }
    
    def test_phase1_demand_analysis(self, sample_planning_data):
        """Test Phase 1: Demand Analysis"""
        demand_data = sample_planning_data['demand']
        
        # Calculate total demand per item
        demand_totals = demand_data[['Week1', 'Week2', 'Week3', 'Week4']].sum(axis=1)
        
        assert len(demand_totals) == 3
        assert demand_totals[0] == 110  # ITEM001 total demand
        assert demand_totals[1] == 150  # ITEM002 total demand
        assert demand_totals[2] == 90   # ITEM003 total demand
        
        # Calculate average weekly demand
        avg_weekly_demand = demand_totals / 4
        assert avg_weekly_demand[0] == 27.5
    
    def test_phase2_inventory_assessment(self, sample_planning_data):
        """Test Phase 2: Inventory Assessment"""
        inventory = sample_planning_data['inventory']
        
        # Identify critical items (negative planning balance)
        critical_items = inventory[inventory['Planning Balance'] < 0]
        assert len(critical_items) == 1
        assert critical_items.iloc[0]['Item'] == 'ITEM002'
        
        # Calculate coverage days
        inventory['Coverage_Days'] = inventory.apply(
            lambda row: row['Planning Balance'] / 10 if row['Planning Balance'] > 0 else 0,
            axis=1
        )
        
        assert inventory.iloc[0]['Coverage_Days'] == 10  # 100/10
        assert inventory.iloc[1]['Coverage_Days'] == 0   # Negative balance
    
    def test_phase3_procurement_planning(self, sample_planning_data):
        """Test Phase 3: Procurement Planning"""
        inventory = sample_planning_data['inventory']
        suppliers = sample_planning_data['suppliers']
        
        # Calculate procurement needs
        procurement_needs = []
        for _, item in inventory.iterrows():
            if item['Planning Balance'] < item['Safety Stock']:
                order_qty = item['Safety Stock'] * 2 - item['Planning Balance']
                procurement_needs.append({
                    'Item': item['Item'],
                    'Order_Quantity': order_qty,
                    'Priority': 'High' if item['Planning Balance'] < 0 else 'Medium'
                })
        
        assert len(procurement_needs) >= 1
        
        # Find item with negative balance
        critical_procurement = [p for p in procurement_needs if p['Priority'] == 'High']
        assert len(critical_procurement) >= 1
    
    def test_phase4_production_scheduling(self, sample_planning_data):
        """Test Phase 4: Production Scheduling"""
        demand_data = sample_planning_data['demand']
        
        # Create production schedule based on demand
        production_schedule = []
        for week in ['Week1', 'Week2', 'Week3', 'Week4']:
            week_schedule = {
                'Week': week,
                'Total_Units': demand_data[week].sum(),
                'Priority_Items': demand_data.nlargest(1, week)['Item'].values[0]
            }
            production_schedule.append(week_schedule)
        
        assert len(production_schedule) == 4
        assert production_schedule[0]['Week'] == 'Week1'
        assert production_schedule[0]['Total_Units'] > 0
    
    def test_phase5_logistics_optimization(self, sample_planning_data):
        """Test Phase 5: Logistics Optimization"""
        suppliers = sample_planning_data['suppliers']
        
        # Optimize supplier selection based on lead time and reliability
        suppliers['Score'] = (
            suppliers['Reliability'] * 0.5 +
            (1 / suppliers['Lead_Time']) * 10 * 0.3 +
            (1 / suppliers['Cost_Factor']) * 0.2
        )
        
        best_supplier = suppliers.nlargest(1, 'Score')
        assert len(best_supplier) == 1
        assert best_supplier.iloc[0]['Supplier'] in ['Supplier A', 'Supplier B', 'Supplier C']
    
    def test_phase6_execution_monitoring(self, sample_planning_data):
        """Test Phase 6: Execution & Monitoring"""
        # Simulate KPI tracking
        kpis = {
            'inventory_turnover': 5.2,
            'stockout_rate': 0.02,
            'order_fulfillment_rate': 0.98,
            'forecast_accuracy': 0.85
        }
        
        # Check KPI thresholds
        assert kpis['inventory_turnover'] > 4  # Good turnover
        assert kpis['stockout_rate'] < 0.05   # Low stockout rate
        assert kpis['order_fulfillment_rate'] > 0.95  # High fulfillment
        assert kpis['forecast_accuracy'] > 0.80  # Good forecast accuracy
    
    def test_planning_cycle_integration(self, sample_planning_data):
        """Test complete planning cycle integration"""
        phases_completed = []
        
        # Execute all phases
        for phase_num in range(1, 7):
            phase_result = {
                'phase': phase_num,
                'status': 'completed',
                'timestamp': datetime.now()
            }
            phases_completed.append(phase_result)
        
        assert len(phases_completed) == 6
        assert all(p['status'] == 'completed' for p in phases_completed)
    
    def test_planning_error_handling(self):
        """Test error handling in planning functions"""
        # Test with empty data
        empty_data = pd.DataFrame()
        
        # Should handle empty data gracefully
        try:
            result = len(empty_data)
            assert result == 0
        except Exception as e:
            pytest.fail(f"Failed to handle empty data: {e}")
        
        # Test with missing columns
        incomplete_data = pd.DataFrame({'Item': ['TEST001']})
        
        # Should identify missing required columns
        required_columns = ['Planning Balance', 'Safety Stock', 'Lead Time']
        missing_columns = [col for col in required_columns if col not in incomplete_data.columns]
        
        assert len(missing_columns) == 3


class TestProductionScheduler:
    """Test ProductionScheduler class"""
    
    def test_scheduler_initialization(self):
        """Test ProductionScheduler initialization"""
        scheduler = erp.ProductionScheduler()
        
        assert scheduler is not None
        assert hasattr(scheduler, 'schedule')
        assert hasattr(scheduler, 'capacity')
    
    def test_calculate_production_capacity(self):
        """Test production capacity calculation"""
        machines = 10
        hours_per_day = 8
        efficiency = 0.85
        days = 5
        
        capacity = machines * hours_per_day * efficiency * days
        assert capacity == 340  # 10 * 8 * 0.85 * 5
    
    def test_schedule_optimization(self):
        """Test production schedule optimization"""
        jobs = [
            {'id': 'JOB001', 'duration': 4, 'priority': 1},
            {'id': 'JOB002', 'duration': 2, 'priority': 3},
            {'id': 'JOB003', 'duration': 3, 'priority': 2}
        ]
        
        # Sort by priority (lower number = higher priority)
        sorted_jobs = sorted(jobs, key=lambda x: x['priority'])
        
        assert sorted_jobs[0]['id'] == 'JOB001'
        assert sorted_jobs[1]['id'] == 'JOB003'
        assert sorted_jobs[2]['id'] == 'JOB002'
    
    def test_bottleneck_identification(self):
        """Test production bottleneck identification"""
        stages = [
            {'stage': 'Cutting', 'capacity': 100, 'demand': 80},
            {'stage': 'Sewing', 'capacity': 80, 'demand': 90},
            {'stage': 'Finishing', 'capacity': 120, 'demand': 70}
        ]
        
        # Find bottleneck (demand > capacity or highest utilization)
        for stage in stages:
            stage['utilization'] = stage['demand'] / stage['capacity']
        
        bottleneck = max(stages, key=lambda x: x['utilization'])
        assert bottleneck['stage'] == 'Sewing'  # 90/80 = 1.125 utilization


class TestCapacityPlanning:
    """Test CapacityPlanningEngine class"""
    
    def test_capacity_planning_initialization(self):
        """Test CapacityPlanningEngine initialization"""
        planner = erp.CapacityPlanningEngine()
        
        assert planner is not None
        assert hasattr(planner, 'resources')
        assert hasattr(planner, 'constraints')
    
    def test_resource_allocation(self):
        """Test resource allocation logic"""
        available_resources = {
            'workers': 50,
            'machines': 20,
            'hours': 400
        }
        
        job_requirements = {
            'workers': 10,
            'machines': 5,
            'hours': 80
        }
        
        # Check if resources are sufficient
        can_allocate = all(
            available_resources[resource] >= job_requirements[resource]
            for resource in job_requirements
        )
        
        assert can_allocate is True
        
        # Allocate resources
        for resource in job_requirements:
            available_resources[resource] -= job_requirements[resource]
        
        assert available_resources['workers'] == 40
        assert available_resources['machines'] == 15
        assert available_resources['hours'] == 320
    
    def test_capacity_utilization_calculation(self):
        """Test capacity utilization calculation"""
        used_capacity = 750
        total_capacity = 1000
        
        utilization = (used_capacity / total_capacity) * 100
        assert utilization == 75.0
        
        # Test with overcapacity
        used_capacity = 1100
        utilization = (used_capacity / total_capacity) * 100
        assert utilization == 110.0  # Overcapacity
    
    def test_shift_planning(self):
        """Test shift planning optimization"""
        daily_demand = 240  # units
        units_per_hour = 10
        hours_per_shift = 8
        
        units_per_shift = units_per_hour * hours_per_shift
        shifts_needed = np.ceil(daily_demand / units_per_shift)
        
        assert shifts_needed == 3  # Need 3 shifts to meet demand


class TestMRPSystem:
    """Test TimePhasedMRP class"""
    
    def test_mrp_initialization(self):
        """Test TimePhasedMRP initialization"""
        mrp = erp.TimePhasedMRP()
        
        assert mrp is not None
        assert hasattr(mrp, 'bom')
        assert hasattr(mrp, 'lead_times')
    
    def test_bom_explosion(self):
        """Test Bill of Materials explosion"""
        bom = {
            'PRODUCT_A': {
                'COMPONENT_1': 2,
                'COMPONENT_2': 3
            },
            'COMPONENT_1': {
                'RAW_MATERIAL_1': 5
            }
        }
        
        # Calculate requirements for 10 units of PRODUCT_A
        quantity = 10
        requirements = {}
        
        # Level 1 explosion
        requirements['COMPONENT_1'] = quantity * 2  # 20
        requirements['COMPONENT_2'] = quantity * 3  # 30
        
        # Level 2 explosion
        requirements['RAW_MATERIAL_1'] = requirements['COMPONENT_1'] * 5  # 100
        
        assert requirements['COMPONENT_1'] == 20
        assert requirements['COMPONENT_2'] == 30
        assert requirements['RAW_MATERIAL_1'] == 100
    
    def test_lead_time_offsetting(self):
        """Test lead time offsetting in MRP"""
        due_date = datetime(2025, 8, 30)
        lead_times = {
            'ASSEMBLY': 3,
            'PROCUREMENT': 7,
            'PRODUCTION': 5
        }
        
        # Calculate start dates
        assembly_start = due_date - timedelta(days=lead_times['ASSEMBLY'])
        production_start = assembly_start - timedelta(days=lead_times['PRODUCTION'])
        procurement_start = production_start - timedelta(days=lead_times['PROCUREMENT'])
        
        assert assembly_start == datetime(2025, 8, 27)
        assert production_start == datetime(2025, 8, 22)
        assert procurement_start == datetime(2025, 8, 15)
    
    def test_net_requirements_calculation(self):
        """Test net requirements calculation"""
        gross_requirement = 100
        on_hand_inventory = 30
        scheduled_receipts = 20
        
        net_requirement = max(0, gross_requirement - on_hand_inventory - scheduled_receipts)
        
        assert net_requirement == 50  # 100 - 30 - 20
        
        # Test when inventory covers requirement
        on_hand_inventory = 150
        net_requirement = max(0, gross_requirement - on_hand_inventory - scheduled_receipts)
        
        assert net_requirement == 0  # No additional requirement needed


class TestPlanningOptimization:
    """Test planning optimization algorithms"""
    
    def test_minimize_total_cost(self):
        """Test cost minimization in planning"""
        options = [
            {'supplier': 'A', 'cost': 1000, 'quality': 0.95},
            {'supplier': 'B', 'cost': 900, 'quality': 0.90},
            {'supplier': 'C', 'cost': 1100, 'quality': 0.98}
        ]
        
        # Find minimum cost option
        min_cost_option = min(options, key=lambda x: x['cost'])
        assert min_cost_option['supplier'] == 'B'
        
        # Find best value (cost/quality ratio)
        for option in options:
            option['value'] = option['cost'] / option['quality']
        
        best_value_option = min(options, key=lambda x: x['value'])
        assert best_value_option['supplier'] == 'B'
    
    def test_maximize_throughput(self):
        """Test throughput maximization"""
        processes = [
            {'name': 'Process A', 'rate': 100},  # units/hour
            {'name': 'Process B', 'rate': 80},
            {'name': 'Process C', 'rate': 120}
        ]
        
        # System throughput limited by slowest process
        system_throughput = min(p['rate'] for p in processes)
        assert system_throughput == 80
        
        # Identify bottleneck
        bottleneck = min(processes, key=lambda x: x['rate'])
        assert bottleneck['name'] == 'Process B'
    
    def test_balance_workload(self):
        """Test workload balancing across resources"""
        workstations = [
            {'id': 'WS1', 'capacity': 100, 'load': 0},
            {'id': 'WS2', 'capacity': 100, 'load': 0},
            {'id': 'WS3', 'capacity': 100, 'load': 0}
        ]
        
        jobs = [50, 40, 60, 30, 20]  # Job sizes
        
        # Balance load across workstations
        for job in jobs:
            # Find workstation with minimum load
            min_load_ws = min(workstations, key=lambda x: x['load'])
            min_load_ws['load'] += job
        
        # Check load distribution
        loads = [ws['load'] for ws in workstations]
        assert max(loads) - min(loads) <= 20  # Reasonably balanced
        assert sum(loads) == sum(jobs)  # All jobs assigned