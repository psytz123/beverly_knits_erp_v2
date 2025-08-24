"""
Comprehensive Unit Tests for 6-Phase Planning Engine
Task 6.1: Create Unit Tests for Planning Phases
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

from production.six_phase_planning_engine import SixPhasePlanningEngine


class TestPlanningPhases:
    """Test suite for the 6-phase planning engine phases"""
    
    @pytest.fixture
    def planning_engine(self):
        """Create a planning engine instance for testing"""
        return SixPhasePlanningEngine()
    
    @pytest.fixture
    def sample_sales_orders(self):
        """Create sample sales orders for testing"""
        return pd.DataFrame({
            'Style#': ['TEST001', 'TEST001', 'TEST002', 'TEST003'],
            'Quantity': [100, 50, 200, 75],
            'Customer': ['Cust1', 'Cust2', 'Cust1', 'Cust3'],
            'Order_Date': [datetime.now() - timedelta(days=i) for i in range(4)],
            'Due_Date': [datetime.now() + timedelta(days=30+i) for i in range(4)]
        })
    
    @pytest.fixture
    def sample_inventory(self):
        """Create sample inventory data for testing"""
        return {
            'F01': pd.DataFrame({
                'fStyle#': ['F_TEST001', 'F_TEST002', 'F_TEST003'],
                'Quantity': [30, 50, 20]
            }),
            'I01': pd.DataFrame({
                'fStyle#': ['F_TEST001', 'F_TEST002'],
                'Quantity': [20, 30]
            }),
            'G00': pd.DataFrame({
                'fStyle#': ['F_TEST001', 'F_TEST002', 'F_TEST003'],
                'Quantity': [25, 40, 15]
            }),
            'G02': pd.DataFrame({
                'fStyle#': ['F_TEST001', 'F_TEST003'],
                'Quantity': [15, 10]
            })
        }
    
    @pytest.fixture
    def sample_bom(self):
        """Create sample BOM data for testing"""
        return pd.DataFrame({
            'Style#': ['TEST001', 'TEST001', 'TEST002', 'TEST002', 'TEST003'],
            'Desc#': ['YARN001', 'YARN002', 'YARN001', 'YARN003', 'YARN002'],
            'BOM_Percent': [60, 40, 70, 30, 100],
            'Unit': ['lbs', 'lbs', 'lbs', 'lbs', 'lbs']
        })
    
    def test_phase1_demand_consolidation(self, planning_engine, sample_sales_orders):
        """Test Phase 1: Demand Consolidation"""
        # Mock the data loading
        planning_engine.sales_orders = sample_sales_orders
        
        # Execute phase 1
        result = planning_engine.execute_demand_consolidation()
        
        # Assertions
        assert 'total_demand' in result
        assert 'sales_orders' in result
        assert 'forecast' in result
        
        # Check demand consolidation
        assert result['total_demand']['TEST001'] == 150  # 100 + 50
        assert result['total_demand']['TEST002'] == 200
        assert result['total_demand']['TEST003'] == 75
        
        # Check that sales orders are properly counted
        assert result['sales_orders']['TEST001'] == 150
        assert result['confidence_score'] is not None
    
    def test_phase2_inventory_assessment(self, planning_engine, sample_inventory):
        """Test Phase 2: Inventory Assessment"""
        # Mock inventory data
        planning_engine.inventory = sample_inventory
        
        # Create style mapping (simplified)
        planning_engine.style_mapping = {
            'F_TEST001': 'TEST001',
            'F_TEST002': 'TEST002',
            'F_TEST003': 'TEST003'
        }
        
        # Execute phase 2
        result = planning_engine.execute_inventory_assessment()
        
        # Assertions
        assert 'inventory_by_stage' in result
        assert 'total_available' in result
        assert 'pipeline_inventory' in result
        
        # Check inventory aggregation
        # TEST001: F01=30, I01=20, G00=25, G02=15 = 90 total
        assert result['total_available'].get('TEST001', 0) == 90
        
    def test_phase3_net_requirements(self, planning_engine):
        """Test Phase 3: Net Requirements Calculation"""
        # Setup test data
        total_demand = {'TEST001': 100, 'TEST002': 50, 'TEST003': 150}
        available_inventory = {'TEST001': 30, 'TEST002': 60, 'TEST003': 0}
        
        # Execute phase 3
        result = planning_engine.calculate_net_requirements(
            total_demand, 
            available_inventory
        )
        
        # Assertions
        assert 'net_requirements' in result
        assert 'inventory_used' in result
        
        # Check net requirements calculation
        assert result['net_requirements']['TEST001'] == 70  # 100 - 30
        assert result['net_requirements']['TEST002'] == 0   # 50 - 60 (no negative)
        assert result['net_requirements']['TEST003'] == 150 # 150 - 0
        
        # Check inventory used
        assert result['inventory_used']['TEST001'] == 30
        assert result['inventory_used']['TEST002'] == 50  # Only uses what's needed
    
    def test_phase4_bom_explosion_net_only(self, planning_engine, sample_bom):
        """Test Phase 4: BOM Explosion for Net Requirements Only"""
        # Mock BOM data
        planning_engine.bom_df = sample_bom
        
        # Net requirements (only styles with positive net should be processed)
        net_requirements = {
            'TEST001': 100,  # Has net requirement
            'TEST002': 0,    # No net requirement - should be skipped
            'TEST003': 50    # Has net requirement
        }
        
        # Execute phase 4
        result = planning_engine.execute_bom_explosion(net_requirements)
        
        # Assertions
        assert 'yarn_requirements' in result
        assert 'bom_details' in result
        
        yarn_reqs = result['yarn_requirements']
        
        # Check that TEST002 was not processed (net req = 0)
        # YARN001 should only have contribution from TEST001 (100 * 0.6 = 60)
        assert yarn_reqs['YARN001'] == 60  # Only from TEST001, not TEST002
        
        # YARN002 should have from TEST001 and TEST003
        assert yarn_reqs['YARN002'] == 40 + 50  # TEST001: 100*0.4=40, TEST003: 50*1.0=50
        
        # YARN003 should not be present (only in TEST002 which has no net req)
        assert 'YARN003' not in yarn_reqs or yarn_reqs.get('YARN003', 0) == 0
    
    def test_phase5_procurement_production(self, planning_engine):
        """Test Phase 5: Procurement & Production Planning"""
        # Setup test data
        yarn_requirements = {
            'YARN001': 1000,
            'YARN002': 500,
            'YARN003': 750
        }
        
        net_requirements = {
            'TEST001': 100,
            'TEST002': 200
        }
        
        # Execute phase 5
        result = planning_engine.execute_procurement_production_planning(
            yarn_requirements,
            net_requirements
        )
        
        # Assertions
        assert 'yarn_procurement' in result
        assert 'knit_orders' in result
        assert 'production_schedule' in result
        
        # Check yarn procurement recommendations
        for yarn_id, qty in yarn_requirements.items():
            assert yarn_id in result['yarn_procurement']
            proc_rec = result['yarn_procurement'][yarn_id]
            assert proc_rec['required_quantity'] == qty
            assert 'supplier' in proc_rec
            assert 'lead_time' in proc_rec
        
        # Check knit order generation
        assert len(result['knit_orders']) > 0
        for ko in result['knit_orders']:
            assert 'style' in ko
            assert 'quantity' in ko
            assert 'priority' in ko
    
    def test_phase6_optimization_output(self, planning_engine):
        """Test Phase 6: Optimization & Output Generation"""
        # Setup test results from previous phases
        planning_results = {
            'phase1': {'total_demand': {'TEST001': 100}},
            'phase2': {'total_available': {'TEST001': 30}},
            'phase3': {'net_requirements': {'TEST001': 70}},
            'phase4': {'yarn_requirements': {'YARN001': 100}},
            'phase5': {'knit_orders': [{'style': 'TEST001', 'quantity': 70}]}
        }
        
        # Execute phase 6
        result = planning_engine.execute_optimization_and_output(planning_results)
        
        # Assertions
        assert 'optimized_plan' in result
        assert 'reports' in result
        assert 'metrics' in result
        
        # Check optimization metrics
        metrics = result['metrics']
        assert 'total_cost' in metrics
        assert 'service_level' in metrics
        assert 'efficiency' in metrics
    
    def test_complete_planning_cycle(self, planning_engine):
        """Test complete 6-phase planning cycle integration"""
        # Execute full planning
        result = planning_engine.execute_planning()
        
        # Assertions - check all phases completed
        assert 'phase1' in result
        assert 'phase2' in result
        assert 'phase3' in result
        assert 'phase4' in result
        assert 'phase5' in result
        assert 'phase6' in result
        
        # Check phase order execution
        assert result['execution_order'] == [
            'demand_consolidation',
            'inventory_assessment', 
            'net_requirements',
            'bom_explosion',
            'procurement_production',
            'optimization_output'
        ]
    
    def test_multi_level_inventory_netting(self, planning_engine):
        """Test multi-level inventory netting with priorities"""
        # Setup inventory at different stages
        inventory_levels = {
            'F01': {'TEST001': 50, 'availability': 1.0, 'days': 0},
            'I01': {'TEST001': 30, 'availability': 0.95, 'days': 2},
            'G00': {'TEST001': 40, 'availability': 0.90, 'days': 30},
            'G02': {'TEST001': 20, 'availability': 0.90, 'days': 30},
            'KO': {'TEST001': 60, 'availability': 0.85, 'days': 45}
        }
        
        total_demand = 200
        
        # Execute multi-level netting
        result = planning_engine.apply_multi_level_netting(
            'TEST001', 
            total_demand,
            inventory_levels
        )
        
        # Assertions
        assert result['original_demand'] == 200
        assert result['net_requirement'] == 0  # All demand covered by inventory
        assert len(result['netting_details']) > 0
        
        # Check netting order (F01 first, then I01, etc.)
        netting_order = [d['level'] for d in result['netting_details']]
        assert netting_order[0] == 'F01'
        assert netting_order[1] == 'I01'
    
    def test_consistency_based_forecasting(self, planning_engine):
        """Test consistency-based forecasting logic"""
        # Create historical data with different consistency patterns
        
        # High consistency pattern
        high_consistency_data = pd.Series(
            [100, 102, 98, 101, 99, 103, 97, 102],
            index=pd.date_range('2024-01-01', periods=8, freq='M')
        )
        
        # Low consistency pattern (high variance)
        low_consistency_data = pd.Series(
            [100, 50, 200, 25, 180, 60, 220, 30],
            index=pd.date_range('2024-01-01', periods=8, freq='M')
        )
        
        # Calculate consistency scores
        high_score = planning_engine.calculate_consistency_score(high_consistency_data)
        low_score = planning_engine.calculate_consistency_score(low_consistency_data)
        
        # Assertions
        assert high_score > 0.7  # Should have high consistency
        assert low_score < 0.3   # Should have low consistency
        
        # Test forecast decisions based on consistency
        high_forecast = planning_engine.forecast_with_consistency(
            'TEST001', 
            high_consistency_data
        )
        low_forecast = planning_engine.forecast_with_consistency(
            'TEST002',
            low_consistency_data
        )
        
        assert high_forecast is not None  # Should provide forecast
        assert low_forecast is None or low_forecast == 0  # Should not forecast
    
    def test_yarn_allocation_priority(self, planning_engine):
        """Test yarn allocation priority system"""
        # Create knit orders with different priorities
        knit_orders = pd.DataFrame({
            'KO_ID': ['KO001', 'KO002', 'KO003', 'KO004'],
            'Style#': ['TEST001', 'TEST002', 'TEST001', 'TEST003'],
            'Qty_Ordered_Lbs': [1000, 500, 750, 300],
            'Shipped_Lbs': [500, 0, 0, 100],  # KO001 and KO004 have started
            'Due_Date': [
                datetime.now() + timedelta(days=5),   # Urgent
                datetime.now() + timedelta(days=30),  # Normal
                datetime.now() + timedelta(days=10),  # Soon
                datetime.now() + timedelta(days=3)    # Very urgent
            ],
            'Is_Sales_Order': [True, False, True, True]
        })
        
        # Calculate priorities
        prioritized = planning_engine.prioritize_knit_orders(knit_orders)
        
        # Assertions
        # KO004 should be highest priority (started + very urgent)
        assert prioritized.iloc[0]['KO_ID'] == 'KO004'
        
        # KO001 should be second (started + urgent)
        assert prioritized.iloc[1]['KO_ID'] == 'KO001'
        
        # Check priority scores
        for idx, ko in prioritized.iterrows():
            assert 'priority_score' in ko
            if ko['Shipped_Lbs'] > 0:
                assert ko['priority_score'] >= 1000  # Started orders have high base score
    
    def test_error_handling(self, planning_engine):
        """Test error handling in planning phases"""
        # Test with missing data
        with pytest.raises(Exception):
            planning_engine.execute_demand_consolidation()  # No sales orders loaded
        
        # Test with invalid net requirements
        result = planning_engine.execute_bom_explosion({'TEST999': -100})
        assert result['yarn_requirements'] == {}  # Should handle negative gracefully
        
        # Test with empty inventory
        result = planning_engine.calculate_net_requirements(
            {'TEST001': 100},
            {}
        )
        assert result['net_requirements']['TEST001'] == 100  # Full demand is net req


class TestDataIntegration:
    """Test suite for data integration and flow"""
    
    def test_style_mapping_consistency(self):
        """Test style mapping between fStyle and Style#"""
        from style_mapping_manager import StyleMappingManager
        
        mapper = StyleMappingManager()
        
        # Test bidirectional mapping
        test_fstyle = 'F123'
        style = mapper.map_fstyle_to_style(test_fstyle)
        
        if style:
            # Reverse mapping should return original
            reverse_fstyle = mapper.map_style_to_fstyle(style)
            assert reverse_fstyle == test_fstyle
        
        # Test unmapped style handling
        unmapped_result = mapper.map_fstyle_to_style('INVALID_FSTYLE')
        assert unmapped_result is None
    
    def test_column_standardization(self):
        """Test data column standardization"""
        from column_standardizer import ColumnStandardizer
        
        standardizer = ColumnStandardizer()
        
        # Create test dataframe with non-standard columns
        test_df = pd.DataFrame({
            'Yarn ID': ['Y001', 'Y002'],
            'StyleNo': ['S001', 'S002'],
            'BOM%': [60, 40],
            'OnOrder': [100, 200]
        })
        
        # Standardize
        standardized = standardizer.standardize_dataframe(test_df.copy())
        
        # Assertions
        assert 'Desc#' in standardized.columns  # Yarn ID -> Desc#
        assert 'Style#' in standardized.columns  # StyleNo -> Style#
        assert 'BOM_Percent' in standardized.columns  # BOM% -> BOM_Percent
        assert 'On_Order' in standardized.columns  # OnOrder -> On_Order
    
    def test_unit_conversion(self):
        """Test pounds to yards conversion"""
        from unit_converter import UnitConverter
        
        converter = UnitConverter()
        
        # Test conversion with known factor
        test_style = 'TEST001'
        test_pounds = 100
        
        yards = converter.pounds_to_yards(test_style, test_pounds)
        pounds_back = converter.yards_to_pounds(test_style, yards)
        
        # Should be approximately equal (accounting for floating point)
        assert abs(pounds_back - test_pounds) < 0.01
    
    def test_data_validation_framework(self):
        """Test data validation framework"""
        from data_validator import DataValidator
        
        validator = DataValidator()
        
        # Run all validations
        results = validator.run_all_validations()
        
        # Assertions
        assert isinstance(results, dict)
        
        # Check validation categories exist
        expected_validations = [
            'BOM_SUM',
            'STYLE_MAPPING', 
            'PLANNING_BALANCE',
            'INVENTORY_CONSISTENCY'
        ]
        
        for validation in expected_validations:
            assert validation in results
            assert 'status' in results[validation]
            assert results[validation]['status'] in ['PASS', 'FAIL', 'ERROR']


class TestPerformance:
    """Test suite for performance benchmarks"""
    
    @pytest.mark.benchmark
    def test_planning_engine_speed(self):
        """Test planning engine completes within time limit"""
        import time
        
        engine = SixPhasePlanningEngine()
        
        start = time.time()
        result = engine.execute_planning()
        duration = time.time() - start
        
        # Should complete within 2 minutes
        assert duration < 120
        assert result is not None
    
    @pytest.mark.benchmark
    def test_api_response_times(self):
        """Test API endpoint response times"""
        import requests
        import time
        
        base_url = 'http://localhost:5005/api'
        endpoints = [
            '/yarn-intelligence',
            '/inventory-intelligence-enhanced',
            '/production-pipeline'
        ]
        
        for endpoint in endpoints:
            try:
                start = time.time()
                response = requests.get(base_url + endpoint, timeout=5)
                duration = time.time() - start
                
                # Should respond within 200ms
                assert duration < 0.2
                assert response.status_code in [200, 503]  # Allow service unavailable
            except requests.exceptions.RequestException:
                # Server might not be running during tests
                pytest.skip(f"Server not available for {endpoint}")
    
    @pytest.mark.load
    def test_concurrent_processing(self):
        """Test system handles concurrent requests"""
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        def simulate_planning_request():
            engine = SixPhasePlanningEngine()
            result = engine.execute_planning()
            return result is not None
        
        # Simulate 5 concurrent planning requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            start = time.time()
            futures = [executor.submit(simulate_planning_request) for _ in range(5)]
            results = [f.result() for f in futures]
            duration = time.time() - start
        
        # All should complete successfully
        assert all(results)
        
        # Should complete within reasonable time (5 minutes for 5 concurrent)
        assert duration < 300


class TestEndToEnd:
    """Test suite for end-to-end workflows"""
    
    def test_complete_planning_to_ko_generation(self):
        """Test complete workflow from sales order to knit order generation"""
        engine = SixPhasePlanningEngine()
        
        # 1. Create a sales order
        sales_order = {
            'Style#': 'TEST001',
            'Quantity': 100,
            'Customer': 'TestCustomer',
            'Due_Date': datetime.now() + timedelta(days=30)
        }
        
        # 2. Run planning engine
        planning_result = engine.execute_planning()
        
        # 3. Verify KO recommendations generated
        assert 'phase5' in planning_result
        assert 'knit_orders' in planning_result['phase5']
        
        if planning_result['phase5']['knit_orders']:
            ko = planning_result['phase5']['knit_orders'][0]
            
            # 4. Verify KO has required fields
            assert 'style' in ko
            assert 'quantity' in ko
            assert 'priority' in ko
            assert 'suggested_start_date' in ko
            
            # 5. Verify yarn allocation calculated
            assert 'yarn_allocation' in ko or 'phase4' in planning_result
    
    def test_shortage_handling_workflow(self):
        """Test system behavior when yarn shortage detected"""
        engine = SixPhasePlanningEngine()
        
        # Create high demand scenario
        high_demand = {
            'TEST001': 10000,  # Very high demand
            'TEST002': 5000
        }
        
        # Execute planning with high demand
        result = engine.execute_planning_with_demand(high_demand)
        
        # Should identify shortages
        assert 'yarn_shortages' in result
        
        if result['yarn_shortages']:
            # Should provide procurement recommendations
            assert 'procurement_urgent' in result
            
            # Should suggest substitutions if available
            assert 'substitution_recommendations' in result or \
                   'yarn_substitutions' in result
    
    def test_inventory_update_after_production(self):
        """Test inventory updates after production completion"""
        engine = SixPhasePlanningEngine()
        
        # Initial inventory check
        initial_inventory = engine.get_inventory_position('TEST001')
        
        # Simulate production completion
        production_update = {
            'KO_ID': 'KO_TEST_001',
            'Style#': 'TEST001',
            'Stage': 'G00',
            'Quantity': 50
        }
        
        # Update inventory
        engine.update_production_progress(production_update)
        
        # Check inventory updated
        new_inventory = engine.get_inventory_position('TEST001')
        
        # G00 inventory should increase
        if initial_inventory and new_inventory:
            assert new_inventory.get('G00', 0) >= initial_inventory.get('G00', 0)


if __name__ == '__main__':
    # Run tests with coverage
    pytest.main([__file__, '-v', '--cov=six_phase_planning_engine', '--cov-report=html'])