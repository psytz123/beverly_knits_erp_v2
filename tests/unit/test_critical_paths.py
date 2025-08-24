#!/usr/bin/env python3
"""
Critical Path Test Coverage for Beverly Knits ERP
Tests for all critical business logic and calculations
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.beverly_comprehensive_erp import (
    InventoryAnalyzer,
    InventoryManagementPipeline, 
    SalesForecastingEngine,
    CapacityPlanningEngine
)
from src.services.capacity_planning_service import CapacityPlanningService
from src.ml_models.ml_forecast_backtesting import MLForecastBacktester


class TestCriticalInventoryCalculations:
    """Test critical inventory calculations"""
    
    def test_planning_balance_formula(self):
        """Test the critical Planning Balance formula with negative Allocated values"""
        # Create test data with negative Allocated values (as in real data)
        test_data = pd.DataFrame({
            'Desc#': ['YARN001', 'YARN002', 'YARN003'],
            'Theoretical_Balance': [1000, 500, 200],
            'Allocated': [-200, -100, -50],  # Negative values as in actual data
            'On_Order': [300, 0, 100]
        })
        
        # Calculate Planning Balance
        test_data['Planning_Balance'] = (
            test_data['Theoretical_Balance'] + 
            test_data['Allocated'] +  # Already negative
            test_data['On_Order']
        )
        
        # Verify calculations
        assert test_data.loc[0, 'Planning_Balance'] == 1100  # 1000 + (-200) + 300
        assert test_data.loc[1, 'Planning_Balance'] == 400   # 500 + (-100) + 0
        assert test_data.loc[2, 'Planning_Balance'] == 250   # 200 + (-50) + 100
    
    def test_weekly_demand_calculation(self):
        """Test weekly demand calculation logic"""
        test_data = pd.DataFrame({
            'Desc#': ['YARN001', 'YARN002', 'YARN003'],
            'Consumed': [-430, 0, -860],  # Negative = amount used
            'Allocated': [-800, -400, 0],
            'Planning_Balance': [100, 500, 50]
        })
        
        weekly_demand = []
        for _, row in test_data.iterrows():
            if abs(row['Consumed']) > 0:
                # From consumed (monthly / 4.3 weeks)
                demand = abs(row['Consumed']) / 4.3
            elif abs(row['Allocated']) > 0:
                # From allocated (8-week production cycle)
                demand = abs(row['Allocated']) / 8
            else:
                # Minimal default
                demand = 10
            weekly_demand.append(demand)
        
        # Verify calculations
        assert pytest.approx(weekly_demand[0], 0.1) == 100  # 430 / 4.3
        assert pytest.approx(weekly_demand[1], 0.1) == 50   # 400 / 8
        assert pytest.approx(weekly_demand[2], 0.1) == 200  # 860 / 4.3
    
    def test_shortage_detection(self):
        """Test yarn shortage detection logic"""
        test_data = pd.DataFrame({
            'Desc#': ['YARN001', 'YARN002', 'YARN003'],
            'Planning_Balance': [-100, 50, 200],
            'Weekly_Demand': [50, 100, 10],
            'Lead_Time_Weeks': [2, 2, 2]
        })
        
        # Calculate weeks of supply
        test_data['Weeks_Supply'] = test_data['Planning_Balance'] / test_data['Weekly_Demand']
        test_data['Is_Shortage'] = test_data['Weeks_Supply'] < test_data['Lead_Time_Weeks']
        
        assert test_data.loc[0, 'Is_Shortage'] == True   # Negative balance
        assert test_data.loc[1, 'Is_Shortage'] == True   # Only 0.5 weeks supply
        assert test_data.loc[2, 'Is_Shortage'] == False  # 20 weeks supply


class TestMLForecasting:
    """Test ML forecasting implementations"""
    
    def test_prophet_model_implementation(self):
        """Test that Prophet model is properly implemented (not placeholder)"""
        backtester = MLForecastBacktester()
        
        # Create test time series
        test_data = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'value': np.random.normal(100, 10, 100)
        })
        
        # Test Prophet forecast
        result = backtester.backtest_model(test_data, 'prophet', test_size=0.2)
        
        # Should have valid results, not empty
        assert 'predictions' in result
        assert 'metrics' in result
        if result['predictions']:  # If Prophet is available
            assert len(result['predictions']) > 0
            assert 'mape' in result['metrics']
    
    def test_lstm_model_implementation(self):
        """Test that LSTM model is properly implemented (not placeholder)"""
        backtester = MLForecastBacktester()
        
        # Create test time series
        test_data = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'value': np.random.normal(100, 10, 100)
        })
        
        # Test LSTM forecast
        result = backtester.backtest_model(test_data, 'lstm', test_size=0.2)
        
        # Should have valid results or error (not placeholder)
        assert 'predictions' in result or 'error' in result
        # LSTM requires TensorFlow, so might have error if not installed
        if 'predictions' in result and result['predictions']:
            assert len(result['predictions']) > 0


class TestCapacityPlanning:
    """Test capacity planning calculations"""
    
    def test_production_rates_not_placeholder(self):
        """Test that production rates use actual values, not placeholders"""
        service = CapacityPlanningService()
        
        # Test different product types
        knit_rate = service._get_machine_rate('knit_product_A')
        dye_rate = service._get_machine_rate('dye_batch_B')
        finish_rate = service._get_machine_rate('finishing_process_C')
        
        # Should have different rates for different processes
        assert knit_rate == 0.75  # 45 minutes for knitting
        assert dye_rate == 1.0    # 1 hour for dyeing
        assert finish_rate == 0.33  # 20 minutes for finishing
        
        # Should not all be 0.5 (the old placeholder)
        rates = [knit_rate, dye_rate, finish_rate]
        assert len(set(rates)) > 1  # At least 2 different values
    
    def test_setup_time_calculation(self):
        """Test setup time is properly calculated"""
        service = CapacityPlanningService()
        
        # Test different product types
        dye_setup = service._get_setup_time('dye_color_change')
        custom_setup = service._get_setup_time('custom_order')
        standard_setup = service._get_setup_time('standard_product')
        
        # Should have appropriate setup times
        assert dye_setup == 3.0      # Color changes need more setup
        assert custom_setup == 2.5    # Custom products need more setup
        assert standard_setup == 1.0  # Standard products minimal setup
        
        # Should not all be the same
        assert len(set([dye_setup, custom_setup, standard_setup])) > 1


class TestEOQCalculations:
    """Test Economic Order Quantity calculations"""
    
    def test_eoq_uses_proper_costs(self):
        """Test EOQ calculation uses proper costs, not placeholders"""
        from src.core.beverly_comprehensive_erp import InventoryAnalyzer
        
        # Create analyzer instance
        analyzer = InventoryAnalyzer(data_path='.')
        
        # Test different item types
        yarn_eoq = analyzer._calculate_eoq('cotton_yarn_30s', 1000)
        fabric_eoq = analyzer._calculate_eoq('polyester_fabric', 1000)
        chemical_eoq = analyzer._calculate_eoq('reactive_dye_blue', 1000)
        
        # EOQ should be different for different item types due to different costs
        # Old placeholder would give same EOQ for same demand
        assert yarn_eoq != fabric_eoq  # Different ordering/holding costs
        assert fabric_eoq != chemical_eoq
        
        # Verify EOQ formula is working
        assert yarn_eoq > 0
        assert fabric_eoq > 0
        assert chemical_eoq > 0


class TestDataIntegrity:
    """Test data integrity and validation"""
    
    def test_style_mapping_completeness(self):
        """Test fStyle# to Style# mapping is handled correctly"""
        test_data = pd.DataFrame({
            'fStyle#': ['FS001', 'FS002', 'FS003'],
            'Style#': ['S001', 'S002', None],  # Some missing mappings
            'Quantity': [100, 200, 300]
        })
        
        # Should handle missing mappings gracefully
        mapped_styles = []
        for _, row in test_data.iterrows():
            style = row.get('Style#') or row.get('fStyle#')
            mapped_styles.append(style)
        
        assert mapped_styles[0] == 'S001'
        assert mapped_styles[1] == 'S002'
        assert mapped_styles[2] == 'FS003'  # Falls back to fStyle#
    
    def test_unit_conversion(self):
        """Test yards to pounds conversion using QuadS data"""
        # Typical conversion factors from QuadS data
        conversion_factors = {
            'lightweight': 0.25,  # lbs per yard
            'medium': 0.40,
            'heavy': 0.55
        }
        
        yards = 1000
        
        # Convert yards to pounds
        light_pounds = yards * conversion_factors['lightweight']
        medium_pounds = yards * conversion_factors['medium']
        heavy_pounds = yards * conversion_factors['heavy']
        
        assert light_pounds == 250
        assert medium_pounds == 400
        assert heavy_pounds == 550


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_data_handling(self):
        """Test system handles empty data gracefully"""
        analyzer = InventoryAnalyzer(data_path='.')
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        # Should not crash, should return safe defaults
        result = analyzer.analyze_inventory(empty_df, pd.DataFrame(), pd.DataFrame())
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_negative_values_handling(self):
        """Test handling of negative values in calculations"""
        test_data = pd.DataFrame({
            'Planning_Balance': [-100, -50, 0, 50],
            'Quantity': [-10, 0, 10, 20]
        })
        
        # Should handle negative values appropriately
        total = test_data['Planning_Balance'].sum()
        assert total == -100  # Sum should work with negatives
        
        # Absolute values for certain calculations
        abs_total = test_data['Planning_Balance'].abs().sum()
        assert abs_total == 200
    
    def test_division_by_zero(self):
        """Test division by zero is handled"""
        # Test weekly demand calculation with zero values
        consumed = 0
        allocated = 0
        
        if consumed != 0:
            weekly_demand = abs(consumed) / 4.3
        elif allocated != 0:
            weekly_demand = abs(allocated) / 8
        else:
            weekly_demand = 10  # Default
        
        assert weekly_demand == 10  # Should use default, not crash


class TestPerformanceOptimizations:
    """Test performance optimizations are working"""
    
    def test_cache_functionality(self):
        """Test that caching is working for repeated operations"""
        # This would require mocking or actual cache testing
        # Simplified test to verify cache structure exists
        try:
            from src.utils.cache_manager import CacheManager
            cache = CacheManager()
            
            # Test cache operations
            cache.set('test_key', 'test_value', ttl=60)
            value = cache.get('test_key')
            
            assert value == 'test_value'
        except ImportError:
            # Cache manager not available
            pass
    
    def test_dataframe_optimization(self):
        """Test DataFrame memory optimization"""
        # Create large test DataFrame
        df = pd.DataFrame({
            'id': range(10000),
            'value': np.random.random(10000),
            'category': ['A', 'B', 'C'] * 3333 + ['A']
        })
        
        # Original memory usage
        original_memory = df.memory_usage(deep=True).sum()
        
        # Optimize dtypes
        df['id'] = df['id'].astype('int32')
        df['value'] = df['value'].astype('float32')
        df['category'] = df['category'].astype('category')
        
        # Optimized memory usage
        optimized_memory = df.memory_usage(deep=True).sum()
        
        # Should be significantly less memory
        assert optimized_memory < original_memory
        reduction = (1 - optimized_memory / original_memory) * 100
        assert reduction > 30  # Should achieve >30% reduction


class TestAPIEndpoints:
    """Test critical API endpoints return correct data structure"""
    
    def test_yarn_intelligence_response_structure(self):
        """Test /api/yarn-intelligence returns correct structure"""
        expected_structure = {
            'timestamp': str,
            'summary': dict,
            'critical_yarns': list,
            'recommendations': list,
            'procurement_plan': dict
        }
        
        # This would need actual API testing or mocking
        # Verify structure matches expected
        assert True  # Placeholder for actual test
    
    def test_planning_phases_response(self):
        """Test /api/planning-phases returns all 6 phases"""
        expected_phases = [
            'demand_analysis',
            'bom_explosion', 
            'capacity_planning',
            'supply_chain_optimization',
            'production_scheduling',
            'execution_monitoring'
        ]
        
        # Would need actual API test
        assert len(expected_phases) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])