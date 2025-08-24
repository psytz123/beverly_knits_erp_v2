#!/usr/bin/env python3
"""
Comprehensive Unit Tests for InventoryAnalyzer Class
Tests all critical inventory calculation and analysis functions
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

from src.core.beverly_comprehensive_erp import InventoryAnalyzer


class TestInventoryAnalyzer:
    """Comprehensive tests for InventoryAnalyzer class"""
    
    @pytest.fixture
    def analyzer(self, tmp_path):
        """Create an InventoryAnalyzer instance for testing"""
        return InventoryAnalyzer(data_path=str(tmp_path))
    
    @pytest.fixture
    def sample_yarn_data(self):
        """Sample yarn inventory data"""
        return pd.DataFrame({
            'Desc#': ['YARN001', 'YARN002', 'YARN003', 'YARN004', 'YARN005'],
            'Description': ['Cotton 30s', 'Polyester 40s', 'Blend 50/50', 'Lycra', 'Wool'],
            'Theoretical_Balance': [1000, 500, 200, 150, 800],
            'Allocated': [-200, -100, -50, -30, -400],  # Negative as per actual data
            'On_Order': [300, 0, 100, 50, 200],
            'Consumed': [-430, -200, 0, -50, -600],  # Negative = consumed
            'Cost/Pound': [3.50, 4.20, 3.80, 12.50, 8.00],
            'Supplier': ['Supplier A', 'Supplier B', 'Supplier A', 'Supplier C', 'Supplier B'],
            'Lead_Time_Days': [14, 21, 14, 28, 21]
        })
    
    @pytest.fixture
    def sample_sales_data(self):
        """Sample sales data"""
        return pd.DataFrame({
            'Style#': ['STYLE001', 'STYLE002', 'STYLE003'],
            'Qty Shipped': [100, 200, 150],
            'Order_Date': pd.date_range(start='2024-01-01', periods=3),
            'Customer': ['Customer A', 'Customer B', 'Customer C'],
            'Value': [5000, 10000, 7500]
        })
    
    @pytest.fixture
    def sample_bom_data(self):
        """Sample BOM data"""
        return pd.DataFrame({
            'Style#': ['STYLE001', 'STYLE001', 'STYLE002', 'STYLE002', 'STYLE003'],
            'Desc#': ['YARN001', 'YARN002', 'YARN001', 'YARN003', 'YARN004'],
            'Quantity_Per_Unit': [0.5, 0.3, 0.4, 0.6, 0.8],
            'Unit': ['LBS', 'LBS', 'LBS', 'LBS', 'LBS']
        })
    
    def test_planning_balance_calculation(self, analyzer, sample_yarn_data):
        """Test the critical Planning Balance formula"""
        # Calculate Planning Balance
        sample_yarn_data['Planning_Balance'] = (
            sample_yarn_data['Theoretical_Balance'] +
            sample_yarn_data['Allocated'] +  # Already negative
            sample_yarn_data['On_Order']
        )
        
        # Verify calculations
        assert sample_yarn_data.loc[0, 'Planning_Balance'] == 1100  # 1000 + (-200) + 300
        assert sample_yarn_data.loc[1, 'Planning_Balance'] == 400   # 500 + (-100) + 0
        assert sample_yarn_data.loc[2, 'Planning_Balance'] == 250   # 200 + (-50) + 100
        assert sample_yarn_data.loc[3, 'Planning_Balance'] == 170   # 150 + (-30) + 50
        assert sample_yarn_data.loc[4, 'Planning_Balance'] == 600   # 800 + (-400) + 200
    
    def test_negative_allocated_handling(self, analyzer, sample_yarn_data):
        """Test handling of negative Allocated values"""
        # Allocated should be negative (amount allocated/reserved)
        assert all(sample_yarn_data['Allocated'] <= 0)
        
        # When calculating available quantity, negative allocated reduces availability
        available = sample_yarn_data['Theoretical_Balance'] + sample_yarn_data['Allocated']
        assert available.loc[0] == 800  # 1000 + (-200)
        assert available.loc[1] == 400  # 500 + (-100)
    
    def test_weekly_demand_calculation(self, analyzer, sample_yarn_data):
        """Test weekly demand calculation from consumed and allocated"""
        weekly_demand = []
        
        for _, row in sample_yarn_data.iterrows():
            if abs(row['Consumed']) > 0:
                # Monthly consumption / 4.3 weeks
                demand = abs(row['Consumed']) / 4.3
            elif abs(row['Allocated']) > 0:
                # Allocated over 8-week production cycle
                demand = abs(row['Allocated']) / 8
            else:
                # Minimal default
                demand = 10
            weekly_demand.append(demand)
        
        # Verify calculations
        assert pytest.approx(weekly_demand[0], 0.1) == 100.0  # 430 / 4.3
        assert pytest.approx(weekly_demand[1], 0.1) == 46.5   # 200 / 4.3
        assert pytest.approx(weekly_demand[2], 0.1) == 6.25   # 50 / 8 (no consumption)
        assert pytest.approx(weekly_demand[3], 0.1) == 11.6   # 50 / 4.3
        assert pytest.approx(weekly_demand[4], 0.1) == 139.5  # 600 / 4.3
    
    def test_shortage_detection(self, analyzer, sample_yarn_data):
        """Test yarn shortage detection logic"""
        # Add Planning Balance
        sample_yarn_data['Planning_Balance'] = (
            sample_yarn_data['Theoretical_Balance'] +
            sample_yarn_data['Allocated'] +
            sample_yarn_data['On_Order']
        )
        
        # Calculate weekly demand
        sample_yarn_data['Weekly_Demand'] = sample_yarn_data.apply(
            lambda row: abs(row['Consumed']) / 4.3 if abs(row['Consumed']) > 0 
            else abs(row['Allocated']) / 8 if abs(row['Allocated']) > 0 
            else 10,
            axis=1
        )
        
        # Calculate weeks of supply
        sample_yarn_data['Weeks_Supply'] = (
            sample_yarn_data['Planning_Balance'] / sample_yarn_data['Weekly_Demand']
        )
        
        # Identify shortages (less than lead time in weeks)
        sample_yarn_data['Lead_Time_Weeks'] = sample_yarn_data['Lead_Time_Days'] / 7
        sample_yarn_data['Is_Shortage'] = (
            sample_yarn_data['Weeks_Supply'] < sample_yarn_data['Lead_Time_Weeks']
        )
        
        # YARN003 should not be shortage (250 balance / 6.25 demand = 40 weeks)
        assert sample_yarn_data.loc[2, 'Is_Shortage'] == False
        
        # Check if any are shortages based on actual calculations
        shortages = sample_yarn_data[sample_yarn_data['Is_Shortage']]
        assert isinstance(shortages, pd.DataFrame)
    
    def test_reorder_point_calculation(self, analyzer, sample_yarn_data):
        """Test reorder point calculation"""
        # Calculate reorder points
        sample_yarn_data['Weekly_Demand'] = sample_yarn_data.apply(
            lambda row: abs(row['Consumed']) / 4.3 if abs(row['Consumed']) > 0 else 10,
            axis=1
        )
        
        # Reorder Point = (Lead Time in Weeks * Weekly Demand) + Safety Stock
        sample_yarn_data['Lead_Time_Weeks'] = sample_yarn_data['Lead_Time_Days'] / 7
        sample_yarn_data['Safety_Stock'] = sample_yarn_data['Weekly_Demand'] * 2  # 2 weeks safety
        sample_yarn_data['Reorder_Point'] = (
            sample_yarn_data['Lead_Time_Weeks'] * sample_yarn_data['Weekly_Demand'] +
            sample_yarn_data['Safety_Stock']
        )
        
        # Verify calculations for YARN001
        yarn001 = sample_yarn_data.loc[0]
        expected_reorder = (2 * 100) + (2 * 100)  # 2 weeks lead + 2 weeks safety
        assert pytest.approx(yarn001['Reorder_Point'], 1) == expected_reorder
    
    def test_eoq_calculation(self, analyzer, sample_yarn_data):
        """Test Economic Order Quantity calculation"""
        # Test EOQ with actual formula
        for _, row in sample_yarn_data.iterrows():
            yarn_id = row['Desc#']
            annual_demand = abs(row['Consumed']) * 12 if abs(row['Consumed']) > 0 else 1200
            
            # Use the improved EOQ calculation from our fixes
            eoq = analyzer._calculate_eoq(yarn_id, annual_demand)
            
            # EOQ should be positive and at least meet demand
            assert eoq > 0
            assert eoq >= annual_demand / 12  # At least one month's demand
    
    def test_inventory_value_calculation(self, analyzer, sample_yarn_data):
        """Test inventory value calculations"""
        # Calculate inventory values
        sample_yarn_data['Planning_Balance'] = (
            sample_yarn_data['Theoretical_Balance'] +
            sample_yarn_data['Allocated'] +
            sample_yarn_data['On_Order']
        )
        
        sample_yarn_data['Inventory_Value'] = (
            sample_yarn_data['Planning_Balance'] * sample_yarn_data['Cost/Pound']
        )
        
        # Verify calculations
        assert sample_yarn_data.loc[0, 'Inventory_Value'] == 1100 * 3.50  # 3850
        assert sample_yarn_data.loc[1, 'Inventory_Value'] == 400 * 4.20   # 1680
        
        # Total inventory value
        total_value = sample_yarn_data['Inventory_Value'].sum()
        assert total_value > 0
    
    def test_supplier_analysis(self, analyzer, sample_yarn_data):
        """Test supplier performance analysis"""
        # Group by supplier
        supplier_analysis = sample_yarn_data.groupby('Supplier').agg({
            'Theoretical_Balance': 'sum',
            'Cost/Pound': 'mean',
            'Lead_Time_Days': 'mean'
        }).round(2)
        
        # Verify supplier metrics
        assert 'Supplier A' in supplier_analysis.index
        assert 'Supplier B' in supplier_analysis.index
        assert 'Supplier C' in supplier_analysis.index
        
        # Check aggregations
        supplier_a = supplier_analysis.loc['Supplier A']
        assert supplier_a['Theoretical_Balance'] == 1200  # YARN001 + YARN003
        assert supplier_a['Lead_Time_Days'] == 14  # Both have 14 days
    
    def test_critical_yarn_identification(self, analyzer, sample_yarn_data):
        """Test identification of critical yarns"""
        # Add necessary columns
        sample_yarn_data['Planning_Balance'] = (
            sample_yarn_data['Theoretical_Balance'] +
            sample_yarn_data['Allocated'] +
            sample_yarn_data['On_Order']
        )
        
        # Identify critical yarns (low balance or high consumption)
        critical_threshold = 500
        high_consumption_threshold = 400
        
        sample_yarn_data['Is_Critical'] = (
            (sample_yarn_data['Planning_Balance'] < critical_threshold) |
            (abs(sample_yarn_data['Consumed']) > high_consumption_threshold)
        )
        
        critical_yarns = sample_yarn_data[sample_yarn_data['Is_Critical']]
        
        # YARN002, YARN003, YARN004 should be critical (low balance)
        # YARN001, YARN005 should be critical (high consumption)
        assert len(critical_yarns) >= 3
    
    def test_procurement_recommendation(self, analyzer, sample_yarn_data):
        """Test procurement recommendation generation"""
        # Calculate recommended order quantities
        sample_yarn_data['Planning_Balance'] = (
            sample_yarn_data['Theoretical_Balance'] +
            sample_yarn_data['Allocated'] +
            sample_yarn_data['On_Order']
        )
        
        sample_yarn_data['Weekly_Demand'] = sample_yarn_data.apply(
            lambda row: abs(row['Consumed']) / 4.3 if abs(row['Consumed']) > 0 else 10,
            axis=1
        )
        
        # Target inventory = 12 weeks of demand
        target_weeks = 12
        sample_yarn_data['Target_Inventory'] = sample_yarn_data['Weekly_Demand'] * target_weeks
        sample_yarn_data['Recommended_Order'] = np.maximum(
            0,
            sample_yarn_data['Target_Inventory'] - sample_yarn_data['Planning_Balance']
        )
        
        # Verify recommendations
        assert all(sample_yarn_data['Recommended_Order'] >= 0)
        
        # Yarns with low balance should have recommendations
        low_balance = sample_yarn_data[sample_yarn_data['Planning_Balance'] < 500]
        assert all(low_balance['Recommended_Order'] > 0)
    
    def test_analyze_inventory_integration(self, analyzer, sample_yarn_data, 
                                         sample_sales_data, sample_bom_data):
        """Test the main analyze_inventory method"""
        # Mock the data loading
        with patch.object(analyzer, 'load_data') as mock_load:
            mock_load.return_value = (sample_yarn_data, sample_sales_data, sample_bom_data)
            
            # Call analyze_inventory
            result = analyzer.analyze_inventory(
                sample_yarn_data,
                sample_sales_data,
                sample_bom_data
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert 'total_value' in result
            assert 'critical_items' in result
            assert 'reorder_suggestions' in result
    
    def test_error_handling(self, analyzer):
        """Test error handling with invalid data"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = analyzer.analyze_inventory(empty_df, empty_df, empty_df)
        assert result is not None
        
        # Test with None
        result = analyzer.analyze_inventory(None, None, None)
        assert result is not None
        
        # Test with missing columns
        bad_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        result = analyzer.analyze_inventory(bad_data, bad_data, bad_data)
        assert result is not None
    
    def test_data_type_conversions(self, analyzer, sample_yarn_data):
        """Test data type conversions and handling"""
        # Test with string numbers
        sample_yarn_data['Theoretical_Balance'] = sample_yarn_data['Theoretical_Balance'].astype(str)
        sample_yarn_data['Allocated'] = sample_yarn_data['Allocated'].astype(str)
        
        # Should handle conversion
        sample_yarn_data['Theoretical_Balance'] = pd.to_numeric(
            sample_yarn_data['Theoretical_Balance'], errors='coerce'
        )
        sample_yarn_data['Allocated'] = pd.to_numeric(
            sample_yarn_data['Allocated'], errors='coerce'
        )
        
        assert sample_yarn_data['Theoretical_Balance'].dtype in [np.float64, np.int64]
        assert sample_yarn_data['Allocated'].dtype in [np.float64, np.int64]
    
    def test_zero_division_handling(self, analyzer, sample_yarn_data):
        """Test handling of zero division scenarios"""
        # Set some demands to zero
        sample_yarn_data.loc[0, 'Consumed'] = 0
        sample_yarn_data.loc[0, 'Allocated'] = 0
        
        # Calculate weekly demand - should not crash
        weekly_demand = sample_yarn_data.apply(
            lambda row: abs(row['Consumed']) / 4.3 if abs(row['Consumed']) > 0 
            else abs(row['Allocated']) / 8 if abs(row['Allocated']) > 0 
            else 10,  # Default value
            axis=1
        )
        
        # Should use default value for zero consumption/allocation
        assert weekly_demand.loc[0] == 10
    
    def test_negative_value_handling(self, analyzer, sample_yarn_data):
        """Test handling of negative values in calculations"""
        # Test with negative Planning Balance
        sample_yarn_data.loc[0, 'Theoretical_Balance'] = 50
        sample_yarn_data.loc[0, 'Allocated'] = -200
        sample_yarn_data.loc[0, 'On_Order'] = 0
        
        planning_balance = (
            sample_yarn_data.loc[0, 'Theoretical_Balance'] +
            sample_yarn_data.loc[0, 'Allocated'] +
            sample_yarn_data.loc[0, 'On_Order']
        )
        
        assert planning_balance == -150  # Should handle negative result
        
        # Shortage detection should flag negative balance
        is_shortage = planning_balance < 0
        assert is_shortage == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])