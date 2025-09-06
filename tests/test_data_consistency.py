#!/usr/bin/env python3
"""
Comprehensive tests for data consistency across the ERP system
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_consistency.consistency_manager import DataConsistencyManager
from data_consistency.validation_rules import DataValidationRules


class TestDataConsistencyManager:
    """Test the centralized data consistency manager"""
    
    def test_column_mappings_available(self):
        """Test that column mappings are defined"""
        assert hasattr(DataConsistencyManager, 'COLUMN_MAPPINGS')
        assert 'yarn_id' in DataConsistencyManager.COLUMN_MAPPINGS
        assert 'planning_balance' in DataConsistencyManager.COLUMN_MAPPINGS
        assert 'style_id' in DataConsistencyManager.COLUMN_MAPPINGS
    
    def test_get_column_name(self):
        """Test getting column names from dataframe"""
        # Create test dataframe with different column name variations
        df = pd.DataFrame({
            'Desc#': [1, 2, 3],
            'Planning Balance': [100, -50, 200],
            'Style#': ['S1', 'S2', 'S3']
        })
        
        # Test yarn_id column detection
        yarn_col = DataConsistencyManager.get_column_name(df, 'yarn_id')
        assert yarn_col == 'Desc#'
        
        # Test planning_balance column detection
        balance_col = DataConsistencyManager.get_column_name(df, 'planning_balance')
        assert balance_col == 'Planning Balance'
        
        # Test non-existent column
        missing_col = DataConsistencyManager.get_column_name(df, 'nonexistent')
        assert missing_col is None
    
    def test_standardize_columns(self):
        """Test column standardization"""
        # Create dataframe with various column names
        df = pd.DataFrame({
            'Desc#': [1, 2, 3],
            'Planning Balance': [100, -50, 200],
            'Style#': ['S1', 'S2', 'S3'],
            'On Order': [0, 10, 0]
        })
        
        standardized = DataConsistencyManager.standardize_columns(df)
        
        # Original dataframe should be unchanged
        assert 'Desc#' in df.columns
        
        # Standardized should have consistent names
        assert 'yarn_id' in standardized.columns or 'Desc#' in standardized.columns
        assert len(standardized) == len(df)
    
    def test_calculate_yarn_shortage_with_shortage(self):
        """Test yarn shortage calculation for yarn with shortage"""
        yarn_row = pd.Series({
            'Desc#': '12345',
            'Description': 'Test Yarn',
            'Planning Balance': -150,
            'Theoretical Balance': -100,
            'Allocated': -50,
            'On Order': 0
        })
        
        shortage = DataConsistencyManager.calculate_yarn_shortage(yarn_row)
        
        assert shortage['yarn_id'] == '12345'
        assert shortage['description'] == 'Test Yarn'
        assert shortage['has_shortage'] == True
        assert shortage['shortage_amount'] == 150  # Most severe shortage
        assert shortage['risk_level'] in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        assert 0 <= shortage['urgency_score'] <= 100
    
    def test_calculate_yarn_shortage_no_shortage(self):
        """Test yarn shortage calculation for yarn without shortage"""
        yarn_row = pd.Series({
            'Desc#': '67890',
            'Description': 'Good Yarn',
            'Planning Balance': 500,
            'Theoretical Balance': 600,
            'Allocated': 0,
            'On Order': 100
        })
        
        shortage = DataConsistencyManager.calculate_yarn_shortage(yarn_row)
        
        assert shortage['yarn_id'] == '67890'
        assert shortage['has_shortage'] == False
        assert shortage['shortage_amount'] == 0
        assert shortage['risk_level'] == 'NONE'
    
    def test_aggregate_yarn_requirements(self):
        """Test yarn requirement aggregation from BOM and production data"""
        # Create test BOM data
        bom_df = pd.DataFrame({
            'Style#': ['S1', 'S1', 'S2', 'S2'],
            'Desc#': ['Y1', 'Y2', 'Y1', 'Y3'],
            'BOM_Percentage': [0.6, 0.4, 0.8, 0.2]
        })
        
        # Create test production data
        production_df = pd.DataFrame({
            'Style #': ['S1', 'S2'],
            'Order #': ['O1', 'O2'],
            'Qty Ordered (lbs)': ['1,000', '500']
        })
        
        requirements = DataConsistencyManager.aggregate_yarn_requirements(bom_df, production_df)
        
        # Check that requirements are calculated correctly
        # S1: 1000 lbs * (0.6 for Y1 + 0.4 for Y2) = 600 Y1 + 400 Y2
        # S2: 500 lbs * (0.8 for Y1 + 0.2 for Y3) = 400 Y1 + 100 Y3
        # Total: Y1 = 1000, Y2 = 400, Y3 = 100
        
        assert 'Y1' in requirements
        assert 'Y2' in requirements
        assert 'Y3' in requirements
        
        assert requirements['Y1'] == 1000  # 600 + 400
        assert requirements['Y2'] == 400
        assert requirements['Y3'] == 100
    
    def test_validate_data_consistency(self):
        """Test comprehensive data validation"""
        # Create test data
        inventory_df = pd.DataFrame({
            'Desc#': ['Y1', 'Y2', 'Y3'],
            'Description': ['Yarn 1', 'Yarn 2', 'Yarn 3'],
            'Planning Balance': [100, -50, 300],
            'Theoretical Balance': [120, -30, 320],
            'Allocated': [0, -20, 0],
            'On Order': [0, 0, 0]
        })
        
        bom_df = pd.DataFrame({
            'Style#': ['S1', 'S1'],
            'Desc#': ['Y1', 'Y2'],
            'BOM_Percentage': [0.7, 0.3]
        })
        
        production_df = pd.DataFrame({
            'Style #': ['S1'],
            'Order #': ['O1'],
            'Qty Ordered (lbs)': ['100']
        })
        
        validation = DataConsistencyManager.validate_data_consistency(
            inventory_df, bom_df, production_df
        )
        
        assert 'is_consistent' in validation
        assert 'discrepancies' in validation
        assert 'statistics' in validation
        assert validation['statistics']['total_shortages'] >= 1  # Y2 has shortage
    
    def test_create_reconciliation_report(self):
        """Test comprehensive reconciliation report generation"""
        # Create test data with known issues
        inventory_df = pd.DataFrame({
            'Desc#': ['Y1', 'Y2'],
            'Description': ['Yarn 1', 'Yarn 2'],
            'Planning Balance': [100, -200],
            'Theoretical Balance': [100, -180],
            'Allocated': [0, -20],
            'On Order': [0, 0]
        })
        
        bom_df = pd.DataFrame({
            'Style#': ['S1'],
            'Desc#': ['Y1'],
            'BOM_Percentage': [1.0]
        })
        
        production_df = pd.DataFrame({
            'Style #': ['S1'],
            'Order #': ['O1'],
            'Qty Ordered (lbs)': ['100']
        })
        
        report = DataConsistencyManager.create_reconciliation_report(
            inventory_df, bom_df, production_df
        )
        
        assert 'timestamp' in report
        assert 'validation' in report
        assert 'shortages' in report
        assert 'requirements' in report
        assert 'recommendations' in report
        
        # Should identify Y2 as having shortage
        assert report['shortages']['summary']['total_yarns_short'] >= 1
        
        # Should have recommendations
        assert len(report['recommendations']) > 0


class TestDataValidationRules:
    """Test data validation rules"""
    
    def test_validate_yarn_inventory_valid(self):
        """Test validation of valid yarn inventory data"""
        df = pd.DataFrame({
            'Desc#': [1, 2, 3],
            'Description': ['Yarn 1', 'Yarn 2', 'Yarn 3'],
            'Planning Balance': [100, 200, -50],
            'Allocated': [0, -10, -25],
            'On Order': [0, 0, 50],
            'Theoretical Balance': [100, 190, -75]
        })
        
        result = DataValidationRules.validate_yarn_inventory(df)
        
        assert result['is_valid'] == True
        assert result['row_count'] == 3
        assert len(result['errors']) == 0
    
    def test_validate_yarn_inventory_missing_columns(self):
        """Test validation with missing required columns"""
        df = pd.DataFrame({
            'Desc#': [1, 2, 3],
            'Description': ['Yarn 1', 'Yarn 2', 'Yarn 3']
            # Missing Planning Balance, Allocated
        })
        
        result = DataValidationRules.validate_yarn_inventory(df)
        
        assert result['is_valid'] == False
        assert len(result['errors']) > 0
        assert any(error['type'] == 'MISSING_COLUMNS' for error in result['errors'])
    
    def test_validate_yarn_inventory_duplicates(self):
        """Test validation with duplicate yarn IDs"""
        df = pd.DataFrame({
            'Desc#': [1, 1, 2],  # Duplicate yarn ID
            'Planning Balance': [100, 200, -50],
            'Allocated': [0, -10, -25]
        })
        
        result = DataValidationRules.validate_yarn_inventory(df)
        
        assert len(result['warnings']) > 0
        assert any(warning['type'] == 'DUPLICATE_YARNS' for warning in result['warnings'])
    
    def test_validate_bom_valid(self):
        """Test validation of valid BOM data"""
        df = pd.DataFrame({
            'Style#': ['S1', 'S1', 'S2'],
            'Desc#': ['Y1', 'Y2', 'Y1'],
            'BOM_Percentage': [0.7, 0.3, 1.0]
        })
        
        result = DataValidationRules.validate_bom(df)
        
        assert result['is_valid'] == True
        assert result['row_count'] == 3
        assert result['statistics']['unique_styles'] == 2
    
    def test_validate_bom_invalid_totals(self):
        """Test BOM validation with invalid percentage totals"""
        df = pd.DataFrame({
            'Style#': ['S1', 'S1'],
            'Desc#': ['Y1', 'Y2'],
            'BOM_Percentage': [0.4, 0.4]  # Only sums to 0.8, should be 1.0
        })
        
        result = DataValidationRules.validate_bom(df)
        
        assert len(result['warnings']) > 0
        assert any(warning['type'] == 'INVALID_BOM_TOTAL' for warning in result['warnings'])
    
    def test_validate_production_orders_valid(self):
        """Test validation of valid production orders"""
        df = pd.DataFrame({
            'Style #': ['S1', 'S2'],
            'Order #': ['O1', 'O2'],
            'Qty Ordered (lbs)': ['1,000', '500'],
            'Shipped (lbs)': ['200', '0'],
            'Balance (lbs)': ['800', '500']
        })
        
        result = DataValidationRules.validate_production_orders(df)
        
        assert result['is_valid'] == True
        assert result['row_count'] == 2
    
    def test_validate_production_orders_duplicates(self):
        """Test validation with duplicate order numbers"""
        df = pd.DataFrame({
            'Style #': ['S1', 'S2'],
            'Order #': ['O1', 'O1'],  # Duplicate order number
            'Qty Ordered (lbs)': ['1,000', '500']
        })
        
        result = DataValidationRules.validate_production_orders(df)
        
        assert result['is_valid'] == False
        assert len(result['errors']) > 0
        assert any(error['type'] == 'DUPLICATE_ORDERS' for error in result['errors'])
    
    def test_cross_validate_data_consistent(self):
        """Test cross-validation of consistent data"""
        inventory_df = pd.DataFrame({
            'Desc#': ['Y1', 'Y2'],
            'Description': ['Yarn 1', 'Yarn 2'],
            'Planning Balance': [100, 200]
        })
        
        bom_df = pd.DataFrame({
            'Style#': ['S1'],
            'Desc#': ['Y1'],
            'BOM_Percentage': [1.0]
        })
        
        production_df = pd.DataFrame({
            'Style #': ['S1'],
            'Order #': ['O1'],
            'Qty Ordered (lbs)': ['100']
        })
        
        result = DataValidationRules.cross_validate_data(inventory_df, bom_df, production_df)
        
        assert result['is_valid'] == True
        assert len(result['missing_references']) == 0
    
    def test_cross_validate_missing_bom(self):
        """Test cross-validation with missing BOM entries"""
        inventory_df = pd.DataFrame({
            'Desc#': ['Y1', 'Y2'],
            'Planning Balance': [100, 200]
        })
        
        bom_df = pd.DataFrame({
            'Style#': ['S1'],
            'Desc#': ['Y1'],
            'BOM_Percentage': [1.0]
        })
        
        production_df = pd.DataFrame({
            'Style #': ['S1', 'S2'],  # S2 has no BOM
            'Order #': ['O1', 'O2'],
            'Qty Ordered (lbs)': ['100', '50']
        })
        
        result = DataValidationRules.cross_validate_data(inventory_df, bom_df, production_df)
        
        assert result['is_valid'] == False
        assert len(result['missing_references']) > 0
        assert any(ref['type'] == 'PRODUCTION_WITHOUT_BOM' for ref in result['missing_references'])


class TestIntegration:
    """Integration tests for data consistency across different scenarios"""
    
    def test_real_world_scenario_consistent(self):
        """Test a realistic scenario where data should be consistent"""
        # Create realistic test data
        inventory_df = pd.DataFrame({
            'Desc#': [13565, 13566, 13567],
            'Description': ['Cotton 40s', 'Lycra 70s', 'Polyester 30s'],
            'Planning Balance': [500, -200, 1000],
            'Theoretical Balance': [500, -180, 1000],
            'Allocated': [0, -20, 0],
            'On Order': [0, 300, 0],
            'Beginning Balance': [600, 100, 1200],
            'Consumed': [100, 300, 200]
        })
        
        bom_df = pd.DataFrame({
            'Style#': ['FF 10006/0001', 'FF 10006/0001', 'CT3085/1DU'],
            'Desc#': [13565, 13566, 13567],
            'BOM_Percentage': [0.7, 0.3, 1.0],
            'unit': ['lbs', 'lbs', 'lbs']
        })
        
        production_df = pd.DataFrame({
            'Style #': ['FF 10006/0001', 'CT3085/1DU'],
            'Order #': ['K2508091', 'K2508090'],
            'Qty Ordered (lbs)': ['1,200', '4,000'],
            'Balance (lbs)': ['1,200', '4,000'],
            'Machine': [83, None]
        })
        
        # Test consistency manager
        report = DataConsistencyManager.create_reconciliation_report(
            inventory_df, bom_df, production_df
        )
        
        # Should identify shortage for yarn 13566
        shortage_yarns = [s['yarn_id'] for s in report['shortages']['top_shortages']]
        assert '13566' in shortage_yarns or 13566 in shortage_yarns
        
        # Should calculate requirements correctly
        # FF 10006/0001: 1200 * (0.7 for 13565 + 0.3 for 13566) = 840 + 360
        # CT3085/1DU: 4000 * 1.0 for 13567 = 4000
        requirements = report['requirements']['top_requirements']
        
        # Find requirements for each yarn
        req_dict = dict(requirements)
        assert req_dict.get('13565', req_dict.get(13565, 0)) == 840
        assert req_dict.get('13566', req_dict.get(13566, 0)) == 360
        assert req_dict.get('13567', req_dict.get(13567, 0)) == 4000
        
        # Should have recommendations
        assert len(report['recommendations']) > 0
    
    def test_data_consistency_across_modules(self):
        """Test that different modules would calculate the same shortage"""
        yarn_row = pd.Series({
            'Desc#': 13565,
            'Description': 'Test Yarn',
            'Planning Balance': -150,
            'Theoretical Balance': -120,
            'Allocated': -30,
            'On Order': 100
        })
        
        # Method 1: Using consistency manager
        consistent_result = DataConsistencyManager.calculate_yarn_shortage(yarn_row)
        
        # Method 2: Manual calculation (simulating legacy logic)
        planning_balance = yarn_row.get('Planning Balance', 0)
        theoretical_balance = yarn_row.get('Theoretical Balance', 0)
        
        has_shortage_legacy = planning_balance < 0 or theoretical_balance < 0
        shortage_amount_legacy = abs(min(planning_balance, theoretical_balance))
        
        # Both methods should agree on basic shortage detection
        assert consistent_result['has_shortage'] == has_shortage_legacy
        assert consistent_result['shortage_amount'] == shortage_amount_legacy
        
        # Consistent method should provide more detailed analysis
        assert consistent_result['risk_level'] in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        assert 0 <= consistent_result['urgency_score'] <= 100


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])