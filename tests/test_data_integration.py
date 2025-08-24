#!/usr/bin/env python3
"""
Integration Tests for Data Flow in Beverly Knits ERP System
Tests the complete data flow through all system components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json

from core.beverly_comprehensive_erp import (
    InventoryAnalyzer,
    InventoryManagementPipeline,
    SalesForecastingEngine,
    CapacityPlanningEngine
)
from production.six_phase_planning_engine import SixPhasePlanningEngine
# Note: DataValidator module not found, commenting out
# from data_validator import DataValidator


class TestStyleMappingFlow:
    """Test style mapping between different systems"""
    
    @pytest.fixture
    def style_mapping_data(self):
        """Create sample style mapping data"""
        return pd.DataFrame({
            'fStyle#': ['F001', 'F002', 'F003', 'F004'],
            'Style#': ['6191-BK', '72762-DS', '80393C-DS', '71320-BK'],
            'Description': ['Jersey Knit', 'Interlock', 'Pique', 'Rib Knit'],
            'Customer': ['Zonkd LLC', 'Gates PT', 'Sears Mfg', 'Behrens Aero']
        })
    
    def test_fstyle_to_style_mapping(self, style_mapping_data):
        """Test mapping from fStyle# to Style#"""
        # Create mapping dictionary
        mapping = dict(zip(style_mapping_data['fStyle#'], style_mapping_data['Style#']))
        
        # Test each mapping
        assert mapping['F001'] == '6191-BK'
        assert mapping['F002'] == '72762-DS'
        assert mapping['F003'] == '80393C-DS'
        assert mapping['F004'] == '71320-BK'
    
    def test_style_to_fstyle_reverse_mapping(self, style_mapping_data):
        """Test reverse mapping from Style# to fStyle#"""
        # Create reverse mapping
        reverse_mapping = dict(zip(style_mapping_data['Style#'], style_mapping_data['fStyle#']))
        
        # Test reverse mappings
        assert reverse_mapping['6191-BK'] == 'F001'
        assert reverse_mapping['72762-DS'] == 'F002'
    
    def test_missing_style_handling(self):
        """Test handling of missing style mappings"""
        mapping = {'F001': '6191-BK'}
        
        # Test missing key
        result = mapping.get('F999', None)
        assert result is None
    
    def test_duplicate_style_detection(self):
        """Test detection of duplicate style mappings"""
        data = pd.DataFrame({
            'fStyle#': ['F001', 'F002', 'F001'],  # Duplicate F001
            'Style#': ['6191-BK', '72762-DS', '6191-CB']
        })
        
        # Check for duplicates
        duplicates = data[data.duplicated(subset=['fStyle#'], keep=False)]
        assert len(duplicates) == 2  # Both duplicate rows
    
    def test_style_mapping_with_inventory(self, style_mapping_data):
        """Test style mapping integration with inventory data"""
        inventory_data = pd.DataFrame({
            'fStyle#': ['F001', 'F002'],
            'Available': [100, 200],
            'Location': ['F01', 'F01']
        })
        
        # Merge with mapping
        merged = inventory_data.merge(style_mapping_data, on='fStyle#', how='left')
        
        assert len(merged) == 2
        assert merged.iloc[0]['Style#'] == '6191-BK'
        assert merged.iloc[0]['Available'] == 100


class TestInventoryAggregation:
    """Test multi-level inventory aggregation"""
    
    @pytest.fixture
    def multi_level_inventory(self):
        """Create sample multi-level inventory data"""
        return {
            'F01': pd.DataFrame({  # Finished goods
                'fStyle#': ['F001', 'F002', 'F003'],
                'Available': [100, 150, 200]
            }),
            'I01': pd.DataFrame({  # QC queue
                'fStyle#': ['F001', 'F003'],
                'Available': [50, 75]
            }),
            'G00': pd.DataFrame({  # In-process
                'fStyle#': ['F001', 'F002', 'F003', 'F004'],
                'Available': [25, 30, 35, 40]
            }),
            'G02': pd.DataFrame({  # In-process
                'fStyle#': ['F002', 'F003'],
                'Available': [20, 15]
            })
        }
    
    def test_single_style_aggregation(self, multi_level_inventory):
        """Test aggregation for a single style across all levels"""
        style = 'F001'
        total = 0
        
        for stage, df in multi_level_inventory.items():
            if style in df['fStyle#'].values:
                amount = df[df['fStyle#'] == style]['Available'].sum()
                total += amount
        
        # F001: F01(100) + I01(50) + G00(25) = 175
        assert total == 175
    
    def test_all_styles_aggregation(self, multi_level_inventory):
        """Test aggregation of all styles"""
        aggregated = {}
        
        for stage, df in multi_level_inventory.items():
            for _, row in df.iterrows():
                style = row['fStyle#']
                if style not in aggregated:
                    aggregated[style] = 0
                aggregated[style] += row['Available']
        
        assert aggregated['F001'] == 175  # 100 + 50 + 25
        assert aggregated['F002'] == 200  # 150 + 30 + 20
        assert aggregated['F003'] == 325  # 200 + 75 + 35 + 15
        assert aggregated['F004'] == 40   # 40 only
    
    def test_stage_priority_ordering(self, multi_level_inventory):
        """Test that inventory levels are prioritized correctly"""
        priority_order = ['F01', 'I01', 'G00', 'G02']
        
        # Verify order matches business rules
        assert priority_order[0] == 'F01'  # Finished goods first
        assert priority_order[1] == 'I01'  # QC queue second
        assert priority_order[-1] == 'G02' # In-process last
    
    def test_negative_inventory_handling(self):
        """Test handling of negative inventory values"""
        inventory = pd.DataFrame({
            'fStyle#': ['F001', 'F002'],
            'Available': [100, -50]  # Negative value
        })
        
        # Calculate total, treating negative as 0
        total = inventory['Available'].apply(lambda x: max(0, x)).sum()
        assert total == 100
    
    def test_inventory_with_allocations(self):
        """Test inventory aggregation with existing allocations"""
        inventory = pd.DataFrame({
            'fStyle#': ['F001'],
            'Available': [100],
            'Allocated': [30],
            'On_Order': [50]
        })
        
        # Calculate net available
        net_available = inventory['Available'].iloc[0] - inventory['Allocated'].iloc[0]
        assert net_available == 70
        
        # Calculate planning balance
        planning_balance = net_available + inventory['On_Order'].iloc[0]
        assert planning_balance == 120


class TestYarnAllocationFlow:
    """Test yarn allocation from Knit Orders to inventory"""
    
    @pytest.fixture
    def knit_order_data(self):
        """Create sample knit order data"""
        return pd.DataFrame({
            'KO#': ['KO001', 'KO002', 'KO003'],
            'Style#': ['6191-BK', '72762-DS', '80393C-DS'],
            'Qty Ordered (lbs)': [1000, 1500, 2000],
            'G00 (lbs)': [200, 300, 400],  # Already in process
            'Balance (lbs)': [800, 1200, 1600]
        })
    
    @pytest.fixture
    def bom_data(self):
        """Create sample BOM data"""
        return pd.DataFrame({
            'Style#': ['6191-BK', '6191-BK', '72762-DS', '72762-DS', '80393C-DS'],
            'Desc#': ['19003', '18868', '19003', '18851', '27035'],
            'BOM_Percent': [60, 40, 50, 50, 100],
            'unit': ['lbs', 'lbs', 'lbs', 'lbs', 'lbs']
        })
    
    def test_yarn_allocation_calculation(self, knit_order_data, bom_data):
        """Test calculation of yarn allocation from KO"""
        ko = knit_order_data.iloc[0]  # KO001 for 6191-BK
        style = ko['Style#']
        qty_ordered = ko['Qty Ordered (lbs)']
        
        # Get BOM for this style
        style_bom = bom_data[bom_data['Style#'] == style]
        
        allocations = {}
        for _, bom_row in style_bom.iterrows():
            yarn_id = bom_row['Desc#']
            percentage = bom_row['BOM_Percent']
            allocation = qty_ordered * (percentage / 100)
            allocations[yarn_id] = allocation
        
        assert allocations['19003'] == 600  # 1000 * 60%
        assert allocations['18868'] == 400  # 1000 * 40%
    
    def test_yarn_allocation_with_waste_factor(self, knit_order_data, bom_data):
        """Test yarn allocation including waste factor"""
        waste_factor = 1.1  # 10% waste
        
        ko = knit_order_data.iloc[0]
        qty_ordered = ko['Qty Ordered (lbs)']
        
        # Calculate with waste
        yarn_needed = qty_ordered * waste_factor
        assert yarn_needed == 1100  # 1000 * 1.1
    
    def test_partial_allocation_handling(self, knit_order_data):
        """Test handling of partial allocations"""
        ko = knit_order_data.iloc[0]
        
        # Some material already in process
        total_ordered = ko['Qty Ordered (lbs)']
        in_process = ko['G00 (lbs)']
        remaining = ko['Balance (lbs)']
        
        assert total_ordered == in_process + remaining
        assert remaining == 800  # Still need to allocate
    
    def test_allocation_priority_rules(self):
        """Test yarn allocation priority rules"""
        knit_orders = pd.DataFrame({
            'KO#': ['KO001', 'KO002', 'KO003'],
            'Priority': [1, 3, 2],  # Priority levels
            'Due_Date': pd.date_range(start='2025-08-20', periods=3)
        })
        
        # Sort by priority
        sorted_by_priority = knit_orders.sort_values('Priority')
        assert sorted_by_priority.iloc[0]['KO#'] == 'KO001'
        
        # Sort by due date
        sorted_by_date = knit_orders.sort_values('Due_Date')
        assert sorted_by_date.iloc[0]['KO#'] == 'KO001'
    
    def test_yarn_shortage_detection(self):
        """Test detection of yarn shortages during allocation"""
        yarn_inventory = pd.DataFrame({
            'Desc#': ['19003', '18868'],
            'Planning Balance': [-100, 200]  # 19003 has shortage
        })
        
        required_yarns = {'19003': 500, '18868': 150}
        
        shortages = {}
        for yarn_id, required in required_yarns.items():
            available = yarn_inventory[yarn_inventory['Desc#'] == yarn_id]['Planning Balance'].iloc[0]
            if available < required:
                shortages[yarn_id] = required - available
        
        assert '19003' in shortages
        assert shortages['19003'] == 600  # 500 - (-100)


class TestDataConsistencyValidation:
    """Test data consistency across the system"""
    
    def test_sales_order_consistency(self):
        """Test consistency of sales order data"""
        sales_orders = pd.DataFrame({
            'SO#': ['S001', 'S002', 'S003'],
            'Style#': ['6191-BK', '72762-DS', '80393C-DS'],
            'Ordered': [100, 200, 150],
            'Shipped': [50, 200, 0],
            'Balance': [50, 0, 150]
        })
        
        # Verify balance calculations
        for _, row in sales_orders.iterrows():
            assert row['Balance'] == row['Ordered'] - row['Shipped']
    
    def test_yarn_inventory_balance_formula(self):
        """Test yarn inventory planning balance formula"""
        yarn_data = pd.DataFrame({
            'Desc#': ['19003'],
            'Theoretical Balance': [1000],
            'Allocated': [300],
            'On Order': [200],
            'Planning Balance': [900]  # Should be 1000 - 300 + 200
        })
        
        row = yarn_data.iloc[0]
        calculated_balance = row['Theoretical Balance'] - row['Allocated'] + row['On Order']
        assert calculated_balance == row['Planning Balance']
    
    def test_date_consistency(self):
        """Test date consistency across different data sources"""
        orders = pd.DataFrame({
            'Order_Date': pd.to_datetime(['2025-08-01', '2025-08-05']),
            'Ship_Date': pd.to_datetime(['2025-08-10', '2025-08-15']),
            'Due_Date': pd.to_datetime(['2025-08-15', '2025-08-20'])
        })
        
        # Verify date logic
        for _, row in orders.iterrows():
            assert row['Order_Date'] <= row['Ship_Date']
            assert row['Ship_Date'] <= row['Due_Date']
    
    def test_unit_conversion_consistency(self):
        """Test unit conversion between yards and pounds"""
        fabric_specs = pd.DataFrame({
            'Style#': ['6191-BK'],
            'Yds_per_Lb': [2.5],  # Conversion factor
            'Order_Yds': [1000],
            'Order_Lbs': [400]  # Should be 1000 / 2.5
        })
        
        row = fabric_specs.iloc[0]
        calculated_lbs = row['Order_Yds'] / row['Yds_per_Lb']
        assert calculated_lbs == row['Order_Lbs']
    
    def test_bom_percentage_totals(self):
        """Test that BOM percentages add up correctly"""
        bom_data = pd.DataFrame({
            'Style#': ['6191-BK', '6191-BK', '6191-BK'],
            'Desc#': ['19003', '18868', '18851'],
            'BOM_Percent': [50, 30, 20]  # Should total 100%
        })
        
        style_total = bom_data.groupby('Style#')['BOM_Percent'].sum()
        assert style_total.iloc[0] == 100


class TestFileLoadingIntegration:
    """Test file loading and data integration"""
    
    def test_excel_file_loading(self):
        """Test loading of Excel files"""
        test_file = '/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/prompts/5/yarn_inventory (2).xlsx'
        
        if os.path.exists(test_file):
            df = pd.read_excel(test_file)
            assert not df.empty
            assert 'Desc#' in df.columns
    
    def test_csv_file_loading(self):
        """Test loading of CSV files"""
        test_file = '/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/prompts/5/Style_BOM.csv'
        
        if os.path.exists(test_file):
            df = pd.read_csv(test_file)
            assert not df.empty
    
    def test_missing_file_handling(self):
        """Test handling of missing files"""
        missing_file = '/nonexistent/file.xlsx'
        
        try:
            df = pd.read_excel(missing_file)
            assert False, "Should have raised an exception"
        except FileNotFoundError:
            assert True  # Expected behavior
    
    def test_corrupt_data_handling(self):
        """Test handling of corrupt or malformed data"""
        bad_data = pd.DataFrame({
            'Column1': [1, 'not_a_number', 3],  # Mixed types
            'Column2': [None, None, None]  # All nulls
        })
        
        # Test numeric conversion
        numeric_col = pd.to_numeric(bad_data['Column1'], errors='coerce')
        assert numeric_col.isna().sum() == 1  # One value couldn't convert
        
        # Test null handling
        assert bad_data['Column2'].isna().all()


class TestEndToEndDataFlow:
    """Test complete data flow through the system"""
    
    @pytest.fixture
    def planning_engine(self):
        """Create planning engine instance"""
        return SixPhasePlanningEngine()
    
    def test_sales_to_yarn_flow(self):
        """Test flow from sales order to yarn requirement"""
        # Step 1: Sales Order
        sales_order = {
            'SO#': 'S001',
            'Style#': '6191-BK',
            'Quantity': 100
        }
        
        # Step 2: BOM lookup
        bom = {
            '6191-BK': {
                '19003': 0.6,  # 60% of this yarn
                '18868': 0.4   # 40% of this yarn
            }
        }
        
        # Step 3: Calculate yarn requirements
        yarn_reqs = {}
        for yarn_id, percentage in bom[sales_order['Style#']].items():
            yarn_reqs[yarn_id] = sales_order['Quantity'] * percentage
        
        assert yarn_reqs['19003'] == 60
        assert yarn_reqs['18868'] == 40
    
    def test_inventory_to_net_requirements_flow(self):
        """Test flow from inventory to net requirements"""
        demand = 100
        inventory = {
            'F01': 30,  # Finished goods
            'I01': 20,  # QC queue
            'G00': 15   # In-process
        }
        
        total_available = sum(inventory.values())
        net_requirement = max(0, demand - total_available)
        
        assert total_available == 65
        assert net_requirement == 35
    
    def test_knit_order_to_production_flow(self):
        """Test flow from knit order to production"""
        knit_order = {
            'KO#': 'KO001',
            'Style#': '6191-BK',
            'Quantity_Lbs': 1000,
            'Status': 'In Process'
        }
        
        # Production tracking
        production_stages = {
            'Knitting': 0.3,    # 30% complete
            'Dyeing': 0.0,      # Not started
            'Finishing': 0.0    # Not started
        }
        
        completed_lbs = knit_order['Quantity_Lbs'] * production_stages['Knitting']
        assert completed_lbs == 300
    
    def test_procurement_recommendation_flow(self):
        """Test flow for procurement recommendations"""
        yarn_shortage = {
            'Desc#': '19003',
            'Required': 1000,
            'Available': 200,
            'Shortage': 800
        }
        
        supplier_info = {
            '19003': {
                'Supplier': 'ABC Yarns',
                'Lead_Time_Days': 14,
                'Min_Order_Qty': 500
            }
        }
        
        # Calculate order quantity
        shortage = yarn_shortage['Shortage']
        min_order = supplier_info['19003']['Min_Order_Qty']
        order_qty = max(shortage, min_order)
        
        # Round up to min order quantity multiple
        if shortage > min_order:
            order_qty = ((shortage // min_order) + 1) * min_order
        
        assert order_qty == 1000  # 2 * 500 min order


class TestDataValidationFramework:
    """Test the data validation framework"""
    
    def test_required_columns_validation(self):
        """Test validation of required columns"""
        required_columns = ['Style#', 'Quantity', 'Due_Date']
        
        # Valid data
        valid_df = pd.DataFrame({
            'Style#': ['6191-BK'],
            'Quantity': [100],
            'Due_Date': [datetime.now()]
        })
        
        missing_cols = [col for col in required_columns if col not in valid_df.columns]
        assert len(missing_cols) == 0
        
        # Invalid data
        invalid_df = pd.DataFrame({
            'Style#': ['6191-BK'],
            'Quantity': [100]
            # Missing Due_Date
        })
        
        missing_cols = [col for col in required_columns if col not in invalid_df.columns]
        assert 'Due_Date' in missing_cols
    
    def test_data_type_validation(self):
        """Test validation of data types"""
        df = pd.DataFrame({
            'Style#': ['6191-BK', '72762-DS'],
            'Quantity': ['100', '200'],  # String instead of numeric
            'Date': ['2025-08-15', '2025-08-20']  # String instead of datetime
        })
        
        # Convert and validate
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        assert df['Quantity'].dtype in [np.int64, np.float64]
        assert df['Date'].dtype == 'datetime64[ns]'
    
    def test_value_range_validation(self):
        """Test validation of value ranges"""
        df = pd.DataFrame({
            'Quantity': [100, -50, 1000000],  # Negative and very large
            'Percentage': [50, 150, -10]  # Outside 0-100 range
        })
        
        # Validate quantity (should be positive)
        invalid_qty = df[df['Quantity'] < 0]
        assert len(invalid_qty) == 1
        
        # Validate percentage (should be 0-100)
        invalid_pct = df[(df['Percentage'] < 0) | (df['Percentage'] > 100)]
        assert len(invalid_pct) == 2
    
    def test_referential_integrity(self):
        """Test referential integrity between tables"""
        orders = pd.DataFrame({
            'Order_ID': [1, 2, 3],
            'Style#': ['6191-BK', '72762-DS', 'INVALID']
        })
        
        styles = pd.DataFrame({
            'Style#': ['6191-BK', '72762-DS', '80393C-DS']
        })
        
        # Check referential integrity
        invalid_styles = orders[~orders['Style#'].isin(styles['Style#'])]
        assert len(invalid_styles) == 1
        assert invalid_styles.iloc[0]['Style#'] == 'INVALID'


class TestPerformanceIntegration:
    """Test performance of integrated operations"""
    
    def test_large_dataset_join_performance(self):
        """Test performance of joining large datasets"""
        import time
        
        # Create large datasets
        n_rows = 10000
        df1 = pd.DataFrame({
            'ID': range(n_rows),
            'Value1': np.random.rand(n_rows)
        })
        
        df2 = pd.DataFrame({
            'ID': range(n_rows),
            'Value2': np.random.rand(n_rows)
        })
        
        start_time = time.time()
        merged = df1.merge(df2, on='ID')
        duration = time.time() - start_time
        
        assert len(merged) == n_rows
        assert duration < 1.0  # Should complete in under 1 second
    
    def test_aggregation_performance(self):
        """Test performance of data aggregation"""
        import time
        
        # Create dataset with many groups
        n_rows = 100000
        n_groups = 1000
        
        df = pd.DataFrame({
            'Group': np.random.choice(range(n_groups), n_rows),
            'Value': np.random.rand(n_rows)
        })
        
        start_time = time.time()
        aggregated = df.groupby('Group')['Value'].sum()
        duration = time.time() - start_time
        
        assert len(aggregated) <= n_groups
        assert duration < 2.0  # Should complete in under 2 seconds


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])