"""
Test v2 APIs with different column name formats
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.v2.yarn import YarnV2Handler
from api.v2.inventory import InventoryV2Handler
from api.v2.production import ProductionV2Handler


class TestV2ColumnNames:
    """Test v2 APIs handle different column name variations"""
    
    def test_yarn_handler_with_desc_variations(self):
        """Test yarn handler handles Desc# column variations"""
        # Test data with different column names
        test_data = pd.DataFrame({
            'yarn_id': ['Y001', 'Y002', 'Y003'],  # Different column name
            'Planning_Balance': [-100, 200, -500],  # Underscore variation
            'Theoretical_Balance': [100, 300, 100],
            'allocated': [150, 50, 400],  # Lowercase
            'On_Order': [50, 0, 200]  # Underscore variation
        })
        
        handler = YarnV2Handler()
        
        # Test shortage analysis
        result = handler._analyze_shortages(test_data, 'data', False)
        
        assert result['total_yarns'] == 3
        assert result['yarns_with_shortage'] == 2  # Y001 and Y003
        assert result['total_shortage_lbs'] == 600  # 100 + 500
        
    def test_yarn_handler_with_planning_balance_variations(self):
        """Test yarn handler calculates Planning Balance correctly"""
        # Test data without Planning Balance column
        test_data = pd.DataFrame({
            'Desc#': ['Y001', 'Y002'],
            'Theoretical Balance': [100, 200],
            'Allocated': [-50, -100],  # Negative for shortage
            'On Order': [25, 50]
        })
        
        handler = YarnV2Handler()
        
        # Test shortage analysis - should calculate Planning Balance
        result = handler._analyze_shortages(test_data, 'summary', False)
        
        # Planning Balance = Theoretical + Allocated + On Order
        # Y001: 100 + (-50) + 25 = 75 (no shortage)
        # Y002: 200 + (-100) + 50 = 150 (no shortage)
        assert result['yarns_with_shortage'] == 0
        
    def test_inventory_handler_with_column_variations(self):
        """Test inventory handler handles column variations"""
        test_data = pd.DataFrame({
            'desc_num': ['Y001', 'Y002', 'Y003'],  # Different column name
            'planning_balance': [-1500, -50, 100],  # Lowercase
            'theoretical_balance': [500, 150, 200],
            'Reserved': [1800, 150, 50],  # Different name for Allocated
            'Ordered': [200, 50, 0]  # Different name for On Order
        })
        
        handler = InventoryV2Handler()
        
        # Test processing
        result = handler._process_yarn_data(
            test_data, 'overview', 'shortage', False
        )
        
        assert result['summary']['total_yarns'] == 3
        assert result['summary']['critical_count'] == 1  # Y001 < -1000
        assert result['summary']['medium_count'] == 1  # Y002 between -100 and 0
        assert result['summary']['low_count'] == 1  # Y003 >= 0
        
    def test_production_handler_with_column_variations(self):
        """Test production handler handles column variations"""
        test_data = pd.DataFrame({
            'KO #': ['KO001', 'KO002', 'KO003'],  # Space variation
            'Style #': ['S001', 'S002', 'S003'],  # Space variation
            'Qty_Ordered': [1000, 2000, 1500],  # Different column name
            'machine': ['M1', None, 'M3']  # Lowercase
        })
        
        handler = ProductionV2Handler()
        
        # Mock ko_data
        handler.analyzer = Mock()
        handler.analyzer.ko_data = test_data
        
        # Test list orders summary
        result = handler._list_orders({'view': 'summary'})
        
        assert result['summary']['total_orders'] == 3
        assert result['summary']['assigned_orders'] == 2  # KO001 and KO003
        assert result['summary']['unassigned_orders'] == 1  # KO002
        assert result['summary']['total_production_lbs'] == 4500
        
    def test_production_handler_detailed_view(self):
        """Test production handler detailed view with column variations"""
        test_data = pd.DataFrame({
            'Actions': ['KO001', 'KO002'],  # Alternative KO# column
            'fStyle#': ['FS001', 'FS002'],  # fStyle instead of Style
            'quantity_lbs': [1500, 2500],  # Different column name
            'Equipment': ['E1', None]  # Alternative to Machine
        })
        
        handler = ProductionV2Handler()
        
        # Mock ko_data
        handler.analyzer = Mock()
        handler.analyzer.ko_data = test_data
        
        # Test detailed view
        result = handler._list_orders({'view': 'detailed'})
        
        assert len(result['orders']) == 2
        assert result['orders'][0]['order_id'] == 'KO001'
        assert result['orders'][0]['style'] == 'FS001'
        assert result['orders'][0]['quantity_lbs'] == 1500
        assert result['orders'][0]['machine'] == 'E1'
        assert result['orders'][0]['status'] == 'ASSIGNED'
        assert result['orders'][1]['status'] == 'UNASSIGNED'
        
    def test_yarn_aggregation_with_planning_ballance_typo(self):
        """Test handling of Planning_Ballance typo"""
        test_data = pd.DataFrame({
            'Desc#': ['Y001', 'Y002', 'Y003'],
            'Planning_Ballance': [-1500, -500, 200],  # Typo variation
        })
        
        handler = YarnV2Handler()
        
        # Test aggregation
        result = handler._aggregate_yarn_data(test_data)
        
        assert result['aggregation']['by_status']['critical'] == 1
        assert result['aggregation']['by_status']['warning'] == 1
        assert result['aggregation']['by_status']['ok'] == 1
        
    def test_inventory_action_items_with_variations(self):
        """Test inventory action items with column variations"""
        test_data = pd.DataFrame({
            'YarnID': ['YARN001', 'YARN002'],  # Different ID column
            'description': ['Cotton Yarn', 'Polyester Yarn'],  # Lowercase
            'Planning Balance': [-2000, -1500]  # With space
        })
        
        handler = InventoryV2Handler()
        
        # Test action items view
        result = handler._process_yarn_data(
            test_data, 'action-items', None, False
        )
        
        assert 'action_items' in result
        assert len(result['action_items']) == 2
        assert result['action_items'][0]['yarn_id'] == 'YARN001'
        assert result['action_items'][0]['description'] == 'Cotton Yarn'
        assert result['action_items'][0]['shortage'] == 2000
        assert result['action_items'][0]['priority'] == 'CRITICAL'


if __name__ == '__main__':
    # Run tests
    test_class = TestV2ColumnNames()
    
    print("Testing Yarn Handler with column variations...")
    test_class.test_yarn_handler_with_desc_variations()
    print("✓ Yarn handler handles Desc# variations")
    
    test_class.test_yarn_handler_with_planning_balance_variations()
    print("✓ Yarn handler calculates Planning Balance correctly")
    
    print("\nTesting Inventory Handler with column variations...")
    test_class.test_inventory_handler_with_column_variations()
    print("✓ Inventory handler handles column variations")
    
    print("\nTesting Production Handler with column variations...")
    test_class.test_production_handler_with_column_variations()
    print("✓ Production handler handles column variations")
    
    test_class.test_production_handler_detailed_view()
    print("✓ Production handler detailed view works")
    
    print("\nTesting special cases...")
    test_class.test_yarn_aggregation_with_planning_ballance_typo()
    print("✓ Handles Planning_Ballance typo")
    
    test_class.test_inventory_action_items_with_variations()
    print("✓ Inventory action items with variations")
    
    print("\n✅ All v2 API column name tests passed!")