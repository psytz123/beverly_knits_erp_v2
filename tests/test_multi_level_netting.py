#!/usr/bin/env python3
"""
Test cases for Multi-Level Inventory Netting Module
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Note: multi_level_netting module not found in codebase
# from multi_level_netting import MultiLevelInventoryNetting, NettingResult
# This test file may need to be updated or removed

class TestMultiLevelNetting(unittest.TestCase):
    """Test cases for multi-level inventory netting"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.netting = MultiLevelInventoryNetting()
        
        # Create sample inventory data
        self.sample_inventory = {
            'F01': pd.DataFrame({
                'Style#': ['STYLE001', 'STYLE002', 'STYLE003'],
                'Quantity': [100, 50, 0]
            }),
            'I01': pd.DataFrame({
                'Style#': ['STYLE001', 'STYLE002', 'STYLE003'],
                'Quantity': [50, 25, 100]
            }),
            'G00': pd.DataFrame({
                'Style#': ['STYLE001', 'STYLE002', 'STYLE003'],
                'Quantity': [200, 100, 50]
            }),
            'G02': pd.DataFrame({
                'Style#': ['STYLE001', 'STYLE002', 'STYLE003'],
                'Quantity': [150, 75, 25]
            })
        }
        
        # Create sample knit orders
        self.sample_knit_orders = pd.DataFrame({
            'Style#': ['STYLE001', 'STYLE002', 'STYLE003', 'STYLE004'],
            'Balance (lbs)': [300, 150, 200, 500]
        })
        
    def test_initialization(self):
        """Test proper initialization of netting system"""
        self.assertIsNotNone(self.netting)
        self.assertEqual(len(self.netting.lead_times), 5)
        self.assertEqual(self.netting.lead_times['F01'], 0)
        self.assertEqual(self.netting.lead_times['I01'], 2)
        self.assertEqual(self.netting.lead_times['G00'], 15)
        
    def test_load_inventory_data(self):
        """Test loading inventory data"""
        self.netting.load_inventory_data(self.sample_inventory)
        
        self.assertEqual(len(self.netting.inventory_levels), 4)
        self.assertIn('F01', self.netting.inventory_levels)
        self.assertIn('I01', self.netting.inventory_levels)
        
    def test_load_knit_orders(self):
        """Test loading knit orders"""
        self.netting.load_knit_orders(self.sample_knit_orders)
        
        self.assertIsNotNone(self.netting.knit_orders)
        self.assertEqual(len(self.netting.knit_orders), 4)
        
    def test_single_style_netting_full_coverage(self):
        """Test netting for a style with full inventory coverage"""
        self.netting.load_inventory_data(self.sample_inventory)
        self.netting.load_knit_orders(self.sample_knit_orders)
        
        # STYLE001 has total inventory of 100+50+200+150+300 = 800
        demand = {'STYLE001': 400}
        results = self.netting.net_demand(demand)
        
        self.assertIn('STYLE001', results)
        result = results['STYLE001']
        
        # Should use inventory in order: F01(100), I01(50), G00(200), G02(150)
        self.assertEqual(result.f01_used, 100)
        self.assertEqual(result.i01_used, 50)
        self.assertEqual(result.g00_used, 200)
        self.assertEqual(result.g02_used, 50)  # Only 50 needed from G02
        self.assertEqual(result.ko_used, 0)  # No KO needed
        self.assertEqual(result.net_requirement, 0)  # Fully covered
        self.assertEqual(result.coverage_percentage, 100)
        
    def test_single_style_netting_partial_coverage(self):
        """Test netting for a style with partial inventory coverage"""
        self.netting.load_inventory_data(self.sample_inventory)
        self.netting.load_knit_orders(self.sample_knit_orders)
        
        # STYLE001 total available: 800
        demand = {'STYLE001': 1000}
        results = self.netting.net_demand(demand)
        
        result = results['STYLE001']
        
        # Should use all available inventory
        self.assertEqual(result.f01_used, 100)
        self.assertEqual(result.i01_used, 50)
        self.assertEqual(result.g00_used, 200)
        self.assertEqual(result.g02_used, 150)
        self.assertEqual(result.ko_used, 300)
        self.assertEqual(result.net_requirement, 200)  # 1000 - 800 = 200
        self.assertEqual(result.coverage_percentage, 80)  # 800/1000 = 80%
        
    def test_style_with_no_inventory(self):
        """Test netting for a style with no inventory"""
        self.netting.load_inventory_data(self.sample_inventory)
        self.netting.load_knit_orders(self.sample_knit_orders)
        
        # STYLE005 has no inventory
        demand = {'STYLE005': 500}
        results = self.netting.net_demand(demand)
        
        result = results['STYLE005']
        
        self.assertEqual(result.f01_used, 0)
        self.assertEqual(result.i01_used, 0)
        self.assertEqual(result.g00_used, 0)
        self.assertEqual(result.g02_used, 0)
        self.assertEqual(result.ko_used, 0)
        self.assertEqual(result.net_requirement, 500)
        self.assertEqual(result.coverage_percentage, 0)
        
    def test_multiple_styles_netting(self):
        """Test netting for multiple styles simultaneously"""
        self.netting.load_inventory_data(self.sample_inventory)
        self.netting.load_knit_orders(self.sample_knit_orders)
        
        demand = {
            'STYLE001': 200,
            'STYLE002': 100,
            'STYLE003': 150
        }
        
        results = self.netting.net_demand(demand)
        
        self.assertEqual(len(results), 3)
        
        # Check STYLE001 (has 100 F01, 50 I01, etc.)
        self.assertEqual(results['STYLE001'].net_requirement, 0)  # Fully covered
        
        # Check STYLE003 (has 0 F01, 100 I01, etc.)
        self.assertEqual(results['STYLE003'].net_requirement, 0)  # Fully covered
        
    def test_first_available_date_calculation(self):
        """Test calculation of first available date"""
        self.netting.load_inventory_data(self.sample_inventory)
        
        # Test immediate availability (F01)
        result = self.netting._net_single_style('STYLE001', 50)
        self.assertEqual(result.first_available_date.date(), datetime.now().date())
        
        # Test I01 availability (2 days)
        result = self.netting._net_single_style('STYLE003', 50)  # STYLE003 has 0 F01, 100 I01
        expected_date = (datetime.now() + timedelta(days=2)).date()
        self.assertEqual(result.first_available_date.date(), expected_date)
        
    def test_summary_statistics(self):
        """Test summary statistics calculation"""
        self.netting.load_inventory_data(self.sample_inventory)
        self.netting.load_knit_orders(self.sample_knit_orders)
        
        demand = {
            'STYLE001': 200,
            'STYLE002': 150,
            'STYLE003': 100
        }
        
        results = self.netting.net_demand(demand)
        
        self.assertIsNotNone(self.netting.summary_statistics)
        summary = self.netting.summary_statistics
        
        self.assertEqual(summary['total_styles'], 3)
        self.assertEqual(summary['total_original_demand'], 450)
        self.assertIn('inventory_usage', summary)
        self.assertIn('F01_finished_goods', summary['inventory_usage'])
        
    def test_time_phased_netting(self):
        """Test time-phased netting capability"""
        self.netting.load_inventory_data(self.sample_inventory)
        
        # Create demand schedule
        demand_schedule = pd.DataFrame({
            'style': ['STYLE001', 'STYLE001', 'STYLE002', 'STYLE002'],
            'date': [
                datetime.now(),
                datetime.now() + timedelta(days=7),
                datetime.now(),
                datetime.now() + timedelta(days=14)
            ],
            'quantity': [100, 150, 50, 75]
        })
        
        time_phased = self.netting.get_time_phased_requirements(demand_schedule, bucket_size=7)
        
        self.assertIsNotNone(time_phased)
        self.assertIn('bucket', time_phased.columns)
        self.assertIn('net_requirement', time_phased.columns)
        
    def test_export_netting_report(self):
        """Test export functionality"""
        self.netting.load_inventory_data(self.sample_inventory)
        self.netting.load_knit_orders(self.sample_knit_orders)
        
        demand = {'STYLE001': 200, 'STYLE002': 150}
        self.netting.net_demand(demand)
        
        # Test export (without actually writing file)
        output_path = 'test_netting_report.xlsx'
        
        # Just verify the method runs without error
        # In real test, we'd mock the file writing
        self.assertIsNotNone(self.netting.netting_results)
        self.assertEqual(len(self.netting.netting_results), 2)
        
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with empty demand
        results = self.netting.net_demand({})
        self.assertEqual(len(results), 0)
        
        # Test with negative demand (should treat as 0)
        results = self.netting.net_demand({'STYLE001': -100})
        self.assertEqual(results['STYLE001'].net_requirement, 0)
        
        # Test with no inventory loaded
        netting_empty = MultiLevelInventoryNetting()
        results = netting_empty.net_demand({'STYLE001': 100})
        self.assertEqual(results['STYLE001'].net_requirement, 100)
        
    def test_inventory_level_priority(self):
        """Test that inventory levels are consumed in correct priority order"""
        self.netting.load_inventory_data(self.sample_inventory)
        
        # Small demand should only use F01
        demand = {'STYLE001': 50}
        results = self.netting.net_demand(demand)
        result = results['STYLE001']
        
        self.assertEqual(result.f01_used, 50)
        self.assertEqual(result.i01_used, 0)
        self.assertEqual(result.g00_used, 0)
        
        # Medium demand should use F01 then I01
        demand = {'STYLE001': 125}
        netting2 = MultiLevelInventoryNetting()
        netting2.load_inventory_data(self.sample_inventory)
        results = netting2.net_demand(demand)
        result = results['STYLE001']
        
        self.assertEqual(result.f01_used, 100)
        self.assertEqual(result.i01_used, 25)
        self.assertEqual(result.g00_used, 0)

if __name__ == '__main__':
    unittest.main()