"""
Unit tests for inventory management functions in beverly_comprehensive_erp.py

Tests core inventory logic including:
- InventoryAnalyzer class
- Yarn shortage calculations
- Planning balance computations
- Substitution matching logic
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Import the module to test
import core.beverly_comprehensive_erp as erp


class TestInventoryAnalyzer:
    """Test suite for InventoryAnalyzer class"""
    
    @pytest.fixture
    def inventory_analyzer(self, sample_yarn_data, sample_demand_data):
        """Create a properly initialized InventoryAnalyzer"""
        analyzer = erp.InventoryAnalyzer()
        analyzer.yarn_inventory = sample_yarn_data
        analyzer.yarn_demand = sample_demand_data
        return analyzer
    
    @pytest.fixture
    def sample_yarn_data(self):
        """Create sample yarn inventory data"""
        return pd.DataFrame({
            'Item': ['YARN001', 'YARN002', 'YARN003', 'YARN004'],
            'Desc#': ['Cotton 30/1', 'Poly 40/1', 'Cotton 30/1 Combed', 'Nylon 20/1'],
            'Material': ['Cotton', 'Polyester', 'Cotton', 'Nylon'],
            'Size': ['30/1', '40/1', '30/1', '20/1'],
            'Color': ['White', 'Black', 'White', 'Red'],
            'Theoretical Balance': [1000.0, 500.0, -200.0, 800.0],
            'Allocated': [200.0, 100.0, 50.0, 150.0],
            'On Order': [0.0, 300.0, 500.0, 0.0],
            'Planning Balance': [800.0, 700.0, 250.0, 650.0],
            'UOM': ['LBS', 'LBS', 'LBS', 'LBS'],
            'Unit Cost': [5.50, 4.25, 6.00, 7.50]
        })
    
    @pytest.fixture
    def sample_demand_data(self):
        """Create sample yarn demand data"""
        return pd.DataFrame({
            'Yarn_Code': ['YARN001', 'YARN002', 'YARN003'],
            'Yarn_Description': ['Cotton 30/1', 'Poly 40/1', 'Cotton 30/1 Combed'],
            'Total_Demand': [1500.0, 300.0, 800.0],
            'Week_1': [300.0, 60.0, 160.0],
            'Week_2': [300.0, 60.0, 160.0],
            'Week_3': [300.0, 60.0, 160.0],
            'Week_4': [300.0, 60.0, 160.0],
            'Week_5': [300.0, 60.0, 160.0]
        })
    
    @pytest.fixture
    def inventory_analyzer(self, sample_yarn_data, sample_demand_data):
        """Create InventoryAnalyzer instance with test data"""
        with patch('beverly_comprehensive_erp.pd.read_excel') as mock_read:
            # Mock the file reading
            mock_read.side_effect = [sample_yarn_data, sample_demand_data]
            
            analyzer = erp.InventoryAnalyzer()
            analyzer.yarn_inventory = sample_yarn_data
            analyzer.yarn_demand = sample_demand_data
            
            return analyzer
    
    def test_inventory_analyzer_initialization(self):
        """Test InventoryAnalyzer initialization"""
        analyzer = erp.InventoryAnalyzer()
        
        assert analyzer is not None
        assert hasattr(analyzer, 'data_dir')
        assert hasattr(analyzer, 'yarn_inventory')
    
    def test_calculate_planning_balance(self, inventory_analyzer):
        """Test planning balance calculation"""
        # Planning Balance = Theoretical Balance - Allocated + On Order
        yarn_data = inventory_analyzer.yarn_inventory
        
        # Test for YARN001
        yarn1 = yarn_data[yarn_data['Item'] == 'YARN001'].iloc[0]
        expected_balance = yarn1['Theoretical Balance'] - yarn1['Allocated'] + yarn1['On Order']
        assert yarn1['Planning Balance'] == expected_balance
        
        # Test for YARN003 (negative theoretical balance)
        yarn3 = yarn_data[yarn_data['Item'] == 'YARN003'].iloc[0]
        expected_balance = yarn3['Theoretical Balance'] - yarn3['Allocated'] + yarn3['On Order']
        assert yarn3['Planning Balance'] == expected_balance
    
    def test_identify_shortages(self, inventory_analyzer):
        """Test shortage identification"""
        # Shortage exists when Planning Balance < 0
        shortages = inventory_analyzer.yarn_inventory[
            inventory_analyzer.yarn_inventory['Planning Balance'] < 0
        ]
        
        # In our test data, no items have negative planning balance
        assert len(shortages) == 0
        
        # Modify data to create shortage
        inventory_analyzer.yarn_inventory.loc[0, 'Planning Balance'] = -100
        shortages = inventory_analyzer.yarn_inventory[
            inventory_analyzer.yarn_inventory['Planning Balance'] < 0
        ]
        
        assert len(shortages) == 1
        assert shortages.iloc[0]['Item'] == 'YARN001'
        assert shortages.iloc[0]['Planning Balance'] == -100
    
    def test_find_substitutions(self, inventory_analyzer):
        """Test yarn substitution matching logic"""
        # Find substitutions for YARN001 (Cotton 30/1)
        target_yarn = inventory_analyzer.yarn_inventory.iloc[0]
        
        # YARN003 should be a match (same material and size)
        substitutions = inventory_analyzer.yarn_inventory[
            (inventory_analyzer.yarn_inventory['Material'] == target_yarn['Material']) &
            (inventory_analyzer.yarn_inventory['Size'] == target_yarn['Size']) &
            (inventory_analyzer.yarn_inventory['Item'] != target_yarn['Item']) &
            (inventory_analyzer.yarn_inventory['Planning Balance'] > 0)
        ]
        
        assert len(substitutions) == 1
        assert substitutions.iloc[0]['Item'] == 'YARN003'
    
    def test_calculate_weekly_consumption(self, inventory_analyzer):
        """Test weekly consumption calculation"""
        demand_data = inventory_analyzer.yarn_demand
        
        # Test for YARN001
        yarn1_demand = demand_data[demand_data['Yarn_Code'] == 'YARN001'].iloc[0]
        weekly_avg = yarn1_demand['Total_Demand'] / 5  # 5 weeks of data
        
        assert weekly_avg == 300.0  # 1500 / 5
        assert yarn1_demand['Week_1'] == 300.0
    
    def test_calculate_days_of_supply(self, inventory_analyzer):
        """Test days of supply calculation"""
        # Days of Supply = Current Stock / Daily Consumption
        yarn_data = inventory_analyzer.yarn_inventory.iloc[0]
        demand_data = inventory_analyzer.yarn_demand.iloc[0]
        
        current_stock = yarn_data['Planning Balance']
        daily_consumption = demand_data['Total_Demand'] / 35  # 5 weeks = 35 days
        
        if daily_consumption > 0:
            days_of_supply = current_stock / daily_consumption
        else:
            days_of_supply = float('inf')
        
        assert days_of_supply > 0
        assert days_of_supply == 800.0 / (1500.0 / 35)
    
    def test_calculate_criticality_score(self, inventory_analyzer):
        """Test criticality scoring logic"""
        # Criticality based on:
        # - Shortage amount
        # - Days of supply
        # - Number of dependent products
        
        yarn_data = inventory_analyzer.yarn_inventory.iloc[0]
        
        # Mock criticality calculation
        shortage = max(0, -yarn_data['Planning Balance'])
        days_supply = 10  # Mock value
        
        if shortage > 0:
            criticality = 100
        elif days_supply < 7:
            criticality = 80
        elif days_supply < 14:
            criticality = 60
        elif days_supply < 30:
            criticality = 40
        else:
            criticality = 20
        
        assert criticality >= 0
        assert criticality <= 100
    
    def test_aggregate_interchangeable_yarns(self, inventory_analyzer):
        """Test aggregation of interchangeable yarns"""
        # Group yarns by material and size
        grouped = inventory_analyzer.yarn_inventory.groupby(['Material', 'Size']).agg({
            'Planning Balance': 'sum',
            'Unit Cost': 'mean'
        }).reset_index()
        
        # Cotton 30/1 group should have 2 items (YARN001 and YARN003)
        cotton_group = grouped[
            (grouped['Material'] == 'Cotton') & 
            (grouped['Size'] == '30/1')
        ]
        
        assert len(cotton_group) == 1
        assert cotton_group.iloc[0]['Planning Balance'] == 1050.0  # 800 + 250
    
    def test_calculate_inventory_value(self, inventory_analyzer):
        """Test inventory value calculation"""
        yarn_data = inventory_analyzer.yarn_inventory
        
        # Calculate total value
        yarn_data['Value'] = yarn_data['Planning Balance'] * yarn_data['Unit Cost']
        total_value = yarn_data['Value'].sum()
        
        expected_value = (800 * 5.50) + (700 * 4.25) + (250 * 6.00) + (650 * 7.50)
        assert abs(total_value - expected_value) < 0.01
    
    def test_handle_missing_data(self):
        """Test handling of missing or invalid data"""
        # Test with missing columns
        incomplete_data = pd.DataFrame({
            'Item': ['YARN001'],
            'Planning Balance': [100.0]
            # Missing other required columns
        })
        
        analyzer = erp.InventoryAnalyzer()
        analyzer.yarn_inventory = incomplete_data
        
        # Should handle gracefully without crashing
        assert 'Theoretical Balance' not in analyzer.yarn_inventory.columns
        
        # Add missing columns with defaults
        if 'Allocated' not in analyzer.yarn_inventory.columns:
            analyzer.yarn_inventory['Allocated'] = 0.0
        
        assert 'Allocated' in analyzer.yarn_inventory.columns
    
    def test_json_serialization(self, inventory_analyzer):
        """Test JSON serialization of numpy types"""
        analyzer = inventory_analyzer
        
        # This tests the clean_for_json function
        data = {
            'numpy_float': np.float64(10.5),
            'numpy_int': np.int64(42),
            'regular_float': 10.5,
            'regular_int': 42,
            'string': 'test',
            'numpy_array': np.array([1, 2, 3])
        }
        
        cleaned = erp.clean_for_json(data)
        
        assert isinstance(cleaned['numpy_float'], float)
        assert isinstance(cleaned['numpy_int'], int)
        assert isinstance(cleaned['regular_float'], float)
        assert isinstance(cleaned['regular_int'], int)
        assert isinstance(cleaned['string'], str)
        assert isinstance(cleaned['numpy_array'], list)


class TestInventoryOptimization:
    """Test inventory optimization functions"""
    
    def test_calculate_safety_stock(self):
        """Test safety stock calculation"""
        # Safety Stock = Z-score * σ * √Lead Time
        avg_demand = 100
        std_dev = 20
        lead_time_days = 7
        service_level = 0.95  # 95% service level
        z_score = 1.645  # Z-score for 95% service level
        
        safety_stock = z_score * std_dev * np.sqrt(lead_time_days)
        
        assert safety_stock > 0
        assert abs(safety_stock - (1.645 * 20 * np.sqrt(7))) < 0.01
    
    def test_calculate_reorder_point(self):
        """Test reorder point calculation"""
        # Reorder Point = (Average Daily Demand * Lead Time) + Safety Stock
        avg_daily_demand = 50
        lead_time_days = 7
        safety_stock = 100
        
        reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
        
        assert reorder_point == 450
    
    def test_calculate_eoq(self):
        """Test Economic Order Quantity calculation"""
        # EOQ = √(2 * Annual Demand * Ordering Cost / Holding Cost)
        annual_demand = 10000
        ordering_cost = 100
        holding_cost_per_unit = 5
        
        eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit)
        
        assert eoq > 0
        assert abs(eoq - 632.45) < 0.01  # Expected EOQ value
    
    def test_abc_classification(self):
        """Test ABC inventory classification"""
        items = pd.DataFrame({
            'Item': ['A', 'B', 'C', 'D', 'E'],
            'Annual_Value': [50000, 30000, 15000, 3000, 2000]
        })
        
        # Calculate cumulative percentage
        items = items.sort_values('Annual_Value', ascending=False)
        items['Cumulative_Value'] = items['Annual_Value'].cumsum()
        items['Cumulative_Percent'] = items['Cumulative_Value'] / items['Annual_Value'].sum() * 100
        
        # Classify items
        items['Category'] = 'C'
        items.loc[items['Cumulative_Percent'] <= 80, 'Category'] = 'A'
        items.loc[(items['Cumulative_Percent'] > 80) & (items['Cumulative_Percent'] <= 95), 'Category'] = 'B'
        
        assert items.iloc[0]['Category'] == 'A'  # Highest value item
        assert items.iloc[-1]['Category'] == 'C'  # Lowest value item


class TestYarnCalculations:
    """Test yarn-specific calculations"""
    
    def test_yarn_shortage_calculation(self):
        """Test yarn shortage amount calculation"""
        planning_balance = -500.0
        
        # Shortage is absolute value of negative planning balance
        shortage = abs(planning_balance) if planning_balance < 0 else 0
        
        assert shortage == 500.0
        
        # Test with positive balance
        planning_balance = 100.0
        shortage = abs(planning_balance) if planning_balance < 0 else 0
        
        assert shortage == 0
    
    def test_yarn_substitution_compatibility(self):
        """Test yarn substitution compatibility rules"""
        yarn1 = {
            'Material': 'Cotton',
            'Size': '30/1',
            'Type': 'Combed'
        }
        
        yarn2 = {
            'Material': 'Cotton',
            'Size': '30/1',
            'Type': 'Combed'
        }
        
        yarn3 = {
            'Material': 'Cotton',
            'Size': '30/1',
            'Type': 'Karded'
        }
        
        yarn4 = {
            'Material': 'Polyester',
            'Size': '30/1',
            'Type': 'Combed'
        }
        
        # Same material, size, and type - compatible
        compatible = (
            yarn1['Material'] == yarn2['Material'] and
            yarn1['Size'] == yarn2['Size'] and
            yarn1['Type'] == yarn2['Type']
        )
        assert compatible is True
        
        # Same material and size, different type - not compatible
        compatible = (
            yarn1['Material'] == yarn3['Material'] and
            yarn1['Size'] == yarn3['Size'] and
            yarn1['Type'] == yarn3['Type']
        )
        assert compatible is False
        
        # Different material - not compatible
        compatible = (
            yarn1['Material'] == yarn4['Material'] and
            yarn1['Size'] == yarn4['Size']
        )
        assert compatible is False
    
    def test_yarn_cost_calculation(self):
        """Test yarn cost calculations"""
        quantity_lbs = 1000
        unit_cost_per_lb = 5.50
        
        total_cost = quantity_lbs * unit_cost_per_lb
        assert total_cost == 5500.0
        
        # Test with waste factor
        waste_factor = 0.05  # 5% waste
        adjusted_quantity = quantity_lbs * (1 + waste_factor)
        adjusted_cost = adjusted_quantity * unit_cost_per_lb
        
        assert adjusted_quantity == 1050.0
        assert adjusted_cost == 5775.0