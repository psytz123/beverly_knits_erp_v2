#!/usr/bin/env python3
"""
Test script to verify column standardization is working correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from pathlib import Path
from src.utils.column_standardization import ColumnStandardizer
from src.data_loaders.unified_data_loader import UnifiedDataLoader

def test_column_standardization():
    """Test that column standardization is working"""
    print("=" * 80)
    print("TESTING COLUMN STANDARDIZATION")
    print("=" * 80)
    
    # Initialize standardizer
    standardizer = ColumnStandardizer()
    
    # Test 1: Test common column variations
    print("\n1. Testing column variation mappings:")
    test_variations = [
        (['Desc#', 'Yarn', 'Yarn_ID'], 'Desc#'),
        (['Style#', 'Style #', 'style'], 'Style#'),
        (['Planning Balance', 'Planning_Balance', 'Planning_Ballance'], 'Planning Balance'),
        (['BOM_Percent', 'BOM_Percentage'], 'BOM_Percent'),
        (['On Order', 'On_Order'], 'On Order'),
    ]
    
    for variations, expected in test_variations:
        for var in variations:
            result = standardizer.get_standard_name(var)
            print(f"  {var:20} -> {result:20} {'✓' if result == expected else '✗'}")
    
    # Test 2: Test file loading with actual data
    print("\n2. Testing actual data file loading:")
    
    data_paths = [
        "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data",
        "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5"
    ]
    
    test_files = [
        ("yarn_inventory.xlsx", "yarn_inventory"),
        ("Style_BOM.csv", "bom"),
        ("BOM_updated.csv", "bom"),
        ("eFab_Knit_Orders.xlsx", "knit_orders"),
        ("Yarn_ID.csv", "yarn_id"),
    ]
    
    for file_name, file_type in test_files:
        print(f"\n  Testing {file_name}:")
        
        # Try to find and load the file
        file_found = False
        for data_path in data_paths:
            file_path = Path(data_path) / file_name
            if not file_path.exists():
                # Also check in subdirectories
                file_path = Path(data_path) / "ERP Data" / file_name
            
            if file_path.exists():
                file_found = True
                print(f"    Found at: {file_path}")
                
                # Load file
                if file_name.endswith('.xlsx'):
                    df = pd.read_excel(file_path, nrows=5)
                else:
                    df = pd.read_csv(file_path, nrows=5)
                
                print(f"    Original columns: {list(df.columns)[:5]}...")
                
                # Apply standardization
                df_standard = standardizer.standardize_dataframe(df, file_type)
                print(f"    Standardized columns: {list(df_standard.columns)[:5]}...")
                
                # Check for critical columns
                if file_type == 'yarn_inventory':
                    critical_cols = ['Desc#', 'Planning Balance']
                elif file_type == 'bom':
                    critical_cols = ['Style#', 'Desc#', 'BOM_Percent']
                elif file_type == 'knit_orders':
                    critical_cols = ['Style#']
                elif file_type == 'yarn_id':
                    critical_cols = ['Desc#', 'Planning Balance']
                else:
                    critical_cols = []
                
                for col in critical_cols:
                    if col in df_standard.columns:
                        print(f"      ✓ {col} found")
                    else:
                        # Check if any variation exists
                        found_var = standardizer.find_column(df_standard, [col])
                        if found_var:
                            print(f"      ⚠ {col} mapped to {found_var}")
                        else:
                            print(f"      ✗ {col} missing!")
                
                break
        
        if not file_found:
            print(f"    ✗ File not found in any data path")
    
    # Test 3: Test with UnifiedDataLoader
    print("\n3. Testing UnifiedDataLoader integration:")
    try:
        loader = UnifiedDataLoader()
        print("  ✓ UnifiedDataLoader initialized")
        
        # Try loading yarn inventory
        yarn_inv = loader.load_yarn_inventory()
        if yarn_inv is not None:
            print(f"  ✓ Loaded yarn inventory: {len(yarn_inv)} rows")
            print(f"    Columns: {list(yarn_inv.columns)[:5]}...")
            
            # Check for standardized columns
            desc_col = standardizer.find_column(yarn_inv, ['Desc#'])
            planning_col = standardizer.find_column(yarn_inv, ['Planning Balance', 'Planning_Balance', 'Planning_Ballance'])
            
            if desc_col:
                print(f"    ✓ Yarn ID column found as: {desc_col}")
            else:
                print(f"    ✗ Yarn ID column not found")
            
            if planning_col:
                print(f"    ✓ Planning Balance column found as: {planning_col}")
            else:
                print(f"    ✗ Planning Balance column not found")
        else:
            print("  ✗ Failed to load yarn inventory")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
    
    print("\n" + "=" * 80)
    print("COLUMN STANDARDIZATION TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_column_standardization()