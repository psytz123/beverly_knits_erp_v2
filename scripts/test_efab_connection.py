#!/usr/bin/env python3
"""
Test script for eFab API connection
"""

import sys
import os
sys.path.insert(0, '/mnt/c/finalee/beverly_knits_erp_v2')

from src.data_sync.efab_api_connector import eFabAPIConnector
import logging
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_efab_connection():
    """
    Test eFab API connection and data retrieval
    """
    print("\n" + "="*60)
    print("eFab API Connection Test")
    print("="*60 + "\n")
    
    # Initialize connector
    connector = eFabAPIConnector()
    
    # Test 1: Connection test
    print("🔌 Testing API connection...")
    if connector.test_connection():
        print("✅ Connection successful!\n")
    else:
        print("❌ Connection failed. Please check your session cookie.\n")
        return False
    
    # Test 2: Fetch sales orders
    print("📄 Fetching sales order plan list...")
    sales_orders = connector.get_sales_order_plan_list()
    if sales_orders is not None and not sales_orders.empty:
        print(f"✅ Retrieved {len(sales_orders)} sales orders")
        print(f"   Columns: {list(sales_orders.columns)[:5]}...")
        print(f"   Sample data:")
        print(sales_orders.head(3).to_string(max_cols=5))
    else:
        print("⚠️  No sales orders retrieved")
    
    print("\n" + "-"*40 + "\n")
    
    # Test 3: Fetch knit orders
    print("🧵 Fetching knit orders...")
    knit_orders = connector.get_knit_orders()
    if knit_orders is not None and not knit_orders.empty:
        print(f"✅ Retrieved {len(knit_orders)} knit orders")
        print(f"   Columns: {list(knit_orders.columns)[:5]}...")
    else:
        print("⚠️  No knit orders retrieved")
    
    print("\n" + "-"*40 + "\n")
    
    # Test 4: Fetch inventory
    print("📦 Fetching inventory data...")
    for warehouse in ['F01', 'G00', 'G02', 'I01']:
        inventory = connector.get_inventory_data(warehouse)
        if inventory is not None and not inventory.empty:
            print(f"✅ Warehouse {warehouse}: {len(inventory)} items")
        else:
            print(f"⚠️  Warehouse {warehouse}: No data")
    
    print("\n" + "="*60)
    print("🎆 All tests completed!")
    print("="*60 + "\n")
    
    return True


def sync_and_save():
    """
    Perform full data sync and save to CSV files
    """
    print("\n🔄 Starting full data synchronization...\n")
    
    connector = eFabAPIConnector()
    saved_files = connector.sync_all_data()
    
    if saved_files:
        print("\n✅ Data synchronization successful!")
        print("\nSaved files:")
        for key, path in saved_files.items():
            print(f"  📁 {key}: {os.path.basename(path)}")
    else:
        print("❌ No files were saved")
    
    return saved_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test eFab API connection')
    parser.add_argument('--sync', action='store_true', help='Perform full data sync')
    parser.add_argument('--test-only', action='store_true', help='Only test connection')
    
    args = parser.parse_args()
    
    if args.sync:
        sync_and_save()
    else:
        test_efab_connection()
        
        if not args.test_only:
            response = input("\n💾 Would you like to sync and save the data? (y/n): ")
            if response.lower() == 'y':
                sync_and_save()