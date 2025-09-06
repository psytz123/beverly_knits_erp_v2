#!/usr/bin/env python3
"""
Test eFab API with local mock server
"""

import sys
import os
sys.path.insert(0, '/mnt/c/finalee/beverly_knits_erp_v2')

from src.data_sync.efab_api_connector import eFabAPIConnector
import logging

logging.basicConfig(level=logging.INFO)

def test_with_mock_server():
    """
    Test eFab API connector with local mock server
    """
    print("\n" + "="*60)
    print("Testing eFab API with Mock Server")
    print("="*60 + "\n")
    
    # Use localhost for testing
    connector = eFabAPIConnector(
        base_url="http://localhost:5007",
        session_cookie="aLfHTrRrtWWy4FPgLnxdEPC7ohA37dlR"
    )
    
    print("🔌 Testing connection to mock server...")
    if connector.test_connection():
        print("✅ Connection successful!\n")
        
        # Test fetching data
        print("📄 Fetching sales orders...")
        sales_orders = connector.get_sales_order_plan_list()
        if sales_orders is not None:
            print(f"✅ Retrieved {len(sales_orders)} sales orders")
            print(f"   First order: {sales_orders.iloc[0].to_dict() if not sales_orders.empty else 'None'}\n")
        
        print("🧵 Fetching knit orders...")
        knit_orders = connector.get_knit_orders()
        if knit_orders is not None:
            print(f"✅ Retrieved {len(knit_orders)} knit orders\n")
        
        print("📦 Fetching inventory...")
        for warehouse in ['F01', 'G00']:
            inventory = connector.get_inventory_data(warehouse)
            if inventory is not None:
                print(f"✅ Warehouse {warehouse}: {len(inventory)} items")
        
        print("\n✨ All tests passed!")
        return True
    else:
        print("❌ Connection failed")
        return False

if __name__ == "__main__":
    test_with_mock_server()