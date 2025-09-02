#!/usr/bin/env python3
"""
Integration Test Script for Beverly Knits ERP
Tests the integration of PostgreSQL database with file-based system
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
from pathlib import Path
from datetime import datetime


def test_configuration():
    """Test unified configuration loading"""
    print("=" * 60)
    print("Testing Configuration Loading...")
    print("-" * 60)
    
    config_path = Path('src/config/unified_config.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✓ Unified configuration loaded successfully")
        print(f"  Primary source: {config['data_source']['primary']}")
        print(f"  Database host: {config['data_source']['database']['host']}")
        print(f"  File path: {config['data_source']['files']['primary_path']}")
    else:
        print("✗ Configuration file not found")
        return False
    
    return True


def test_data_paths():
    """Test data path configuration"""
    print("\n" + "=" * 60)
    print("Testing Data Path Configuration...")
    print("-" * 60)
    
    try:
        from utils.data_config import DATA_BASE_DIR, DATA_FILES
        print(f"✓ Data configuration loaded")
        print(f"  Base directory: {DATA_BASE_DIR}")
        
        # Check if directory exists
        if os.path.exists(DATA_BASE_DIR):
            print(f"  ✓ Directory exists")
            files = os.listdir(DATA_BASE_DIR)[:5]  # Show first 5 files
            if files:
                print(f"  Sample files: {files}")
        else:
            print(f"  ✗ Directory does not exist")
            
    except Exception as e:
        print(f"✗ Error loading data config: {e}")
        return False
    
    return True


def test_database_loader():
    """Test database loader functionality"""
    print("\n" + "=" * 60)
    print("Testing Database Loader...")
    print("-" * 60)
    
    try:
        from data_loaders.database_loader import DatabaseDataLoader
        
        loader = DatabaseDataLoader()
        print("✓ Database loader initialized")
        
        # Test connection
        loader.connect()
        status = loader.get_data_source_status()
        
        print(f"  Primary source: {status['primary_source']}")
        print(f"  Database connected: {status['database_connected']}")
        
        if status['database_connected']:
            print("  ✓ Database connection successful")
            print(f"  Database records: {status.get('database_records', 0)}")
        else:
            print("  ⚠ Database not connected, will use file fallback")
        
        loader.disconnect()
        return True
        
    except Exception as e:
        print(f"✗ Error with database loader: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_server():
    """Test if API server can be imported"""
    print("\n" + "=" * 60)
    print("Testing API Server Module...")
    print("-" * 60)
    
    try:
        from api.database_api_server import app, get_db_connection
        print("✓ API server module loaded successfully")
        print("  Available endpoints will be on port 5007")
        return True
    except Exception as e:
        print(f"✗ Error loading API server: {e}")
        return False


def test_etl_pipeline():
    """Test if ETL pipeline can be imported"""
    print("\n" + "=" * 60)
    print("Testing ETL Pipeline Module...")
    print("-" * 60)
    
    try:
        from data_sync.database_etl_pipeline import BeverlyKnitsETL
        print("✓ ETL pipeline module loaded successfully")
        
        etl = BeverlyKnitsETL()
        print(f"  Data path: {etl.data_path}")
        print(f"  Database: {etl.config['database']}")
        return True
    except Exception as e:
        print(f"✗ Error loading ETL pipeline: {e}")
        return False


def test_data_loading():
    """Test actual data loading"""
    print("\n" + "=" * 60)
    print("Testing Data Loading...")
    print("-" * 60)
    
    try:
        from data_loaders.database_loader import DatabaseDataLoader
        
        loader = DatabaseDataLoader()
        
        # Try to load some data
        print("Attempting to load yarn inventory...")
        yarn_df = loader.load_yarn_inventory()
        if not yarn_df.empty:
            print(f"  ✓ Loaded {len(yarn_df)} yarn records")
        else:
            print("  ⚠ No yarn data loaded")
        
        print("Attempting to load sales orders...")
        sales_df = loader.load_sales_orders()
        if not sales_df.empty:
            print(f"  ✓ Loaded {len(sales_df)} sales order records")
        else:
            print("  ⚠ No sales order data loaded")
        
        loader.disconnect()
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False


def main():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("BEVERLY KNITS ERP INTEGRATION TEST")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Data Paths", test_data_paths),
        ("Database Loader", test_database_loader),
        ("API Server", test_api_server),
        ("ETL Pipeline", test_etl_pipeline),
        ("Data Loading", test_data_loading)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:20} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Integration successful.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)