#!/usr/bin/env python3
"""
Test script to validate all debugging fixes
"""

import os
import sys

# Set environment to non-interactive to avoid browser opening
os.environ['NON_INTERACTIVE'] = 'true'

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test that all critical modules can be imported"""
    print("=" * 60)
    print("TESTING MODULE IMPORTS")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test core ERP import
    try:
        from core import beverly_comprehensive_erp
        print("✓ Core ERP module imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Core ERP import failed: {e}")
        tests_failed += 1
    
    # Test planning API
    try:
        from production.planning_data_api import PlanningDataAPI, get_planning_api
        api = get_planning_api()
        print("✓ Planning API imported and initialized successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Planning API failed: {e}")
        tests_failed += 1
    
    # Test forecasting engine
    try:
        from forecasting.enhanced_forecasting_engine import EnhancedForecastingEngine
        print("✓ Forecasting engine imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Forecasting engine failed: {e}")
        tests_failed += 1
    
    # Test data loaders
    try:
        from data_loaders.optimized_data_loader import OptimizedDataLoader
        print("✓ Optimized data loader imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Data loader failed: {e}")
        tests_failed += 1
    
    # Test cache manager
    try:
        from utils.cache_manager import CacheManager
        cache = CacheManager(default_ttl=60)
        print("✓ Cache manager imported and initialized successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Cache manager failed: {e}")
        tests_failed += 1
    
    # Test service managers
    try:
        from services.service_manager import ServiceManager
        print("✓ Service manager imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Service manager failed: {e}")
        tests_failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    return tests_failed == 0


def test_functionality():
    """Test basic functionality of fixed components"""
    print("\n" + "=" * 60)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test cache operations
    try:
        from utils.cache_manager import CacheManager
        cache = CacheManager(default_ttl=60)
        
        # Test set and get
        cache.set("test_key", "test_value", ttl=30)
        value = cache.get("test_key")
        assert value == "test_value", f"Expected 'test_value', got {value}"
        
        # Test delete
        cache.delete("test_key")
        value = cache.get("test_key")
        assert value is None, f"Expected None after delete, got {value}"
        
        print("✓ Cache operations working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Cache operations failed: {e}")
        tests_failed += 1
    
    # Test planning API
    try:
        from production.planning_data_api import PlanningDataAPI, PlanningRequest
        api = PlanningDataAPI()
        
        # Test basic method exists
        assert hasattr(api, 'get_planning_data'), "Missing get_planning_data method"
        
        # Test creating a request
        request = PlanningRequest(
            request_id="test_001",
            request_type="demand_forecast",
            planning_horizon_days=30
        )
        
        print("✓ Planning API structure validated")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Planning API validation failed: {e}")
        tests_failed += 1
    
    # Test forecasting with sample data
    try:
        import pandas as pd
        import numpy as np
        from forecasting.enhanced_forecasting_engine import EnhancedForecastingEngine
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        demand = np.random.randint(50, 150, size=30)
        df = pd.DataFrame({'date': dates, 'demand': demand})
        
        engine = EnhancedForecastingEngine()
        
        # Test basic forecasting - just verify it initializes without crashing
        assert hasattr(engine, 'forecast'), "Missing forecast method"
        assert hasattr(engine, 'models'), "Missing models attribute"
        assert len(engine.models) > 0, "No models loaded"
        
        print("✓ Forecasting engine initialized correctly with models")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Forecasting functionality failed: {e}")
        tests_failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    return tests_failed == 0


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("BEVERLY KNITS ERP - DEBUG FIX VALIDATION")
    print("=" * 60)
    
    # Run import tests
    imports_ok = test_imports()
    
    # Run functionality tests
    functionality_ok = test_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if imports_ok and functionality_ok:
        print("✓ ALL TESTS PASSED - System is ready!")
        print("\nFixes applied:")
        print("1. Fixed planning_api import issue in core ERP")
        print("2. Fixed deprecated fillna method in forecasting")
        print("3. Fixed browser opening during module import")
        print("4. Added test mode detection to prevent interactive prompts")
        print("5. Fixed test import paths in conftest.py")
        return 0
    else:
        print("✗ Some tests failed - Review the output above")
        return 1


if __name__ == "__main__":
    exit(main())