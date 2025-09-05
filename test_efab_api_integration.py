#!/usr/bin/env python3
"""
End-to-End Test Script for eFab.ai API Integration
Run this script to test the complete API integration pipeline
"""

import asyncio
import sys
import os
import logging
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.config.secure_api_config import get_api_config
from src.api_clients.efab_api_client import EFabAPIClient
from src.api_clients.efab_auth_manager import EFabAuthManager
from src.api_clients.efab_transformers import EFabDataTransformer
from src.data_loaders.efab_api_loader import EFabAPIDataLoader
from src.monitoring.api_monitor import get_monitor


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_status(test_name: str, passed: bool, details: str = ""):
    """Print test status"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {test_name}: {status}")
    if details:
        print(f"    {details}")


async def test_configuration():
    """Test configuration loading"""
    print_section("1. Configuration Test")
    
    try:
        config_manager = get_api_config()
        config = config_manager.get_credentials()
        
        # Check configuration
        has_url = bool(config.get('base_url'))
        has_username = bool(config.get('username'))
        has_password = bool(config.get('password'))
        api_enabled = config.get('api_enabled', False)
        
        print_status("Base URL configured", has_url, config.get('base_url', 'Not set'))
        print_status("Username configured", has_username, 
                    config.get('username', 'Not set')[:3] + '***' if has_username else 'Not set')
        print_status("Password configured", has_password, 
                    '***' if has_password else 'Not set')
        print_status("API enabled", api_enabled)
        
        # Check feature flags
        flags = config_manager.get_feature_flags()
        print("\n  Feature Flags:")
        for flag, value in flags.items():
            print(f"    {flag}: {value}")
        
        return api_enabled
        
    except Exception as e:
        print_status("Configuration loading", False, str(e))
        return False


async def test_authentication():
    """Test authentication with eFab.ai"""
    print_section("2. Authentication Test")
    
    try:
        config = get_api_config().get_credentials()
        
        if not config.get('api_enabled'):
            print("  ⚠️  API not enabled (missing credentials)")
            return False
        
        auth_manager = EFabAuthManager(config)
        
        # Test authentication
        print("  Attempting authentication...")
        await auth_manager.initialize()
        token = await auth_manager.authenticate()
        
        print_status("Authentication successful", bool(token))
        
        # Check session info
        session_info = auth_manager.get_session_info()
        if session_info:
            print(f"    User: {session_info.get('username')}")
            print(f"    Session expires: {session_info.get('expires_at')}")
            print(f"    Time remaining: {session_info.get('time_remaining', 0):.0f} seconds")
        
        # Cleanup
        await auth_manager.cleanup()
        
        return bool(token)
        
    except Exception as e:
        print_status("Authentication", False, str(e))
        return False


async def test_api_health():
    """Test API health check"""
    print_section("3. API Health Check")
    
    try:
        config = get_api_config().get_credentials()
        
        if not config.get('api_enabled'):
            print("  ⚠️  API not enabled")
            return False
        
        async with EFabAPIClient(config) as client:
            # Test health endpoint
            is_healthy = await client.health_check()
            print_status("API health check", is_healthy)
            
            # Get client status
            status = client.get_status()
            print("\n  Client Status:")
            print(f"    Authenticated: {status.get('authenticated')}")
            print(f"    Circuit Breaker: {status['circuit_breaker']['state']}")
            
            if status.get('cache'):
                print(f"    Cache entries: {status['cache']['total_entries']}")
            
            return is_healthy
            
    except Exception as e:
        print_status("Health check", False, str(e))
        return False


async def test_data_loading():
    """Test loading data from API"""
    print_section("4. Data Loading Test")
    
    results = {}
    
    try:
        config = get_api_config().get_credentials()
        
        if not config.get('api_enabled'):
            print("  ⚠️  API not enabled")
            return results
        
        async with EFabAPIClient(config) as client:
            # Test individual endpoints
            endpoints = [
                ('Yarn Inventory', client.get_yarn_active),
                ('Knit Orders', client.get_knit_orders),
                ('Yarn PO', client.get_yarn_po),
                ('Yarn Expected', client.get_yarn_expected),
                ('Sales Activity', client.get_sales_activity),
                ('Yarn Demand', client.get_yarn_demand),
                ('Yarn Demand KO', client.get_yarn_demand_ko)
            ]
            
            for name, func in endpoints:
                try:
                    df = await func()
                    success = not df.empty
                    results[name] = df
                    print_status(name, success, f"{len(df)} records" if success else "No data")
                except Exception as e:
                    print_status(name, False, str(e)[:50])
                    results[name] = pd.DataFrame()
            
            return results
            
    except Exception as e:
        print_status("Data loading", False, str(e))
        return results


async def test_data_transformation():
    """Test data transformation"""
    print_section("5. Data Transformation Test")
    
    try:
        # Create sample data
        sample_data = {
            'data': [
                {
                    'yarn_id': '18884',
                    'description': '100% COTTON 30/1 ROYAL BLUE',
                    'theoretical_balance': '2,506.18',
                    'allocated': '-30,859.80',
                    'on_order': '36,161.30'
                }
            ]
        }
        
        transformer = EFabDataTransformer()
        
        # Test yarn transformation
        yarn_df = transformer.transform_yarn_active(sample_data)
        has_planning_balance = 'Planning Balance' in yarn_df.columns
        has_desc = 'Desc#' in yarn_df.columns
        
        print_status("Yarn transformation", not yarn_df.empty)
        print_status("Planning Balance calculated", has_planning_balance)
        print_status("Field mapping applied", has_desc)
        
        if not yarn_df.empty:
            print(f"    Sample Planning Balance: {yarn_df['Planning Balance'].iloc[0]:.2f}")
        
        return True
        
    except Exception as e:
        print_status("Data transformation", False, str(e))
        return False


async def test_api_loader():
    """Test API data loader with fallback"""
    print_section("6. API Data Loader Test")
    
    try:
        loader = EFabAPIDataLoader()
        
        # Get loader status
        status = loader.get_loader_status()
        print(f"  API Enabled: {status['api_enabled']}")
        print(f"  API Available: {status['api_available']}")
        print(f"  Fallback Enabled: {status['fallback_enabled']}")
        
        # Test loading with API or fallback
        print("\n  Testing data loading...")
        
        # Test yarn inventory
        yarn_df = loader.load_yarn_inventory()
        print_status("Yarn inventory loading", not yarn_df.empty, 
                    f"{len(yarn_df)} records" if not yarn_df.empty else "")
        
        # Test knit orders
        orders_df = loader.load_knit_orders()
        print_status("Knit orders loading", not orders_df.empty,
                    f"{len(orders_df)} records" if not orders_df.empty else "")
        
        return True
        
    except Exception as e:
        print_status("API loader", False, str(e))
        return False


async def test_monitoring():
    """Test monitoring functionality"""
    print_section("7. Monitoring Test")
    
    try:
        monitor = get_monitor()
        
        # Record some test metrics
        monitor.record_api_call('/api/test', 0.5, True, 200)
        monitor.record_cache_access('/api/test', True)
        monitor.record_auth_event(True, 'test_user')
        
        # Get statistics
        stats = monitor.get_statistics()
        health = monitor.get_health_status()
        
        print(f"  Health Status: {health['status']}")
        print(f"  Health Score: {health['health_score']}/100")
        print(f"  Total API Calls: {stats['total_api_calls']}")
        print(f"  API Success Rate: {stats['api_success_rate']:.1f}%")
        print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.1f}%")
        
        if health['issues']:
            print("\n  Issues detected:")
            for issue in health['issues']:
                print(f"    - {issue}")
        
        return True
        
    except Exception as e:
        print_status("Monitoring", False, str(e))
        return False


async def test_parallel_loading():
    """Test parallel data loading"""
    print_section("8. Parallel Loading Test")
    
    try:
        config = get_api_config().get_credentials()
        
        if not config.get('api_enabled'):
            print("  ⚠️  API not enabled, skipping parallel test")
            return False
        
        async with EFabAPIClient(config) as client:
            import time
            
            print("  Starting parallel data load...")
            start_time = time.time()
            
            all_data = await client.get_all_data_parallel()
            
            elapsed = time.time() - start_time
            
            print(f"  Completed in {elapsed:.2f} seconds")
            print(f"  Loaded {len(all_data)} datasets:")
            
            for name, df in all_data.items():
                print(f"    {name}: {len(df)} records")
            
            return True
            
    except Exception as e:
        print_status("Parallel loading", False, str(e))
        return False


async def main():
    """Run all tests"""
    print("\n" + "🔧" * 30)
    print(" eFab.ai API Integration Test Suite")
    print("🔧" * 30)
    print(f"\nTest started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track test results
    results = {
        'configuration': await test_configuration(),
        'authentication': await test_authentication(),
        'health': await test_api_health(),
        'transformation': await test_data_transformation(),
        'loader': await test_api_loader(),
        'monitoring': await test_monitoring()
    }
    
    # Test data loading if API is available
    if results['authentication']:
        data = await test_data_loading()
        results['data_loading'] = bool(data)
        results['parallel'] = await test_parallel_loading()
    
    # Summary
    print_section("Test Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print(f"\n  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {total_tests - passed_tests}")
    print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\n  Test Results:")
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"    {status} {test_name}")
    
    # Save test results
    try:
        results_file = Path('test_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': {k: v for k, v in results.items()},
                'summary': {
                    'total': total_tests,
                    'passed': passed_tests,
                    'failed': total_tests - passed_tests,
                    'success_rate': (passed_tests/total_tests)*100
                }
            }, f, indent=2)
        print(f"\n  Results saved to: {results_file}")
    except Exception as e:
        print(f"\n  Failed to save results: {e}")
    
    print("\n" + "=" * 60)
    
    # Return exit code
    return 0 if passed_tests == total_tests else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)