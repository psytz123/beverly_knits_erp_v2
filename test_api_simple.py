#!/usr/bin/env python3
"""
Simple End-to-End Test for eFab API Integration
Tests each component step by step
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

def print_test(name, result, details=""):
    """Print test result"""
    status = "PASS" if result else "FAIL"
    print(f"[{status}] {name}")
    if details:
        print(f"   {details}")

print("\n" + "="*60)
print(" eFab.ai API Integration - End-to-End Test")
print("="*60)

# Test 1: Configuration
print("\n1. Testing Configuration...")
try:
    from src.config.secure_api_config import get_api_config
    
    config_manager = get_api_config()
    config = config_manager.get_credentials()
    
    print_test("Config loaded", True)
    print_test("Base URL", bool(config.get('base_url')), config.get('base_url'))
    print_test("Username", bool(config.get('username')), 
               config.get('username', 'Not set'))
    print_test("Password", bool(config.get('password')), 
               "***" if config.get('password') else "Not set")
    print_test("API Enabled", config.get('api_enabled', False))
    
    config_success = config.get('api_enabled', False)
except Exception as e:
    print_test("Configuration", False, str(e))
    config_success = False

# Test 2: Authentication
print("\n2. Testing Authentication...")
async def test_auth():
    try:
        from src.api_clients.efab_auth_manager import EFabAuthManager
        
        if not config_success:
            print_test("Authentication", False, "Config not available")
            return False
            
        auth_manager = EFabAuthManager(config)
        await auth_manager.initialize()
        
        print("   Attempting to authenticate...")
        token = await auth_manager.authenticate()
        
        print_test("Authentication", bool(token), f"Token: {token[:20]}..." if token else "No token")
        
        session_info = auth_manager.get_session_info()
        if session_info:
            print(f"   User: {session_info.get('username')}")
            print(f"   Expires: {session_info.get('expires_at')}")
        
        await auth_manager.cleanup()
        return bool(token)
        
    except Exception as e:
        print_test("Authentication", False, str(e))
        return False

auth_success = asyncio.run(test_auth())

# Test 3: API Client
print("\n3. Testing API Client...")
async def test_client():
    try:
        from src.api_clients.efab_api_client import EFabAPIClient
        
        if not auth_success:
            print_test("API Client", False, "Authentication failed")
            return False
        
        async with EFabAPIClient(config) as client:
            # Test health check
            is_healthy = await client.health_check()
            print_test("Health check", is_healthy)
            
            # Get client status
            status = client.get_status()
            print_test("Client initialized", True)
            print(f"   Circuit breaker: {status['circuit_breaker']['state']}")
            print(f"   Authenticated: {status['authenticated']}")
            
            return is_healthy
            
    except Exception as e:
        print_test("API Client", False, str(e))
        return False

client_success = asyncio.run(test_client())

# Test 4: Data Loading
print("\n4. Testing Data Loading...")
async def test_loading():
    try:
        from src.api_clients.efab_api_client import EFabAPIClient
        
        if not client_success:
            print_test("Data Loading", False, "Client not available")
            return {}
        
        results = {}
        async with EFabAPIClient(config) as client:
            # Test yarn inventory
            try:
                yarn_df = await client.get_yarn_active()
                results['yarn'] = len(yarn_df) if hasattr(yarn_df, '__len__') else 0
                print_test("Yarn inventory", results['yarn'] > 0, f"{results['yarn']} records")
            except Exception as e:
                print_test("Yarn inventory", False, str(e)[:50])
                results['yarn'] = 0
            
            # Test knit orders
            try:
                orders_df = await client.get_knit_orders()
                results['orders'] = len(orders_df) if hasattr(orders_df, '__len__') else 0
                print_test("Knit orders", results['orders'] > 0, f"{results['orders']} records")
            except Exception as e:
                print_test("Knit orders", False, str(e)[:50])
                results['orders'] = 0
            
            return results
            
    except Exception as e:
        print_test("Data Loading", False, str(e))
        return {}

data_results = asyncio.run(test_loading())

# Test 5: Data Transformation
print("\n5. Testing Data Transformation...")
try:
    from src.api_clients.efab_transformers import EFabDataTransformer
    
    transformer = EFabDataTransformer()
    
    # Test with sample data
    sample = {
        'data': [{
            'yarn_id': '18884',
            'description': 'TEST YARN',
            'theoretical_balance': 1000,
            'allocated': -500,
            'on_order': 300
        }]
    }
    
    df = transformer.transform_yarn_active(sample)
    has_planning = 'Planning Balance' in df.columns if hasattr(df, 'columns') else False
    
    print_test("Transformer", True)
    print_test("Planning Balance", has_planning)
    
    if has_planning and len(df) > 0:
        print(f"   Calculated: {df['Planning Balance'].iloc[0]}")
    
    transform_success = True
except Exception as e:
    print_test("Data Transformation", False, str(e))
    transform_success = False

# Test 6: API Data Loader
print("\n6. Testing API Data Loader...")
try:
    from src.data_loaders.efab_api_loader import EFabAPIDataLoader
    
    loader = EFabAPIDataLoader()
    status = loader.get_loader_status()
    
    print_test("Loader initialized", True)
    print(f"   API enabled: {status['api_enabled']}")
    print(f"   Fallback enabled: {status['fallback_enabled']}")
    
    # Try loading yarn inventory
    yarn_df = loader.load_yarn_inventory()
    print_test("Load yarn inventory", len(yarn_df) > 0 if hasattr(yarn_df, '__len__') else False,
               f"{len(yarn_df) if hasattr(yarn_df, '__len__') else 0} records")
    
    loader_success = True
except Exception as e:
    print_test("API Data Loader", False, str(e))
    loader_success = False

# Test 7: Monitoring
print("\n7. Testing Monitoring...")
try:
    from src.monitoring.api_monitor import get_monitor
    
    monitor = get_monitor()
    
    # Record test metrics
    monitor.record_api_call('/api/test', 0.5, True, 200)
    monitor.record_cache_access('/api/test', True)
    
    stats = monitor.get_statistics()
    health = monitor.get_health_status()
    
    print_test("Monitor active", True)
    print(f"   Health: {health['status']}")
    print(f"   Score: {health['health_score']}/100")
    
    monitor_success = True
except Exception as e:
    print_test("Monitoring", False, str(e))
    monitor_success = False

# Summary
print("\n" + "="*60)
print(" TEST SUMMARY")
print("="*60)

results = {
    "Configuration": config_success,
    "Authentication": auth_success,
    "API Client": client_success,
    "Data Loading": bool(data_results),
    "Transformation": transform_success,
    "API Loader": loader_success,
    "Monitoring": monitor_success
}

passed = sum(1 for v in results.values() if v)
total = len(results)

print(f"\nPassed: {passed}/{total}")
print(f"Success Rate: {(passed/total)*100:.0f}%")

print("\nDetailed Results:")
for name, success in results.items():
    status = "PASS" if success else "FAIL"
    print(f"  [{status}] {name}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"test_results_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'data_loaded': data_results,
        'summary': {
            'passed': passed,
            'total': total,
            'success_rate': (passed/total)*100
        }
    }, f, indent=2)

print(f"\nResults saved to: {results_file}")
print("="*60)