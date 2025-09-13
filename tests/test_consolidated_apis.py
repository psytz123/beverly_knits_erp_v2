#!/usr/bin/env python3
"""
Test script for consolidated API endpoints
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:5005"

def test_endpoint(url, params=None):
    """Test a single endpoint"""
    try:
        response = requests.get(url, params=params, timeout=5)
        return {
            'status_code': response.status_code,
            'success': response.status_code == 200,
            'data': response.json() if response.status_code == 200 else None,
            'error': None
        }
    except Exception as e:
        return {
            'status_code': None,
            'success': False,
            'data': None,
            'error': str(e)
        }

def main():
    """Test all consolidated endpoints"""
    print("=" * 60)
    print("Testing Consolidated API Endpoints")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Test consolidated endpoints
    endpoints = [
        {
            'name': 'Inventory Unified',
            'url': f'{BASE_URL}/api/inventory/unified',
            'params': {'type': 'all'}
        },
        {
            'name': 'Forecast Unified',
            'url': f'{BASE_URL}/api/forecast/unified',
            'params': {'type': 'all', 'horizon': 30}
        },
        {
            'name': 'Production Unified',
            'url': f'{BASE_URL}/api/production/unified',
            'params': {'type': 'all'}
        },
        {
            'name': 'Yarn Unified',
            'url': f'{BASE_URL}/api/yarn-intelligence',
            'params': {'type': 'all'}
        },
        {
            'name': 'Planning Unified',
            'url': f'{BASE_URL}/api/planning/unified',
            'params': {'type': 'all'}
        },
        {
            'name': 'System Unified',
            'url': f'{BASE_URL}/api/system/unified',
            'params': {'operation': 'status'}
        }
    ]
    
    results = []
    
    for endpoint in endpoints:
        print(f"\nTesting: {endpoint['name']}")
        print(f"URL: {endpoint['url']}")
        print(f"Params: {endpoint.get('params', {})}")
        
        result = test_endpoint(endpoint['url'], endpoint.get('params'))
        results.append({
            'endpoint': endpoint['name'],
            'result': result
        })
        
        if result['success']:
            print(f"✓ SUCCESS - Status: {result['status_code']}")
            if result['data'] and 'summary' in result['data']:
                print(f"  Summary: {result['data']['summary']}")
        else:
            print(f"✗ FAILED - Error: {result['error']}")
    
    # Test admin endpoints
    print("\n" + "=" * 60)
    print("Testing Admin Endpoints")
    print("=" * 60)
    
    admin_endpoints = [
        {
            'name': 'Deprecation Report',
            'url': f'{BASE_URL}/api/admin/deprecation-report'
        },
        {
            'name': 'Feature Flags',
            'url': f'{BASE_URL}/api/admin/feature-flags'
        }
    ]
    
    for endpoint in admin_endpoints:
        print(f"\nTesting: {endpoint['name']}")
        print(f"URL: {endpoint['url']}")
        
        result = test_endpoint(endpoint['url'])
        
        if result['success']:
            print(f"✓ SUCCESS - Status: {result['status_code']}")
            if result['data']:
                print(f"  Data: {json.dumps(result['data'], indent=2)[:200]}...")
        else:
            print(f"✗ FAILED - Error: {result['error']}")
    
    # Test old endpoints to verify redirect
    print("\n" + "=" * 60)
    print("Testing Deprecated Endpoint Redirects")
    print("=" * 60)
    
    deprecated = [
        {
            'name': 'Old Inventory Status',
            'old': f'{BASE_URL}/api/inventory-status',
            'new': f'{BASE_URL}/api/inventory/unified'
        },
        {
            'name': 'Old Yarn Intelligence',
            'old': f'{BASE_URL}/api/yarn-intelligence',
            'new': f'{BASE_URL}/api/yarn-intelligence'
        },
        {
            'name': 'Old ML Forecast',
            'old': f'{BASE_URL}/api/ml-forecast',
            'new': f'{BASE_URL}/api/forecast/unified'
        }
    ]
    
    for endpoint in deprecated:
        print(f"\nTesting: {endpoint['name']}")
        print(f"Old URL: {endpoint['old']}")
        print(f"Expected redirect to: {endpoint['new']}")
        
        try:
            response = requests.get(endpoint['old'], allow_redirects=False, timeout=5)
            if 'X-Deprecated-Endpoint' in response.headers:
                print(f"✓ Deprecation header found")
                print(f"  New endpoint: {response.headers.get('X-New-Endpoint')}")
            else:
                print(f"✗ No deprecation header")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['result']['success'])
    failed = len(results) - successful
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/len(results)*100):.1f}%")
    
    return successful == len(results)

if __name__ == "__main__":
    # First check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=2)
        print("Server is running, proceeding with tests...")
        success = main()
        exit(0 if success else 1)
    except:
        print("ERROR: Server is not running on port 5005")
        print("Please start the server first: python3 src/core/beverly_comprehensive_erp.py")
        exit(1)