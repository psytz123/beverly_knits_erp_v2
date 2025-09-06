#!/usr/bin/env python3
"""
Test script for the ERP wrapper service
Run the wrapper first: cd erp-wrapper && uvicorn app.main:app --port 8000
"""

import requests
import json
from datetime import datetime

WRAPPER_URL = "http://localhost:8000"

def test_wrapper():
    print("\n" + "="*60)
    print("Testing ERP Wrapper Service")
    print("="*60 + "\n")
    
    # Test 1: Root endpoint
    print("🏠 Testing root endpoint...")
    try:
        response = requests.get(f"{WRAPPER_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service: {data['service']} v{data['version']}")
            print(f"   Status: {data['status']}")
            print(f"   Endpoints: {len(data['endpoints'])} available\n")
        else:
            print(f"❌ Root endpoint returned {response.status_code}\n")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to wrapper service")
        print("⚠️  Make sure the wrapper is running:")
        print("   cd erp-wrapper")
        print("   uvicorn app.main:app --port 8000\n")
        return False
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False
    
    # Test 2: Health check
    print("🆗 Testing health endpoint...")
    try:
        response = requests.get(f"{WRAPPER_URL}/health")
        data = response.json()
        status_icon = "✅" if data['status'] == 'healthy' else "⚠️"
        print(f"{status_icon} Status: {data['status']}")
        print(f"   ERP Connected: {data.get('erp_connected', False)}\n")
    except Exception as e:
        print(f"❌ Health check failed: {e}\n")
    
    # Test 3: Service status
    print("📊 Testing service status...")
    try:
        response = requests.get(f"{WRAPPER_URL}/api/status")
        data = response.json()
        print(f"✅ Service running")
        print(f"   Session active: {data['session']['has_cookie']}")
        print(f"   Base URL: {data['config']['base_url']}")
        print(f"   Username: {data['config']['username']}\n")
    except Exception as e:
        print(f"❌ Status check failed: {e}\n")
    
    # Test 4: Sales orders (with cache)
    print("📄 Testing sales orders endpoint...")
    try:
        response = requests.get(f"{WRAPPER_URL}/api/sales-orders")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Sales orders: {data['count']} records")
            print(f"   Source: {data['source']}")
            if data['data'] and len(data['data']) > 0:
                print(f"   Sample: {data['data'][0].get('order_id', 'N/A')}\n")
        else:
            print(f"❌ Sales orders returned {response.status_code}")
            print(f"   Response: {response.text[:200]}\n")
    except Exception as e:
        print(f"❌ Sales orders failed: {e}\n")
    
    # Test 5: Inventory
    print("📦 Testing inventory endpoint...")
    try:
        response = requests.get(f"{WRAPPER_URL}/api/inventory/F01")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Inventory F01: {data['count']} items")
            print(f"   Source: {data['source']}\n")
        else:
            print(f"❌ Inventory returned {response.status_code}\n")
    except Exception as e:
        print(f"❌ Inventory failed: {e}\n")
    
    # Test 6: Cache management
    print("🗝️ Testing cache clear...")
    try:
        response = requests.post(f"{WRAPPER_URL}/api/cache/clear")
        if response.status_code == 200:
            print(f"✅ Cache cleared successfully\n")
        else:
            print(f"❌ Cache clear returned {response.status_code}\n")
    except Exception as e:
        print(f"❌ Cache clear failed: {e}\n")
    
    print("="*60)
    print("✨ Wrapper service test complete!")
    print("="*60 + "\n")
    
    print("💡 Next steps:")
    print("1. Check API docs: http://localhost:8000/docs")
    print("2. Update Beverly Knits ERP to use wrapper")
    print("3. Deploy with Docker for production")
    
    return True

if __name__ == "__main__":
    test_wrapper()