#!/usr/bin/env python3
"""
Test ERP Server and Endpoints
"""

import requests
import time
import subprocess
import sys
import os
from pathlib import Path

print("="*60)
print(" Testing ERP Server")
print("="*60)

# Check if server is already running
print("\n1. Checking if server is already running...")
try:
    response = requests.get("http://localhost:5006/api/health", timeout=2)
    if response.status_code == 200:
        print("   Server is already running on port 5006")
        server_running = True
    else:
        print(f"   Server responded with status {response.status_code}")
        server_running = True
except:
    print("   Server is not running")
    server_running = False

if not server_running:
    print("\n2. Starting ERP server...")
    print("   Please start the server manually with:")
    print("   py src/core/beverly_comprehensive_erp.py")
    print("\n   Then run this script again to test the endpoints.")
    sys.exit(1)

# Test endpoints
print("\n2. Testing endpoints...")

endpoints_to_test = [
    "/api/health",
    "/api/inventory-intelligence-enhanced?realtime=true",
    "/api/advanced-optimization",
    "/api/yarn-intelligence",
    "/api/comprehensive-kpis"
]

results = []

for endpoint in endpoints_to_test:
    url = f"http://localhost:5006{endpoint}"
    print(f"\n   Testing: {endpoint}")
    
    try:
        response = requests.get(url, timeout=10)
        status = response.status_code
        
        if status == 200:
            print(f"      [OK] Status: {status}")
            
            # Check response content
            try:
                data = response.json()
                if isinstance(data, dict):
                    if 'error' in data:
                        print(f"      Warning: Error in response: {data['error']}")
                        results.append((endpoint, "ERROR", data['error']))
                    else:
                        keys = list(data.keys())[:5]
                        print(f"      Response keys: {keys}")
                        results.append((endpoint, "OK", None))
                else:
                    print(f"      Response type: {type(data)}")
                    results.append((endpoint, "OK", None))
            except:
                print(f"      Response not JSON")
                results.append((endpoint, "OK", None))
        else:
            print(f"      [FAIL] Status: {status}")
            
            # Try to get error message
            try:
                error_data = response.json()
                if 'error' in error_data:
                    print(f"      Error: {error_data['error']}")
                    results.append((endpoint, "FAIL", error_data['error']))
                else:
                    results.append((endpoint, "FAIL", f"Status {status}"))
            except:
                results.append((endpoint, "FAIL", f"Status {status}"))
                
    except requests.exceptions.Timeout:
        print(f"      [FAIL] Timeout")
        results.append((endpoint, "TIMEOUT", None))
    except Exception as e:
        print(f"      [FAIL] Error: {str(e)[:50]}")
        results.append((endpoint, "ERROR", str(e)))

print("\n" + "="*60)
print(" Test Summary")
print("="*60)

ok_count = sum(1 for _, status, _ in results if status == "OK")
fail_count = sum(1 for _, status, _ in results if status in ["FAIL", "ERROR", "TIMEOUT"])

print(f"\nPassed: {ok_count}/{len(results)}")
print(f"Failed: {fail_count}/{len(results)}")

print("\nDetailed Results:")
for endpoint, status, error in results:
    if status == "OK":
        print(f"  [OK] {endpoint}")
    else:
        print(f"  [{status}] {endpoint}")
        if error:
            print(f"       {error[:80]}")

if fail_count > 0:
    print("\n" + "="*60)
    print(" Troubleshooting Tips")
    print("="*60)
    print("\n1. Check if data files exist:")
    print("   - yarn_inventory.xlsx in data/production/5/ERP Data/")
    print("   - BOM_updated.csv")
    print("   - Sales Activity Report.csv")
    print("\n2. Check server console for error messages")
    print("\n3. Try restarting the server:")
    print("   - Kill existing: Ctrl+C or close terminal")
    print("   - Start again: py src/core/beverly_comprehensive_erp.py")
    print("\n4. Check .env file has correct DATA_BASE_PATH")

print("\n" + "="*60)