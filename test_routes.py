#!/usr/bin/env python3
"""Test script to verify Flask routes are accessible"""

import requests
import sys

# Base URL
BASE_URL = "http://localhost:5006"

# Routes to test
routes = [
    "/",
    "/test-early",
    "/api/sales",
    "/api/inventory-overview",
    "/hello"
]

print(f"Testing Flask routes at {BASE_URL}")
print("-" * 50)

for route in routes:
    url = BASE_URL + route
    try:
        response = requests.get(url, timeout=5)
        print(f"[OK] {route} - Status: {response.status_code}")
        if response.status_code == 200 and len(response.text) < 200:
            print(f"  Response: {response.text[:100]}...")
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] {route} - Connection refused (Is the Flask app running?)")
    except requests.exceptions.Timeout:
        print(f"[FAIL] {route} - Timeout")
    except Exception as e:
        print(f"[FAIL] {route} - Error: {str(e)}")
    print()

print("\nIf you see 'Connection refused', make sure to:")
print("1. Stop the current Flask app (Ctrl+C)")
print("2. Restart it: python src/core/beverly_comprehensive_erp.py")