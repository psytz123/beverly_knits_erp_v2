#!/usr/bin/env python3
"""Test API endpoints like a browser."""
import requests

url = "http://localhost:5006/api/comprehensive-kpis"
print(f"Testing: {url}")

try:
    response = requests.get(url, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Headers: {response.headers.get('Content-Type')}")
    print(f"Body (first 200 chars): {response.text[:200]}")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ SUCCESS - Got {len(data)} fields")
        print(f"Data source: {data.get('source', 'unknown')}")
    else:
        print(f"❌ FAILED - Status {response.status_code}")

except Exception as e:
    print(f"❌ ERROR: {e}")
