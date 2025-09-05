#!/usr/bin/env python3
"""
Test eFab API Endpoints with Authentication
"""

import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get credentials from environment
EFAB_BASE_URL = os.getenv('EFAB_BASE_URL', 'https://efab.bkiapps.com')
EFAB_USERNAME = os.getenv('EFAB_USERNAME')
EFAB_PASSWORD = os.getenv('EFAB_PASSWORD')

print("=" * 80)
print(f"eFab API Authentication Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print(f"Base URL: {EFAB_BASE_URL}")
print(f"Username: {EFAB_USERNAME}")
print(f"Password: {'*' * len(EFAB_PASSWORD) if EFAB_PASSWORD else 'NOT SET'}")

# Create session
session = requests.Session()

# Step 1: Authenticate
login_url = f"{EFAB_BASE_URL}/login"
print(f"\n1. Attempting login to {login_url}...")

login_data = {
    'username': EFAB_USERNAME,
    'password': EFAB_PASSWORD
}

try:
    # Try to login
    response = session.post(login_url, data=login_data, allow_redirects=False)
    print(f"   Login response status: {response.status_code}")
    
    # Check cookies
    if session.cookies:
        print(f"   Cookies received: {list(session.cookies.keys())}")
    else:
        print(f"   No cookies received")
        
    # Check for redirect
    if 'Location' in response.headers:
        print(f"   Redirect to: {response.headers['Location']}")
        
except Exception as e:
    print(f"   Login failed: {e}")
    
# Step 2: Test endpoints with session
endpoints = [
    "/api/yarn/active",
    "/api/greige/g00",
    "/api/greige/g02", 
    "/api/finished/i01",
    "/api/finished/f01",
    "/api/yarn-po",
    "/fabric/knitorder/list",
    "/api/styles",
    "/api/report/yarn_expected",
    "/api/report/sales_activity",
    "/api/report/yarn_demand",
    "/api/sales-order/plan/list"
]

print("\n2. Testing API endpoints with authentication...")
print("-" * 80)

successful = []
failed = []

for endpoint in endpoints:
    url = f"{EFAB_BASE_URL}{endpoint}"
    try:
        response = session.get(url, timeout=10)
        
        # Check if it's JSON
        try:
            data = response.json()
            is_json = True
            if isinstance(data, list):
                record_count = len(data)
            elif isinstance(data, dict):
                record_count = len(data.get('data', data))
            else:
                record_count = 1
        except:
            is_json = False
            record_count = 0
            
        content_type = response.headers.get('Content-Type', 'Unknown')
        
        if response.status_code == 200 and is_json:
            print(f"[OK] {endpoint}")
            print(f"     JSON: Yes | Records: {record_count} | Type: {type(data).__name__}")
            successful.append(endpoint)
        else:
            print(f"[FAIL] {endpoint}")
            print(f"     Status: {response.status_code} | JSON: {is_json}")
            if not is_json and response.status_code == 200:
                # Check if still redirecting to login
                if 'login' in response.text[:200].lower():
                    print(f"     Still redirecting to login")
            failed.append(endpoint)
            
    except Exception as e:
        print(f"[ERROR] {endpoint}")
        print(f"     Error: {str(e)[:100]}")
        failed.append(endpoint)

# Summary
print("\n" + "=" * 80)
print("SUMMARY:")
print(f"  Total endpoints: {len(endpoints)}")
print(f"  Successful (JSON): {len(successful)}")
print(f"  Failed: {len(failed)}")

if successful:
    print(f"\nWorking endpoints:")
    for ep in successful:
        print(f"  - {ep}")
        
if failed:
    print(f"\nFailed endpoints:")
    for ep in failed:
        print(f"  - {ep}")

print("=" * 80)