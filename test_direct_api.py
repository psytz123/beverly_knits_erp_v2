#!/usr/bin/env python3
"""
Direct API Test - Tests eFab.ai endpoints directly
"""

import requests
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Get credentials
username = os.getenv('EFAB_USERNAME')
password = os.getenv('EFAB_PASSWORD')
base_url = os.getenv('EFAB_BASE_URL', 'https://efab.bkiapps.com')

print("="*60)
print(" Testing eFab.ai API Endpoints Directly")
print("="*60)
print(f"\nBase URL: {base_url}")
print(f"Username: {username}")
print(f"Password: {'***' if password else 'Not set'}")

# Test endpoints
endpoints = [
    '/api/yarn/active',
    '/api/greige/g00',
    '/api/finished/f01',
    '/api/yarn-po',
    '/fabric/knitorder/list',
    '/api/report/yarn_expected',
    '/api/report/yarn_demand',
    '/api/report/yarn_demand_ko',
]

print("\n" + "="*60)
print(" Testing Endpoints")
print("="*60)

# Create a session for authentication
session = requests.Session()

# Try different authentication methods
print("\nTrying authentication methods:")

# Method 1: Basic Auth
print("\n1. Basic Authentication:")
for endpoint in endpoints[:1]:  # Test first endpoint
    url = f"{base_url}{endpoint}"
    print(f"\n   Testing: {endpoint}")
    
    try:
        response = session.get(
            url,
            auth=HTTPBasicAuth(username, password),
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            if isinstance(data, dict):
                print(f"   Response keys: {list(data.keys())[:5]}")
            else:
                print(f"   Response type: {type(data)}")
    except Exception as e:
        print(f"   Error: {str(e)[:100]}")

# Method 2: Session-based login
print("\n2. Session-based Authentication:")
login_endpoints = [
    '/login',
    '/api/login',
    '/api/auth/login',
    '/auth/login'
]

for login_endpoint in login_endpoints:
    url = f"{base_url}{login_endpoint}"
    print(f"\n   Trying login at: {login_endpoint}")
    
    try:
        response = session.post(
            url,
            json={'username': username, 'password': password},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   Login successful!")
            break
    except Exception as e:
        print(f"   Error: {str(e)[:50]}")

# Test endpoints with session
print("\n3. Testing endpoints with session:")
for endpoint in endpoints[:3]:  # Test first 3 endpoints
    url = f"{base_url}{endpoint}"
    print(f"\n   Testing: {endpoint}")
    
    try:
        response = session.get(url, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            if 'json' in content_type:
                data = response.json()
                if isinstance(data, dict):
                    print(f"   Response keys: {list(data.keys())[:5]}")
                elif isinstance(data, list):
                    print(f"   Response: List with {len(data)} items")
            else:
                print(f"   Response type: {content_type}")
                print(f"   Response length: {len(response.text)} chars")
    except Exception as e:
        print(f"   Error: {str(e)[:100]}")

# Test without authentication (some endpoints might be public)
print("\n4. Testing without authentication:")
for endpoint in ['/api/health', '/health', '/api/status']:
    url = f"{base_url}{endpoint}"
    print(f"\n   Testing: {endpoint}")
    
    try:
        response = requests.get(url, timeout=5)
        print(f"   Status: {response.status_code}")
    except Exception as e:
        print(f"   Error: {str(e)[:50]}")

print("\n" + "="*60)
print(" Test Complete")
print("="*60)