#!/usr/bin/env python3
"""
Test specific eFab API endpoint: /api/knitorder/list
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
print(f"Testing /api/knitorder/list endpoint")
print("=" * 80)

# Create session and authenticate
session = requests.Session()
login_url = f"{EFAB_BASE_URL}/login"
login_data = {
    'username': EFAB_USERNAME,
    'password': EFAB_PASSWORD
}

print(f"1. Authenticating...")
try:
    response = session.post(login_url, data=login_data, allow_redirects=False)
    if response.status_code == 302 and session.cookies:
        print(f"   [OK] Authentication successful")
        print(f"   Session cookie: {list(session.cookies.keys())[0]}")
    else:
        print(f"   [FAIL] Authentication failed: {response.status_code}")
except Exception as e:
    print(f"   [ERROR] Login error: {e}")
    exit(1)

# Test the correct endpoint
endpoints_to_test = [
    "/api/knitorder/list",        # Correct API path
    "/fabric/knitorder/list",      # Original path that returns HTML
    "/api/knit-order/list",        # Alternative naming
    "/api/knitorders",             # Alternative naming
    "/api/production/knitorders"  # Alternative path
]

print(f"\n2. Testing knit order endpoints...")
print("-" * 80)

for endpoint in endpoints_to_test:
    url = f"{EFAB_BASE_URL}{endpoint}"
    print(f"\nTesting: {endpoint}")
    
    try:
        response = session.get(url, timeout=10)
        print(f"  Status: {response.status_code}")
        
        # Check content type
        content_type = response.headers.get('Content-Type', 'Unknown')
        print(f"  Content-Type: {content_type}")
        
        # Try to parse as JSON
        try:
            data = response.json()
            is_json = True
            
            # Analyze the data structure
            if isinstance(data, list):
                record_count = len(data)
                print(f"  [JSON] Array with {record_count} records")
                
                # Show sample record if available
                if record_count > 0:
                    sample = data[0]
                    print(f"  Sample fields: {list(sample.keys())[:5]}...")
                    
            elif isinstance(data, dict):
                keys = list(data.keys())
                print(f"  [JSON] Object with keys: {keys}")
                
                # Check for data field
                if 'data' in data and isinstance(data['data'], list):
                    record_count = len(data['data'])
                    print(f"  Data field contains {record_count} records")
                    if record_count > 0:
                        sample = data['data'][0]
                        print(f"  Sample fields: {list(sample.keys())[:5]}...")
                else:
                    record_count = len(data)
                    
            else:
                print(f"  [JSON] Type: {type(data).__name__}")
                record_count = 1
                
            # Save successful endpoint data
            if response.status_code == 200 and record_count > 0:
                with open(f'knitorder_sample_{endpoint.replace("/", "_")}.json', 'w') as f:
                    if isinstance(data, list) and len(data) > 5:
                        json.dump(data[:5], f, indent=2)
                    else:
                        json.dump(data, f, indent=2)
                print(f"  -> Sample data saved to file")
                
        except json.JSONDecodeError:
            print(f"  [NOT JSON] Format not parseable")
            if response.status_code == 200:
                # Check if it's HTML
                if 'text/html' in content_type:
                    print(f"  Returns HTML page")
                    if 'login' in response.text[:500].lower():
                        print(f"  Still requires authentication")
                        
    except requests.exceptions.Timeout:
        print(f"  [TIMEOUT] Request took >10 seconds")
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] Connection failed")
    except Exception as e:
        print(f"  [ERROR] {str(e)[:100]}")

print("\n" + "=" * 80)
print("Test complete. Check above for working endpoints.")
print("=" * 80)