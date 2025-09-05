#!/usr/bin/env python3
"""
Test different API endpoint formats and authentication methods
"""

import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

username = os.getenv('EFAB_USERNAME')
password = os.getenv('EFAB_PASSWORD')
base_url = os.getenv('EFAB_BASE_URL', 'https://efab.bkiapps.com')

print("Testing Different API Formats")
print("="*60)

# Create session
session = requests.Session()

# Try login first
login_response = session.post(f"{base_url}/login", json={'username': username, 'password': password})
print(f"Login status: {login_response.status_code}")

# Check cookies
print("\nSession cookies:")
for cookie in session.cookies:
    print(f"  {cookie.name}: {cookie.value[:20]}..." if len(cookie.value) > 20 else f"  {cookie.name}: {cookie.value}")

# Different endpoint variations to try
endpoint_variations = [
    # Standard REST API formats
    '/api/v1/yarn/active',
    '/api/v2/yarn/active',
    '/rest/yarn/active',
    '/data/yarn/active',
    
    # Alternative formats
    '/yarn/active.json',
    '/api/yarn/active.json',
    '/api/yarn/active?format=json',
    '/api/yarn/active?output=json',
    
    # GraphQL style
    '/graphql?query={yarn_active}',
    '/query/yarn/active',
    
    # Direct data endpoints
    '/export/yarn/active',
    '/download/yarn/active',
    '/report/yarn/active',
    
    # Without /api prefix
    '/yarn/active',
    '/yarns',
    '/yarn',
    
    # Check if there's a different API documentation
    '/api',
    '/api/docs',
    '/api/swagger',
    '/api/help',
]

print("\nTesting endpoint variations:")
print("-"*40)

found_json = False

for endpoint in endpoint_variations:
    url = f"{base_url}{endpoint}"
    
    try:
        # Try with different headers
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        response = session.get(url, headers=headers, timeout=5)
        
        # Check if we got JSON
        content_type = response.headers.get('content-type', '')
        
        if response.status_code == 200:
            if 'json' in content_type:
                print(f"[JSON] {endpoint} - Status {response.status_code}")
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        print(f"       Keys: {list(data.keys())[:3]}")
                    elif isinstance(data, list) and data:
                        print(f"       List with {len(data)} items")
                    found_json = True
                except:
                    pass
            elif response.status_code == 200:
                # Check if response text looks like JSON
                text = response.text.strip()
                if text.startswith('{') or text.startswith('['):
                    print(f"[MAYBE JSON] {endpoint} - Status {response.status_code}")
                    try:
                        data = json.loads(text)
                        print(f"       Parsed successfully!")
                        found_json = True
                    except:
                        print(f"       Failed to parse")
                else:
                    print(f"[HTML] {endpoint} - Status {response.status_code}")
        elif response.status_code in [301, 302, 303, 307, 308]:
            print(f"[REDIRECT] {endpoint} -> {response.headers.get('location', 'unknown')}")
        elif response.status_code == 401:
            print(f"[AUTH REQUIRED] {endpoint}")
        elif response.status_code == 404:
            # Skip 404s to reduce noise
            pass
        else:
            print(f"[{response.status_code}] {endpoint}")
            
    except requests.exceptions.Timeout:
        print(f"[TIMEOUT] {endpoint}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {endpoint} - {str(e)[:30]}")

if not found_json:
    print("\n" + "="*60)
    print("No JSON endpoints found. Checking for API in page content...")
    
    # Check main page for API information
    response = session.get(base_url)
    if 'api' in response.text.lower():
        # Count occurrences
        api_count = response.text.lower().count('api')
        json_count = response.text.lower().count('json')
        print(f"Found 'api' {api_count} times in main page")
        print(f"Found 'json' {json_count} times in main page")
    
    # Check for common API frameworks
    frameworks = ['swagger', 'openapi', 'graphql', 'rest', 'jsonapi']
    for framework in frameworks:
        if framework in response.text.lower():
            print(f"Found reference to '{framework}'")

print("\n" + "="*60)
print("Test Complete")