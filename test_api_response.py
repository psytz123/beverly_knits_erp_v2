#!/usr/bin/env python3
"""
Test API Response Content
"""

import requests
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

username = os.getenv('EFAB_USERNAME')
password = os.getenv('EFAB_PASSWORD')
base_url = os.getenv('EFAB_BASE_URL', 'https://efab.bkiapps.com')

print("Testing API Response Content")
print("="*60)

# Create session and login
session = requests.Session()

# Login
login_url = f"{base_url}/login"
response = session.post(
    login_url,
    json={'username': username, 'password': password}
)
print(f"Login status: {response.status_code}")

# Get yarn active data
yarn_url = f"{base_url}/api/yarn/active"
response = session.get(yarn_url)

print(f"\nAPI Response for /api/yarn/active:")
print(f"Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('content-type')}")
print(f"Response length: {len(response.text)} chars")

# Check if it's HTML
if 'html' in response.headers.get('content-type', ''):
    print("\nResponse is HTML - parsing...")
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Check for JSON in the HTML
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string and 'data' in script.string:
            print(f"Found script with potential data: {script.string[:200]}...")
            break
    
    # Check for tables
    tables = soup.find_all('table')
    if tables:
        print(f"Found {len(tables)} tables in HTML")
        # Show first table structure
        if tables[0]:
            rows = tables[0].find_all('tr')
            print(f"First table has {len(rows)} rows")
            if rows:
                # Show headers
                headers = rows[0].find_all(['th', 'td'])
                if headers:
                    print(f"Headers: {[h.text.strip()[:20] for h in headers[:5]]}")
    
    # Check for pre-formatted JSON
    pre_tags = soup.find_all('pre')
    for pre in pre_tags:
        if '{' in pre.text or '[' in pre.text:
            print(f"Found potential JSON in <pre> tag: {pre.text[:200]}...")
            break
    
    # Show first 500 chars of response
    print(f"\nFirst 500 chars of HTML response:")
    print(response.text[:500])
else:
    # Try to parse as JSON
    try:
        data = response.json()
        print(f"\nJSON Response:")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            # Show first item if it's a list
            for key, value in data.items():
                if isinstance(value, list) and value:
                    print(f"{key}[0]: {value[0]}")
                    break
        elif isinstance(data, list):
            print(f"List with {len(data)} items")
            if data:
                print(f"First item: {data[0]}")
    except:
        print(f"\nRaw response (first 500 chars):")
        print(response.text[:500])

# Try a different endpoint with explicit JSON accept header
print("\n" + "="*60)
print("Testing with JSON Accept header:")
headers = {'Accept': 'application/json'}
response = session.get(yarn_url, headers=headers)
print(f"Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('content-type')}")

if response.status_code == 200:
    try:
        data = response.json()
        print("Successfully got JSON!")
        if isinstance(data, list):
            print(f"Data is a list with {len(data)} items")
        elif isinstance(data, dict):
            print(f"Data keys: {list(data.keys())}")
    except:
        print("Still not JSON, response starts with:")
        print(response.text[:200])