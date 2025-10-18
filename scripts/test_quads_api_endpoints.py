#!/usr/bin/env python3
"""
Test QuadS API endpoints with authenticated session
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

QUADS_BASE_URL = "https://quads.bkiapps.com"
username = os.getenv("QUADS_USERNAME", "psytz")
password = os.getenv("QUADS_PASSWORD", "big$cat")

# Create persistent client
client = httpx.Client(follow_redirects=True, timeout=60.0)

# Login
print("Logging in...")
response = client.post(
    f"{QUADS_BASE_URL}/login",
    data={"username": username, "password": password},
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

if 'x-dancer-username' in response.headers and response.headers['x-dancer-username'] == username:
    print(f"[OK] Logged in as {username}")
    session_cookie = client.cookies.get('dancer.session')
    print(f"Session cookie: {session_cookie[:30]}...\n")
else:
    print("Login failed!")
    exit(1)

# Test different API endpoints
endpoints = [
    "/api/styles/finished/active",
    "/api/styles/greige/active",
    "/api/finished-fabric/list",
    "/api/greige-fabric/list",
    "/finished-fabric/list",
    "/greige-fabric/list",
]

for endpoint in endpoints:
    url = f"{QUADS_BASE_URL}{endpoint}"
    print(f"{'='*70}")
    print(f"Testing: {endpoint}")
    print(f"{'='*70}")

    try:
        response = client.get(url, timeout=10.0)

        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"Content-Length: {len(response.text)} bytes")

        if response.status_code == 200:
            # Check if JSON
            if 'application/json' in response.headers.get('content-type', ''):
                import json
                try:
                    data = response.json()
                    print(f"Response type: {type(data)}")

                    if isinstance(data, list):
                        print(f"[SUCCESS] Got {len(data)} records")
                        if len(data) > 0:
                            print(f"\nFirst record keys: {list(data[0].keys())}")
                            print(f"\nFirst record:")
                            print(json.dumps(data[0], indent=2))

                            if len(data) > 1:
                                print(f"\nSecond record:")
                                print(json.dumps(data[1], indent=2))

                            # This is probably the right endpoint!
                            print(f"\n{'*'*70}")
                            print(f"*** FOUND WORKING ENDPOINT: {endpoint} ***")
                            print(f"{'*'*70}\n")
                            break
                    elif isinstance(data, dict):
                        print(f"Response keys: {list(data.keys())}")
                        print(f"\nResponse:")
                        print(json.dumps(data, indent=2)[:1000])
                except:
                    print(f"JSON parse error")
                    print(f"Response (first 500 chars):")
                    print(response.text[:500])
            else:
                print(f"Non-JSON response (first 500 chars):")
                print(response.text[:500])
        else:
            print(f"Error response (first 500 chars):")
            print(response.text[:500])

    except httpx.ReadTimeout:
        print("[TIMEOUT] Endpoint timed out after 10 seconds")
    except Exception as e:
        print(f"[ERROR] {e}")

    print()

client.close()
