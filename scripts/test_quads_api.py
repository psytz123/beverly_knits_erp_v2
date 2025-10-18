#!/usr/bin/env python3
"""
Test QuadS API connection and view response structure
"""

import os
import httpx
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

QUADS_BASE_URL = "https://quads.bkiapps.com"
QUADS_API_PREFIX = "/api"


def login_to_quads() -> Optional[str]:
    """Login to QuadS and get session token."""
    username = os.getenv("QUADS_USERNAME", "")
    password = os.getenv("QUADS_PASSWORD", "")

    if not username or not password:
        return None

    login_url = f"{QUADS_BASE_URL}/api/auth/login"

    try:
        print(f"Attempting login as {username}...")

        response = httpx.post(
            login_url,
            json={"username": username, "password": password},
            headers={"Content-Type": "application/json"},
            timeout=30.0
        )

        response.raise_for_status()
        data = response.json()

        # Try to extract token
        token = (
            data.get('token') or
            data.get('session_token') or
            data.get('access_token') or
            response.cookies.get('session')
        )

        if token:
            print(f"Login successful! Token: {token[:20]}...")
            return token
        else:
            print(f"Login succeeded but no token found. Response keys: {list(data.keys())}")
            return None

    except Exception as e:
        print(f"Login failed: {e}")
        return None


# Try automatic login first
session_token = login_to_quads()

# Fall back to manual token
if not session_token:
    session_token = os.getenv("QUADS_SESSION_TOKEN", "")

if not session_token:
    print("\nError: Could not get QuadS session token")
    print("\nPlease add to your .env file either:")
    print("  Option 1 (recommended):")
    print("    QUADS_USERNAME=your_username")
    print("    QUADS_PASSWORD=your_password")
    print("  Option 2:")
    print("    QUADS_SESSION_TOKEN=your_token_here")
    exit(1)

print("="*70)
print("QuadS API Connection Test")
print("="*70)
print(f"\nBase URL: {QUADS_BASE_URL}")
print(f"Session Token: {session_token[:20]}..." if len(session_token) > 20 else f"Session Token: {session_token}")

# Test endpoints
endpoints = [
    "/styles/finished/active",
    "/styles/greige/active"
]

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {session_token}",
    "Cookie": f"session={session_token}"
}

for endpoint in endpoints:
    url = f"{QUADS_BASE_URL}{QUADS_API_PREFIX}{endpoint}"

    print(f"\n{'-'*70}")
    print(f"Testing: {endpoint}")
    print(f"{'-'*70}")

    try:
        response = httpx.get(url, headers=headers, timeout=30.0)

        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")

        if response.status_code == 200:
            data = response.json()

            print(f"\nResponse Type: {type(data)}")

            if isinstance(data, list):
                print(f"Number of records: {len(data)}")
                if len(data) > 0:
                    print(f"\nFirst record keys: {list(data[0].keys())}")
                    print(f"\nFirst record (formatted):")
                    print(json.dumps(data[0], indent=2))

                    if len(data) > 1:
                        print(f"\nSecond record (formatted):")
                        print(json.dumps(data[1], indent=2))

            elif isinstance(data, dict):
                print(f"Response keys: {list(data.keys())}")
                print(f"\nFull response (formatted):")
                print(json.dumps(data, indent=2))

            else:
                print(f"Unexpected response format: {data}")

        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text[:500]}")
    except httpx.RequestError as e:
        print(f"Request Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

print(f"\n{'='*70}")
print("Test Complete")
print("="*70)
