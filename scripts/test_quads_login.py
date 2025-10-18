#!/usr/bin/env python3
"""
Test QuadS login and extract session token
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

QUADS_BASE_URL = "https://quads.bkiapps.com"
username = os.getenv("QUADS_USERNAME", "psytz")
password = os.getenv("QUADS_PASSWORD", "big$cat")

print("="*70)
print("QuadS Login Test")
print("="*70)
print(f"Base URL: {QUADS_BASE_URL}")
print(f"Username: {username}")
print(f"Password: {'*' * len(password)}")

# Try different login methods
login_configs = [
    {
        "url": f"{QUADS_BASE_URL}/login",
        "method": "POST",
        "content_type": "application/x-www-form-urlencoded",
        "payload_type": "form"
    },
    {
        "url": f"{QUADS_BASE_URL}/login",
        "method": "POST",
        "content_type": "application/json",
        "payload_type": "json"
    },
    {
        "url": f"{QUADS_BASE_URL}/api/auth/login",
        "method": "POST",
        "content_type": "application/json",
        "payload_type": "json"
    }
]

for idx, config in enumerate(login_configs, 1):
    print(f"\n{'-'*70}")
    print(f"Test {idx}: {config['url']}")
    print(f"Method: {config['method']}")
    print(f"Content-Type: {config['content_type']}")
    print(f"{'-'*70}")

    try:
        if config['payload_type'] == 'json':
            response = httpx.post(
                config['url'],
                json={"username": username, "password": password},
                headers={"Content-Type": config['content_type']},
                timeout=30.0,
                follow_redirects=True
            )
        else:
            response = httpx.post(
                config['url'],
                data={"username": username, "password": password},
                headers={"Content-Type": config['content_type']},
                timeout=30.0,
                follow_redirects=True
            )

        print(f"Status: {response.status_code}")
        print(f"Final URL: {response.url}")
        print(f"Headers: {dict(response.headers)}")

        # Check for tokens in cookies
        print(f"\nCookies:")
        for name, value in response.cookies.items():
            print(f"  {name}: {value[:50]}..." if len(value) > 50 else f"  {name}: {value}")

        # Check response body
        print(f"\nResponse body (first 500 chars):")
        try:
            data = response.json()
            print(f"JSON: {str(data)[:500]}")
        except:
            print(f"Text: {response.text[:500]}")

        # Try to extract session token
        token = None
        if 'session' in response.cookies:
            token = response.cookies['session']
            print(f"\n✓ Found session token in cookies: {token[:30]}...")
        elif 'connect.sid' in response.cookies:
            token = response.cookies['connect.sid']
            print(f"\n✓ Found connect.sid token in cookies: {token[:30]}...")

        if token:
            print(f"\n{'='*70}")
            print(f"SUCCESS! Token: {token}")
            print(f"{'='*70}")
            break

    except Exception as e:
        print(f"Error: {e}")

print(f"\n{'='*70}")
print("Test Complete")
print("="*70)
