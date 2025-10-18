#!/usr/bin/env python3
"""
Inspect QuadS HTML structure to determine how to parse the data
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

QUADS_BASE_URL = "https://quads.bkiapps.com"
username = os.getenv("QUADS_USERNAME", "psytz")
password = os.getenv("QUADS_PASSWORD", "big$cat")

# Create persistent client
client = httpx.Client(follow_redirects=True, timeout=30.0)

# Login
print("Logging in...")
response = client.post(
    f"{QUADS_BASE_URL}/login",
    data={"username": username, "password": password},
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

if 'x-dancer-username' in response.headers and response.headers['x-dancer-username'] == username:
    print(f"[OK] Logged in as {username}\n")
else:
    print("Login failed!")
    exit(1)

# Fetch finished styles page
print("Fetching finished styles page...")
response = client.get(f"{QUADS_BASE_URL}/knit-style/list/finished")

print(f"Status: {response.status_code}")
print(f"Content-Length: {len(response.text)} bytes\n")

# Save HTML to file for inspection
with open("quads_finished_styles.html", "w", encoding="utf-8") as f:
    f.write(response.text)

print("[OK] Saved HTML to: quads_finished_styles.html")
print("\nFirst 2000 characters:")
print("="*70)
print(response.text[:2000])
print("="*70)

# Try to find any data patterns
print("\nLooking for data patterns...")

# Check for JSON data embedded in HTML
if '"data"' in response.text or "'data'" in response.text:
    print("[OK] Found 'data' in response - might have embedded JSON")

# Check for table
if '<table' in response.text:
    print("[OK] Found <table> tag")
    # Find table start
    table_start = response.text.find('<table')
    table_snippet = response.text[table_start:table_start+500]
    print(f"\nTable snippet:\n{table_snippet}")

# Check for script tags with data
if '<script' in response.text:
    print("\n[OK] Found <script> tags")
    import re
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', response.text, re.DOTALL)
    for i, script in enumerate(scripts[:3]):
        if len(script.strip()) > 50 and ('data' in script.lower() or '{' in script):
            print(f"\nScript {i+1} (first 500 chars):")
            print(script[:500])

client.close()
