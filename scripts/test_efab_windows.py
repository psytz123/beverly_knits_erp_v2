#!/usr/bin/env python3
"""
Test eFab API connection from Windows Python
Run this from Windows (not WSL) if DNS doesn't resolve in WSL
"""

import requests
import json
import socket
from datetime import datetime

print("\n" + "="*60)
print("eFab API Connection Test (Windows)")
print("="*60 + "\n")

# Configuration
base_url = "https://efab.bklapps.com"
session_cookie = "aLfHTrRrtWWy4FPgLnxdEPC7ohA37dlR"

# Test 1: DNS Resolution
print("🔍 Testing DNS resolution...")
try:
    ip = socket.gethostbyname("efab.bklapps.com")
    print(f"✅ DNS resolved: efab.bklapps.com -> {ip}\n")
except socket.gaierror as e:
    print(f"❌ DNS resolution failed: {e}")
    print("⚠️  Make sure you're on the corporate network/VPN\n")
    exit(1)

# Test 2: HTTPS Connection
print("🌐 Testing HTTPS connection...")
try:
    response = requests.get(base_url, timeout=5, verify=True)
    print(f"✅ HTTPS connection successful (Status: {response.status_code})\n")
except requests.exceptions.SSLError as e:
    print(f"⚠️  SSL certificate issue: {e}")
    print("Trying without certificate verification...\n")
    try:
        response = requests.get(base_url, timeout=5, verify=False)
        print(f"✅ Connection successful without SSL verification\n")
    except Exception as e:
        print(f"❌ Connection failed: {e}\n")
except Exception as e:
    print(f"❌ Connection failed: {e}\n")

# Test 3: API Endpoint
print("📡 Testing API endpoint with session cookie...")
headers = {
    'Cookie': f'dancer.session={session_cookie}',
    'Accept': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

try:
    api_url = f"{base_url}/api/sales-order/plan/list"
    response = requests.get(api_url, headers=headers, timeout=10, verify=False)
    
    if response.status_code == 200:
        print(f"✅ API call successful!")
        
        try:
            data = response.json()
            if isinstance(data, list):
                print(f"   Retrieved {len(data)} records")
                if len(data) > 0:
                    print(f"\n   Sample record:")
                    print(json.dumps(data[0], indent=2)[:500] + "...")
            else:
                print(f"   Response type: {type(data)}")
        except:
            print(f"   Response: {response.text[:200]}...")
    elif response.status_code == 401:
        print(f"❌ Authentication failed (401)")
        print("⚠️  Session cookie may have expired. Get a fresh one from browser.")
    else:
        print(f"❌ API returned status {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
        
except Exception as e:
    print(f"❌ API call failed: {e}")

print("\n" + "="*60)
print("Test complete!")
print("="*60 + "\n")

# Instructions for WSL integration
print("💡 If this works from Windows but not WSL:")
print("   1. Add eFab IP to WSL /etc/hosts:")
print(f"      echo '{ip if 'ip' in locals() else 'IP_ADDRESS'} efab.bklapps.com' | sudo tee -a /etc/hosts")
print("   2. Or configure WSL to use Windows DNS:")
print("      sudo bash -c 'echo nameserver 8.8.8.8 > /etc/resolv.conf'")
print("   3. Or run a local proxy on Windows and connect through it from WSL")