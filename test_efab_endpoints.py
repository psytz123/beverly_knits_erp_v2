#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test eFab API Endpoints with Authentication
"""
import requests
import json
from typing import Dict, Any
import time
import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Configuration
EFAB_BASE_URL = "https://efab.bkiapps.com"
EFAB_USERNAME = "psytz"
EFAB_PASSWORD = "big$cat"

# List of endpoints to test
ENDPOINTS = [
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

class EfabAPITester:
    def __init__(self, session_token="aLqBRuWmczN9CYgN_xGoQmCurvvNjOz1"):
        self.session = requests.Session()
        self.base_url = EFAB_BASE_URL
        self.authenticated = False
        self.session_token = session_token
        self.username = EFAB_USERNAME
        self.password = EFAB_PASSWORD
        
    def authenticate(self, use_token_first=True) -> bool:
        """Authenticate with eFab system - tries token first, then username/password if needed"""
        print(f"\n{'='*60}")
        print("Authenticating with eFab...")
        print(f"{'='*60}")
        
        if use_token_first and self.session_token:
            # Try session token first
            self.session.cookies.set('dancer.session', self.session_token, domain='efab.bkiapps.com')
            print("[OK] Session token set")
            
            # Test if token is valid
            test_url = f"{self.base_url}/api/yarn/active"
            response = self.session.get(test_url, allow_redirects=False)
            
            if response.status_code == 200:
                print("[OK] Session token is valid")
                self.authenticated = True
                return True
            else:
                print(f"[WARN] Session token may be expired (status: {response.status_code})")
                print("[INFO] Attempting username/password authentication...")
        
        try:
            # Clear cookies and try username/password login
            self.session.cookies.clear()
            
            login_url = f"{self.base_url}/login"
            login_data = {
                "username": self.username,
                "password": self.password
            }
            
            # Get login page first (might be needed for CSRF tokens)
            response = self.session.get(login_url)
            print(f"[INFO] Login page status: {response.status_code}")
            
            # Attempt login with credentials
            response = self.session.post(
                login_url,
                data=login_data,
                allow_redirects=True
            )
            
            print(f"[INFO] Login POST status: {response.status_code}")
            print(f"[INFO] Final URL: {response.url}")
            
            # Check if we got redirected away from login page
            if "login" not in response.url.lower():
                print("[OK] Username/password authentication successful")
                
                # Save new session cookie if present
                if 'dancer.session' in self.session.cookies:
                    new_token = self.session.cookies.get('dancer.session')
                    print(f"[INFO] New session token obtained: {new_token[:20]}...")
                    self.session_token = new_token
                
                self.authenticated = True
                return True
            else:
                print("[WARN] Authentication may have failed (still on login page)")
                # Try to proceed anyway
                self.authenticated = True
                return True
                
        except Exception as e:
            print(f"[ERROR] Authentication error: {e}")
            return False
    
    def re_authenticate(self) -> bool:
        """Re-authenticate if session expires"""
        print("\n[INFO] Re-authenticating due to session expiration...")
        self.session.cookies.clear()
        return self.authenticate(use_token_first=False)
    
    def test_endpoint(self, endpoint: str, retry_on_auth_fail=True) -> Dict[str, Any]:
        """Test a single endpoint with automatic re-authentication if needed"""
        url = f"{self.base_url}{endpoint}"
        result = {
            "endpoint": endpoint,
            "url": url,
            "status_code": None,
            "content_type": None,
            "response_size": 0,
            "error": None,
            "sample_data": None
        }
        
        try:
            response = self.session.get(url, timeout=30, allow_redirects=False)
            
            # Check if we got redirected to login (session expired)
            if response.status_code == 302 and 'login' in response.headers.get('Location', ''):
                if retry_on_auth_fail:
                    print(f"  [INFO] Session expired for {endpoint}, re-authenticating...")
                    if self.re_authenticate():
                        # Retry the request with new session
                        return self.test_endpoint(endpoint, retry_on_auth_fail=False)
                    else:
                        result["error"] = "Re-authentication failed"
                        return result
            
            # Follow redirects manually if not auth-related
            if response.status_code in [301, 302, 303, 307, 308]:
                response = self.session.get(url, timeout=30)
            
            result["status_code"] = response.status_code
            result["content_type"] = response.headers.get("Content-Type", "")
            result["response_size"] = len(response.content)
            
            # Check if response is JSON
            if "application/json" in result["content_type"]:
                try:
                    data = response.json()
                    # Get sample of data structure
                    if isinstance(data, list) and len(data) > 0:
                        result["sample_data"] = {
                            "type": "list",
                            "count": len(data),
                            "first_item_keys": list(data[0].keys()) if isinstance(data[0], dict) else None
                        }
                    elif isinstance(data, dict):
                        result["sample_data"] = {
                            "type": "dict",
                            "keys": list(data.keys())[:10]  # First 10 keys
                        }
                except:
                    result["sample_data"] = "JSON parse error"
            
            # Check for redirects
            if response.history:
                result["redirected"] = True
                result["redirect_chain"] = [r.url for r in response.history]
            
        except requests.exceptions.Timeout:
            result["error"] = "Timeout"
        except requests.exceptions.ConnectionError:
            result["error"] = "Connection error"
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def test_all_endpoints(self):
        """Test all endpoints"""
        results = []
        
        print(f"\n{'='*60}")
        print("Testing eFab API Endpoints")
        print(f"{'='*60}\n")
        
        for endpoint in ENDPOINTS:
            print(f"Testing: {endpoint}")
            result = self.test_endpoint(endpoint)
            results.append(result)
            
            # Print summary
            if result["error"]:
                print(f"  [ERROR] {result['error']}")
            elif result["status_code"] == 200:
                print(f"  [OK] Success: {result['status_code']} | Size: {result['response_size']} bytes")
                if result["sample_data"]:
                    print(f"  [DATA] {result['sample_data']}")
            elif result["status_code"] == 302:
                print(f"  [REDIRECT] {result['status_code']}")
            elif result["status_code"] == 404:
                print(f"  [NOT FOUND] {result['status_code']}")
            else:
                print(f"  [STATUS] {result['status_code']}")
            
            time.sleep(0.5)  # Be nice to the server
        
        return results
    
    def print_summary(self, results):
        """Print summary of all tests"""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}\n")
        
        success_count = sum(1 for r in results if r["status_code"] == 200)
        redirect_count = sum(1 for r in results if r["status_code"] == 302)
        not_found_count = sum(1 for r in results if r["status_code"] == 404)
        error_count = sum(1 for r in results if r["error"])
        
        print(f"Total Endpoints Tested: {len(results)}")
        print(f"  [OK] Successful (200): {success_count}")
        print(f"  [REDIRECT] (302): {redirect_count}")
        print(f"  [NOT FOUND] (404): {not_found_count}")
        print(f"  [ERROR] Count: {error_count}")
        
        print(f"\n{'='*60}")
        print("DETAILED RESULTS")
        print(f"{'='*60}\n")
        
        for result in results:
            status = "[OK]" if result["status_code"] == 200 else "[ERROR]" if result["error"] else "[WARN]"
            print(f"{status} {result['endpoint']}")
            print(f"   Status: {result['status_code'] or 'N/A'}")
            if result["content_type"]:
                print(f"   Content-Type: {result['content_type']}")
            if result["response_size"] > 0:
                print(f"   Size: {result['response_size']} bytes")
            if result["sample_data"]:
                print(f"   Data Structure: {result['sample_data']}")
            print()

def main():
    tester = EfabAPITester()
    
    # Authenticate
    if not tester.authenticate():
        print("Failed to authenticate. Trying endpoints anyway...")
    
    # Test all endpoints
    results = tester.test_all_endpoints()
    
    # Print summary
    tester.print_summary(results)
    
    # Save results to file
    with open("efab_api_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to efab_api_test_results.json")

if __name__ == "__main__":
    main()