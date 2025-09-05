#!/usr/bin/env python3
"""
Fetch QuadS fabric style data from BKI Apps
"""
import requests
import json
import csv
import os
from pathlib import Path

# Configuration
QUADS_BASE_URL = "https://quads.bkiapps.com"
QUADS_USERNAME = "psytz"
QUADS_PASSWORD = "big$cat"

# Data path where files should be saved
DATA_PATH = Path(r"D:\AI\Workspaces\efab.ai\beverly_knits_erp_v2\data\production\5\ERP Data")

class QuadSDataFetcher:
    def __init__(self, session_token=None):
        self.session = requests.Session()
        self.base_url = QUADS_BASE_URL
        self.authenticated = False
        self.session_token = session_token or "aLqIljNOrvabt2CP_6ZLaC9zlNM7HXlR"
        
    def authenticate(self):
        """Authenticate with QuadS system"""
        print("\n" + "="*60)
        print("Authenticating with QuadS...")
        print("="*60)
        
        # Use session token if available
        if self.session_token:
            self.session.cookies.set('Dancer.session', self.session_token, domain='quads.bkiapps.com')
            print("[OK] Session token set")
            self.authenticated = True
            return True
        
        try:
            login_url = f"{self.base_url}/login"
            login_data = {
                "username": QUADS_USERNAME,
                "password": QUADS_PASSWORD
            }
            
            # Attempt login
            response = self.session.post(
                login_url,
                data=login_data,
                allow_redirects=True
            )
            
            print(f"Login status: {response.status_code}")
            
            # Check if we got redirected away from login
            if response.status_code == 200 and "login" not in response.url.lower():
                print("[OK] Authentication successful")
                self.authenticated = True
                return True
            else:
                print("[WARN] Authentication may have failed")
                # Try to proceed anyway
                return True
                
        except Exception as e:
            print(f"[ERROR] Authentication error: {e}")
            return False
    
    def fetch_fabric_list(self, fabric_type="finished"):
        """Fetch fabric list from QuadS"""
        # Use API endpoints instead of web endpoints
        if fabric_type == "greige":
            endpoint = "/api/styles/greige/active"
        elif fabric_type == "finished":
            endpoint = "/api/styles/finished/active"
        else:
            endpoint = f"/api/styles/{fabric_type}/active"
        
        url = f"{self.base_url}{endpoint}"
        
        print(f"\nFetching {fabric_type} fabric list...")
        
        try:
            response = self.session.get(url, timeout=30)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                # Check content type
                content_type = response.headers.get('Content-Type', '')
                print(f"Content type: {content_type}")
                
                if 'application/json' in content_type:
                    data = response.json()
                    print(f"[OK] Retrieved {len(data) if isinstance(data, list) else 'data'} records")
                    return data
                else:
                    # Might be HTML or CSV
                    print(f"[INFO] Response is not JSON, saving raw content")
                    return response.text
            else:
                print(f"[ERROR] Failed to fetch data: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to fetch {fabric_type} list: {e}")
            return None
    
    def save_fabric_data(self, data, fabric_type):
        """Save fabric data to CSV file"""
        if not data:
            print(f"[WARN] No data to save for {fabric_type}")
            return False
            
        # Create data directory if it doesn't exist
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        filename = f"QuadS_{fabric_type}FabricList.csv"
        filepath = DATA_PATH / filename
        
        print(f"Saving to: {filepath}")
        
        try:
            if isinstance(data, list):
                # Save JSON data as CSV
                if len(data) > 0:
                    # Get all unique keys from all records
                    keys = set()
                    for item in data:
                        if isinstance(item, dict):
                            keys.update(item.keys())
                    
                    keys = sorted(list(keys))
                    
                    with open(filepath, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=keys)
                        writer.writeheader()
                        writer.writerows(data)
                    
                    print(f"[OK] Saved {len(data)} records to {filename}")
                else:
                    print(f"[WARN] No records to save")
                    
            elif isinstance(data, str):
                # Save raw text/HTML
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(data)
                print(f"[OK] Saved raw data to {filename}")
                
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save data: {e}")
            return False
    
    def fetch_all_data(self):
        """Fetch all QuadS fabric data"""
        if not self.authenticate():
            print("[ERROR] Authentication failed, trying to proceed anyway...")
        
        # Fetch finished fabric list
        finished_data = self.fetch_fabric_list("finished")
        if finished_data:
            self.save_fabric_data(finished_data, "finished")
        
        # Fetch greige fabric list
        greige_data = self.fetch_fabric_list("greige")
        if greige_data:
            self.save_fabric_data(greige_data, "greige")
        
        print("\n" + "="*60)
        print("QuadS Data Fetch Complete")
        print("="*60)
        
        # List saved files
        if DATA_PATH.exists():
            print("\nSaved files:")
            for file in DATA_PATH.glob("QuadS_*.csv"):
                size = file.stat().st_size
                print(f"  - {file.name} ({size:,} bytes)")

def main():
    fetcher = QuadSDataFetcher()
    fetcher.fetch_all_data()

if __name__ == "__main__":
    main()