#!/usr/bin/env python3
"""
Test script to verify data deduplication in API responses
"""

import requests
import json
from collections import Counter

BASE_URL = "http://localhost:5006/api"

def test_endpoint(endpoint, key_field, name):
    """Test an endpoint for duplicate data"""
    print(f"\n=== Testing {name} ({endpoint}) ===")
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}")
        data = response.json()
        
        # Extract the data array based on endpoint structure
        if 'orders' in data:
            items = data['orders']
        elif 'yarn_netting' in data:
            items = data['yarn_netting']
        elif 'criticality_analysis' in data and 'yarns' in data['criticality_analysis']:
            items = data['criticality_analysis']['yarns']
        elif 'pipeline' in data:
            items = data['pipeline']
        elif 'data' in data:
            items = data['data']
        else:
            print(f"  Could not find data array in response")
            return
            
        if not items:
            print(f"  No data items found")
            return
            
        print(f"  Total items: {len(items)}")
        
        # Check for duplicates
        if items and len(items) > 0 and key_field in items[0]:
            keys = [item[key_field] for item in items if key_field in item]
            unique_keys = set(keys)
            duplicates = len(keys) - len(unique_keys)
            
            print(f"  Unique {key_field}: {len(unique_keys)}")
            print(f"  Duplicates: {duplicates}")
            
            if duplicates > 0:
                # Find which keys are duplicated
                key_counts = Counter(keys)
                dup_keys = {k: v for k, v in key_counts.items() if v > 1}
                print(f"  Duplicated keys: {dup_keys}")
        else:
            print(f"  Key field '{key_field}' not found in data")
            
    except Exception as e:
        print(f"  Error: {e}")

# Test all major endpoints
endpoints = [
    ("yarn-intelligence", "yarn_id", "Yarn Intelligence"),
    ("knit-orders", "ko_id", "Knit Orders"),
    ("production-planning", "po_number", "Production Planning"),
    ("production-pipeline", "stage", "Production Pipeline"),
    ("inventory-netting", "yarn_id", "Inventory Netting"),
]

print("Testing API endpoints for data deduplication...")
for endpoint, key, name in endpoints:
    test_endpoint(endpoint, key, name)

print("\n=== Deduplication Test Complete ===")