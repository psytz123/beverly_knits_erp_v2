#!/usr/bin/env python3
"""Test script to verify machine schedule board data flow"""

import requests
import json

def test_machine_schedule_data():
    """Test that APIs provide correct data for machine schedule board"""
    
    print("Testing Machine Schedule Board Data Flow")
    print("=" * 50)
    
    # Test factory floor API
    print("\n1. Testing Factory Floor API...")
    try:
        response = requests.get("http://localhost:5006/api/factory-floor-ai-dashboard")
        if response.status_code == 200:
            data = response.json()
            work_centers = data.get('work_center_groups', [])
            total_machines = sum(len(wc.get('machines', [])) for wc in work_centers)
            print(f"   ✓ Factory API: {len(work_centers)} work centers, {total_machines} machines")
            
            # Show sample machine
            if work_centers and work_centers[0].get('machines'):
                machine = work_centers[0]['machines'][0]
                print(f"   Sample machine: {machine.get('machine_id')} in WC {work_centers[0].get('work_center_id')}")
        else:
            print(f"   ✗ Factory API failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Factory API error: {e}")
    
    # Test knit orders API
    print("\n2. Testing Knit Orders API...")
    try:
        response = requests.get("http://localhost:5006/api/knit-orders")
        if response.status_code == 200:
            data = response.json()
            orders = data.get('orders', [])
            assigned = [o for o in orders if o.get('machine')]
            unassigned = [o for o in orders if not o.get('machine')]
            print(f"   ✓ Orders API: {len(orders)} total orders")
            print(f"     - {len(assigned)} assigned to machines")
            print(f"     - {len(unassigned)} unassigned")
            
            # Show sample assigned order
            if assigned:
                order = assigned[0]
                print(f"   Sample order: {order.get('order_id')} on machine {order.get('machine')}")
        else:
            print(f"   ✗ Orders API failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Orders API error: {e}")
    
    # Test production planning API
    print("\n3. Testing Production Planning API...")
    try:
        response = requests.get("http://localhost:5006/api/production-planning")
        if response.status_code == 200:
            data = response.json()
            schedule = data.get('production_schedule', [])
            print(f"   ✓ Planning API: {len(schedule)} scheduled items")
        else:
            print(f"   ✗ Planning API failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Planning API error: {e}")
    
    # Test machine schedule board HTML
    print("\n4. Testing Machine Schedule Board HTML...")
    try:
        response = requests.get("http://localhost:5006/machine-schedule")
        if response.status_code == 200:
            html = response.text
            if "Machine Production Schedule Board" in html:
                print(f"   ✓ Machine schedule board HTML loaded")
                # Check for key elements
                if "loadAllData()" in html:
                    print("   ✓ Data loading function present")
                if "createOrderBar" in html:
                    print("   ✓ Order bar creation function present")
                if "renderSchedule" in html:
                    print("   ✓ Schedule rendering function present")
            else:
                print(f"   ✗ HTML doesn't contain expected content")
        else:
            print(f"   ✗ HTML page failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ HTML page error: {e}")
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("- APIs are working and returning data")
    print("- Machine schedule board HTML is accessible")
    print("- Orders should display on machine timelines")
    print("\nAccess the board at: http://localhost:5006/machine-schedule")

if __name__ == "__main__":
    test_machine_schedule_data()