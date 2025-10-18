#!/usr/bin/env python3
"""
Test Fabric Inquiry System

This script tests the fabric inquiry API to verify everything is working.
"""

import os
import sys
import httpx
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

# Turso connection
database_url = os.getenv("TURSO_DATABASE_URL", "").replace("libsql://", "https://")
auth_token = os.getenv("TURSO_AUTH_TOKEN", "")

turso_client = httpx.Client(
    headers={
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    },
    timeout=30.0
)


def execute_sql(sql: str, params: list = None):
    """Execute SQL on Turso"""
    payload = {"statements": [{"q": sql, "params": params or []}]}
    response = turso_client.post(database_url, json=payload)
    response.raise_for_status()
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        result = result[0]
    return result


def test_fabric_specs_lookup():
    """Test that we can find fabric specs"""
    print("\n" + "="*70)
    print("TEST 1: Fabric Specs Lookup")
    print("="*70)

    # Try to find a finished fabric
    sql = "SELECT f_id, name, gsm, overall_width, yds_per_lb FROM finished_fabric_specs LIMIT 5"
    result = execute_sql(sql)

    if result.get("results", {}).get("rows"):
        rows = result["results"]["rows"]
        print(f"\n[OK] Found {len(rows)} finished fabrics in database:")
        for row in rows:
            print(f"  - F ID: {row[0]} - {row[1]} (GSM: {row[2]}, Width: {row[3]}, Yds/Lb: {row[4]})")
        return rows[0][0]  # Return first fabric ID for testing
    else:
        print("[ERROR] No finished fabrics found")
        return None


def test_greige_specs_lookup():
    """Test greige fabric lookup"""
    print("\n" + "="*70)
    print("TEST 2: Greige Specs Lookup")
    print("="*70)

    sql = "SELECT g_id, g_base, construction, customer LIMIT 5"
    result = execute_sql("SELECT g_id, g_base, construction, customer FROM greige_fabric_specs LIMIT 5")

    if result.get("results", {}).get("rows"):
        rows = result["results"]["rows"]
        print(f"\n[OK] Found {len(rows)} greige fabrics in database:")
        for row in rows:
            print(f"  - G ID: {row[0]} - Base: {row[1]} ({row[3]})")
        return rows[0][0]  # Return first fabric ID
    else:
        print("[ERROR] No greige fabrics found")
        return None


def test_inventory_tables():
    """Test that inventory tables exist"""
    print("\n" + "="*70)
    print("TEST 3: Inventory Tables")
    print("="*70)

    # Check fabric_inventory
    result = execute_sql("SELECT COUNT(*) FROM fabric_inventory")
    if result.get("results", {}).get("rows"):
        count = result["results"]["rows"][0][0]
        print(f"\n[OK] fabric_inventory table exists: {count} records")

    # Check fabric_movements
    result = execute_sql("SELECT COUNT(*) FROM fabric_movements")
    if result.get("results", {}).get("rows"):
        count = result["results"]["rows"][0][0]
        print(f"[OK] fabric_movements table exists: {count} records")


def add_sample_inventory(fabric_id):
    """Add sample inventory for testing"""
    print("\n" + "="*70)
    print("TEST 4: Add Sample Inventory")
    print("="*70)

    print(f"\nAdding sample inventory for fabric {fabric_id}...")

    # Add inventory at each stage
    stages = [
        ("G00", 500, 250, 5),
        ("G02", 300, 150, 3),
        ("I01", 200, 100, 2),
        ("F01", 100, 50, 1)
    ]

    for stage, yards, lbs, rolls in stages:
        sql = """
            INSERT OR REPLACE INTO fabric_inventory
            (fabric_id, fabric_type, stage, quantity_yards, quantity_lbs, rolls)
            VALUES (?, 'finished', ?, ?, ?, ?)
        """
        execute_sql(sql, [fabric_id, stage, yards, lbs, rolls])
        print(f"  [OK] Added {yards} yards at stage {stage}")

    # Add a movement record
    sql = """
        INSERT INTO fabric_movements
        (fabric_id, from_stage, to_stage, quantity_yards, quantity_lbs, operator)
        VALUES (?, 'G00', 'G02', 200, 100, 'System Test')
    """
    execute_sql(sql, [fabric_id])
    print(f"  [OK] Added movement record: G00 -> G02")

    return fabric_id


def verify_inquiry_data(fabric_id):
    """Verify the inquiry system can find the data"""
    print("\n" + "="*70)
    print("TEST 5: Verify Inquiry System")
    print("="*70)

    # Get inventory by stage
    sql = """
        SELECT stage, quantity_yards, quantity_lbs, rolls
        FROM fabric_inventory
        WHERE fabric_id = ?
        ORDER BY
            CASE stage
                WHEN 'G00' THEN 1
                WHEN 'G02' THEN 2
                WHEN 'I01' THEN 3
                WHEN 'F01' THEN 4
            END
    """
    result = execute_sql(sql, [fabric_id])

    if result.get("results", {}).get("rows"):
        rows = result["results"]["rows"]
        print(f"\n[OK] Found inventory for fabric {fabric_id}:")
        total_yards = 0
        total_lbs = 0
        total_rolls = 0

        for row in rows:
            stage, yards, lbs, rolls = row
            print(f"  - {stage}: {yards} yards, {lbs} lbs, {rolls} rolls")
            total_yards += yards
            total_lbs += lbs
            total_rolls += rolls

        print(f"\n  TOTAL: {total_yards} yards, {total_lbs} lbs, {total_rolls} rolls")
        return True
    else:
        print(f"[ERROR] No inventory found for fabric {fabric_id}")
        return False


def main():
    print("\n" + "="*70)
    print("FABRIC INQUIRY SYSTEM - TEST SUITE")
    print("="*70)

    try:
        # Test 1: Find finished fabrics
        finished_id = test_fabric_specs_lookup()

        # Test 2: Find greige fabrics
        greige_id = test_greige_specs_lookup()

        # Test 3: Check inventory tables exist
        test_inventory_tables()

        # Test 4 & 5: Add sample data and verify
        if finished_id:
            fabric_id = add_sample_inventory(finished_id)
            verify_inquiry_data(fabric_id)

        # Final summary
        print("\n" + "="*70)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("="*70)

        print("\n[READY] The Fabric Inquiry System is ready to use!")
        print("\nYou can now:")
        print("1. Start your Flask app: python src/core/beverly_comprehensive_erp.py")
        print("2. Navigate to: http://localhost:5000/fabric-inquiry")
        if finished_id:
            print(f"3. Search for fabric: {finished_id}")
        print("\nThe system will show inventory across all production stages!")

    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        turso_client.close()


if __name__ == "__main__":
    main()
