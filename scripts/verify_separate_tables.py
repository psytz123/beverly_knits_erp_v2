#!/usr/bin/env python3
"""
Verify the separate tables import
"""

import os
import sys
import httpx
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

database_url = os.getenv("TURSO_DATABASE_URL", "").replace("libsql://", "https://")
auth_token = os.getenv("TURSO_AUTH_TOKEN", "")

turso_client = httpx.Client(
    headers={
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    },
    timeout=60.0
)


def execute_sql(sql: str) -> dict:
    """Execute SQL on Turso."""
    payload = {"statements": [{"q": sql, "params": []}]}
    response = turso_client.post(database_url, json=payload)
    response.raise_for_status()
    return response.json()


def main():
    print("=" * 70)
    print("Separate Tables Verification")
    print("=" * 70)

    # Count finished fabrics
    result = execute_sql("SELECT COUNT(*) FROM finished_fabric_specs")
    # Handle list response
    if isinstance(result, list) and len(result) > 0:
        result = result[0]
    if result and "results" in result:
        results_data = result["results"]
        # Results can be dict or list
        if isinstance(results_data, dict) and "rows" in results_data:
            rows = results_data["rows"]
            if rows:
                count = rows[0][0]
                print(f"\nFinished fabrics: {count}")
        elif isinstance(results_data, list) and len(results_data) > 0:
            rows = results_data[0].get("rows", [])
            if rows:
                count = rows[0][0]
                print(f"\nFinished fabrics: {count}")

    # Count greige fabrics
    result = execute_sql("SELECT COUNT(*) FROM greige_fabric_specs")
    if isinstance(result, list) and len(result) > 0:
        result = result[0]
    if result and "results" in result:
        results_data = result["results"]
        if isinstance(results_data, dict) and "rows" in results_data:
            rows = results_data["rows"]
            if rows:
                count = rows[0][0]
                print(f"Greige fabrics: {count}")
        elif isinstance(results_data, list) and len(results_data) > 0:
            rows = results_data[0].get("rows", [])
            if rows:
                count = rows[0][0]
                print(f"Greige fabrics: {count}")

    # Sample finished fabrics
    result = execute_sql("""
        SELECT f_id, gsm, overall_width, yds_per_lb, construction
        FROM finished_fabric_specs
        WHERE yds_per_lb IS NOT NULL
        LIMIT 5
    """)
    if isinstance(result, list) and len(result) > 0:
        result = result[0]
    if result and "results" in result:
        results_data = result["results"]
        rows = None
        if isinstance(results_data, dict) and "rows" in results_data:
            rows = results_data["rows"]
        elif isinstance(results_data, list) and len(results_data) > 0:
            rows = results_data[0].get("rows", [])

        if rows:
            print("\nSample finished fabrics:")
            print("-" * 70)
            for row in rows:
                f_id = row[0]
                gsm = row[1] or 'N/A'
                width = row[2] or 'N/A'
                yds_lb = row[3] or 'N/A'
                constr = (row[4] or '')[:35]
                print(f"  {f_id:10} | GSM: {str(gsm):6} | Width: {str(width):6} | Yds/Lb: {str(yds_lb):6} | {constr}")

    # Sample greige fabrics
    result = execute_sql("""
        SELECT g_id, g_base, construction, customer
        FROM greige_fabric_specs
        LIMIT 5
    """)
    if isinstance(result, list) and len(result) > 0:
        result = result[0]
    if result and "results" in result:
        results_data = result["results"]
        rows = None
        if isinstance(results_data, dict) and "rows" in results_data:
            rows = results_data["rows"]
        elif isinstance(results_data, list) and len(results_data) > 0:
            rows = results_data[0].get("rows", [])

        if rows:
            print("\nSample greige fabrics:")
            print("-" * 70)
            for row in rows:
                g_id = row[0] or ''
                g_base = row[1] or ''
                constr = (row[2] or '')[:25]
                customer = (row[3] or '')[:20]
                print(f"  {g_id:10} | Base: {g_base:12} | {constr:25} | {customer}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Verification complete!")
    print("=" * 70)

    turso_client.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        turso_client.close()
        sys.exit(1)
