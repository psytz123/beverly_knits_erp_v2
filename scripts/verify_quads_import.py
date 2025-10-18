#!/usr/bin/env python3
"""Quick verification of QuadS fabric specs in Turso"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()


def execute_turso_sql(sql: str):
    """Execute SQL on Turso database."""
    database_url = os.getenv("TURSO_DATABASE_URL", "").replace("libsql://", "https://")
    auth_token = os.getenv("TURSO_AUTH_TOKEN", "")

    payload = {"statements": [{"q": sql, "params": []}]}

    response = httpx.post(
        database_url,
        json=payload,
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        },
        timeout=10.0
    )

    response.raise_for_status()
    result = response.json()

    # Handle list or dict response
    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    return result


# Count records
print("Verifying QuadS Fabric Specs in Turso...")
print("=" * 70)

result = execute_turso_sql("SELECT COUNT(*) FROM fabric_specs")

if result and "results" in result:
    # Turso returns results as a dict with various keys
    results_data = result["results"]

    # Check if it's a list (multiple results) or dict (single result)
    if isinstance(results_data, list) and len(results_data) > 0:
        rows = results_data[0].get("rows", [])
    elif isinstance(results_data, dict) and "rows" in results_data:
        rows = results_data["rows"]
    else:
        rows = []

    if rows and len(rows) > 0:
        count = rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]
        print(f"[OK] Total fabric_specs records: {count}")

# Sample records
result = execute_turso_sql("""
    SELECT style, gsm, width
    FROM fabric_specs
    WHERE style IS NOT NULL
    LIMIT 10
""")

if result and "results" in result:
    results_data = result["results"]

    if isinstance(results_data, list) and len(results_data) > 0:
        rows = results_data[0].get("rows", [])
    elif isinstance(results_data, dict) and "rows" in results_data:
        rows = results_data["rows"]
    else:
        rows = []

    if rows:
        print("\nSample fabric specs:")
        print("-" * 70)
        for row in rows[:10]:
            if len(row) >= 3:
                style, gsm, width = row[0], row[1], row[2]
                print(f"  Style: {str(style)[:20]:20} | GSM: {str(gsm or 'N/A')[:6]:6} | Width: {str(width or 'N/A')[:6]:6}")

print("\n" + "=" * 70)
print("[SUCCESS] Verification complete!")
