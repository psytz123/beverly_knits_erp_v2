#!/usr/bin/env python3
"""Test importing a few records from QuadS"""

import os
import sys
import pandas as pd
import httpx
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()


def execute_batch(statements):
    """Execute multiple SQL statements in batch."""
    database_url = os.getenv("TURSO_DATABASE_URL", "").replace("libsql://", "https://")
    auth_token = os.getenv("TURSO_AUTH_TOKEN", "")

    payload = {"statements": statements}

    response = httpx.post(
        database_url,
        json=payload,
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        },
        timeout=30.0
    )

    response.raise_for_status()
    return response.json()


# Load QuadS data
project_root = Path(__file__).parent.parent
quads_file = project_root / 'startingdocs' / 'QuadS_finishedFabricList_ (6).xlsx'

print("Loading QuadS data...")
df = pd.read_excel(quads_file)
print(f"Loaded {len(df)} records")
print(f"Columns: {df.columns.tolist()}")

# Take just first 5 records for testing
test_records = []
for idx, row in df.head(5).iterrows():
    f_id = row.get('F ID')
    gsm = row.get('GSM')
    width = row.get('Overall Width')

    if pd.notna(f_id):
        test_records.append({
            'style': str(f_id),
            'yds_per_lb': None,
            'gsm': int(gsm) if pd.notna(gsm) else None,
            'width': float(width) if pd.notna(width) else None,
            'fabric_type': None,
            'description': None
        })

print(f"\nPrepared {len(test_records)} test records:")
for rec in test_records:
    print(f"  {rec}")

# Insert test records
statements = []
for record in test_records:
    sql = """
        INSERT OR REPLACE INTO fabric_specs
        (style, yds_per_lb, gsm, width, fabric_type, description, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """
    params = [
        record['style'],
        record['yds_per_lb'],
        record['gsm'],
        record['width'],
        record['fabric_type'],
        record['description']
    ]
    statements.append({"q": sql, "params": params})

print(f"\nInserting {len(statements)} records...")
result = execute_batch(statements)
print("Result:", result)

# Verify
def execute_sql(sql):
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
    return response.json()

print("\nVerifying...")
result = execute_sql("SELECT COUNT(*) FROM fabric_specs")
print("Count result:", result)

result = execute_sql("SELECT * FROM fabric_specs LIMIT 5")
print("Sample records:", result)
