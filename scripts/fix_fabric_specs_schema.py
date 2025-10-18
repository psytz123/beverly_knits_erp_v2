#!/usr/bin/env python3
"""Add missing columns to fabric_specs table"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()


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


print("Checking fabric_specs schema...")
result = execute_sql("PRAGMA table_info(fabric_specs)")
print("Current schema:", result)

print("\nAdding missing columns...")
try:
    result = execute_sql("ALTER TABLE fabric_specs ADD COLUMN description TEXT")
    print("Added description column:", result)
except Exception as e:
    print(f"Note: {e}")

try:
    result = execute_sql("ALTER TABLE fabric_specs ADD COLUMN updated_at TEXT DEFAULT CURRENT_TIMESTAMP")
    print("Added updated_at column:", result)
except Exception as e:
    print(f"Note: {e}")

print("\nVerifying updated schema...")
result = execute_sql("PRAGMA table_info(fabric_specs)")
print("Updated schema:", result)
