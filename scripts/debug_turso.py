#!/usr/bin/env python3
"""Debug Turso database structure"""

import os
import httpx
import json
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
    return response.json()


print("Turso Database Debug Info")
print("=" * 70)

# List all tables
result = execute_turso_sql("SELECT name FROM sqlite_master WHERE type='table'")
print("\nRaw response for tables query:")
print(json.dumps(result, indent=2))

# Check fabric_specs table specifically
result = execute_turso_sql("SELECT COUNT(*) FROM fabric_specs")
print("\nRaw response for count query:")
print(json.dumps(result, indent=2))
