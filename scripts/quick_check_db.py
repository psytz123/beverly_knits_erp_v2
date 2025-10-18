#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.turso_client import get_turso_client

turso = get_turso_client()

# Direct count
result = turso.execute("SELECT COUNT(*) as cnt FROM historical_sales")
print(f"Total records: {result[0]['cnt'] if result else 0}")

# Sample records
result = turso.execute("SELECT * FROM historical_sales LIMIT 5")
if result:
    for row in result:
        print(row)
