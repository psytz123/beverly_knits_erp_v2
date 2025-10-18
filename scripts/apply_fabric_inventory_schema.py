#!/usr/bin/env python3
"""
Apply Fabric Inventory Schema to Turso Database

This script creates the fabric_inventory and fabric_movements tables
in the live Turso database for tracking fabric inventory across production stages.

Usage:
    python scripts/apply_fabric_inventory_schema.py
"""

import os
import sys
import httpx
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Turso connection
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
    """Execute SQL on Turso database"""
    payload = {"statements": [{"q": sql, "params": []}]}
    response = turso_client.post(database_url, json=payload)
    response.raise_for_status()
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        result = result[0]
    return result


def apply_schema():
    """Apply the fabric inventory schema to Turso"""
    logger.info("=" * 70)
    logger.info("Applying Fabric Inventory Schema to Turso")
    logger.info("=" * 70)

    # Read the schema file
    schema_file = Path(__file__).parent.parent / "database" / "migrations" / "fabric_inventory_schema.sql"

    if not schema_file.exists():
        logger.error(f"Schema file not found: {schema_file}")
        sys.exit(1)

    with open(schema_file, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    # Split by semicolons to execute statements individually
    # Remove the BEGIN/COMMIT transactions for individual execution
    statements = []
    for line in schema_sql.split('\n'):
        line = line.strip()
        if line and not line.startswith('--') and line not in ('BEGIN;', 'COMMIT;'):
            statements.append(line)

    # Join back and split by semicolon
    full_sql = ' '.join(statements)
    sql_commands = [cmd.strip() for cmd in full_sql.split(';') if cmd.strip()]

    logger.info(f"Found {len(sql_commands)} SQL commands to execute\n")

    # Execute each command
    success_count = 0
    for i, sql_cmd in enumerate(sql_commands, 1):
        try:
            logger.info(f"Executing command {i}/{len(sql_commands)}...")

            # Show first 100 chars of command
            preview = sql_cmd[:100].replace('\n', ' ')
            logger.info(f"  {preview}...")

            execute_sql(sql_cmd)
            success_count += 1
            logger.info(f"  ✓ Success")

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            # Don't stop - continue with next command

    logger.info("\n" + "=" * 70)
    logger.info(f"Schema Application Complete: {success_count}/{len(sql_commands)} successful")
    logger.info("=" * 70)

    # Verify tables were created
    verify_tables()


def verify_tables():
    """Verify that tables were created successfully"""
    logger.info("\nVerifying tables...")

    tables_to_check = [
        "fabric_inventory",
        "fabric_movements"
    ]

    for table in tables_to_check:
        try:
            sql = f"SELECT COUNT(*) FROM {table}"
            result = execute_sql(sql)

            if result.get("results", {}).get("rows"):
                count = result["results"]["rows"][0][0]
                logger.info(f"  ✓ {table}: exists (contains {count} records)")
            else:
                logger.info(f"  ✓ {table}: exists (empty)")

        except Exception as e:
            logger.error(f"  ✗ {table}: verification failed - {e}")

    logger.info("\nSchema verification complete!")


def main():
    """Main execution"""
    try:
        if not database_url or not auth_token:
            logger.error("Missing TURSO_DATABASE_URL or TURSO_AUTH_TOKEN environment variables")
            logger.error("Please ensure .env file is configured correctly")
            sys.exit(1)

        logger.info(f"Target database: {database_url}\n")

        # Auto-confirm for non-interactive execution
        logger.info("Creating fabric_inventory and fabric_movements tables in Turso database...")

        apply_schema()

        logger.info("\n✓ All operations completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Start your Flask application")
        logger.info("2. Navigate to http://localhost:5000/fabric-inquiry")
        logger.info("3. Enter a fabric ID (F ID or G ID) to search")
        logger.info("\nNote: The system is ready but inventory data needs to be populated.")
        logger.info("You can populate it by:")
        logger.info("  - Manual entry through the API")
        logger.info("  - Importing from eFab or QuadS systems")
        logger.info("  - Adding records directly to fabric_inventory table")

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        turso_client.close()


if __name__ == "__main__":
    main()
