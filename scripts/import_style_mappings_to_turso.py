#!/usr/bin/env python3
"""
Import eFab_Styles mapping from Excel to Turso database
Replaces Excel file dependency with database table
"""

import os
import sys
import pandas as pd
import httpx
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def execute_turso_sql(sql: str, params: list = None) -> dict:
    """Execute SQL on Turso database."""
    database_url = os.getenv("TURSO_DATABASE_URL", "")
    auth_token = os.getenv("TURSO_AUTH_TOKEN", "")

    if not database_url or not auth_token:
        raise ValueError("TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set in .env")

    if database_url.startswith("libsql://"):
        database_url = database_url.replace("libsql://", "https://")

    payload = {
        "statements": [{"q": sql, "params": params or []}]
    }

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


def execute_batch(statements: list) -> dict:
    """Execute multiple SQL statements in batch."""
    database_url = os.getenv("TURSO_DATABASE_URL", "")
    auth_token = os.getenv("TURSO_AUTH_TOKEN", "")

    if database_url.startswith("libsql://"):
        database_url = database_url.replace("libsql://", "https://")

    payload = {"statements": statements}

    response = httpx.post(
        database_url,
        json=payload,
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        },
        timeout=120.0
    )

    response.raise_for_status()
    return response.json()


def create_style_mappings_table():
    """Create style_mappings table in Turso"""
    logger.info("📊 Creating style_mappings table...")

    sql = """
    CREATE TABLE IF NOT EXISTS style_mappings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fstyle TEXT NOT NULL,
        gbase TEXT NOT NULL,
        style TEXT,
        description TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(fstyle, gbase)
    )
    """

    try:
        execute_turso_sql(sql)
        logger.info("✓ Table created successfully")

        # Create index for faster lookups
        index_sql = """
        CREATE INDEX IF NOT EXISTS idx_style_mappings_fstyle ON style_mappings(fstyle)
        """
        execute_turso_sql(index_sql)
        logger.info("✓ Index created")

        return True
    except Exception as e:
        logger.error(f"❌ Error creating table: {e}")
        return False


def import_style_mappings(file_path: str):
    """Import eFab_Styles from Excel to Turso"""
    logger.info(f"📂 Importing style mappings from: {file_path}")

    # Read Excel file
    try:
        df = pd.read_excel(file_path)
        logger.info(f"✓ Loaded {len(df)} rows from Excel")
        logger.info(f"✓ Columns: {df.columns.tolist()}")
    except Exception as e:
        logger.error(f"❌ Error reading Excel: {e}")
        return

    # Display sample data
    logger.info("\n" + "="*80)
    logger.info("Sample data:")
    logger.info(df.head(10).to_string())
    logger.info("="*80 + "\n")

    # Map column names (check actual column names in file)
    # Common variations: fStyle, gBase, Style, Description
    column_mapping = {}

    for col in df.columns:
        col_lower = col.lower().strip()
        if 'fstyle' in col_lower:
            column_mapping['fstyle'] = col
        elif 'gbase' in col_lower or 'base' in col_lower:
            column_mapping['gbase'] = col
        elif col_lower == 'style':
            column_mapping['style'] = col
        elif 'desc' in col_lower and 'description' not in column_mapping:
            column_mapping['description'] = col

    logger.info(f"Column mapping: {column_mapping}")

    if 'fstyle' not in column_mapping or 'gbase' not in column_mapping:
        logger.error("❌ Required columns 'fstyle' and 'gbase' not found")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return

    # Prepare data for insertion
    statements = []
    inserted_count = 0
    skipped_count = 0

    for idx, row in df.iterrows():
        try:
            fstyle = str(row[column_mapping['fstyle']]).strip() if pd.notna(row[column_mapping['fstyle']]) else None
            gbase = str(row[column_mapping['gbase']]).strip() if pd.notna(row[column_mapping['gbase']]) else None

            # Skip if either is missing
            if not fstyle or not gbase or fstyle == 'nan' or gbase == 'nan':
                skipped_count += 1
                continue

            style = str(row[column_mapping['style']]).strip() if 'style' in column_mapping and pd.notna(row[column_mapping['style']]) else None
            description = str(row[column_mapping['description']]).strip() if 'description' in column_mapping and pd.notna(row[column_mapping['description']]) else None

            # Clean None values
            if style == 'nan':
                style = None
            if description == 'nan':
                description = None

            sql = """
            INSERT OR REPLACE INTO style_mappings (fstyle, gbase, style, description)
            VALUES (?, ?, ?, ?)
            """

            statements.append({
                "q": sql,
                "params": [fstyle, gbase, style, description]
            })

            inserted_count += 1

            # Batch insert every 100 rows
            if len(statements) >= 100:
                try:
                    execute_batch(statements)
                    logger.info(f"✓ Inserted batch ({inserted_count} total)")
                    statements = []
                except Exception as e:
                    logger.error(f"❌ Batch insert error: {e}")
                    statements = []

        except Exception as e:
            logger.error(f"❌ Error processing row {idx}: {e}")
            skipped_count += 1

    # Insert remaining rows
    if statements:
        try:
            execute_batch(statements)
            logger.info(f"✓ Inserted final batch")
        except Exception as e:
            logger.error(f"❌ Final batch error: {e}")

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Import complete!")
    logger.info(f"   Inserted: {inserted_count}")
    logger.info(f"   Skipped: {skipped_count}")
    logger.info(f"{'='*80}\n")

    # Verify import
    verify_import()


def verify_import():
    """Verify the imported data"""
    logger.info("🔍 Verifying import...")

    try:
        # Count total records
        result = execute_turso_sql("SELECT COUNT(*) as cnt FROM style_mappings")
        if result and 'results' in result and result['results']:
            count_data = result['results'][0]
            if 'rows' in count_data and count_data['rows']:
                total = count_data['rows'][0][0]
                logger.info(f"✓ Total mappings in database: {total}")

        # Get sample records
        result = execute_turso_sql("SELECT * FROM style_mappings LIMIT 5")
        if result and 'results' in result and result['results']:
            data = result['results'][0]
            if 'rows' in data and data['rows']:
                logger.info("\nSample records:")
                columns = data.get('columns', [])
                for row in data['rows']:
                    record = dict(zip(columns, row))
                    logger.info(f"  fStyle: {record.get('fstyle')} -> gBase: {record.get('gbase')}")

    except Exception as e:
        logger.error(f"❌ Verification error: {e}")


def main():
    """Main import process"""
    import argparse

    parser = argparse.ArgumentParser(description='Import eFab_Styles to Turso')
    parser.add_argument('--file', type=str, default=r'C:\Users\psytz\Downloads\eFab_Styles_20251018.xlsx',
                        help='Path to eFab_Styles Excel file')
    parser.add_argument('--create-table', action='store_true',
                        help='Create table before importing')

    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        logger.error(f"❌ File not found: {file_path}")
        return

    # Create table if requested
    if args.create_table:
        if not create_style_mappings_table():
            logger.error("❌ Failed to create table, aborting")
            return

    # Import data
    import_style_mappings(str(file_path))


if __name__ == "__main__":
    main()
