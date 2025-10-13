"""
Import BOM and Fabric Specifications from CSV files to Turso database
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


def create_tables():
    """Create BOM and fabric_specs tables if they don't exist."""
    logger.info("Creating BOM table...")
    execute_turso_sql("""
        CREATE TABLE IF NOT EXISTS bom (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            style TEXT NOT NULL,
            yarn_id TEXT NOT NULL,
            bom_percentage REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(style, yarn_id)
        )
    """)
    logger.info("✓ BOM table created")

    logger.info("Creating fabric_specs table...")
    execute_turso_sql("""
        CREATE TABLE IF NOT EXISTS fabric_specs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            style TEXT NOT NULL UNIQUE,
            yds_per_lb REAL,
            gsm INTEGER,
            width REAL,
            fabric_type TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    logger.info("✓ fabric_specs table created")


def import_bom_data():
    """Import BOM data from CSV."""
    logger.info("📊 Importing BOM data...")

    # Look for BOM CSV file
    project_root = Path(__file__).parent.parent
    bom_files = [
        project_root / 'data' / 'production' / '5' / 'BOM_updated.csv',
        project_root / 'data' / 'production' / '5' / 'Style_BOM.csv',
        project_root / 'data' / 'BOM_updated.csv'
    ]

    bom_file = None
    for f in bom_files:
        if f.exists():
            bom_file = f
            break

    if not bom_file:
        logger.warning("⚠ BOM file not found, skipping BOM import")
        return 0

    logger.info(f"Loading BOM from: {bom_file}")
    df = pd.read_csv(bom_file)
    logger.info(f"✓ Loaded {len(df)} BOM records from CSV")
    logger.info(f"✓ Columns: {df.columns.tolist()}")

    # Map column names
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'style' in col_lower:
            column_mapping['style'] = col
        elif 'desc' in col_lower and ('#' in col or 'id' in col_lower):
            column_mapping['yarn_id'] = col
        elif 'bom' in col_lower and 'percentage' in col_lower:
            column_mapping['bom_percentage'] = col

    logger.info(f"Column mapping: {column_mapping}")

    # Prepare BOM records
    records_to_insert = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            style = row.get(column_mapping.get('style', 'Style#'))
            yarn_id = row.get(column_mapping.get('yarn_id', 'Desc#'))
            bom_pct = row.get(column_mapping.get('bom_percentage', 'BOM_Percentage'))

            if pd.isna(style) or pd.isna(yarn_id) or pd.isna(bom_pct):
                skipped += 1
                continue

            bom_float = float(bom_pct)
            if bom_float <= 0:
                skipped += 1
                continue

            records_to_insert.append({
                'style': str(style),
                'yarn_id': str(yarn_id),
                'bom_percentage': bom_float
            })

        except Exception as e:
            logger.warning(f"Skipping row {idx}: {e}")
            skipped += 1

    logger.info(f"✓ Prepared {len(records_to_insert)} BOM records")
    if skipped > 0:
        logger.info(f"⚠ Skipped {skipped} rows")

    # Batch insert
    batch_size = 50
    total_inserted = 0

    for i in range(0, len(records_to_insert), batch_size):
        batch = records_to_insert[i:i + batch_size]
        statements = []

        for record in batch:
            sql = """
                INSERT OR REPLACE INTO bom
                (style, yarn_id, bom_percentage)
                VALUES (?, ?, ?)
            """
            params = [
                record['style'],
                record['yarn_id'],
                record['bom_percentage']
            ]
            statements.append({"q": sql, "params": params})

        try:
            execute_batch(statements)
            total_inserted += len(batch)
            logger.info(f"✓ Inserted BOM batch {i//batch_size + 1}: {total_inserted}/{len(records_to_insert)}")
        except Exception as e:
            logger.error(f"❌ Error inserting BOM batch: {e}")

    return total_inserted


def import_fabric_specs():
    """Import fabric specifications from CSV."""
    logger.info("📊 Importing Fabric Specifications...")

    # Look for fabric specs file (Excel)
    project_root = Path(__file__).parent.parent
    spec_files = [
        project_root / 'eFab_Styles_sample.xlsx',
        project_root / 'data' / 'eFab_Styles_sample.xlsx',
        project_root / 'data' / 'fabric_specs.csv'
    ]

    spec_file = None
    for f in spec_files:
        if f.exists():
            spec_file = f
            break

    if not spec_file:
        logger.warning("⚠ Fabric specs file not found, skipping fabric specs import")
        return 0

    logger.info(f"Loading fabric specs from: {spec_file}")
    # Handle both Excel and CSV
    if str(spec_file).endswith('.xlsx'):
        df = pd.read_excel(spec_file)
    else:
        df = pd.read_csv(spec_file)
    logger.info(f"✓ Loaded {len(df)} fabric spec records")
    logger.info(f"✓ Columns: {df.columns.tolist()}")

    # Prepare fabric spec records
    records_to_insert = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            style = row.get('Name', row.get('F ID', row.get('Style', '')))
            yds_lbs = row.get('Yds/Lbs', row.get('yds_per_lb'))
            gsm = row.get('GSM', row.get('gsm'))
            width = row.get('Overall Width', row.get('width'))
            fabric_type = row.get('Fabric Type', row.get('fabric_type', ''))

            if pd.isna(style):
                skipped += 1
                continue

            records_to_insert.append({
                'style': str(style),
                'yds_per_lb': float(yds_lbs) if not pd.isna(yds_lbs) else None,
                'gsm': int(gsm) if not pd.isna(gsm) else None,
                'width': float(width) if not pd.isna(width) else None,
                'fabric_type': str(fabric_type) if not pd.isna(fabric_type) else None
            })

        except Exception as e:
            logger.warning(f"Skipping row {idx}: {e}")
            skipped += 1

    logger.info(f"✓ Prepared {len(records_to_insert)} fabric spec records")
    if skipped > 0:
        logger.info(f"⚠ Skipped {skipped} rows")

    # Batch insert
    batch_size = 50
    total_inserted = 0

    for i in range(0, len(records_to_insert), batch_size):
        batch = records_to_insert[i:i + batch_size]
        statements = []

        for record in batch:
            sql = """
                INSERT OR REPLACE INTO fabric_specs
                (style, yds_per_lb, gsm, width, fabric_type)
                VALUES (?, ?, ?, ?, ?)
            """
            params = [
                record['style'],
                record['yds_per_lb'],
                record['gsm'],
                record['width'],
                record['fabric_type']
            ]
            statements.append({"q": sql, "params": params})

        try:
            execute_batch(statements)
            total_inserted += len(batch)
            logger.info(f"✓ Inserted fabric specs batch {i//batch_size + 1}: {total_inserted}/{len(records_to_insert)}")
        except Exception as e:
            logger.error(f"❌ Error inserting fabric specs batch: {e}")

    return total_inserted


def verify_data():
    """Verify imported data."""
    logger.info("\n" + "="*60)
    logger.info("Verifying imported data...")
    logger.info("="*60)

    # Check BOM
    result = execute_turso_sql("SELECT COUNT(*) as count FROM bom")
    # Handle Turso response format: [{'results': {...}}]
    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if result.get("results"):
        count = result["results"][0]["rows"][0][0]
        logger.info(f"✓ BOM records: {count}")

        # Sample BOM data
        result = execute_turso_sql("SELECT * FROM bom LIMIT 5")
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        if result.get("results") and result["results"][0].get("rows"):
            logger.info("Sample BOM records:")
            for row in result["results"][0]["rows"][:3]:
                logger.info(f"  Style: {row[1]}, Yarn: {row[2]}, BOM%: {row[3]}")

    # Check fabric specs
    result = execute_turso_sql("SELECT COUNT(*) as count FROM fabric_specs")
    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if result.get("results"):
        count = result["results"][0]["rows"][0][0]
        logger.info(f"✓ Fabric specs records: {count}")

        # Sample fabric specs
        result = execute_turso_sql("SELECT * FROM fabric_specs LIMIT 5")
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        if result.get("results") and result["results"][0].get("rows"):
            logger.info("Sample fabric specs:")
            for row in result["results"][0]["rows"][:3]:
                logger.info(f"  Style: {row[1]}, Yds/Lb: {row[2]}, GSM: {row[3]}")


if __name__ == "__main__":
    try:
        logger.info("🚀 Starting BOM and Fabric Specs import to Turso")
        logger.info("="*60)

        # Create tables
        create_tables()

        # Import BOM data
        bom_count = import_bom_data()

        # Import fabric specs
        specs_count = import_fabric_specs()

        # Verify
        verify_data()

        logger.info("\n" + "="*60)
        logger.info("✅ Import complete!")
        logger.info(f"BOM records imported: {bom_count}")
        logger.info(f"Fabric specs imported: {specs_count}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
