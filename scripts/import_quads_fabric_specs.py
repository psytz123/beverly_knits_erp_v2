#!/usr/bin/env python3
"""
Import Fabric Specifications from QuadS Excel file to Turso database
"""

import os
import sys
import pandas as pd
import httpx
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_yds_per_lb(gsm: Optional[float], width_inches: Optional[float]) -> Optional[float]:
    """
    Calculate yards per pound from GSM and width.

    Formula: yds_per_lb = 16129.032 / (gsm * width_in_inches)

    Args:
        gsm: Grams per square meter
        width_inches: Fabric width in inches

    Returns:
        Yards per pound, or None if calculation not possible
    """
    if gsm is None or width_inches is None:
        return None

    if gsm <= 0 or width_inches <= 0:
        return None

    try:
        yds_per_lb = 16129.032 / (gsm * width_inches)
        return round(yds_per_lb, 2)
    except (ZeroDivisionError, ValueError):
        return None


def execute_turso_sql(sql: str, params: Optional[List] = None) -> dict:
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


def execute_batch(statements: List[Dict]) -> dict:
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


def ensure_fabric_specs_table() -> None:
    """Create fabric_specs table if it doesn't exist."""
    logger.info("Ensuring fabric_specs table exists...")
    execute_turso_sql("""
        CREATE TABLE IF NOT EXISTS fabric_specs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            style TEXT NOT NULL UNIQUE,
            yds_per_lb REAL,
            gsm INTEGER,
            width REAL,
            fabric_type TEXT,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    logger.info("✓ fabric_specs table ready")


def load_quads_fabric_data() -> pd.DataFrame:
    """Load fabric data from QuadS Excel file."""
    project_root = Path(__file__).parent.parent

    # Look for QuadS finished fabric list
    quads_files = [
        project_root / 'startingdocs' / 'QuadS_finishedFabricList_ (6).xlsx',
        project_root / 'startingdocs' / 'QuadS_finishedFabricList.xlsx',
        project_root / 'data' / 'QuadS_finishedFabricList.xlsx',
    ]

    quads_file = None
    for f in quads_files:
        if f.exists():
            quads_file = f
            break

    if not quads_file:
        raise FileNotFoundError(
            "QuadS fabric list not found. Expected files:\n" +
            "\n".join(f"  - {f}" for f in quads_files)
        )

    logger.info(f"Loading QuadS data from: {quads_file}")
    df = pd.read_excel(quads_file)
    logger.info(f"✓ Loaded {len(df)} records from QuadS")
    logger.info(f"✓ Columns: {df.columns.tolist()}")

    return df


def map_quads_columns(df: pd.DataFrame) -> List[Dict]:
    """Map QuadS columns to fabric_specs schema."""
    records = []
    skipped = 0

    # Common column name variations
    style_cols = ['Style#', 'Style #', 'StyleNumber', 'Style', 'Name', 'F ID']
    yds_lb_cols = ['Yds/Lbs', 'Yds/Lb', 'yds_per_lb', 'YdsPerLb', 'Yards per Pound']
    gsm_cols = ['GSM', 'gsm', 'Weight']
    width_cols = ['Overall Width', 'Width', 'width', 'Fabric Width']
    type_cols = ['Fabric Type', 'Type', 'fabric_type', 'FabricType']
    desc_cols = ['Description', 'Desc', 'description']

    # Find actual column names
    style_col = next((c for c in df.columns if c in style_cols), None)
    yds_lb_col = next((c for c in df.columns if c in yds_lb_cols), None)
    gsm_col = next((c for c in df.columns if c in gsm_cols), None)
    width_col = next((c for c in df.columns if c in width_cols), None)
    type_col = next((c for c in df.columns if c in type_cols), None)
    desc_col = next((c for c in df.columns if c in desc_cols), None)

    logger.info("\nColumn mapping:")
    logger.info(f"  Style: {style_col}")
    logger.info(f"  Yds/Lb: {yds_lb_col}")
    logger.info(f"  GSM: {gsm_col}")
    logger.info(f"  Width: {width_col}")
    logger.info(f"  Type: {type_col}")
    logger.info(f"  Description: {desc_col}")

    if not style_col:
        raise ValueError(f"Could not find style column. Available: {df.columns.tolist()}")

    # Process each row
    for idx, row in df.iterrows():
        try:
            style = row.get(style_col)

            # Skip if no style
            if pd.isna(style) or str(style).strip() == '':
                skipped += 1
                continue

            # Extract values
            yds_per_lb = row.get(yds_lb_col) if yds_lb_col else None
            gsm = row.get(gsm_col) if gsm_col else None
            width = row.get(width_col) if width_col else None
            fabric_type = row.get(type_col) if type_col else None
            description = row.get(desc_col) if desc_col else None

            # Convert to appropriate types
            yds_per_lb_value = float(yds_per_lb) if not pd.isna(yds_per_lb) else None
            gsm_value = int(float(gsm)) if not pd.isna(gsm) else None
            width_value = float(width) if not pd.isna(width) else None

            # Calculate yds_per_lb if not provided but GSM and width are available
            if yds_per_lb_value is None and gsm_value is not None and width_value is not None:
                yds_per_lb_value = calculate_yds_per_lb(gsm_value, width_value)
                if yds_per_lb_value is not None:
                    logger.debug(f"Calculated yds/lb for {style}: {yds_per_lb_value}")

            record = {
                'style': str(style).strip(),
                'yds_per_lb': yds_per_lb_value,
                'gsm': gsm_value,
                'width': width_value,
                'fabric_type': str(fabric_type).strip() if not pd.isna(fabric_type) else None,
                'description': str(description).strip() if not pd.isna(description) else None
            }

            records.append(record)

        except Exception as e:
            logger.warning(f"Skipping row {idx}: {e}")
            skipped += 1

    logger.info(f"\n✓ Prepared {len(records)} fabric spec records")
    if skipped > 0:
        logger.warning(f"⚠ Skipped {skipped} rows")

    return records


def import_to_turso(records: List[Dict]) -> int:
    """Import fabric specs to Turso database."""
    batch_size = 50
    total_inserted = 0

    logger.info(f"\nImporting {len(records)} records to Turso...")

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        statements = []

        for record in batch:
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

        try:
            execute_batch(statements)
            total_inserted += len(batch)
            logger.info(
                f"✓ Batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}: "
                f"{total_inserted}/{len(records)} records"
            )
        except Exception as e:
            logger.error(f"❌ Error inserting batch: {e}")
            raise

    return total_inserted


def verify_import() -> None:
    """Verify imported data."""
    logger.info("\n" + "="*70)
    logger.info("Verifying imported data...")
    logger.info("="*70)

    # Count records
    result = execute_turso_sql("SELECT COUNT(*) as count FROM fabric_specs")

    # Handle list or dict response
    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if result and "results" in result and len(result["results"]) > 0:
        rows = result["results"][0].get("rows", [])
        if rows and len(rows) > 0 and len(rows[0]) > 0:
            count = rows[0][0]
            logger.info(f"✓ Total fabric_specs records: {count}")

            # Sample data
            result = execute_turso_sql("""
                SELECT style, yds_per_lb, gsm, width, fabric_type
                FROM fabric_specs
                LIMIT 5
            """)

            # Handle list or dict response
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            if result and "results" in result and len(result["results"]) > 0:
                rows = result["results"][0].get("rows", [])
                if rows:
                    logger.info("\nSample records:")
                    logger.info("-" * 70)
                    for row in rows[:5]:
                        if len(row) >= 5:
                            style, yds_lb, gsm, width, ftype = row[0], row[1], row[2], row[3], row[4]
                            logger.info(
                                f"  Style: {str(style)[:15]:15} | Yds/Lb: {str(yds_lb or 'N/A')[:6]:6} | "
                                f"GSM: {str(gsm or 'N/A')[:4]:4} | Width: {str(width or 'N/A')[:5]:5} | Type: {str(ftype or 'N/A')[:20]:20}"
                            )
        else:
            logger.warning("No records found in fabric_specs table")
    else:
        logger.warning("Could not verify import - unexpected response format")


def main() -> None:
    """Main import process."""
    try:
        logger.info("="*70)
        logger.info("🚀 QuadS Fabric Specs Import to Turso")
        logger.info("="*70 + "\n")

        # Step 1: Ensure table exists
        ensure_fabric_specs_table()

        # Step 2: Load QuadS data
        df = load_quads_fabric_data()

        # Step 3: Map columns
        records = map_quads_columns(df)

        # Step 4: Import to Turso
        inserted_count = import_to_turso(records)

        # Step 5: Verify
        verify_import()

        # Summary
        logger.info("\n" + "="*70)
        logger.info("✅ Import Complete!")
        logger.info(f"   Records imported: {inserted_count}")
        logger.info("="*70)

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
