#!/usr/bin/env python3
"""
Import CORRECT fabric specifications from QuadS Excel files to Turso database
Clears old incorrect data and imports fresh data with proper calculations
"""

import os
import sys
import pandas as pd
import httpx
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_yds_per_lb(gsm: Optional[float], width_inches: Optional[float]) -> Optional[float]:
    """Calculate yards per pound from GSM and width."""
    if gsm is None or width_inches is None or gsm <= 0 or width_inches <= 0:
        return None
    try:
        yds_per_lb = 16129.032 / (gsm * width_inches)
        return round(yds_per_lb, 2)
    except (ZeroDivisionError, ValueError):
        return None


def execute_turso_sql(sql: str, params: Optional[List] = None) -> dict:
    """Execute SQL on Turso database."""
    database_url = os.getenv("TURSO_DATABASE_URL", "").replace("libsql://", "https://")
    auth_token = os.getenv("TURSO_AUTH_TOKEN", "")

    payload = {"statements": [{"q": sql, "params": params or []}]}

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
        timeout=120.0
    )

    response.raise_for_status()
    return response.json()


def clear_old_data():
    """Clear old incorrect data from fabric_specs table."""
    logger.info("Clearing old incorrect data...")
    try:
        execute_turso_sql("DELETE FROM fabric_specs")
        logger.info("[OK] Old data cleared")
    except Exception as e:
        logger.error(f"Error clearing old data: {e}")
        raise


def import_finished_fabrics(file_path: str) -> int:
    """Import finished fabric specifications."""
    logger.info(f"\nImporting finished fabrics from: {file_path}")

    df = pd.read_excel(file_path)
    logger.info(f"[OK] Loaded {len(df)} finished fabric records")

    records = []
    skipped = 0
    calculated_yds_lb = 0

    for idx, row in df.iterrows():
        try:
            # Extract F ID (style identifier)
            f_id = row.get('F ID')
            if pd.isna(f_id):
                skipped += 1
                continue

            # Extract GSM and width
            gsm = row.get('GSM')
            width = row.get('Overall Width')

            # Convert to appropriate types
            gsm_value = float(gsm) if not pd.isna(gsm) else None
            width_value = float(width) if not pd.isna(width) else None

            # Calculate yds_per_lb
            yds_per_lb = calculate_yds_per_lb(gsm_value, width_value)
            if yds_per_lb:
                calculated_yds_lb += 1

            # Extract other fields
            name = row.get('Name')
            composition = row.get('Composition')
            construction = row.get('Construction')
            fiber = row.get('Fiber')

            # Build description
            desc_parts = []
            if not pd.isna(construction):
                desc_parts.append(str(construction))
            if not pd.isna(composition):
                desc_parts.append(str(composition))
            if not pd.isna(fiber):
                desc_parts.append(f"Fiber: {fiber}")

            description = " | ".join(desc_parts) if desc_parts else None

            records.append({
                'style': str(f_id),
                'yds_per_lb': yds_per_lb,
                'gsm': int(gsm_value) if gsm_value else None,
                'width': width_value,
                'fabric_type': 'finished',
                'description': description
            })

        except Exception as e:
            logger.warning(f"Skipping row {idx}: {e}")
            skipped += 1

    logger.info(f"[OK] Prepared {len(records)} finished fabric records")
    logger.info(f"[OK] Calculated yds/lb for {calculated_yds_lb} records")
    if skipped > 0:
        logger.warning(f"[!] Skipped {skipped} rows")

    # Batch import
    return import_records_to_turso(records, "finished")


def import_greige_fabrics(file_path: str) -> int:
    """Import greige fabric specifications."""
    logger.info(f"\nImporting greige fabrics from: {file_path}")

    df = pd.read_excel(file_path)
    logger.info(f"[OK] Loaded {len(df)} greige fabric records")

    records = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            # Extract G ID (style identifier)
            g_id = row.get('G ID')
            if pd.isna(g_id):
                skipped += 1
                continue

            # Extract other fields
            g_base = row.get('G Base')
            construction = row.get('Construction')
            customer = row.get('Customer')
            ref_style = row.get('Ref Style')

            # Build description
            desc_parts = []
            if not pd.isna(construction):
                desc_parts.append(str(construction))
            if not pd.isna(g_base):
                desc_parts.append(f"Base: {g_base}")
            if not pd.isna(customer):
                desc_parts.append(f"Customer: {customer}")

            description = " | ".join(desc_parts) if desc_parts else None

            # Note: Greige fabrics don't have GSM/width, so no yds/lb calculation
            records.append({
                'style': str(g_id),
                'yds_per_lb': None,  # No GSM/width data available
                'gsm': None,
                'width': None,
                'fabric_type': 'greige',
                'description': description
            })

        except Exception as e:
            logger.warning(f"Skipping row {idx}: {e}")
            skipped += 1

    logger.info(f"[OK] Prepared {len(records)} greige fabric records")
    if skipped > 0:
        logger.warning(f"[!] Skipped {skipped} rows")
    logger.info(f"[!] Note: Greige fabrics have no GSM/width data - yds/lb not calculated")

    # Batch import
    return import_records_to_turso(records, "greige")


def import_records_to_turso(records: List[Dict], fabric_type: str) -> int:
    """Import records to Turso in batches."""
    batch_size = 50
    total_inserted = 0

    logger.info(f"Importing {len(records)} {fabric_type} records to Turso...")

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
            logger.info(f"  Batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}: {total_inserted}/{len(records)}")
        except Exception as e:
            logger.error(f"Error inserting batch: {e}")
            raise

    return total_inserted


def verify_import():
    """Verify imported data."""
    logger.info("\n" + "="*70)
    logger.info("Verifying imported data...")
    logger.info("="*70)

    # Count total records
    result = execute_turso_sql("SELECT COUNT(*) FROM fabric_specs")
    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if result and "results" in result:
        results_data = result["results"]
        if isinstance(results_data, dict) and "rows" in results_data:
            total_count = results_data["rows"][0][0]
        else:
            total_count = 0

        logger.info(f"[OK] Total records: {total_count}")

    # Count by type
    result = execute_turso_sql("""
        SELECT fabric_type, COUNT(*) as count
        FROM fabric_specs
        GROUP BY fabric_type
    """)

    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if result and "results" in result:
        results_data = result["results"]
        if isinstance(results_data, dict) and "rows" in results_data:
            logger.info("\nRecords by type:")
            for row in results_data["rows"]:
                logger.info(f"  {row[0]}: {row[1]} records")

    # Count with yds/lb
    result = execute_turso_sql("""
        SELECT COUNT(*) FROM fabric_specs
        WHERE yds_per_lb IS NOT NULL
    """)

    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if result and "results" in result:
        results_data = result["results"]
        if isinstance(results_data, dict) and "rows" in results_data:
            with_yds_lb = results_data["rows"][0][0]
            logger.info(f"\n[OK] Records with calculated yds/lb: {with_yds_lb}")

    # Sample finished fabrics
    result = execute_turso_sql("""
        SELECT style, gsm, width, yds_per_lb, fabric_type
        FROM fabric_specs
        WHERE fabric_type = 'finished' AND yds_per_lb IS NOT NULL
        LIMIT 5
    """)

    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if result and "results" in result:
        results_data = result["results"]
        if isinstance(results_data, dict) and "rows" in results_data:
            logger.info("\nSample finished fabrics with yds/lb:")
            logger.info("-" * 70)
            for row in results_data["rows"]:
                style, gsm, width, yds_lb, ftype = row
                logger.info(f"  Style: {style:10} | GSM: {gsm or 'N/A':6} | Width: {width or 'N/A':6} | Yds/Lb: {yds_lb or 'N/A':6}")


def main():
    """Main import process."""
    try:
        logger.info("="*70)
        logger.info("CORRECT QuadS Data Import to Turso")
        logger.info("="*70 + "\n")

        # File paths
        finished_file = r"c:\Users\psytz\Downloads\QuadS_finishedFabricList_ (7).xlsx"
        greige_file = r"c:\Users\psytz\Downloads\QuadS_greigeFabricList_ (7).xlsx"

        # Verify files exist
        if not Path(finished_file).exists():
            logger.error(f"Finished fabric file not found: {finished_file}")
            sys.exit(1)

        if not Path(greige_file).exists():
            logger.error(f"Greige fabric file not found: {greige_file}")
            sys.exit(1)

        # Clear old incorrect data
        clear_old_data()

        # Import finished fabrics
        finished_count = import_finished_fabrics(finished_file)

        # Import greige fabrics
        greige_count = import_greige_fabrics(greige_file)

        # Verify
        verify_import()

        # Summary
        logger.info("\n" + "="*70)
        logger.info("[SUCCESS] Import Complete!")
        logger.info(f"  Finished fabrics: {finished_count}")
        logger.info(f"  Greige fabrics: {greige_count}")
        logger.info(f"  Total imported: {finished_count + greige_count}")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
