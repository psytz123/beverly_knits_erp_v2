#!/usr/bin/env python3
"""
Create two separate tables for finished and greige fabrics
Import correct QuadS data into separate tables
"""

import os
import sys
import pandas as pd
import httpx
import logging
import time
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
    """Execute SQL on Turso."""
    payload = {"statements": [{"q": sql, "params": []}]}
    response = turso_client.post(database_url, json=payload)
    response.raise_for_status()
    return response.json()


def execute_batch_with_retry(statements: List[Dict], max_retries: int = 3) -> bool:
    """Execute batch with retry logic."""
    payload = {"statements": statements}

    for attempt in range(max_retries):
        try:
            response = turso_client.post(database_url, json=payload)
            response.raise_for_status()
            return True
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return False
        except Exception as e:
            logger.error(f"Batch error: {e}")
            return False
    return False


def create_tables():
    """Create two separate tables for finished and greige fabrics."""
    logger.info("Creating separate tables for finished and greige fabrics...")

    # Drop old table if exists
    logger.info("Dropping old fabric_specs table...")
    try:
        execute_sql("DROP TABLE IF EXISTS fabric_specs")
    except:
        pass

    # Create finished_fabric_specs table
    logger.info("Creating finished_fabric_specs table...")
    execute_sql("""
        CREATE TABLE IF NOT EXISTS finished_fabric_specs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            f_id TEXT NOT NULL UNIQUE,
            name TEXT,
            gsm INTEGER,
            overall_width REAL,
            cuttable_width REAL,
            yds_per_lb REAL,
            oz_lin_yd REAL,
            oz_sq_yd REAL,
            gum REAL,
            trim REAL,
            stage TEXT,
            finish_code TEXT,
            finish_type TEXT,
            fiber TEXT,
            composition TEXT,
            construction TEXT,
            specialty TEXT,
            custodian TEXT,
            g_base TEXT,
            fbase_id TEXT,
            g_base_id TEXT,
            level TEXT,
            by_eaches TEXT,
            roll_wt REAL,
            roll_length REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create greige_fabric_specs table
    logger.info("Creating greige_fabric_specs table...")
    execute_sql("""
        CREATE TABLE IF NOT EXISTS greige_fabric_specs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            g_id TEXT NOT NULL UNIQUE,
            g_base TEXT,
            g_base_id TEXT,
            g_version TEXT,
            g_stage TEXT,
            customer TEXT,
            work_center TEXT,
            c_id TEXT,
            construction TEXT,
            custodian TEXT,
            ref_style TEXT,
            knit_price REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    logger.info("[OK] Tables created successfully\n")


def import_finished_fabrics():
    """Import finished fabric data."""
    logger.info("="*70)
    logger.info("Importing Finished Fabrics")
    logger.info("="*70 + "\n")

    file_path = r"c:\Users\psytz\Downloads\QuadS_finishedFabricList_ (7).xlsx"
    df = pd.read_excel(file_path)
    logger.info(f"Loaded {len(df)} finished fabric records\n")

    records = []
    calculated = 0

    for idx, row in df.iterrows():
        try:
            f_id = row.get('F ID')
            if pd.isna(f_id):
                continue

            # Extract all fields
            gsm = float(row.get('GSM')) if not pd.isna(row.get('GSM')) else None
            overall_width = float(row.get('Overall Width')) if not pd.isna(row.get('Overall Width')) else None

            # Calculate yds/lb
            yds_per_lb = calculate_yds_per_lb(gsm, overall_width)
            if yds_per_lb:
                calculated += 1

            records.append({
                'f_id': str(f_id),
                'name': str(row.get('Name')) if not pd.isna(row.get('Name')) else None,
                'gsm': int(gsm) if gsm else None,
                'overall_width': overall_width,
                'cuttable_width': float(row.get('Cuttable Width')) if not pd.isna(row.get('Cuttable Width')) else None,
                'yds_per_lb': yds_per_lb,
                'oz_lin_yd': float(row.get('Oz/Lin Yd')) if not pd.isna(row.get('Oz/Lin Yd')) else None,
                'oz_sq_yd': float(row.get('Oz / Sq Yd')) if not pd.isna(row.get('Oz / Sq Yd')) else None,
                'gum': float(row.get('Gum')) if not pd.isna(row.get('Gum')) else None,
                'trim': float(row.get('Trim')) if not pd.isna(row.get('Trim')) else None,
                'stage': str(row.get('Stage')) if not pd.isna(row.get('Stage')) else None,
                'finish_code': str(row.get('Finish Code')) if not pd.isna(row.get('Finish Code')) else None,
                'finish_type': str(row.get('Finish Type')) if not pd.isna(row.get('Finish Type')) else None,
                'fiber': str(row.get('Fiber')) if not pd.isna(row.get('Fiber')) else None,
                'composition': str(row.get('Composition')) if not pd.isna(row.get('Composition')) else None,
                'construction': str(row.get('Construction')) if not pd.isna(row.get('Construction')) else None,
                'specialty': str(row.get('Specialty')) if not pd.isna(row.get('Specialty')) else None,
                'custodian': str(row.get('Custodian')) if not pd.isna(row.get('Custodian')) else None,
                'g_base': str(row.get('G Base')) if not pd.isna(row.get('G Base')) else None,
                'fbase_id': str(row.get('FBase ID')) if not pd.isna(row.get('FBase ID')) else None,
                'g_base_id': str(row.get('G Base ID')) if not pd.isna(row.get('G Base ID')) else None,
                'level': str(row.get('Level')) if not pd.isna(row.get('Level')) else None,
                'by_eaches': str(row.get('By Eaches?')) if not pd.isna(row.get('By Eaches?')) else None,
                'roll_wt': float(row.get('Roll Wt')) if not pd.isna(row.get('Roll Wt')) else None,
                'roll_length': float(row.get('Roll Length')) if not pd.isna(row.get('Roll Length')) else None,
            })

        except Exception as e:
            logger.warning(f"Skipping row {idx}: {e}")
            continue

    logger.info(f"Prepared {len(records)} finished fabric records")
    logger.info(f"Calculated yds/lb for {calculated} records\n")

    # Import in batches
    batch_size = 25
    total_inserted = 0
    failed = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        statements = []

        for record in batch:
            sql = """
                INSERT OR REPLACE INTO finished_fabric_specs
                (f_id, name, gsm, overall_width, cuttable_width, yds_per_lb,
                 oz_lin_yd, oz_sq_yd, gum, trim, stage, finish_code, finish_type,
                 fiber, composition, construction, specialty, custodian, g_base,
                 fbase_id, g_base_id, level, by_eaches, roll_wt, roll_length, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """
            params = [
                record['f_id'], record['name'], record['gsm'], record['overall_width'],
                record['cuttable_width'], record['yds_per_lb'], record['oz_lin_yd'],
                record['oz_sq_yd'], record['gum'], record['trim'], record['stage'],
                record['finish_code'], record['finish_type'], record['fiber'],
                record['composition'], record['construction'], record['specialty'],
                record['custodian'], record['g_base'], record['fbase_id'],
                record['g_base_id'], record['level'], record['by_eaches'],
                record['roll_wt'], record['roll_length']
            ]
            statements.append({"q": sql, "params": params})

        if execute_batch_with_retry(statements):
            total_inserted += len(batch)
            logger.info(f"Batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}: {total_inserted}/{len(records)}")
            if (i // batch_size + 1) % 10 == 0:
                time.sleep(1)
        else:
            failed += 1
            if failed > 5:
                logger.error("Too many failures, stopping")
                break

    logger.info(f"\n[OK] Inserted {total_inserted} finished fabric records\n")
    return total_inserted


def import_greige_fabrics():
    """Import greige fabric data."""
    logger.info("="*70)
    logger.info("Importing Greige Fabrics")
    logger.info("="*70 + "\n")

    file_path = r"c:\Users\psytz\Downloads\QuadS_greigeFabricList_ (7).xlsx"
    df = pd.read_excel(file_path)
    logger.info(f"Loaded {len(df)} greige fabric records\n")

    records = []

    for idx, row in df.iterrows():
        try:
            g_id = row.get('G ID')
            if pd.isna(g_id):
                continue

            records.append({
                'g_id': str(g_id),
                'g_base': str(row.get('G Base')) if not pd.isna(row.get('G Base')) else None,
                'g_base_id': str(row.get('G Base ID')) if not pd.isna(row.get('G Base ID')) else None,
                'g_version': str(row.get('G Version')) if not pd.isna(row.get('G Version')) else None,
                'g_stage': str(row.get('G Stage')) if not pd.isna(row.get('G Stage')) else None,
                'customer': str(row.get('Customer')) if not pd.isna(row.get('Customer')) else None,
                'work_center': str(row.get('Work Center')) if not pd.isna(row.get('Work Center')) else None,
                'c_id': str(row.get('C ID')) if not pd.isna(row.get('C ID')) else None,
                'construction': str(row.get('Construction')) if not pd.isna(row.get('Construction')) else None,
                'custodian': str(row.get('Custodian')) if not pd.isna(row.get('Custodian')) else None,
                'ref_style': str(row.get('Ref Style')) if not pd.isna(row.get('Ref Style')) else None,
                'knit_price': float(row.get('Knit Price')) if not pd.isna(row.get('Knit Price')) else None,
            })

        except Exception as e:
            logger.warning(f"Skipping row {idx}: {e}")
            continue

    logger.info(f"Prepared {len(records)} greige fabric records\n")

    # Import in batches
    batch_size = 25
    total_inserted = 0
    failed = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        statements = []

        for record in batch:
            sql = """
                INSERT OR REPLACE INTO greige_fabric_specs
                (g_id, g_base, g_base_id, g_version, g_stage, customer, work_center,
                 c_id, construction, custodian, ref_style, knit_price, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """
            params = [
                record['g_id'], record['g_base'], record['g_base_id'], record['g_version'],
                record['g_stage'], record['customer'], record['work_center'], record['c_id'],
                record['construction'], record['custodian'], record['ref_style'], record['knit_price']
            ]
            statements.append({"q": sql, "params": params})

        if execute_batch_with_retry(statements):
            total_inserted += len(batch)
            logger.info(f"Batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}: {total_inserted}/{len(records)}")
            if (i // batch_size + 1) % 10 == 0:
                time.sleep(1)
        else:
            failed += 1
            if failed > 5:
                logger.error("Too many failures, stopping")
                break

    logger.info(f"\n[OK] Inserted {total_inserted} greige fabric records\n")
    return total_inserted


def verify_tables():
    """Verify both tables."""
    logger.info("="*70)
    logger.info("Verification")
    logger.info("="*70 + "\n")

    # Count finished fabrics
    result = execute_sql("SELECT COUNT(*) FROM finished_fabric_specs")
    if result and "results" in result:
        results_data = result["results"]
        if isinstance(results_data, list) and len(results_data) > 0:
            rows = results_data[0].get("rows", [])
            if rows:
                count = rows[0][0]
                logger.info(f"Finished fabrics: {count}")

    # Count greige fabrics
    result = execute_sql("SELECT COUNT(*) FROM greige_fabric_specs")
    if result and "results" in result:
        results_data = result["results"]
        if isinstance(results_data, list) and len(results_data) > 0:
            rows = results_data[0].get("rows", [])
            if rows:
                count = rows[0][0]
                logger.info(f"Greige fabrics: {count}")

    # Sample finished fabric with yds/lb
    result = execute_sql("""
        SELECT f_id, gsm, overall_width, yds_per_lb
        FROM finished_fabric_specs
        WHERE yds_per_lb IS NOT NULL
        LIMIT 5
    """)
    if result and "results" in result:
        results_data = result["results"]
        if isinstance(results_data, list) and len(results_data) > 0:
            rows = results_data[0].get("rows", [])
            if rows:
                logger.info("\nSample finished fabrics:")
                logger.info("-" * 70)
                for row in rows:
                    logger.info(f"  F ID: {row[0]:10} | GSM: {row[1] or 'N/A':6} | Width: {row[2] or 'N/A':6} | Yds/Lb: {row[3] or 'N/A':6}")


def main():
    """Main import process."""
    try:
        logger.info("="*70)
        logger.info("QuadS Data Import - Separate Tables")
        logger.info("="*70 + "\n")

        # Create tables
        create_tables()

        # Import finished fabrics
        finished_count = import_finished_fabrics()

        # Import greige fabrics
        greige_count = import_greige_fabrics()

        # Verify
        verify_tables()

        # Summary
        logger.info("\n" + "="*70)
        logger.info("[SUCCESS] Import Complete!")
        logger.info(f"  Finished fabrics: {finished_count}")
        logger.info(f"  Greige fabrics: {greige_count}")
        logger.info(f"  Total: {finished_count + greige_count}")
        logger.info("="*70)

        turso_client.close()

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        turso_client.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
