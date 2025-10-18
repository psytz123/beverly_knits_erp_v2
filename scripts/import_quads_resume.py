#!/usr/bin/env python3
"""
Resume QuadS import with retry logic and smaller batches
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


# Create persistent client
database_url = os.getenv("TURSO_DATABASE_URL", "").replace("libsql://", "https://")
auth_token = os.getenv("TURSO_AUTH_TOKEN", "")

turso_client = httpx.Client(
    headers={
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    },
    timeout=60.0
)


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
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                logger.error("Max retries reached")
                return False

        except Exception as e:
            logger.error(f"Batch error: {e}")
            return False

    return False


def import_all_data():
    """Import all data with resume capability."""
    logger.info("="*70)
    logger.info("QuadS Data Import (Resumable)")
    logger.info("="*70 + "\n")

    # File paths
    finished_file = r"c:\Users\psytz\Downloads\QuadS_finishedFabricList_ (7).xlsx"
    greige_file = r"c:\Users\psytz\Downloads\QuadS_greigeFabricList_ (7).xlsx"

    # Load data
    logger.info("Loading Excel files...")
    df_finished = pd.read_excel(finished_file)
    df_greige = pd.read_excel(greige_file)

    logger.info(f"Finished fabrics: {len(df_finished)} records")
    logger.info(f"Greige fabrics: {len(df_greige)} records\n")

    # Prepare all records
    all_records = []

    # Process finished fabrics
    logger.info("Processing finished fabrics...")
    for idx, row in df_finished.iterrows():
        try:
            f_id = row.get('F ID')
            if pd.isna(f_id):
                continue

            gsm = float(row.get('GSM')) if not pd.isna(row.get('GSM')) else None
            width = float(row.get('Overall Width')) if not pd.isna(row.get('Overall Width')) else None
            yds_per_lb = calculate_yds_per_lb(gsm, width)

            construction = row.get('Construction')
            composition = row.get('Composition')

            desc_parts = []
            if not pd.isna(construction):
                desc_parts.append(str(construction))
            if not pd.isna(composition):
                desc_parts.append(str(composition))

            all_records.append({
                'style': str(f_id),
                'yds_per_lb': yds_per_lb,
                'gsm': int(gsm) if gsm else None,
                'width': width,
                'fabric_type': 'finished',
                'description': " | ".join(desc_parts) if desc_parts else None
            })
        except:
            continue

    # Process greige fabrics
    logger.info("Processing greige fabrics...")
    for idx, row in df_greige.iterrows():
        try:
            g_id = row.get('G ID')
            if pd.isna(g_id):
                continue

            construction = row.get('Construction')
            g_base = row.get('G Base')

            desc_parts = []
            if not pd.isna(construction):
                desc_parts.append(str(construction))
            if not pd.isna(g_base):
                desc_parts.append(f"Base: {g_base}")

            all_records.append({
                'style': str(g_id),
                'yds_per_lb': None,
                'gsm': None,
                'width': None,
                'fabric_type': 'greige',
                'description': " | ".join(desc_parts) if desc_parts else None
            })
        except:
            continue

    logger.info(f"\n[OK] Prepared {len(all_records)} total records\n")

    # Import in small batches with pause
    batch_size = 25  # Smaller batches to avoid rate limits
    total_inserted = 0
    failed_batches = 0

    for i in range(0, len(all_records), batch_size):
        batch = all_records[i:i + batch_size]
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

        # Execute with retry
        success = execute_batch_with_retry(statements)

        if success:
            total_inserted += len(batch)
            logger.info(f"Batch {i//batch_size + 1}/{(len(all_records)-1)//batch_size + 1}: {total_inserted}/{len(all_records)}")

            # Small pause to avoid rate limiting
            if (i // batch_size + 1) % 10 == 0:
                time.sleep(1)
        else:
            failed_batches += 1
            logger.error(f"Failed to insert batch {i//batch_size + 1}")

            if failed_batches > 5:
                logger.error("Too many failed batches, stopping")
                break

    # Summary
    logger.info("\n" + "="*70)
    logger.info(f"[COMPLETE] Inserted {total_inserted}/{len(all_records)} records")
    logger.info(f"Failed batches: {failed_batches}")
    logger.info("="*70)

    turso_client.close()


if __name__ == "__main__":
    try:
        import_all_data()
    except KeyboardInterrupt:
        logger.info("\nImport interrupted by user")
        turso_client.close()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        turso_client.close()
