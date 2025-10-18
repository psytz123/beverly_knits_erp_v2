#!/usr/bin/env python3
"""
Import fabric specifications from QuadS API to Turso database
Uses working API endpoints discovered through testing
"""

import os
import sys
import httpx
import logging
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

QUADS_BASE_URL = "https://quads.bkiapps.com"
QUADS_TIMEOUT = 120.0  # Increased for large responses


def calculate_yds_per_lb(gsm: Optional[float], width_inches: Optional[float]) -> Optional[float]:
    """Calculate yards per pound from GSM and width."""
    if gsm is None or width_inches is None or gsm <= 0 or width_inches <= 0:
        return None
    try:
        yds_per_lb = 16129.032 / (gsm * width_inches)
        return round(yds_per_lb, 2)
    except (ZeroDivisionError, ValueError):
        return None


def login_to_quads() -> Optional[httpx.Client]:
    """Login to QuadS and return authenticated client."""
    username = os.getenv("QUADS_USERNAME", "")
    password = os.getenv("QUADS_PASSWORD", "")

    if not username or not password:
        logger.error("QUADS_USERNAME or QUADS_PASSWORD not set in .env")
        return None

    client = httpx.Client(follow_redirects=True, timeout=QUADS_TIMEOUT)

    try:
        logger.info(f"Logging in to QuadS as {username}...")

        response = client.post(
            f"{QUADS_BASE_URL}/login",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        if 'x-dancer-username' in response.headers:
            logged_in_user = response.headers['x-dancer-username']
            if logged_in_user != '-' and logged_in_user == username:
                logger.info(f"Successfully logged in to QuadS as {logged_in_user}")
                return client

        logger.error("Login failed - check credentials")
        return None

    except Exception as e:
        logger.error(f"Error during login: {e}")
        return None


def fetch_styles_from_quads(client: httpx.Client, style_type: str) -> List[Dict]:
    """
    Fetch fabric styles from QuadS API.

    Args:
        client: Authenticated httpx client
        style_type: "finished" or "greige"

    Returns:
        List of style records
    """
    endpoint = f"/api/styles/{style_type}/active"
    url = f"{QUADS_BASE_URL}{endpoint}"

    try:
        logger.info(f"Fetching {style_type} styles from {url}...")

        response = client.get(url, timeout=QUADS_TIMEOUT)
        response.raise_for_status()

        # Parse JSON (even though content-type might say text/html)
        data = response.json()

        if isinstance(data, list):
            logger.info(f"Fetched {len(data)} {style_type} styles from QuadS")

            # Log sample record structure
            if len(data) > 0:
                logger.info(f"Sample record keys: {list(data[0].keys())[:10]}")

            return data
        else:
            logger.warning(f"Unexpected response type: {type(data)}")
            return []

    except httpx.ReadTimeout:
        logger.error(f"Timeout fetching {style_type} styles (endpoint may be slow or not available)")
        return []
    except Exception as e:
        logger.error(f"Error fetching {style_type} styles: {e}")
        return []


def map_quads_record_to_fabric_spec(record: Dict, style_type: str) -> Optional[Dict]:
    """
    Map QuadS API record to fabric_specs schema.

    QuadS fields (discovered from greige endpoint):
    - style_id: Style ID number
    - base: Base style code
    - ref_style: Reference style
    - construction: Fabric construction
    - customer: Customer name
    - etc.
    """
    try:
        # Try to find style identifier
        style = (
            record.get('base') or
            record.get('ref_style') or
            record.get('style_id') or
            record.get('id')
        )

        if not style:
            return None

        # Extract GSM, width, and other specs
        # Note: QuadS might use different field names
        gsm = record.get('gsm') or record.get('weight')
        width = record.get('width') or record.get('overall_width')

        # Try to parse from strings if needed
        if gsm and isinstance(gsm, str):
            try:
                gsm = float(gsm)
            except:
                gsm = None

        if width and isinstance(width, str):
            try:
                width = float(width)
            except:
                width = None

        # Calculate yds/lb
        yds_per_lb = calculate_yds_per_lb(gsm, width)

        # Get description
        description = (
            record.get('construction') or
            record.get('description') or
            record.get('ref_style')
        )

        return {
            'style': str(style).strip(),
            'yds_per_lb': yds_per_lb,
            'gsm': int(gsm) if gsm else None,
            'width': width,
            'fabric_type': style_type,
            'description': str(description).strip() if description else None
        }

    except Exception as e:
        logger.warning(f"Error mapping record: {e}")
        return None


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


def import_to_turso(records: List[Dict]) -> int:
    """Import fabric specs to Turso database."""
    if not records:
        return 0

    batch_size = 50
    total_inserted = 0

    logger.info(f"Importing {len(records)} records to Turso...")

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
            logger.info(f"Batch {i//batch_size + 1}: {total_inserted}/{len(records)} records")
        except Exception as e:
            logger.error(f"Error inserting batch: {e}")
            raise

    return total_inserted


def main():
    """Main import process."""
    try:
        logger.info("="*70)
        logger.info("QuadS API Import to Turso")
        logger.info("="*70 + "\n")

        # Login to QuadS
        client = login_to_quads()

        if not client:
            logger.error("Cannot proceed without QuadS session")
            sys.exit(1)

        # Fetch greige styles (known working endpoint)
        greige_raw = fetch_styles_from_quads(client, "greige")

        # Try finished styles (may timeout)
        finished_raw = fetch_styles_from_quads(client, "finished")

        # Map records
        greige_records = []
        for record in greige_raw:
            mapped = map_quads_record_to_fabric_spec(record, "greige")
            if mapped:
                greige_records.append(mapped)

        finished_records = []
        for record in finished_raw:
            mapped = map_quads_record_to_fabric_spec(record, "finished")
            if mapped:
                finished_records.append(mapped)

        logger.info(f"\nMapped {len(greige_records)} greige records")
        logger.info(f"Mapped {len(finished_records)} finished records")

        # Combine and import
        all_records = greige_records + finished_records

        if not all_records:
            logger.warning("No records to import")
            sys.exit(0)

        total_imported = import_to_turso(all_records)

        # Summary
        logger.info("\n" + "="*70)
        logger.info("Import Complete!")
        logger.info(f"  Greige styles: {len(greige_records)}")
        logger.info(f"  Finished styles: {len(finished_records)}")
        logger.info(f"  Total imported: {total_imported}")
        logger.info("="*70)

        client.close()

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
