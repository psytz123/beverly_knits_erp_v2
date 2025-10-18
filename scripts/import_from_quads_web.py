#!/usr/bin/env python3
"""
Import fabric specifications from QuadS web interface to Turso database
Uses web scraping since QuadS uses a web interface rather than REST API
"""

import os
import sys
import httpx
import logging
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


QUADS_BASE_URL = "https://quads.bkiapps.com"
QUADS_TIMEOUT = 30.0


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
    """
    Login to QuadS and return authenticated client.

    Returns:
        Authenticated httpx.Client with session cookie, or None if login failed
    """
    username = os.getenv("QUADS_USERNAME", "")
    password = os.getenv("QUADS_PASSWORD", "")

    if not username or not password:
        logger.error("QUADS_USERNAME or QUADS_PASSWORD not set in .env")
        return None

    # Create persistent client
    client = httpx.Client(follow_redirects=True, timeout=QUADS_TIMEOUT)

    try:
        logger.info(f"Logging in to QuadS as {username}...")

        response = client.post(
            f"{QUADS_BASE_URL}/login",
            data={
                "username": username,
                "password": password
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        # Check for successful login
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


def fetch_fabric_styles(client: httpx.Client, style_type: str = "finished") -> List[Dict]:
    """
    Fetch fabric styles from QuadS web interface.

    Args:
        client: Authenticated httpx client
        style_type: "finished" or "greige"

    Returns:
        List of fabric spec dictionaries
    """
    url = f"{QUADS_BASE_URL}/knit-style/list/{style_type}"

    try:
        logger.info(f"Fetching {style_type} fabric styles from QuadS...")

        response = client.get(url)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the data table (adapt selectors based on actual HTML structure)
        # This is a placeholder - you'll need to inspect the actual HTML
        table = soup.find('table')

        if not table:
            logger.warning("Could not find fabric table in response")
            return []

        records = []

        # Parse table rows
        rows = table.find_all('tr')[1:]  # Skip header row

        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 3:  # Minimum expected columns
                continue

            try:
                # Extract data from columns (adapt based on actual table structure)
                style = cols[0].get_text(strip=True)
                gsm_text = cols[1].get_text(strip=True) if len(cols) > 1 else None
                width_text = cols[2].get_text(strip=True) if len(cols) > 2 else None

                # Parse numeric values
                gsm = int(gsm_text) if gsm_text and gsm_text.isdigit() else None
                width = float(width_text) if width_text and width_text.replace('.', '').isdigit() else None

                # Calculate yds/lb
                yds_per_lb = calculate_yds_per_lb(gsm, width)

                records.append({
                    'style': style,
                    'yds_per_lb': yds_per_lb,
                    'gsm': gsm,
                    'width': width,
                    'fabric_type': style_type,
                    'description': None
                })

            except Exception as e:
                logger.warning(f"Error parsing row: {e}")
                continue

        logger.info(f"Parsed {len(records)} {style_type} fabric styles")
        return records

    except Exception as e:
        logger.error(f"Error fetching {style_type} styles: {e}")
        return []


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
        logger.info("QuadS Web Import to Turso")
        logger.info("="*70 + "\n")

        # Login to QuadS
        client = login_to_quads()

        if not client:
            logger.error("Cannot proceed without QuadS session")
            logger.info("\nPlease add to your .env file:")
            logger.info("  QUADS_USERNAME=your_username")
            logger.info("  QUADS_PASSWORD=your_password")
            sys.exit(1)

        # Fetch finished styles
        finished_records = fetch_fabric_styles(client, "finished")

        # Fetch greige styles
        greige_records = fetch_fabric_styles(client, "greige")

        # Combine records
        all_records = finished_records + greige_records

        # Import to Turso
        total_imported = import_to_turso(all_records)

        # Summary
        logger.info("\n" + "="*70)
        logger.info("Import Complete!")
        logger.info(f"  Finished styles: {len(finished_records)}")
        logger.info(f"  Greige styles: {len(greige_records)}")
        logger.info(f"  Total imported: {total_imported}")
        logger.info("="*70)

        # Close client
        client.close()

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
