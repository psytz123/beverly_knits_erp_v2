#!/usr/bin/env python3
"""
Import fabric specifications from QuadS API to Turso database
"""

import os
import sys
import httpx
import logging
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# QuadS API Configuration
QUADS_BASE_URL = "https://quads.bkiapps.com"
QUADS_API_PREFIX = "/api"
QUADS_TIMEOUT = 30.0


def login_to_quads() -> Optional[str]:
    """
    Login to QuadS using credentials from environment.

    Returns:
        Session token if successful, None otherwise
    """
    username = os.getenv("QUADS_USERNAME", "")
    password = os.getenv("QUADS_PASSWORD", "")

    if not username or not password:
        logger.warning("QUADS_USERNAME or QUADS_PASSWORD not set in .env")
        return None

    # Try multiple login endpoints
    login_urls = [
        os.getenv("QUADS_LOGIN_URL", f"{QUADS_BASE_URL}/LOGIN"),
        f"{QUADS_BASE_URL}/api/auth/login",
        f"{QUADS_BASE_URL}/login"
    ]

    for login_url in login_urls:
        try:
            logger.info(f"Attempting login to {login_url} as {username}...")

            # Try JSON payload first
            response = httpx.post(
                login_url,
                json={
                    "username": username,
                    "password": password
                },
                headers={"Content-Type": "application/json"},
                timeout=QUADS_TIMEOUT,
                follow_redirects=True
            )

            # If JSON fails, try form data
            if response.status_code >= 400:
                response = httpx.post(
                    login_url,
                    data={
                        "username": username,
                        "password": password
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=QUADS_TIMEOUT,
                    follow_redirects=True
                )

            if response.status_code == 200:
                # Check for token in response
                try:
                    data = response.json()
                    token = (
                        data.get('token') or
                        data.get('session_token') or
                        data.get('access_token') or
                        data.get('sessionToken')
                    )
                except:
                    token = None

                # Check cookies
                if not token and 'session' in response.cookies:
                    token = response.cookies['session']

                # Check Set-Cookie header
                if not token and 'Set-Cookie' in response.headers:
                    cookie_header = response.headers['Set-Cookie']
                    if 'session=' in cookie_header:
                        token = cookie_header.split('session=')[1].split(';')[0]

                # Check connect.sid cookie (common in Express apps)
                if not token and 'connect.sid' in response.cookies:
                    token = response.cookies['connect.sid']

                if token:
                    logger.info(f"Successfully logged in to QuadS via {login_url}")
                    return token
                else:
                    logger.warning(f"Login to {login_url} succeeded but no token found")

        except httpx.HTTPStatusError as e:
            logger.debug(f"Login to {login_url} failed: {e.response.status_code}")
            continue
        except Exception as e:
            logger.debug(f"Error with {login_url}: {e}")
            continue

    # No login URL worked
    logger.error("Could not login to QuadS with any endpoint")
    return None


def get_quads_session_token() -> str:
    """
    Get QuadS session token.
    First tries to login with credentials, then falls back to manual token.
    """
    # Try automatic login first
    token = login_to_quads()

    if token:
        return token

    # Fall back to manual token from .env
    token = os.getenv("QUADS_SESSION_TOKEN", "")
    if token:
        logger.info("Using QUADS_SESSION_TOKEN from .env")
        return token

    raise ValueError(
        "Cannot get QuadS session token. Please either:\n"
        "  1. Add QUADS_USERNAME and QUADS_PASSWORD to .env for automatic login, OR\n"
        "  2. Add QUADS_SESSION_TOKEN to .env manually"
    )


def fetch_from_quads(endpoint: str, session_token: str) -> List[Dict]:
    """
    Fetch data from QuadS API.

    Args:
        endpoint: API endpoint path (e.g., '/styles/finished/active')
        session_token: QuadS session token

    Returns:
        List of records from API response
    """
    url = f"{QUADS_BASE_URL}{QUADS_API_PREFIX}{endpoint}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {session_token}",
        "Cookie": f"session={session_token}"
    }

    logger.info(f"Fetching from QuadS: {url}")

    try:
        response = httpx.get(
            url,
            headers=headers,
            timeout=QUADS_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()

        # Handle different response formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Check for common response wrapper patterns
            if 'data' in data:
                return data['data'] if isinstance(data['data'], list) else [data['data']]
            elif 'items' in data:
                return data['items']
            elif 'results' in data:
                return data['results']
            else:
                return [data]
        else:
            logger.warning(f"Unexpected response type: {type(data)}")
            return []

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching from QuadS: {e.response.status_code} - {e.response.text}")
        raise
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error fetching from QuadS: {e}")
        raise


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


def calculate_yds_per_lb(gsm: Optional[float], width_inches: Optional[float]) -> Optional[float]:
    """
    Calculate yards per pound from GSM and width.

    Formula: yds_per_lb = 16129.032 / (gsm * width_in_inches)

    This is the standard textile industry formula derived from:
    - 1 pound = 453.592 grams
    - 1 yard = 0.9144 meters
    - 1 inch = 2.54 cm
    - GSM = grams per square meter

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
        # Standard textile formula
        yds_per_lb = 16129.032 / (gsm * width_inches)
        return round(yds_per_lb, 2)
    except (ZeroDivisionError, ValueError):
        return None


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


def map_quads_style_to_fabric_spec(style_data: Dict) -> Optional[Dict]:
    """
    Map QuadS style data to fabric_specs schema.

    Expected QuadS fields (adapt based on actual API response):
    - id, fid, f_id, style_id, style_number (style identifier)
    - gsm, weight (fabric weight)
    - width, overall_width, cuttable_width
    - yds_per_lb, yards_per_pound
    - type, fabric_type
    - name, description
    """
    try:
        # Try multiple possible field names for style ID
        style = (
            style_data.get('id') or
            style_data.get('fid') or
            style_data.get('f_id') or
            style_data.get('F ID') or
            style_data.get('style_id') or
            style_data.get('style_number') or
            style_data.get('styleNumber') or
            style_data.get('Style#')
        )

        if not style:
            logger.warning(f"No style ID found in record: {list(style_data.keys())[:5]}")
            return None

        # Extract other fields with fallbacks
        yds_per_lb = (
            style_data.get('yds_per_lb') or
            style_data.get('yards_per_pound') or
            style_data.get('Yds/Lbs') or
            style_data.get('YdsPerLb')
        )

        gsm = (
            style_data.get('gsm') or
            style_data.get('GSM') or
            style_data.get('weight') or
            style_data.get('Weight')
        )

        width = (
            style_data.get('width') or
            style_data.get('Width') or
            style_data.get('overall_width') or
            style_data.get('Overall Width') or
            style_data.get('overallWidth')
        )

        fabric_type = (
            style_data.get('type') or
            style_data.get('Type') or
            style_data.get('fabric_type') or
            style_data.get('Fabric Type') or
            style_data.get('fabricType')
        )

        description = (
            style_data.get('description') or
            style_data.get('Description') or
            style_data.get('name') or
            style_data.get('Name')
        )

        # Convert to appropriate types
        gsm_value = int(float(gsm)) if gsm is not None else None
        width_value = float(width) if width is not None else None
        yds_per_lb_value = float(yds_per_lb) if yds_per_lb is not None else None

        # Calculate yds_per_lb if not provided but GSM and width are available
        if yds_per_lb_value is None and gsm_value is not None and width_value is not None:
            yds_per_lb_value = calculate_yds_per_lb(gsm_value, width_value)
            if yds_per_lb_value is not None:
                logger.debug(f"Calculated yds/lb for {style}: {yds_per_lb_value}")

        return {
            'style': str(style).strip(),
            'yds_per_lb': yds_per_lb_value,
            'gsm': gsm_value,
            'width': width_value,
            'fabric_type': str(fabric_type).strip() if fabric_type else None,
            'description': str(description).strip() if description else None
        }

    except Exception as e:
        logger.warning(f"Error mapping style data: {e}")
        return None


def import_finished_styles(session_token: str) -> int:
    """Import finished fabric styles from QuadS API."""
    logger.info("Fetching finished fabric styles from QuadS API...")

    try:
        # Fetch active finished styles
        styles = fetch_from_quads("/styles/finished/active", session_token)
        logger.info(f"Fetched {len(styles)} finished styles from QuadS")

        if not styles:
            logger.warning("No styles returned from API")
            return 0

        # Log first record structure for debugging
        if styles:
            logger.info(f"Sample QuadS record keys: {list(styles[0].keys())}")

        # Map to fabric_specs schema
        records = []
        skipped = 0

        for style_data in styles:
            record = map_quads_style_to_fabric_spec(style_data)
            if record:
                records.append(record)
            else:
                skipped += 1

        logger.info(f"Prepared {len(records)} fabric spec records")
        if skipped > 0:
            logger.warning(f"Skipped {skipped} records")

        if not records:
            logger.error("No valid records to import")
            return 0

        # Import to Turso in batches
        batch_size = 50
        total_inserted = 0

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
                    f"Batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}: "
                    f"{total_inserted}/{len(records)} records"
                )
            except Exception as e:
                logger.error(f"Error inserting batch: {e}")
                raise

        return total_inserted

    except Exception as e:
        logger.error(f"Error importing finished styles: {e}")
        raise


def import_greige_styles(session_token: str) -> int:
    """Import greige fabric styles from QuadS API."""
    logger.info("Fetching greige fabric styles from QuadS API...")

    try:
        # Fetch active greige styles
        styles = fetch_from_quads("/styles/greige/active", session_token)
        logger.info(f"Fetched {len(styles)} greige styles from QuadS")

        if not styles:
            logger.warning("No greige styles returned from API")
            return 0

        # Map and import (similar to finished styles)
        records = []
        skipped = 0

        for style_data in styles:
            record = map_quads_style_to_fabric_spec(style_data)
            if record:
                records.append(record)
            else:
                skipped += 1

        logger.info(f"Prepared {len(records)} greige spec records")
        if skipped > 0:
            logger.warning(f"Skipped {skipped} records")

        if not records:
            return 0

        # Import to Turso
        batch_size = 50
        total_inserted = 0

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
                logger.info(f"Greige batch {i//batch_size + 1}: {total_inserted}/{len(records)}")
            except Exception as e:
                logger.error(f"Error inserting greige batch: {e}")
                raise

        return total_inserted

    except Exception as e:
        logger.error(f"Error importing greige styles: {e}")
        raise


def verify_import() -> None:
    """Verify imported data."""
    logger.info("\n" + "="*70)
    logger.info("Verifying imported data...")
    logger.info("="*70)

    # Count records
    result = execute_turso_sql("SELECT COUNT(*) FROM fabric_specs")

    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if result and "results" in result:
        results_data = result["results"]

        if isinstance(results_data, dict) and "rows" in results_data:
            rows = results_data["rows"]
        elif isinstance(results_data, list) and len(results_data) > 0:
            rows = results_data[0].get("rows", [])
        else:
            rows = []

        if rows and len(rows) > 0:
            count = rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]
            logger.info(f"Total fabric_specs records: {count}")

            # Sample data
            result = execute_turso_sql("""
                SELECT style, yds_per_lb, gsm, width, fabric_type
                FROM fabric_specs
                ORDER BY updated_at DESC
                LIMIT 10
            """)

            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            if result and "results" in result:
                results_data = result["results"]

                if isinstance(results_data, dict) and "rows" in results_data:
                    rows = results_data["rows"]
                elif isinstance(results_data, list) and len(results_data) > 0:
                    rows = results_data[0].get("rows", [])
                else:
                    rows = []

                if rows:
                    logger.info("\nRecent fabric specs:")
                    logger.info("-" * 70)
                    for row in rows[:10]:
                        if len(row) >= 5:
                            style, yds_lb, gsm, width, ftype = row[0], row[1], row[2], row[3], row[4]
                            logger.info(
                                f"  Style: {str(style)[:15]:15} | Yds/Lb: {str(yds_lb or 'N/A')[:6]:6} | "
                                f"GSM: {str(gsm or 'N/A')[:4]:4} | Width: {str(width or 'N/A')[:5]:5} | "
                                f"Type: {str(ftype or 'N/A')[:20]:20}"
                            )


def main() -> None:
    """Main import process."""
    try:
        logger.info("="*70)
        logger.info("QuadS API to Turso Import")
        logger.info("="*70 + "\n")

        # Get QuadS session token
        try:
            session_token = get_quads_session_token()
        except ValueError as e:
            logger.error(str(e))
            logger.info("\nTo use this script, add your QuadS session token to .env:")
            logger.info("  QUADS_SESSION_TOKEN=your_token_here")
            sys.exit(1)

        # Import finished styles
        finished_count = import_finished_styles(session_token)
        logger.info(f"\nFinished styles imported: {finished_count}")

        # Import greige styles
        greige_count = import_greige_styles(session_token)
        logger.info(f"Greige styles imported: {greige_count}")

        # Verify
        verify_import()

        # Summary
        logger.info("\n" + "="*70)
        logger.info("Import Complete!")
        logger.info(f"  Finished styles: {finished_count}")
        logger.info(f"  Greige styles: {greige_count}")
        logger.info(f"  Total: {finished_count + greige_count}")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
