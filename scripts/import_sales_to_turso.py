"""
Import Sales Activity Report from Excel to Turso database
"""

import os
import sys
import pandas as pd
import httpx
import logging
from datetime import datetime
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


def import_sales_report(file_path: str):
    """Import sales activity report from Excel."""
    logger.info(f"📊 Importing sales data from: {file_path}")

    # Read Excel file
    try:
        df = pd.read_excel(file_path)
        logger.info(f"✓ Loaded {len(df)} rows from Excel")
        logger.info(f"✓ Columns: {df.columns.tolist()}")
    except Exception as e:
        logger.error(f"❌ Error reading Excel: {e}")
        return

    # Display first few rows to understand structure
    logger.info("\n" + "="*60)
    logger.info("Sample data:")
    logger.info(df.head().to_string())
    logger.info("="*60 + "\n")

    # Map column names (adjust based on actual Excel structure)
    # Common patterns in eFab reports:
    # - 'Date', 'Order Date', 'Ship Date'
    # - 'Style', 'Style#', 'Style Number'
    # - 'Customer', 'Customer Name'
    # - 'Quantity', 'Qty', 'Yards'

    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'date' in col_lower and 'date' not in column_mapping:
            column_mapping['date'] = col
        elif 'style' in col_lower and 'style' not in column_mapping:
            column_mapping['style'] = col
        elif 'customer' in col_lower and 'customer' not in column_mapping:
            column_mapping['customer'] = col
        elif any(x in col_lower for x in ['quantity', 'qty', 'yards']) and 'quantity' not in column_mapping:
            column_mapping['quantity'] = col
        elif 'order' in col_lower and '#' in col_lower and 'order_number' not in column_mapping:
            column_mapping['order_number'] = col

    logger.info(f"Column mapping detected: {column_mapping}")

    # Prepare data for insertion
    records_to_insert = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            # Extract fields with fallbacks
            date_val = row.get(column_mapping.get('date', 'Date'))
            style_val = row.get(column_mapping.get('style', 'Style'))
            customer_val = row.get(column_mapping.get('customer', 'Customer'), 'Unknown')
            quantity_val = row.get(column_mapping.get('quantity', 'Quantity'), 0)
            order_number_val = row.get(column_mapping.get('order_number', 'Order#'), '')

            # Skip if missing critical fields
            if pd.isna(date_val) or pd.isna(style_val) or pd.isna(quantity_val):
                skipped += 1
                continue

            # Convert date to string
            if isinstance(date_val, pd.Timestamp):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val)

            # Convert quantity to float
            try:
                qty_float = float(quantity_val)
            except (ValueError, TypeError):
                skipped += 1
                continue

            if qty_float <= 0:
                skipped += 1
                continue

            records_to_insert.append({
                'date': date_str,
                'style': str(style_val),
                'customer': str(customer_val),
                'quantity': qty_float,
                'order_number': str(order_number_val) if not pd.isna(order_number_val) else ''
            })

        except Exception as e:
            logger.warning(f"Skipping row {idx}: {e}")
            skipped += 1
            continue

    logger.info(f"✓ Prepared {len(records_to_insert)} records for insertion")
    if skipped > 0:
        logger.info(f"⚠ Skipped {skipped} rows due to missing/invalid data")

    # Batch insert in groups of 50 for efficiency
    batch_size = 50
    total_inserted = 0

    for i in range(0, len(records_to_insert), batch_size):
        batch = records_to_insert[i:i + batch_size]
        statements = []

        for record in batch:
            # Use INSERT OR REPLACE to handle duplicates
            sql = """
                INSERT OR REPLACE INTO historical_sales
                (date, style, customer, quantity, units, order_number)
                VALUES (?, ?, ?, ?, 'yards', ?)
            """
            params = [
                record['date'],
                record['style'],
                record['customer'],
                record['quantity'],
                record['order_number']
            ]
            statements.append({"q": sql, "params": params})

        try:
            execute_batch(statements)
            total_inserted += len(batch)
            logger.info(f"✓ Inserted batch {i//batch_size + 1}: {total_inserted}/{len(records_to_insert)} records")
        except Exception as e:
            logger.error(f"❌ Error inserting batch: {e}")
            continue

    # Verify data in database
    result = execute_turso_sql("SELECT COUNT(*) as count FROM historical_sales")
    if result.get("results"):
        total_count = result["results"][0]["rows"][0][0]
        logger.info(f"\n✅ Import complete!")
        logger.info(f"Total records in database: {total_count}")

    # Get unique styles
    result = execute_turso_sql("SELECT DISTINCT style FROM historical_sales ORDER BY style")
    if result.get("results") and result["results"][0].get("rows"):
        styles = [row[0] for row in result["results"][0]["rows"]]
        logger.info(f"Unique styles: {len(styles)}")
        logger.info(f"Sample styles: {styles[:10]}")

    # Get date range
    result = execute_turso_sql("""
        SELECT MIN(date) as min_date, MAX(date) as max_date
        FROM historical_sales
    """)
    if result.get("results") and result["results"][0].get("rows"):
        date_range = result["results"][0]["rows"][0]
        logger.info(f"Date range: {date_range[0]} to {date_range[1]}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python import_sales_to_turso.py <path_to_excel_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        sys.exit(1)

    try:
        import_sales_report(file_path)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
