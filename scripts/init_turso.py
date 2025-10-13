"""
Initialize Turso database with ERP schema and test data
"""

import os
import sys
import logging
import httpx
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def execute_turso_sql(sql: str) -> dict:
    """Execute SQL on Turso database."""
    database_url = os.getenv("TURSO_DATABASE_URL", "")
    auth_token = os.getenv("TURSO_AUTH_TOKEN", "")

    if database_url.startswith("libsql://"):
        database_url = database_url.replace("libsql://", "https://")

    payload = {
        "statements": [{"q": sql}]
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


def main():
    """Initialize Turso database."""
    logger.info("🚀 Initializing Turso database for Beverly Knits ERP")

    # Test connection
    logger.info("Testing Turso connection...")
    result = execute_turso_sql("SELECT 1 as test")
    logger.info("✓ Connection successful")

    # Create historical_sales table
    logger.info("Creating historical_sales table...")
    execute_turso_sql("""
        CREATE TABLE IF NOT EXISTS historical_sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            style TEXT NOT NULL,
            customer TEXT,
            quantity REAL NOT NULL,
            units TEXT DEFAULT 'yards',
            order_number TEXT,
            ship_date TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    logger.info("✓ historical_sales table created")

    # Create indexes
    execute_turso_sql("""
        CREATE INDEX IF NOT EXISTS idx_sales_date ON historical_sales(date)
    """)
    execute_turso_sql("""
        CREATE INDEX IF NOT EXISTS idx_sales_style ON historical_sales(style)
    """)
    logger.info("✓ Indexes created")

    # Create yarn_inventory table
    logger.info("Creating yarn_inventory table...")
    execute_turso_sql("""
        CREATE TABLE IF NOT EXISTS yarn_inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            yarn_id TEXT NOT NULL UNIQUE,
            description TEXT,
            supplier TEXT,
            color TEXT,
            theoretical_balance REAL DEFAULT 0,
            planning_balance REAL DEFAULT 0,
            allocated REAL DEFAULT 0,
            on_order REAL DEFAULT 0,
            cost_per_lb REAL,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    logger.info("✓ yarn_inventory table created")

    # Create forecast_results table
    logger.info("Creating forecast_results table...")
    execute_turso_sql("""
        CREATE TABLE IF NOT EXISTS forecast_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            style TEXT NOT NULL,
            forecast_date TEXT NOT NULL,
            week_number INTEGER NOT NULL,
            predicted_quantity REAL NOT NULL,
            confidence REAL,
            model_used TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    logger.info("✓ forecast_results table created")

    # Insert sample sales data
    logger.info("Inserting sample historical sales data...")
    styles = ['6191-BK', '80393C-DS', '72762-GS', '71320-BK']

    for style in styles:
        for i in range(52):  # 52 weeks of data
            date = (datetime.now() - timedelta(weeks=52-i)).strftime('%Y-%m-%d')
            quantity = 1000 + (i * 10) + (i % 7) * 50  # Some variation

            sql = f"""
                INSERT INTO historical_sales (date, style, customer, quantity, units)
                VALUES ('{date}', '{style}', 'Customer-{style[:4]}', {quantity}, 'yards')
            """
            execute_turso_sql(sql)

    logger.info(f"✓ Inserted sample data for {len(styles)} styles")

    # Verify data
    result = execute_turso_sql("SELECT COUNT(*) as count FROM historical_sales")
    if result.get("results"):
        count = result["results"][0]["rows"][0][0]
        logger.info(f"✓ Total records in historical_sales: {count}")

    logger.info("=" * 60)
    logger.info("✅ Turso database initialization complete!")
    logger.info("=" * 60)
    logger.info(f"Database URL: {os.getenv('TURSO_DATABASE_URL')}")
    logger.info("Tables created: historical_sales, yarn_inventory, forecast_results")
    logger.info("Ready for ML forecasting!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)
