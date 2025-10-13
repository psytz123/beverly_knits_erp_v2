"""
Turso Database Client for Beverly Knits ERP
Manages all data storage in Turso edge database using HTTP API
"""

import os
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class TursoClient:
    """Client for interacting with Turso database via HTTP API."""

    def __init__(self):
        """Initialize Turso client with credentials from environment."""
        self.database_url = os.getenv("TURSO_DATABASE_URL", "")
        self.auth_token = os.getenv("TURSO_AUTH_TOKEN", "")

        if not self.database_url or not self.auth_token:
            raise ValueError("TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set")

        # Convert libsql:// to https://
        if self.database_url.startswith("libsql://"):
            self.database_url = self.database_url.replace("libsql://", "https://")

        self.client = httpx.Client()
        self._test_connection()

    def _test_connection(self) -> None:
        """Test connection to Turso database."""
        try:
            result = self.execute("SELECT 1 as test")
            logger.info("✓ Connected to Turso database")
        except Exception as e:
            logger.error(f"Failed to connect to Turso: {e}")
            raise

    def execute(self, sql: str, params: Optional[List] = None) -> List[Dict]:
        """
        Execute SQL query on Turso database.

        Args:
            sql: SQL query to execute
            params: Query parameters

        Returns:
            List of result rows as dictionaries
        """
        try:
            payload = {
                "statements": [
                    {
                        "q": sql,
                        "params": params or []
                    }
                ]
            }

            response = self.client.post(
                self.database_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.auth_token}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )

            response.raise_for_status()
            data = response.json()

            # Parse Turso response format
            if "results" in data and len(data["results"]) > 0:
                result = data["results"][0]

                if "rows" in result and "columns" in result:
                    rows = result["rows"]
                    columns = result["columns"]

                    # Convert to list of dicts
                    return [dict(zip(columns, row)) for row in rows]

            return []

        except httpx.HTTPError as e:
            logger.error(f"HTTP error executing SQL: {e}")
            return []
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            return []

    def execute_many(self, sql: str, params_list: List[List]) -> int:
        """
        Execute SQL query multiple times with different parameters.

        Args:
            sql: SQL query
            params_list: List of parameter lists

        Returns:
            Number of successful executions
        """
        try:
            statements = [
                {"q": sql, "params": params}
                for params in params_list
            ]

            payload = {"statements": statements}

            response = self.client.post(
                self.database_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.auth_token}",
                    "Content-Type": "application/json"
                },
                timeout=60.0
            )

            response.raise_for_status()
            return len(params_list)

        except Exception as e:
            logger.error(f"Error executing batch: {e}")
            return 0

    def initialize_schema(self) -> None:
        """Create all necessary tables for ERP data."""
        try:
            # Historical Sales Data table
            self.execute("""
                CREATE TABLE IF NOT EXISTS historical_sales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    style TEXT NOT NULL,
                    customer TEXT,
                    quantity REAL NOT NULL,
                    units TEXT DEFAULT 'yards',
                    order_number TEXT,
                    ship_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_sales_date
                ON historical_sales(date);
            """)

            self.execute("""
                CREATE INDEX IF NOT EXISTS idx_sales_style
                ON historical_sales(style);
            """)

            # Yarn Inventory table
            cursor.execute("""
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
                );
            """)

            # Knit Orders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knit_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL UNIQUE,
                    style TEXT NOT NULL,
                    customer TEXT,
                    quantity REAL NOT NULL,
                    status TEXT,
                    due_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # BOM (Bill of Materials) table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bom (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    style TEXT NOT NULL,
                    yarn_id TEXT NOT NULL,
                    percentage REAL NOT NULL,
                    PRIMARY KEY (style, yarn_id)
                ) WITHOUT ROWID;
            """)

            # Fabric Specifications table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fabric_specs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    style TEXT NOT NULL UNIQUE,
                    yds_per_lb REAL,
                    gsm INTEGER,
                    width REAL,
                    fabric_type TEXT,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Forecast Results table (ML predictions)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS forecast_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    style TEXT NOT NULL,
                    forecast_date TEXT NOT NULL,
                    week_number INTEGER NOT NULL,
                    predicted_quantity REAL NOT NULL,
                    confidence REAL,
                    model_used TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_forecast_style_week
                ON forecast_results(style, week_number);
            """)

            # Yarn Demand Forecast table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS yarn_demand_forecast (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    yarn_id TEXT NOT NULL,
                    week_number INTEGER NOT NULL,
                    forecasted_demand_lbs REAL NOT NULL,
                    confidence REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

            self.conn.commit()
            logger.info("✓ Turso schema initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
            raise

    # ========== Historical Sales Data Methods ==========

    def insert_sales_data(self, sales_records: List[Dict]) -> int:
        """
        Insert historical sales data for ML forecasting.

        Args:
            sales_records: List of sales records with date, style, quantity

        Returns:
            Number of records inserted
        """
        try:
            cursor = self.conn.cursor()
            inserted = 0

            for record in sales_records:
                cursor.execute("""
                    INSERT OR REPLACE INTO historical_sales
                    (date, style, customer, quantity, units, order_number, ship_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.get('date'),
                    record.get('style'),
                    record.get('customer', 'Unknown'),
                    record.get('quantity', 0),
                    record.get('units', 'yards'),
                    record.get('order_number'),
                    record.get('ship_date')
                ))
                inserted += 1

            self.conn.commit()
            logger.info(f"✓ Inserted {inserted} sales records into Turso")
            return inserted

        except Exception as e:
            logger.error(f"Error inserting sales data: {e}")
            self.conn.rollback()
            return 0

    def get_sales_history(
        self,
        style: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Retrieve historical sales data for ML forecasting.

        Args:
            style: Filter by specific style
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum records to return

        Returns:
            List of sales records
        """
        try:
            cursor = self.conn.cursor()

            query = "SELECT * FROM historical_sales WHERE 1=1"
            params = []

            if style:
                query += " AND style = ?"
                params.append(style)

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert to list of dicts
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]

            logger.info(f"✓ Retrieved {len(results)} sales records from Turso")
            return results

        except Exception as e:
            logger.error(f"Error retrieving sales history: {e}")
            return []

    def get_styles_with_history(self, min_records: int = 10) -> List[str]:
        """
        Get list of styles with sufficient historical data for forecasting.

        Args:
            min_records: Minimum number of historical records required

        Returns:
            List of style codes
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                SELECT style, COUNT(*) as record_count
                FROM historical_sales
                GROUP BY style
                HAVING record_count >= ?
                ORDER BY record_count DESC
            """, (min_records,))

            rows = cursor.fetchall()
            styles = [row[0] for row in rows]

            logger.info(f"✓ Found {len(styles)} styles with {min_records}+ records")
            return styles

        except Exception as e:
            logger.error(f"Error getting styles: {e}")
            return []

    # ========== Yarn Inventory Methods ==========

    def upsert_yarn_inventory(self, yarn_data: List[Dict]) -> int:
        """Insert or update yarn inventory data."""
        try:
            cursor = self.conn.cursor()
            updated = 0

            for yarn in yarn_data:
                cursor.execute("""
                    INSERT OR REPLACE INTO yarn_inventory
                    (yarn_id, description, supplier, color, theoretical_balance,
                     planning_balance, allocated, on_order, cost_per_lb, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    yarn.get('yarn_id'),
                    yarn.get('description'),
                    yarn.get('supplier'),
                    yarn.get('color'),
                    yarn.get('theoretical_balance', 0),
                    yarn.get('planning_balance', 0),
                    yarn.get('allocated', 0),
                    yarn.get('on_order', 0),
                    yarn.get('cost_per_lb')
                ))
                updated += 1

            self.conn.commit()
            logger.info(f"✓ Upserted {updated} yarn records")
            return updated

        except Exception as e:
            logger.error(f"Error upserting yarn inventory: {e}")
            self.conn.rollback()
            return 0

    def get_yarn_inventory(self, yarn_id: Optional[str] = None) -> List[Dict]:
        """Retrieve yarn inventory data."""
        try:
            cursor = self.conn.cursor()

            if yarn_id:
                cursor.execute("""
                    SELECT * FROM yarn_inventory WHERE yarn_id = ?
                """, (yarn_id,))
            else:
                cursor.execute("SELECT * FROM yarn_inventory")

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]

            return results

        except Exception as e:
            logger.error(f"Error retrieving yarn inventory: {e}")
            return []

    # ========== Forecast Storage Methods ==========

    def store_forecast_results(self, forecasts: List[Dict]) -> int:
        """Store ML forecast results."""
        try:
            cursor = self.conn.cursor()
            stored = 0

            for forecast in forecasts:
                cursor.execute("""
                    INSERT INTO forecast_results
                    (style, forecast_date, week_number, predicted_quantity,
                     confidence, model_used)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    forecast.get('style'),
                    forecast.get('forecast_date'),
                    forecast.get('week_number'),
                    forecast.get('predicted_quantity'),
                    forecast.get('confidence'),
                    forecast.get('model_used')
                ))
                stored += 1

            self.conn.commit()
            logger.info(f"✓ Stored {stored} forecast results")
            return stored

        except Exception as e:
            logger.error(f"Error storing forecasts: {e}")
            self.conn.rollback()
            return 0

    def get_latest_forecasts(
        self,
        style: Optional[str] = None,
        weeks_ahead: int = 8
    ) -> List[Dict]:
        """Retrieve latest ML forecasts."""
        try:
            cursor = self.conn.cursor()

            query = """
                SELECT * FROM forecast_results
                WHERE created_at >= date('now', '-7 days')
            """
            params = []

            if style:
                query += " AND style = ?"
                params.append(style)

            query += " ORDER BY week_number ASC LIMIT ?"
            params.append(weeks_ahead)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]

            return results

        except Exception as e:
            logger.error(f"Error retrieving forecasts: {e}")
            return []

    # ========== BOM Methods ==========

    def upsert_bom_data(self, bom_records: List[Dict]) -> int:
        """Insert or update BOM data."""
        try:
            cursor = self.conn.cursor()
            updated = 0

            for bom in bom_records:
                cursor.execute("""
                    INSERT OR REPLACE INTO bom (style, yarn_id, percentage)
                    VALUES (?, ?, ?)
                """, (
                    bom.get('style'),
                    bom.get('yarn_id'),
                    bom.get('percentage', 0)
                ))
                updated += 1

            self.conn.commit()
            logger.info(f"✓ Upserted {updated} BOM records")
            return updated

        except Exception as e:
            logger.error(f"Error upserting BOM: {e}")
            self.conn.rollback()
            return 0

    def get_bom_for_style(self, style: str) -> Dict[str, float]:
        """Get BOM percentages for a style."""
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                SELECT yarn_id, percentage
                FROM bom
                WHERE style = ?
            """, (style,))

            rows = cursor.fetchall()
            bom_dict = {row[0]: row[1] for row in rows}

            return bom_dict

        except Exception as e:
            logger.error(f"Error retrieving BOM for {style}: {e}")
            return {}

    # ========== Utility Methods ==========

    def get_database_stats(self) -> Dict[str, int]:
        """Get statistics about data in Turso."""
        try:
            cursor = self.conn.cursor()

            stats = {}

            tables = [
                'historical_sales',
                'yarn_inventory',
                'knit_orders',
                'bom',
                'fabric_specs',
                'forecast_results'
            ]

            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[table] = count

            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Turso connection closed")


# Singleton instance
_turso_client = None


def get_turso_client() -> TursoClient:
    """Get or create Turso client singleton."""
    global _turso_client
    if _turso_client is None:
        _turso_client = TursoClient()
        _turso_client.initialize_schema()
    return _turso_client
