#!/usr/bin/env python3
"""
Quick script to verify Turso historical_sales data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.turso_client import TursoClient
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main() -> int:
    """Verify data in Turso historical_sales table"""
    try:
        client = TursoClient()

        # Use get_database_stats
        stats = client.get_database_stats()

        print(f"\n{'='*60}")
        print(f"TURSO DATABASE STATUS")
        print(f"{'='*60}")
        print(f"Historical Sales Records: {stats.get('historical_sales_count', 0):,}")
        print(f"Forecast Results: {stats.get('forecast_results_count', 0):,}")
        print(f"BOM Entries: {stats.get('bom_count', 0):,}")
        print(f"Yarn Inventory: {stats.get('yarn_inventory_count', 0):,}")

        total_count = stats.get('historical_sales_count', 0)

        if total_count > 0:
            # Get additional details using execute
            date_range = client.execute("""
                SELECT
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date
                FROM historical_sales
            """)

            # Get style count
            style_count = client.execute("""
                SELECT COUNT(DISTINCT style) as count
                FROM historical_sales
            """)

            # Sample records
            samples = client.execute("""
                SELECT date, style, customer, quantity, units
                FROM historical_sales
                ORDER BY date DESC
                LIMIT 5
            """)

            print(f"\nDate Range:")
            if date_range and len(date_range) > 0:
                print(f"  Earliest: {date_range[0].get('earliest_date', 'N/A')}")
                print(f"  Latest: {date_range[0].get('latest_date', 'N/A')}")

            print(f"\nUnique Styles: {style_count[0]['count'] if style_count and len(style_count) > 0 else 0}")

            print(f"\nSample Records (most recent):")
            for i, rec in enumerate(samples, 1):
                customer_name = rec.get('customer', 'N/A')
                if customer_name and len(customer_name) > 30:
                    customer_name = customer_name[:30]
                print(f"  {i}. {rec['date']} | {rec['style']} | {customer_name} | {rec['quantity']} {rec['units']}")
        else:
            print("\n⚠ No data found - import may still be in progress or failed")

        print(f"{'='*60}\n")
        return 0

    except Exception as e:
        logger.error(f"Error checking Turso data: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
