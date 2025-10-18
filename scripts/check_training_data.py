#!/usr/bin/env python3
"""
Check Training Data Availability
Diagnoses what data is available for training
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.turso_client import get_turso_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_training_data():
    """
    Check what data is available for training
    """
    try:
        logger.info("Checking training data availability...")
        turso_client = get_turso_client()

        # Check historical_sales table
        logger.info("\n1. Checking historical_sales table...")
        result = turso_client.execute("""
            SELECT COUNT(*) as count,
                   COUNT(DISTINCT style) as distinct_styles,
                   MIN(date) as earliest_date,
                   MAX(date) as latest_date
            FROM historical_sales
        """)

        if result:
            row = result[0]
            print(f"  Total Records: {row['count']}")
            print(f"  Distinct Styles: {row['distinct_styles']}")
            print(f"  Date Range: {row['earliest_date']} to {row['latest_date']}")

        # Check styles with sufficient data
        logger.info("\n2. Checking styles with >= 10 records...")
        result = turso_client.execute("""
            SELECT style, COUNT(*) as record_count
            FROM historical_sales
            WHERE units = 'yards'
            GROUP BY style
            HAVING record_count >= 10
            ORDER BY record_count DESC
            LIMIT 10
        """)

        if result:
            print(f"  Found {len(result)} styles with >= 10 records:")
            for row in result[:10]:
                print(f"    {row['style']}: {row['record_count']} records")
        else:
            print("  No styles with sufficient data found")

        # Check other relevant tables
        logger.info("\n3. Checking knit_orders table...")
        result = turso_client.execute("""
            SELECT COUNT(*) as count,
                   COUNT(DISTINCT style) as distinct_styles
            FROM knit_orders
        """)

        if result:
            row = result[0]
            print(f"  Total Orders: {row['count']}")
            print(f"  Distinct Styles: {row['distinct_styles']}")

        # Check fabric_specs
        logger.info("\n4. Checking fabric_specs table...")
        result = turso_client.execute("""
            SELECT COUNT(*) as count
            FROM fabric_specs
        """)

        if result:
            print(f"  Total Fabric Specs: {result[0]['count']}")

        # Check bom
        logger.info("\n5. Checking bom table...")
        result = turso_client.execute("""
            SELECT COUNT(*) as count,
                   COUNT(DISTINCT style) as distinct_styles
            FROM bom
        """)

        if result:
            row = result[0]
            print(f"  Total BOM Entries: {row['count']}")
            print(f"  Distinct Styles: {row['distinct_styles']}")

        logger.info("\n" + "="*80)
        logger.info("DIAGNOSIS")
        logger.info("="*80)

        # Get counts
        hist_result = turso_client.execute("SELECT COUNT(*) as count FROM historical_sales")
        hist_count = hist_result[0]['count'] if hist_result else 0

        if hist_count == 0:
            print("\nISSUE: No historical sales data found!")
            print("\nSOLUTIONS:")
            print("1. Import historical sales data from your sales system")
            print("2. Generate sample data for testing")
            print("3. Ensure sales data is being tracked in the system")
            print("\nTo generate sample data:")
            print("  python scripts/generate_sample_historical_sales.py")
        else:
            trainable_result = turso_client.execute("""
                SELECT COUNT(DISTINCT style) as count
                FROM (
                    SELECT style, COUNT(*) as record_count
                    FROM historical_sales
                    WHERE units = 'yards'
                    GROUP BY style
                    HAVING record_count >= 10
                )
            """)
            trainable_count = trainable_result[0]['count'] if trainable_result else 0

            if trainable_count == 0:
                print(f"\nISSUE: Found {hist_count} historical records but no styles have >= 10 records")
                print("\nSOLUTIONS:")
                print("1. Wait for more sales data to accumulate")
                print("2. Lower the minimum record threshold (current: 10)")
                print("3. Generate additional sample data")
            else:
                print(f"\nSUCCESS: Found {trainable_count} styles ready for training!")
                print("You can proceed with training.")

    except Exception as e:
        logger.exception(f"Error checking training data: {e}")


if __name__ == "__main__":
    check_training_data()
