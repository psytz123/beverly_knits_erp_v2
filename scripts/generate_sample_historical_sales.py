#!/usr/bin/env python3
"""
Generate Sample Historical Sales Data
Creates synthetic sales history for testing the training system
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.turso_client import get_turso_client
from datetime import datetime, timedelta
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_historical_sales(
    num_styles: int = 20,
    weeks_of_history: int = 52,
    base_weekly_sales: float = 1000.0
):
    """
    Generate sample historical sales data for training

    Args:
        num_styles: Number of styles to generate data for
        weeks_of_history: Number of weeks of historical data
        base_weekly_sales: Average weekly sales in yards
    """
    try:
        logger.info("=" * 80)
        logger.info("GENERATING SAMPLE HISTORICAL SALES DATA")
        logger.info("=" * 80)

        turso_client = get_turso_client()

        # Get existing styles from knit_orders or create sample ones
        logger.info("\n[1/4] Getting existing styles from database...")

        result = turso_client.execute("""
            SELECT DISTINCT style
            FROM knit_orders
            LIMIT ?
        """, [num_styles])

        if result and len(result) > 0:
            styles = [row['style'] for row in result]
            logger.info(f"Found {len(styles)} existing styles")
        else:
            # Create sample style codes
            styles = [f"STYLE{str(i+1).zfill(3)}" for i in range(num_styles)]
            logger.info(f"Created {len(styles)} sample style codes")

        # Generate historical sales for each style
        logger.info(f"\n[2/4] Generating {weeks_of_history} weeks of sales data...")

        end_date = datetime.now()
        all_records = []

        for style in styles:
            # Generate weekly sales pattern with some variation
            # Include seasonality, trend, and random noise

            for week_offset in range(weeks_of_history):
                sale_date = end_date - timedelta(weeks=week_offset)

                # Seasonal pattern (higher in certain months)
                month = sale_date.month
                seasonal_factor = 1.0 + 0.3 * abs((month - 6) / 6)  # Peak mid-year

                # Weekly trend (slight growth over time)
                trend_factor = 1.0 + (week_offset / weeks_of_history) * 0.2

                # Random variation
                random_factor = random.uniform(0.7, 1.3)

                # Calculate quantity
                quantity = base_weekly_sales * seasonal_factor * trend_factor * random_factor

                # Add to batch
                all_records.append({
                    'style': style,
                    'date': sale_date.strftime('%Y-%m-%d'),
                    'quantity': round(quantity, 2),
                    'units': 'yards'
                })

        logger.info(f"  Generated {len(all_records)} records, inserting in batches...")

        # Insert in batches of 100
        batch_size = 100
        records_inserted = 0

        for i in range(0, len(all_records), batch_size):
            batch = all_records[i:i+batch_size]

            # Build batch insert SQL
            values_placeholders = ', '.join(['(?, ?, ?, ?)' for _ in batch])
            sql = f"""
                INSERT INTO historical_sales (style, date, quantity, units)
                VALUES {values_placeholders}
            """

            # Flatten batch parameters
            params = []
            for record in batch:
                params.extend([record['style'], record['date'], record['quantity'], record['units']])

            turso_client.execute(sql, params)
            records_inserted += len(batch)

            if records_inserted % 500 == 0 or records_inserted == len(all_records):
                logger.info(f"  Inserted {records_inserted}/{len(all_records)} records...")

        logger.info(f"\n[3/4] Successfully inserted {records_inserted} historical sales records")

        # Verify data
        logger.info("\n[4/4] Verifying inserted data...")

        result = turso_client.execute("""
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT style) as distinct_styles,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                AVG(quantity) as avg_quantity
            FROM historical_sales
        """)

        if result:
            row = result[0]
            print("\nData Summary:")
            print(f"  Total Records: {row['total_records']}")
            print(f"  Distinct Styles: {row['distinct_styles']}")
            print(f"  Date Range: {row['earliest_date']} to {row['latest_date']}")
            print(f"  Average Quantity: {row['avg_quantity']:.2f} yards")

        # Check trainable styles
        result = turso_client.execute("""
            SELECT COUNT(*) as count
            FROM (
                SELECT style, COUNT(*) as record_count
                FROM historical_sales
                WHERE units = 'yards'
                GROUP BY style
                HAVING record_count >= 10
            )
        """)

        if result:
            trainable_count = result[0]['count']
            print(f"\nStyles Ready for Training: {trainable_count}")

        logger.info("\n" + "=" * 80)
        logger.info("SAMPLE DATA GENERATION COMPLETE")
        logger.info("=" * 80)
        print("\nYou can now train the forecast models:")
        print("  python scripts/train_forecast_models.py")

        return True

    except Exception as e:
        logger.exception(f"Error generating sample data: {e}")
        return False


if __name__ == "__main__":
    # Start with smaller dataset for faster testing
    success = generate_sample_historical_sales(
        num_styles=5,   # Start with just 5 styles
        weeks_of_history=20,  # 20 weeks of history
        base_weekly_sales=1000.0
    )

    sys.exit(0 if success else 1)
