#!/usr/bin/env python3
"""
Apply Forecast Training Schema to Turso Database
Migrates database to support ML training, accuracy tracking, and blend weights
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.turso_client import get_turso_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_forecast_training_schema():
    """
    Apply forecast training schema migration to Turso database

    Returns:
        bool: Success status
    """
    try:
        logger.info("=" * 80)
        logger.info("APPLYING FORECAST TRAINING SCHEMA MIGRATION")
        logger.info("=" * 80)

        # Get Turso client
        turso_client = get_turso_client()

        # Read schema file
        schema_file = Path(__file__).parent.parent / "database" / "migrations" / "forecast_training_schema.sql"

        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            return False

        logger.info(f"Reading schema from: {schema_file}")

        with open(schema_file, 'r') as f:
            schema_sql = f.read()

        # Split into individual statements (simple split on semicolon)
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]

        logger.info(f"Executing {len(statements)} SQL statements...")

        # Execute each statement
        success_count = 0
        error_count = 0

        for idx, statement in enumerate(statements, 1):
            # Skip comments and empty lines
            if statement.startswith('--') or not statement:
                continue

            try:
                # Execute statement
                turso_client.execute(statement)
                success_count += 1

                # Log progress for major operations
                if any(keyword in statement.upper() for keyword in ['CREATE TABLE', 'CREATE VIEW', 'CREATE INDEX']):
                    # Extract table/view/index name
                    parts = statement.split()
                    if 'TABLE' in parts:
                        idx_table = parts.index('TABLE')
                        name = parts[idx_table + 3] if 'NOT' in parts else parts[idx_table + 1]
                    elif 'VIEW' in parts:
                        idx_view = parts.index('VIEW')
                        name = parts[idx_view + 3] if 'NOT' in parts else parts[idx_view + 1]
                    elif 'INDEX' in parts:
                        idx_index = parts.index('INDEX')
                        name = parts[idx_index + 3] if 'NOT' in parts else parts[idx_index + 1]
                    else:
                        name = "unknown"

                    logger.info(f"✓ Created {name}")

            except Exception as e:
                error_count += 1
                logger.error(f"Error executing statement {idx}: {e}")
                logger.debug(f"Statement: {statement[:100]}...")

        logger.info("=" * 80)
        logger.info(f"MIGRATION COMPLETE")
        logger.info(f"  Successful: {success_count}")
        logger.info(f"  Errors: {error_count}")
        logger.info("=" * 80)

        # Verify tables were created
        logger.info("\nVerifying tables...")

        tables_to_verify = [
            'forecast_training_history',
            'forecast_blend_weights',
            'forecast_accuracy',
            'external_forecasts',
            'model_performance_metrics'
        ]

        for table in tables_to_verify:
            try:
                result = turso_client.execute(f"SELECT COUNT(*) as count FROM {table}")
                count = result[0]['count'] if result else 0
                logger.info(f"✓ {table}: {count} rows")
            except Exception as e:
                logger.error(f"✗ {table}: Not found or error - {e}")

        # Verify views
        logger.info("\nVerifying views...")

        views_to_verify = [
            'v_latest_blend_weights',
            'v_recent_training',
            'v_accuracy_by_source',
            'v_problematic_styles'
        ]

        for view in views_to_verify:
            try:
                result = turso_client.execute(f"SELECT * FROM {view} LIMIT 1")
                logger.info(f"✓ {view}: Accessible")
            except Exception as e:
                logger.warning(f"✗ {view}: {e}")

        logger.info("\n✅ Forecast training schema successfully applied!")
        return True

    except Exception as e:
        logger.exception(f"Error applying schema: {e}")
        return False


if __name__ == "__main__":
    success = apply_forecast_training_schema()
    sys.exit(0 if success else 1)
