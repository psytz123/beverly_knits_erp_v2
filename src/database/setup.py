#!/usr/bin/env python3
"""
Database Setup Script
Purpose: Initialize database and start sync service
Usage: python setup.py --init --sync --schedule
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_database(reset: bool = False) -> None:
    """
    Setup database tables.

    Args:
        reset: Drop existing tables first
    """
    from database.config import create_tables, drop_tables, test_connection

    logger.info("Setting up database...")

    # Test connection
    if not test_connection():
        logger.error("Database connection failed!")
        sys.exit(1)

    if reset:
        logger.warning("Dropping existing tables...")
        drop_tables()

    # Create tables
    logger.info("Creating tables...")
    create_tables()
    logger.info("✓ Database setup complete")


def run_initial_sync(session_cookie: str = None) -> None:
    """
    Run initial data sync.

    Args:
        session_cookie: eFab session cookie
    """
    from database.config import get_database_url
    from database.efab_api_sync import EFabAPISync

    # Get session cookie
    if not session_cookie:
        session_cookie = os.environ.get('EFAB_SESSION')

    if not session_cookie:
        logger.error("Session cookie required! Set EFAB_SESSION environment variable")
        sys.exit(1)

    logger.info("Running initial data sync...")

    # Initialize sync service
    db_url = get_database_url()
    sync_service = EFabAPISync(db_url, session_cookie)

    # Run full sync
    results = sync_service.sync_all()

    if results['success']:
        logger.info("✓ Initial sync complete")
        logger.info(f"  CF Versions: {results['cf_versions']['created']} created, {results['cf_versions']['updated']} updated")
        logger.info(f"  Production Orders: {results['production_orders']['created']} created, {results['production_orders']['updated']} updated")
    else:
        logger.error(f"✗ Sync failed: {results.get('errors', [])}")
        sys.exit(1)


def start_scheduler(session_cookie: str = None, interval: int = 2) -> None:
    """
    Start sync scheduler.

    Args:
        session_cookie: eFab session cookie
        interval: Sync interval in hours
    """
    from database.config import get_database_url
    from database.scheduler import init_scheduler
    import time

    # Get session cookie
    if not session_cookie:
        session_cookie = os.environ.get('EFAB_SESSION')

    if not session_cookie:
        logger.error("Session cookie required! Set EFAB_SESSION environment variable")
        sys.exit(1)

    logger.info(f"Starting scheduler with {interval} hour interval...")

    # Initialize scheduler
    db_url = get_database_url()
    scheduler = init_scheduler(db_url, session_cookie, interval, auto_start=True)

    logger.info("✓ Scheduler started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(60)
            status = scheduler.get_status()
            if status['last_sync']:
                logger.info(f"Last sync: {status['last_sync']}, Next: {status['next_sync']}")

    except KeyboardInterrupt:
        logger.info("Stopping scheduler...")
        scheduler.stop()
        logger.info("✓ Scheduler stopped")


def show_status() -> None:
    """Show database status."""
    from database.config import get_session, test_connection
    from database.models import (
        CFVersion, YarnRequirement, ProductionOrder,
        YarnInventory, APISync
    )

    if not test_connection():
        logger.error("Database connection failed!")
        sys.exit(1)

    print("\nDatabase Status")
    print("=" * 50)

    try:
        with get_session() as session:
            # Get record counts
            stats = {
                'CF Versions': session.query(CFVersion).count(),
                'Yarn Requirements': session.query(YarnRequirement).count(),
                'Production Orders': session.query(ProductionOrder).count(),
                'Yarn Inventory': session.query(YarnInventory).count(),
                'API Syncs': session.query(APISync).count()
            }

            print("\nRecord Counts:")
            for table, count in stats.items():
                print(f"  {table:20s}: {count:,}")

            # Get last sync
            last_sync = session.query(APISync).order_by(
                APISync.started_at.desc()
            ).first()

            if last_sync:
                print(f"\nLast Sync:")
                print(f"  Started: {last_sync.started_at}")
                print(f"  Status: {last_sync.status}")
                print(f"  Records: {last_sync.records_processed}")

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Database Setup and Management')

    parser.add_argument(
        '--init',
        action='store_true',
        help='Initialize database tables'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset database (drop and recreate tables)'
    )
    parser.add_argument(
        '--sync',
        action='store_true',
        help='Run initial data sync'
    )
    parser.add_argument(
        '--schedule',
        action='store_true',
        help='Start sync scheduler'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show database status'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=2,
        help='Scheduler interval in hours (default: 2)'
    )
    parser.add_argument(
        '--cookie',
        type=str,
        help='eFab session cookie'
    )

    args = parser.parse_args()

    # Default action if none specified
    if not any([args.init, args.sync, args.schedule, args.status]):
        args.status = True

    # Execute requested actions
    if args.init or args.reset:
        setup_database(reset=args.reset)

    if args.sync:
        run_initial_sync(args.cookie)

    if args.schedule:
        start_scheduler(args.cookie, args.interval)

    if args.status:
        show_status()


if __name__ == "__main__":
    main()