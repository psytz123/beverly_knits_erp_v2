#!/usr/bin/env python3
"""
Database Sync Scheduler
Purpose: Schedule automated data pulls from eFab API
Usage: scheduler = DataSyncScheduler(db_url, session_cookie)
       scheduler.start()
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import schedule
import threading
import time
import logging
from database.efab_api_sync import EFabAPISync

logger = logging.getLogger(__name__)


class DataSyncScheduler:
    """
    Scheduler for automated eFab API data synchronization.

    Handles scheduled and on-demand syncs with retry logic.
    """

    def __init__(
        self,
        database_url: str,
        session_cookie: str,
        sync_interval_hours: int = 2
    ):
        """
        Initialize scheduler.

        Args:
            database_url: Database connection string
            session_cookie: eFab authentication cookie
            sync_interval_hours: Hours between syncs
        """
        self.database_url = database_url
        self.session_cookie = session_cookie
        self.sync_interval_hours = sync_interval_hours

        # Initialize sync service
        self.sync_service = EFabAPISync(database_url, session_cookie)

        # Thread control
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_sync: Optional[datetime] = None
        self.next_sync: Optional[datetime] = None
        self.sync_history: list = []

    def sync_job(self) -> None:
        """Execute sync job."""
        try:
            logger.info("Starting scheduled sync...")
            self.last_sync = datetime.utcnow()

            # Run incremental sync
            results = self.sync_service.sync_incremental(hours=self.sync_interval_hours * 2)

            # Store results
            self.sync_history.append({
                'timestamp': self.last_sync,
                'success': results['success'],
                'duration': results['duration'],
                'records': {
                    'cf_versions': results['cf_versions'],
                    'production_orders': results['production_orders']
                }
            })

            # Keep only last 100 entries
            if len(self.sync_history) > 100:
                self.sync_history = self.sync_history[-100:]

            if results['success']:
                logger.info(f"Sync completed successfully in {results['duration']:.2f}s")
            else:
                logger.error(f"Sync failed: {results.get('errors', [])}")

            # Calculate next sync time
            self.next_sync = datetime.utcnow() + timedelta(hours=self.sync_interval_hours)

        except Exception as e:
            logger.exception(f"Sync job failed: {e}")
            self.sync_history.append({
                'timestamp': datetime.utcnow(),
                'success': False,
                'error': str(e)
            })

    def run_scheduler(self) -> None:
        """Run the scheduler loop."""
        # Schedule jobs
        schedule.every(self.sync_interval_hours).hours.do(self.sync_job)

        # Run first sync immediately
        self.sync_job()

        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def start(self) -> None:
        """Start the scheduler in background thread."""
        if self.running:
            logger.warning("Scheduler already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.thread.start()
        logger.info(f"Scheduler started with {self.sync_interval_hours} hour interval")

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self.running:
            logger.warning("Scheduler not running")
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def manual_sync(self) -> Dict[str, Any]:
        """
        Trigger manual sync.

        Returns:
            Sync results
        """
        logger.info("Manual sync triggered")
        return self.sync_service.sync_all()

    def get_status(self) -> Dict[str, Any]:
        """
        Get scheduler status.

        Returns:
            Status information
        """
        return {
            'running': self.running,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'next_sync': self.next_sync.isoformat() if self.next_sync else None,
            'sync_interval_hours': self.sync_interval_hours,
            'total_syncs': len(self.sync_history),
            'successful_syncs': sum(1 for s in self.sync_history if s.get('success', False)),
            'recent_syncs': self.sync_history[-10:]  # Last 10 syncs
        }

    def update_session_cookie(self, new_cookie: str) -> None:
        """
        Update session cookie.

        Args:
            new_cookie: New session cookie value
        """
        self.session_cookie = new_cookie
        self.sync_service.session_cookie = new_cookie
        self.sync_service.headers['Cookie'] = f"dancer.session={new_cookie}"
        logger.info("Session cookie updated")


# Global scheduler instance
_scheduler: Optional[DataSyncScheduler] = None


def get_scheduler() -> Optional[DataSyncScheduler]:
    """Get global scheduler instance."""
    return _scheduler


def init_scheduler(
    database_url: str,
    session_cookie: str,
    sync_interval_hours: int = 2,
    auto_start: bool = True
) -> DataSyncScheduler:
    """
    Initialize and optionally start global scheduler.

    Args:
        database_url: Database connection string
        session_cookie: eFab authentication cookie
        sync_interval_hours: Hours between syncs
        auto_start: Start scheduler immediately

    Returns:
        Scheduler instance
    """
    global _scheduler

    if _scheduler:
        _scheduler.stop()

    _scheduler = DataSyncScheduler(database_url, session_cookie, sync_interval_hours)

    if auto_start:
        _scheduler.start()

    return _scheduler


def main():
    """Test scheduler."""
    import os

    # Configuration
    DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost/efab_erp")
    SESSION_COOKIE = os.environ.get("EFAB_SESSION", "aNEM2YqIXevF7IvZ13r68JFeSlsVo1Lh")

    # Initialize scheduler
    scheduler = init_scheduler(DATABASE_URL, SESSION_COOKIE, sync_interval_hours=1)

    print("Scheduler started. Press Ctrl+C to stop.")
    print(f"Status: {scheduler.get_status()}")

    try:
        while True:
            time.sleep(60)
            status = scheduler.get_status()
            print(f"\n[{datetime.now()}] Status:")
            print(f"  Last sync: {status['last_sync']}")
            print(f"  Next sync: {status['next_sync']}")
            print(f"  Total syncs: {status['total_syncs']}")
            print(f"  Successful: {status['successful_syncs']}")

    except KeyboardInterrupt:
        print("\nStopping scheduler...")
        scheduler.stop()
        print("Scheduler stopped")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()