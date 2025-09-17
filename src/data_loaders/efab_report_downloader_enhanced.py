#!/usr/bin/env python3
"""
Enhanced EFab Report Downloader Module with Auto-Refresh and Retry Logic
Includes session cookie auto-refresh, exponential backoff, and health monitoring
"""

import requests
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any, Tuple
import time
import os
import pickle
from functools import wraps

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator for retry logic with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds between retries
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {e}")
            raise last_exception
        return wrapper
    return decorator


class SessionManager:
    """
    Manages session cookies with auto-refresh and persistence
    """

    def __init__(self, cache_file: Path = Path("/tmp/efab_session_cache.pkl")):
        """
        Initialize session manager with cache file

        Args:
            cache_file: Path to store session cache
        """
        self.cache_file = cache_file
        self.session_data: Dict[str, Any] = {}
        self.load_cache()

    def load_cache(self) -> None:
        """Load cached session data if available"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.session_data = pickle.load(f)
                logger.info(f"Loaded session cache from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Could not load session cache: {e}")
                self.session_data = {}

    def save_cache(self) -> None:
        """Save session data to cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.session_data, f)
            logger.debug(f"Saved session cache to {self.cache_file}")
        except Exception as e:
            logger.warning(f"Could not save session cache: {e}")

    def is_session_valid(self, cookie: str) -> bool:
        """
        Check if session cookie is still valid

        Args:
            cookie: Session cookie to validate

        Returns:
            True if valid, False otherwise
        """
        # Check cache for last validation time
        cache_key = f"valid_{cookie[:10]}"  # Use first 10 chars as key
        last_check = self.session_data.get(cache_key)

        # If checked within last hour, assume still valid
        if last_check and datetime.now() - last_check < timedelta(hours=1):
            return True

        # Test with a simple API call
        try:
            headers = {
                'Cookie': f'dancer.session={cookie}',
                'User-Agent': 'Mozilla/5.0'
            }
            response = requests.get(
                "https://efab.bkiapps.com/api/report/report_queue",
                headers=headers,
                timeout=10
            )

            is_valid = response.status_code == 200

            if is_valid:
                self.session_data[cache_key] = datetime.now()
                self.save_cache()
                logger.info("Session cookie validated successfully")
            else:
                logger.warning(f"Session validation failed: HTTP {response.status_code}")

            return is_valid

        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return False

    def get_new_session(self) -> Optional[str]:
        """
        Attempt to get a new session cookie
        Note: This is a placeholder - manual browser login is required

        Returns:
            New session cookie if available, None otherwise
        """
        # Check environment for updated cookie
        env_cookie = os.environ.get('EFAB_SESSION_NEW')
        if env_cookie:
            logger.info("Found new session cookie in EFAB_SESSION_NEW environment variable")
            return env_cookie

        # Check a file for updated cookie (for automation)
        cookie_file = Path("/tmp/efab_session_new.txt")
        if cookie_file.exists():
            try:
                new_cookie = cookie_file.read_text().strip()
                if new_cookie:
                    logger.info(f"Found new session cookie in {cookie_file}")
                    cookie_file.unlink()  # Remove after reading
                    return new_cookie
            except Exception as e:
                logger.error(f"Error reading cookie file: {e}")

        logger.warning("No new session cookie available. Manual login required.")
        logger.warning("Please update EFAB_SESSION_NEW environment variable or create /tmp/efab_session_new.txt")
        return None


class EFabReportDownloaderEnhanced:
    """
    Enhanced downloader with auto-refresh, retry logic, and health monitoring
    """

    def __init__(self, session_cookie: str):
        """
        Initialize enhanced downloader

        Args:
            session_cookie: Initial dancer.session cookie value
        """
        self.session_cookie = session_cookie
        self.base_url = "https://efab.bkiapps.com"
        self.queue_endpoint = "/api/report/report_queue"
        self.session_manager = SessionManager()

        # Health monitoring
        self.health_stats = {
            'last_successful_download': None,
            'last_failed_download': None,
            'total_downloads': 0,
            'failed_downloads': 0,
            'session_refreshes': 0,
            'last_session_refresh': None
        }

        # Request headers
        self.headers = {
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cookie': f'dancer.session={session_cookie}',
            'Referer': 'https://efab.bkiapps.com/reports/report_queue',
            'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36',
            'X-Requested-With': 'XMLHttpRequest'
        }

    def refresh_session_if_needed(self) -> bool:
        """
        Check and refresh session if needed

        Returns:
            True if session is valid, False if refresh failed
        """
        # First check if current session is valid
        if self.session_manager.is_session_valid(self.session_cookie):
            return True

        logger.warning("Current session invalid, attempting refresh...")

        # Try to get new session
        new_cookie = self.session_manager.get_new_session()
        if new_cookie:
            self.session_cookie = new_cookie
            self.headers['Cookie'] = f'dancer.session={new_cookie}'
            self.health_stats['session_refreshes'] += 1
            self.health_stats['last_session_refresh'] = datetime.now()

            # Update environment variable for other processes
            os.environ['EFAB_SESSION'] = new_cookie

            logger.info("Session cookie refreshed successfully")
            return True

        return False

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def check_queue(self) -> Optional[Dict[str, Any]]:
        """
        Check report queue with retry logic

        Returns:
            Queue data if successful, None if failed
        """
        # Ensure session is valid
        if not self.refresh_session_if_needed():
            raise Exception("Session expired and could not be refreshed")

        response = requests.get(
            f"{self.base_url}{self.queue_endpoint}",
            headers=self.headers,
            timeout=30
        )

        if response.status_code == 401:
            # Session expired, clear cache and retry
            self.session_manager.session_data.clear()
            self.session_manager.save_cache()
            raise Exception("Session expired (401)")

        if response.status_code != 200:
            raise Exception(f"Queue check failed: HTTP {response.status_code}")

        queue_data = response.json()
        logger.info(f"Queue check successful: {len(queue_data) if isinstance(queue_data, list) else len(queue_data.get('reports', []))} reports found")
        return queue_data

    def find_yarn_demand_report(self, queue_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find the latest Yarn Demand report in queue

        Args:
            queue_data: Response from queue check

        Returns:
            Report info if found, None otherwise
        """
        # Handle both list and dict responses
        reports = queue_data if isinstance(queue_data, list) else queue_data.get('reports', [])

        # Look for Yarn Demand reports
        yarn_reports = [
            r for r in reports
            if isinstance(r, dict) and (
                'yarn_demand' in r.get('name', '').lower() or
                'yarn demand' in r.get('name', '').lower() or
                'expected_yarn' in r.get('name', '').lower()
            )
        ]

        if not yarn_reports:
            logger.info("No Yarn Demand reports found in queue")
            return None

        # Sort by date if available, otherwise assume list is ordered
        if yarn_reports[0].get('created_at'):
            yarn_reports.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        latest = yarn_reports[0]
        logger.info(f"Found Yarn Demand report: {latest.get('name')}")
        return latest

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def download_report(self, report_url: str, target_path: Path) -> bool:
        """
        Download report with retry logic

        Args:
            report_url: URL to download report from
            target_path: Where to save the file

        Returns:
            True if successful, False otherwise
        """
        # Ensure session is valid
        if not self.refresh_session_if_needed():
            raise Exception("Session expired and could not be refreshed")

        # Ensure full URL
        if not report_url.startswith('http'):
            report_url = f"{self.base_url}{report_url}"

        logger.info(f"Downloading report from: {report_url}")

        response = requests.get(
            report_url,
            headers=self.headers,
            stream=True,
            timeout=60
        )

        if response.status_code == 401:
            # Session expired
            self.session_manager.session_data.clear()
            self.session_manager.save_cache()
            raise Exception("Session expired during download (401)")

        if response.status_code != 200:
            raise Exception(f"Download failed: HTTP {response.status_code}")

        # Ensure directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file in chunks with progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    if progress % 10 < 0.1:  # Log every 10%
                        logger.debug(f"Download progress: {progress:.0f}%")

        file_size = target_path.stat().st_size
        logger.info(f"Downloaded {file_size:,} bytes to {target_path}")
        return True

    def download_latest(self, target_path: Path) -> bool:
        """
        Main method: Download latest Yarn Demand report with full error handling

        Args:
            target_path: Where to save the Excel file

        Returns:
            True if successful, False otherwise
        """
        start_time = datetime.now()
        logger.info(f"[ENHANCED] Starting Yarn Demand download at {start_time}")

        try:
            # Step 1: Check queue
            queue_data = self.check_queue()
            if not queue_data:
                self.health_stats['failed_downloads'] += 1
                self.health_stats['last_failed_download'] = datetime.now()
                logger.warning("Could not check report queue")
                return False

            # Step 2: Find Yarn Demand report
            report = self.find_yarn_demand_report(queue_data)
            if not report:
                logger.info("No Yarn Demand report available in queue")
                return False

            # Step 3: Get download URL
            download_url = report.get('download_url') or report.get('url')
            if not download_url:
                logger.error("No download URL in report data")
                return False

            # Step 4: Download file
            success = self.download_report(download_url, target_path)

            if success:
                # Update health stats
                self.health_stats['total_downloads'] += 1
                self.health_stats['last_successful_download'] = datetime.now()

                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"[SUCCESS] Downloaded Yarn Demand report in {duration:.1f}s")

                # Archive copy with timestamp
                archive_dir = target_path.parent / "archive"
                archive_dir.mkdir(exist_ok=True)
                archive_name = f"Yarn_Demand_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                archive_path = archive_dir / archive_name

                import shutil
                shutil.copy2(target_path, archive_path)
                logger.info(f"Archived copy saved to {archive_path}")

                # Clean old archives (keep last 30 days)
                self._cleanup_old_archives(archive_dir)
            else:
                self.health_stats['failed_downloads'] += 1
                self.health_stats['last_failed_download'] = datetime.now()

            return success

        except Exception as e:
            self.health_stats['failed_downloads'] += 1
            self.health_stats['last_failed_download'] = datetime.now()
            logger.error(f"Download failed with error: {e}")
            return False

    def _cleanup_old_archives(self, archive_dir: Path, days_to_keep: int = 30) -> None:
        """
        Clean up old archive files

        Args:
            archive_dir: Directory containing archives
            days_to_keep: Number of days to keep archives
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            for file in archive_dir.glob("Yarn_Demand_*.xlsx"):
                if file.stat().st_mtime < cutoff_date.timestamp():
                    file.unlink()
                    logger.debug(f"Removed old archive: {file.name}")

        except Exception as e:
            logger.warning(f"Error cleaning archives: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the downloader

        Returns:
            Dictionary with health metrics
        """
        status = self.health_stats.copy()

        # Add derived metrics
        if status['total_downloads'] > 0:
            status['success_rate'] = ((status['total_downloads'] - status['failed_downloads'])
                                     / status['total_downloads']) * 100
        else:
            status['success_rate'] = 0

        # Check if recent download was successful
        if status['last_successful_download']:
            time_since_success = datetime.now() - status['last_successful_download']
            status['healthy'] = time_since_success < timedelta(hours=24)
            status['time_since_last_success'] = str(time_since_success)
        else:
            status['healthy'] = False
            status['time_since_last_success'] = None

        # Session status
        status['session_valid'] = self.session_manager.is_session_valid(self.session_cookie)

        return status


def main():
    """Test enhanced downloader functionality"""
    import os

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get session cookie
    session_cookie = os.environ.get('EFAB_SESSION', 'aMdcwNLa0ov0pcbWcQ_zb5wyPLSkYF_B')

    # Create enhanced downloader
    downloader = EFabReportDownloaderEnhanced(session_cookie)

    # Test download
    test_path = Path('/tmp/test_yarn_demand_enhanced.xlsx')
    success = downloader.download_latest(test_path)

    # Show health status
    health = downloader.get_health_status()
    print("\n=== Health Status ===")
    for key, value in health.items():
        print(f"  {key}: {value}")

    if success:
        print(f"\n✓ Successfully downloaded to {test_path}")
        print(f"  File size: {test_path.stat().st_size:,} bytes")
    else:
        print("\n✗ Download failed - check logs for details")

    return success


if __name__ == "__main__":
    main()