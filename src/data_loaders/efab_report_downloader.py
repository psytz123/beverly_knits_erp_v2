#!/usr/bin/env python3
"""
EFab Report Downloader Module
Minimal implementation to download Yarn Demand reports from eFab API
Follows Operating Charter: Less is More principle
"""

import requests
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Dict, Any
import time

logger = logging.getLogger(__name__)


class EFabReportDownloader:
    """
    Minimal downloader for eFab Yarn Demand reports
    Purpose: Download Excel reports from eFab report queue
    Usage: downloader = EFabReportDownloader(session_cookie)
           success = downloader.download_latest(target_path)
    """

    def __init__(self, session_cookie: str):
        """
        Initialize downloader with session authentication

        Args:
            session_cookie: dancer.session cookie value for authentication
        """
        self.session_cookie = session_cookie
        self.base_url = "https://efab.bkiapps.com"
        self.queue_endpoint = "/api/report/report_queue"

        # Request headers matching browser request
        self.headers = {
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cookie': f'dancer.session={session_cookie}',
            'Referer': 'https://efab.bkiapps.com/reports/report_queue',
            'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36',
            'X-Requested-With': 'XMLHttpRequest'
        }

    def check_queue(self) -> Optional[Dict[str, Any]]:
        """
        Check report queue for available Yarn Demand reports

        Returns:
            Queue data if successful, None if failed
        """
        try:
            response = requests.get(
                f"{self.base_url}{self.queue_endpoint}",
                headers=self.headers,
                timeout=30
            )

            if response.status_code == 200:
                queue_data = response.json()
                logger.info(f"Queue check successful: {len(queue_data.get('reports', []))} reports found")
                return queue_data
            else:
                logger.error(f"Queue check failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error checking queue: {e}")
            return None

    def find_yarn_demand_report(self, queue_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find the latest Yarn Demand report in queue

        Args:
            queue_data: Response from queue check

        Returns:
            Report info if found, None otherwise
        """
        # Handle list response (direct list of reports)
        if isinstance(queue_data, list):
            reports = queue_data
        elif isinstance(queue_data, dict) and 'reports' in queue_data:
            reports = queue_data['reports']
        else:
            return None

        # Look for Yarn Demand reports
        yarn_reports = [
            r for r in reports
            if isinstance(r, dict) and (
                'yarn_demand' in r.get('name', '').lower() or
                'yarn demand' in r.get('name', '').lower()
            )
        ]

        if not yarn_reports:
            logger.info("No Yarn Demand reports found in queue")
            return None

        # Return most recent (assume list is ordered by date)
        latest = yarn_reports[0]
        logger.info(f"Found Yarn Demand report: {latest.get('name')}")
        return latest

    def download_report(self, report_url: str, target_path: Path) -> bool:
        """
        Download report file from URL

        Args:
            report_url: URL to download report from
            target_path: Where to save the file

        Returns:
            True if successful, False otherwise
        """
        try:
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

            if response.status_code == 200:
                # Ensure directory exists
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Write file in chunks
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_size = target_path.stat().st_size
                logger.info(f"Downloaded {file_size:,} bytes to {target_path}")
                return True
            else:
                logger.error(f"Download failed: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error downloading report: {e}")
            return False

    def download_latest(self, target_path: Path) -> bool:
        """
        Main method: Check queue and download latest Yarn Demand report

        Args:
            target_path: Where to save the Excel file

        Returns:
            True if successful download, False otherwise
        """
        logger.info(f"Starting Yarn Demand download at {datetime.now()}")

        # Step 1: Check queue
        queue_data = self.check_queue()
        if not queue_data:
            logger.warning("Could not check report queue")
            return False

        # Step 2: Find Yarn Demand report
        report = self.find_yarn_demand_report(queue_data)
        if not report:
            logger.info("No Yarn Demand report available")
            return False

        # Step 3: Get download URL
        download_url = report.get('download_url') or report.get('url')
        if not download_url:
            logger.error("No download URL in report data")
            return False

        # Step 4: Download file
        success = self.download_report(download_url, target_path)

        if success:
            logger.info(f"Successfully downloaded Yarn Demand report to {target_path}")

            # Archive copy with timestamp
            archive_dir = target_path.parent / "archive"
            archive_dir.mkdir(exist_ok=True)
            archive_name = f"Yarn_Demand_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            archive_path = archive_dir / archive_name

            import shutil
            shutil.copy2(target_path, archive_path)
            logger.info(f"Archived copy saved to {archive_path}")

        return success


def main():
    """Test function for EFab Report Downloader"""
    import os

    # Get session cookie from environment or use test value
    session_cookie = os.environ.get('EFAB_SESSION', 'aMdcwNLa0ov0pcbWcQ_zb5wyPLSkYF_B')

    downloader = EFabReportDownloader(session_cookie)

    # Test download
    test_path = Path('/tmp/test_yarn_demand.xlsx')
    success = downloader.download_latest(test_path)

    if success:
        print(f"✓ Successfully downloaded to {test_path}")
        print(f"  File size: {test_path.stat().st_size:,} bytes")
    else:
        print("✗ Download failed")

    return success


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()