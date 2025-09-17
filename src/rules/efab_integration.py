"""eFab Integration with Retry Logic.

Handles all communications with eFab API including automatic retries
with exponential backoff for resilient integration.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
import logging
import time
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class eFabIntegration:
    """eFab API integration with exponential backoff retry logic.

    Features:
    - 3 retry attempts with exponential backoff
    - Automatic fallback to cached data
    - Comprehensive error handling
    - Performance monitoring
    """

    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_DELAY = 1.0  # seconds
    BACKOFF_FACTOR = 2.0
    MAX_DELAY = 30.0

    # Timeout configuration
    CONNECT_TIMEOUT = 10  # seconds
    READ_TIMEOUT = 30  # seconds

    # API endpoints
    ENDPOINTS = {
        'yarn_demand': '/ExpectedYarnReport',
        'knit_orders': '/api/knit-orders',
        'machine_status': '/api/machine-status',
        'production_update': '/api/production-update'
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        session_cookie: Optional[str] = None
    ) -> None:
        """Initialize eFab integration.

        Args:
            base_url: eFab API base URL
            session_cookie: Session cookie for authentication
        """
        self.base_url = base_url or "https://efab.local"
        self.session_cookie = session_cookie or os.getenv('EFAB_SESSION')

        self.session = self._create_session()
        self.request_log: List[Dict[str, Any]] = []
        self.error_log: List[Dict[str, Any]] = []
        self.fallback_dir = Path("/mnt/c/finalee/beverly_knits_erp_v2/data/fallback")
        self.fallback_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"eFabIntegration initialized with base_url={self.base_url}")

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy.

        Returns:
            Configured requests session
        """
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=self.BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set headers
        session.headers.update({
            'User-Agent': 'BeverlyKnitsERP/2.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

        # Add session cookie if available
        if self.session_cookie:
            session.cookies.set('session', self.session_cookie)

        return session

    def fetch_with_retry(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch data with exponential backoff retry.

        Args:
            endpoint: API endpoint
            method: HTTP method
            params: Query parameters
            data: Request body

        Returns:
            Response data or None if all retries failed
        """
        url = f"{self.base_url}{endpoint}"
        attempt = 0
        delay = self.INITIAL_DELAY
        last_error = None

        while attempt <= self.MAX_RETRIES:
            try:
                # Add delay for retries
                if attempt > 0:
                    logger.info(f"Retry {attempt}/{self.MAX_RETRIES} after {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * self.BACKOFF_FACTOR, self.MAX_DELAY)

                # Make request
                start_time = time.time()

                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    timeout=(self.CONNECT_TIMEOUT, self.READ_TIMEOUT)
                )

                duration = time.time() - start_time

                # Check status
                response.raise_for_status()

                # Log success
                self._log_request(endpoint, method, attempt + 1, duration, "success")

                # Parse response
                if response.content:
                    return response.json()
                return {}

            except requests.exceptions.RequestException as e:
                last_error = e
                attempt += 1
                self._log_error(endpoint, method, attempt, str(e))

                if attempt > self.MAX_RETRIES:
                    logger.error(f"All retries exhausted for {endpoint}")
                    return self._use_fallback(endpoint)

        return None

    def download_yarn_demand_report(self) -> Optional[pd.DataFrame]:
        """Download yarn demand report from eFab.

        Returns:
            DataFrame with yarn demand data or None
        """
        logger.info("Downloading yarn demand report from eFab...")

        try:
            # Fetch report
            data = self.fetch_with_retry(
                self.ENDPOINTS['yarn_demand'],
                method="GET"
            )

            if data is None:
                logger.error("Failed to download yarn demand report")
                return self._load_fallback_data('yarn_demand')

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Save as fallback
            self._save_fallback_data('yarn_demand', df)

            logger.info(f"Downloaded {len(df)} yarn demand records")
            return df

        except Exception as e:
            logger.exception(f"Error downloading yarn demand: {e}")
            return self._load_fallback_data('yarn_demand')

    def get_knit_orders(self) -> Optional[pd.DataFrame]:
        """Get knit orders from eFab.

        Returns:
            DataFrame with knit orders or None
        """
        logger.info("Fetching knit orders from eFab...")

        try:
            data = self.fetch_with_retry(
                self.ENDPOINTS['knit_orders'],
                method="GET"
            )

            if data is None:
                return self._load_fallback_data('knit_orders')

            df = pd.DataFrame(data)
            self._save_fallback_data('knit_orders', df)

            logger.info(f"Fetched {len(df)} knit orders")
            return df

        except Exception as e:
            logger.exception(f"Error fetching knit orders: {e}")
            return self._load_fallback_data('knit_orders')

    def update_production_status(
        self,
        order_id: str,
        status: str,
        quantity: Optional[float] = None
    ) -> bool:
        """Update production status in eFab.

        Args:
            order_id: Order ID
            status: New status
            quantity: Optional quantity update

        Returns:
            True if successful
        """
        logger.info(f"Updating production status for {order_id}")

        data = {
            'order_id': order_id,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }

        if quantity is not None:
            data['quantity'] = quantity

        result = self.fetch_with_retry(
            self.ENDPOINTS['production_update'],
            method="POST",
            data=data
        )

        success = result is not None
        if success:
            logger.info(f"Successfully updated {order_id}")
        else:
            logger.error(f"Failed to update {order_id}")

        return success

    def _use_fallback(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Use fallback data when API fails.

        Args:
            endpoint: API endpoint that failed

        Returns:
            Fallback data or None
        """
        logger.warning(f"Using fallback data for {endpoint}")

        # Load appropriate fallback
        endpoint_key = endpoint.split('/')[-1]
        return self._load_fallback_data(endpoint_key)

    def _save_fallback_data(self, key: str, df: pd.DataFrame) -> None:
        """Save data as fallback.

        Args:
            key: Fallback key
            df: DataFrame to save
        """
        try:
            fallback_file = self.fallback_dir / f"{key}_fallback.csv"
            df.to_csv(fallback_file, index=False)
            logger.debug(f"Saved fallback data for {key}")
        except Exception as e:
            logger.error(f"Failed to save fallback: {e}")

    def _load_fallback_data(self, key: str) -> Optional[pd.DataFrame]:
        """Load fallback data.

        Args:
            key: Fallback key

        Returns:
            Fallback DataFrame or None
        """
        try:
            fallback_file = self.fallback_dir / f"{key}_fallback.csv"
            if fallback_file.exists():
                df = pd.read_csv(fallback_file)
                logger.info(f"Loaded fallback data for {key}")
                return df
        except Exception as e:
            logger.error(f"Failed to load fallback: {e}")

        return None

    def _log_request(
        self,
        endpoint: str,
        method: str,
        attempt: int,
        duration: float,
        status: str
    ) -> None:
        """Log request details.

        Args:
            endpoint: API endpoint
            method: HTTP method
            attempt: Attempt number
            duration: Request duration
            status: Request status
        """
        self.request_log.append({
            'timestamp': datetime.now(),
            'endpoint': endpoint,
            'method': method,
            'attempt': attempt,
            'duration': duration,
            'status': status
        })

    def _log_error(
        self,
        endpoint: str,
        method: str,
        attempt: int,
        error: str
    ) -> None:
        """Log error details.

        Args:
            endpoint: API endpoint
            method: HTTP method
            attempt: Attempt number
            error: Error message
        """
        self.error_log.append({
            'timestamp': datetime.now(),
            'endpoint': endpoint,
            'method': method,
            'attempt': attempt,
            'error': error
        })

    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status summary.

        Returns:
            Status information
        """
        total_requests = len(self.request_log)
        successful = sum(1 for r in self.request_log if r['status'] == 'success')
        total_errors = len(self.error_log)

        return {
            'connected': self.session_cookie is not None,
            'total_requests': total_requests,
            'successful_requests': successful,
            'failed_requests': total_requests - successful,
            'success_rate': (successful / total_requests * 100) if total_requests > 0 else 0,
            'total_errors': total_errors,
            'avg_attempts': sum(r['attempt'] for r in self.request_log) / total_requests if total_requests > 0 else 0
        }

    def test_connection(self) -> bool:
        """Test connection to eFab.

        Returns:
            True if connection successful
        """
        try:
            result = self.fetch_with_retry("/api/health", method="GET")
            return result is not None
        except Exception:
            return False


if __name__ == "__main__":
    """Validation of eFab integration."""

    # Note: Using mock URL for testing
    efab = eFabIntegration(base_url="http://localhost:5000")

    # Test Case 1: Test connection (will fail but demonstrate retry)
    connected = efab.test_connection()
    logger.info(f"Connection test: {connected}")

    # Test Case 2: Get integration status
    status = efab.get_integration_status()
    logger.info(f"Integration status: {status}")

    # Test Case 3: Fallback handling
    # This will fail and use fallback
    orders = efab.get_knit_orders()
    if orders is not None:
        logger.info(f"Got {len(orders)} orders (likely from fallback)")
    else:
        logger.info("No orders available")

    # Test Case 4: Check retry logging
    if efab.error_log:
        logger.info(f"Logged {len(efab.error_log)} errors with retries")

    print("All validations completed!")