"""
ML Forecast Cache with automatic background refresh.

Provides thread-safe caching for ML-generated yarn forecasts to avoid
re-computing expensive Prophet/XGBoost predictions on every API request.
"""
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ForecastCache:
    """
    Thread-safe cache for ML forecasts with automatic refresh tracking.

    Stores ML-generated yarn demand forecasts and tracks when they were last
    updated to determine staleness.
    """

    def __init__(self, refresh_interval_minutes: int = 15, cache_file: str = "cache/ml_forecasts.json"):
        """
        Initialize forecast cache.

        Args:
            refresh_interval_minutes: How often cache should be refreshed
            cache_file: Path to cache file for persistence
        """
        self._ml_forecasts: Dict[str, float] = {}
        self._last_updated: Optional[datetime] = None
        self._lock = Lock()
        self.refresh_interval = timedelta(minutes=refresh_interval_minutes)
        self.cache_file = Path(cache_file)

        # Create cache directory if it doesn't exist
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Load cached forecasts from disk
        self._load_from_disk()

        logger.info(f"Forecast cache initialized (refresh every {refresh_interval_minutes} min)")

    def get_forecasts(self) -> Dict[str, float]:
        """
        Get cached ML forecasts.

        Returns:
            Dictionary mapping yarn_id -> forecasted_lbs
            Thread-safe copy of cached data
        """
        with self._lock:
            return self._ml_forecasts.copy()

    def update_forecasts(self, forecasts: Dict[str, float]) -> None:
        """
        Update cached forecasts (called by background job).

        Args:
            forecasts: New forecast data to cache
        """
        with self._lock:
            self._ml_forecasts = forecasts
            self._last_updated = datetime.now()
            logger.info(f"✓ Forecast cache updated: {len(forecasts)} yarns at {self._last_updated.strftime('%H:%M:%S')}")

            # Save to disk for persistence across restarts
            self._save_to_disk()

    def is_stale(self) -> bool:
        """
        Check if cache needs refresh.

        Returns:
            True if cache is empty or older than refresh_interval
        """
        if not self._last_updated:
            return True
        return datetime.now() - self._last_updated > self.refresh_interval

    def get_age_seconds(self) -> Optional[int]:
        """
        Get cache age in seconds.

        Returns:
            Age in seconds, or None if cache never populated
        """
        if not self._last_updated:
            return None
        return int((datetime.now() - self._last_updated).total_seconds())

    def get_last_updated(self) -> Optional[datetime]:
        """
        Get timestamp of last cache update.

        Returns:
            Datetime of last update, or None if never updated
        """
        with self._lock:
            return self._last_updated

    def clear(self) -> None:
        """Clear the cache (useful for testing or manual reset)."""
        with self._lock:
            self._ml_forecasts.clear()
            self._last_updated = None
            logger.info("Forecast cache cleared")

            # Remove cache file
            if self.cache_file.exists():
                self.cache_file.unlink()

    def _save_to_disk(self) -> None:
        """Save forecasts to disk for persistence across restarts."""
        try:
            cache_data = {
                "forecasts": self._ml_forecasts,
                "last_updated": self._last_updated.isoformat() if self._last_updated else None
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"✓ Saved {len(self._ml_forecasts)} forecasts to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save forecast cache to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load forecasts from disk on startup."""
        if not self.cache_file.exists():
            logger.info("No cached forecasts found on disk")
            return

        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)

            self._ml_forecasts = cache_data.get("forecasts", {})
            last_updated_str = cache_data.get("last_updated")

            if last_updated_str:
                self._last_updated = datetime.fromisoformat(last_updated_str)

            if self._ml_forecasts:
                age = self.get_age_seconds()
                logger.info(f"✓ Loaded {len(self._ml_forecasts)} cached forecasts from disk (age: {age}s)")
            else:
                logger.warning("Cache file exists but contains no forecasts")

        except Exception as e:
            logger.error(f"Failed to load forecast cache from disk: {e}")
            # Start with empty cache if load fails
            self._ml_forecasts = {}
            self._last_updated = None


# Global cache instance - single source of truth for ML forecasts
# Refresh hourly - provides fresh data without overloading server
forecast_cache = ForecastCache(refresh_interval_minutes=60)
