"""Cache Manager for Beverly Knits ERP.

Manages API response caching and data file change detection
for optimal performance.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Callable
import logging
import json
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for API responses and data operations.

    Features:
    - Memory-based caching with TTL
    - File change detection
    - Immediate invalidation on data changes
    - Performance tracking
    """

    # Default cache configuration
    DEFAULT_TTL = 3600  # 1 hour
    MAX_CACHE_SIZE = 100  # Maximum number of cached items

    # Watched data files
    WATCHED_FILES = [
        'yarn_inventory.xlsx',
        'BOM_updated.csv',
        'eFab_Knit_Orders.csv',
        'Expected_Yarn_Report.xlsx',
        'Sales Activity Report.csv',
        'Machine Report fin1.csv'
    ]

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_ttl: int = DEFAULT_TTL
    ) -> None:
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache files
            default_ttl: Default time-to-live in seconds
        """
        self.cache_dir = cache_dir or Path("/tmp/bki_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.default_ttl = default_ttl
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.file_mtimes: Dict[str, float] = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0
        }

        logger.info(f"CacheManager initialized with TTL={default_ttl}s")

    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Unique cache key
        """
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(
        self,
        key: str,
        fetch_func: Optional[Callable] = None,
        ttl: Optional[int] = None
    ) -> Optional[Any]:
        """Get value from cache or fetch if missing.

        Args:
            key: Cache key
            fetch_func: Function to fetch data if not cached
            ttl: Time-to-live for this entry

        Returns:
            Cached or fetched value
        """
        # Check memory cache
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if datetime.now() < entry['expires']:
                self.cache_stats['hits'] += 1
                logger.debug(f"Cache hit: {key}")
                return entry['value']
            else:
                # Expired
                del self.memory_cache[key]

        self.cache_stats['misses'] += 1
        logger.debug(f"Cache miss: {key}")

        # Fetch new value if function provided
        if fetch_func is not None:
            value = fetch_func()
            self.set(key, value, ttl)
            return value

        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        ttl = ttl or self.default_ttl

        # Enforce cache size limit
        if len(self.memory_cache) >= self.MAX_CACHE_SIZE:
            self._evict_oldest()

        self.memory_cache[key] = {
            'value': value,
            'expires': datetime.now() + timedelta(seconds=ttl),
            'created': datetime.now()
        }

        logger.debug(f"Cached {key} with TTL={ttl}s")

    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries.

        Args:
            pattern: Pattern to match keys (None = all)

        Returns:
            Number of entries invalidated
        """
        if pattern is None:
            # Clear all
            count = len(self.memory_cache)
            self.memory_cache.clear()
            self.cache_stats['invalidations'] += count
            logger.info(f"Invalidated all {count} cache entries")
            return count

        # Clear matching pattern
        to_remove = [
            k for k in self.memory_cache.keys()
            if pattern in k
        ]

        for key in to_remove:
            del self.memory_cache[key]

        self.cache_stats['invalidations'] += len(to_remove)
        logger.info(f"Invalidated {len(to_remove)} entries matching '{pattern}'")

        return len(to_remove)

    def check_file_changes(self, data_dir: Path) -> bool:
        """Check if watched files have changed.

        Args:
            data_dir: Directory containing data files

        Returns:
            True if any files changed
        """
        changed = False

        for filename in self.WATCHED_FILES:
            filepath = data_dir / filename
            if not filepath.exists():
                continue

            mtime = filepath.stat().st_mtime
            last_mtime = self.file_mtimes.get(str(filepath), 0)

            if mtime > last_mtime:
                self.file_mtimes[str(filepath)] = mtime
                logger.info(f"File changed: {filename}")
                changed = True

        if changed:
            # Invalidate all cache on file change
            self.invalidate()

        return changed

    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self.memory_cache:
            return

        oldest_key = min(
            self.memory_cache.keys(),
            key=lambda k: self.memory_cache[k]['created']
        )

        del self.memory_cache[oldest_key]
        logger.debug(f"Evicted oldest cache entry: {oldest_key}")

    def cached_api(
        self,
        ttl: Optional[int] = None,
        key_prefix: Optional[str] = None
    ):
        """Decorator for caching API responses.

        Args:
            ttl: Time-to-live for cache
            key_prefix: Prefix for cache keys

        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_args = [key_prefix or func.__name__] + list(args)
                key = self.cache_key(*cache_args, **kwargs)

                # Try to get from cache
                result = self.get(key)
                if result is not None:
                    return result

                # Fetch and cache
                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                return result

            return wrapper
        return decorator

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache performance statistics
        """
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (
            self.cache_stats['hits'] / total_requests * 100
            if total_requests > 0 else 0
        )

        return {
            'total_requests': total_requests,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'invalidations': self.cache_stats['invalidations'],
            'current_size': len(self.memory_cache),
            'max_size': self.MAX_CACHE_SIZE
        }

    def save_to_disk(self, key: str, value: Any) -> None:
        """Save value to disk cache.

        Args:
            key: Cache key
            value: Value to save
        """
        cache_file = self.cache_dir / f"{key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"Saved to disk: {key}")
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")

    def load_from_disk(self, key: str) -> Optional[Any]:
        """Load value from disk cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        cache_file = self.cache_dir / f"{key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'rb') as f:
                value = pickle.load(f)
            logger.debug(f"Loaded from disk: {key}")
            return value
        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")
            return None


if __name__ == "__main__":
    """Validation of cache manager."""

    cache = CacheManager(default_ttl=5)  # 5 second TTL for testing

    # Test Case 1: Basic caching
    cache.set("test_key", "test_value")
    value = cache.get("test_key")
    assert value == "test_value"
    logger.info("Test 1 passed: Basic caching")

    # Test Case 2: Cache with fetch function
    class FetchCounter:
        count = 0

    def fetch_data():
        FetchCounter.count += 1
        return f"fetched_{FetchCounter.count}"

    # First call should fetch
    result1 = cache.get("fetch_test", fetch_data)
    assert result1 == "fetched_1"
    assert FetchCounter.count == 1

    # Second call should use cache
    result2 = cache.get("fetch_test", fetch_data)
    assert result2 == "fetched_1"
    assert FetchCounter.count == 1  # Not incremented
    logger.info("Test 2 passed: Fetch function caching")

    # Test Case 3: Cache invalidation
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("api_key1", "api_value1")

    # Invalidate pattern
    count = cache.invalidate("api")
    assert count == 1
    assert cache.get("key1") == "value1"  # Still exists
    assert cache.get("api_key1") is None  # Invalidated
    logger.info("Test 3 passed: Pattern invalidation")

    # Test Case 4: Cache statistics
    stats = cache.get_statistics()
    assert stats['hits'] > 0
    assert stats['misses'] > 0
    logger.info(f"Test 4 passed: Statistics = {stats}")

    # Test Case 5: Decorator
    @cache.cached_api(ttl=10, key_prefix="test_api")
    def api_function(param: str) -> str:
        return f"result_{param}"

    # First call
    result1 = api_function("test")
    # Second call (cached)
    result2 = api_function("test")
    assert result1 == result2
    logger.info("Test 5 passed: Decorator caching")

    print("All validations passed!")