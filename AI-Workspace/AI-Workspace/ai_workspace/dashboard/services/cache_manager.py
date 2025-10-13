"""Multi-layer cache manager for dashboard performance optimization.

Implements a 3-layer cache architecture:
- L1: In-memory LRU cache (fastest, ~1ms)
- L2: Redis cache (fast, ~5ms, optional)
- L3: Disk cache (persistent, ~50ms)

Target performance:
- 95% cache hit rate
- <100ms warm queries
- <10ms cached queries
"""

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    from cachetools import LRUCache
except ImportError:
    raise ImportError(
        "cachetools not installed. Install with: pip install cachetools"
    )

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore

logger = logging.getLogger(__name__)


class CacheMetrics:
    """Track cache performance metrics."""

    def __init__(self) -> None:
        """Initialize cache metrics."""
        self.l1_hits: int = 0
        self.l1_misses: int = 0
        self.l2_hits: int = 0
        self.l2_misses: int = 0
        self.l3_hits: int = 0
        self.l3_misses: int = 0
        self.total_queries: int = 0
        self.total_cache_time_ms: float = 0.0

    def record_hit(self, layer: str, duration_ms: float) -> None:
        """Record cache hit.

        Args:
            layer: Cache layer ('l1', 'l2', 'l3')
            duration_ms: Query duration in milliseconds
        """
        self.total_queries += 1
        self.total_cache_time_ms += duration_ms

        if layer == "l1":
            self.l1_hits += 1
        elif layer == "l2":
            self.l2_hits += 1
        elif layer == "l3":
            self.l3_hits += 1

    def record_miss(self, layer: str) -> None:
        """Record cache miss.

        Args:
            layer: Cache layer ('l1', 'l2', 'l3')
        """
        if layer == "l1":
            self.l1_misses += 1
        elif layer == "l2":
            self.l2_misses += 1
        elif layer == "l3":
            self.l3_misses += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache performance metrics
        """
        total_hits = self.l1_hits + self.l2_hits + self.l3_hits
        total_requests = total_hits + self.l1_misses + self.l2_misses + self.l3_misses

        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0
        avg_query_time = (
            self.total_cache_time_ms / self.total_queries
            if self.total_queries > 0 else 0.0
        )

        return {
            "l1_hits": self.l1_hits,
            "l1_misses": self.l1_misses,
            "l1_hit_rate": (
                self.l1_hits / (self.l1_hits + self.l1_misses) * 100
                if (self.l1_hits + self.l1_misses) > 0 else 0.0
            ),
            "l2_hits": self.l2_hits,
            "l2_misses": self.l2_misses,
            "l2_hit_rate": (
                self.l2_hits / (self.l2_hits + self.l2_misses) * 100
                if (self.l2_hits + self.l2_misses) > 0 else 0.0
            ),
            "l3_hits": self.l3_hits,
            "l3_misses": self.l3_misses,
            "l3_hit_rate": (
                self.l3_hits / (self.l3_hits + self.l3_misses) * 100
                if (self.l3_hits + self.l3_misses) > 0 else 0.0
            ),
            "overall_hit_rate": hit_rate,
            "total_requests": total_requests,
            "total_hits": total_hits,
            "avg_query_time_ms": round(avg_query_time, 2),
        }

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0
        self.l3_hits = 0
        self.l3_misses = 0
        self.total_queries = 0
        self.total_cache_time_ms = 0.0


class CacheManager:
    """Multi-layer cache manager for dashboard performance.

    Implements a 3-layer caching strategy:

    Layer 1 (L1): In-memory LRU cache
        - Fastest access (<1ms)
        - Limited size (100 entries)
        - TTL: 5 minutes
        - Best for: Recent search results, agent metadata

    Layer 2 (L2): Redis cache (optional)
        - Fast access (<5ms)
        - Shared across processes
        - TTL: 5 minutes
        - Best for: Shared search results, session data
        - Graceful degradation if unavailable

    Layer 3 (L3): Disk cache
        - Slower access (<50ms)
        - Persistent storage
        - No TTL (invalidate on file change)
        - Best for: Search indexes, agent manifests

    Example:
        >>> cache = CacheManager(l1_size=100, l2_ttl=300)
        >>>
        >>> # Get with automatic fallback
        >>> result = cache.get("search:query123", miss_fn=lambda: expensive_search())
        >>>
        >>> # Manual set
        >>> cache.set("search:query456", results, ttl=300)
        >>>
        >>> # Invalidate pattern
        >>> cache.invalidate("search:*")
        >>>
        >>> # Get metrics
        >>> stats = cache.get_metrics()
    """

    def __init__(
        self,
        l1_size: int = 100,
        l2_ttl: int = 300,
        l3_dir: Optional[Path] = None,
        redis_client: Optional[Any] = None,
        redis_url: Optional[str] = None,
    ) -> None:
        """Initialize cache manager.

        Args:
            l1_size: Maximum number of entries in L1 cache (default: 100)
            l2_ttl: Default TTL for L2 cache in seconds (default: 300)
            l3_dir: Directory for L3 disk cache (default: .ai-workspace/cache)
            redis_client: Pre-configured Redis client (optional)
            redis_url: Redis URL for automatic connection (optional)
        """
        # Layer 1: In-memory LRU cache
        self.l1_cache: LRUCache = LRUCache(maxsize=l1_size)
        self.l1_ttl: Dict[str, float] = {}  # Key -> expiry timestamp
        self.l1_max_size = l1_size

        # Layer 2: Redis cache (optional)
        self.l2_ttl = l2_ttl
        self.redis: Optional[Any] = None

        if redis_client:
            self.redis = redis_client
            logger.info("Using provided Redis client for L2 cache")
        elif redis_url and REDIS_AVAILABLE:
            try:
                self.redis = redis.from_url(redis_url, decode_responses=False)
                self.redis.ping()
                logger.info(f"Connected to Redis L2 cache: {redis_url}")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}. L2 cache disabled.")
                self.redis = None
        elif REDIS_AVAILABLE:
            # Try default local Redis
            try:
                self.redis = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
                self.redis.ping()
                logger.info("Connected to local Redis L2 cache")
            except Exception:
                logger.info("Redis not available, L2 cache disabled (optional)")
                self.redis = None

        # Layer 3: Disk cache
        if l3_dir:
            self.l3_dir = l3_dir
        else:
            self.l3_dir = Path.cwd() / ".ai-workspace" / "cache" / "search"

        self.l3_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"L3 disk cache directory: {self.l3_dir}")

        # Metrics
        self.metrics = CacheMetrics()

    def _is_expired(self, key: str) -> bool:
        """Check if L1 cache entry is expired.

        Args:
            key: Cache key

        Returns:
            True if expired, False otherwise
        """
        if key not in self.l1_ttl:
            return False

        return time.time() > self.l1_ttl[key]

    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key.

        Args:
            key: Original cache key

        Returns:
            MD5 hash of key
        """
        return hashlib.md5(key.encode()).hexdigest()

    def get(
        self,
        key: str,
        miss_fn: Optional[Callable[[], Any]] = None,
        ttl: Optional[int] = None,
    ) -> Optional[Any]:
        """Get value from cache with automatic fallback.

        Tries L1 -> L2 -> L3 -> miss_fn in order.

        Args:
            key: Cache key
            miss_fn: Function to call on cache miss (optional)
            ttl: TTL in seconds for new cache entries (optional)

        Returns:
            Cached value or result of miss_fn, or None if not found
        """
        start_time = time.time()

        # Try L1 (memory)
        if key in self.l1_cache and not self._is_expired(key):
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_hit("l1", duration_ms)
            logger.debug(f"L1 cache HIT: {key} ({duration_ms:.2f}ms)")
            return self.l1_cache[key]

        if key in self.l1_cache:
            # Expired entry
            del self.l1_cache[key]
            del self.l1_ttl[key]

        self.metrics.record_miss("l1")

        # Try L2 (Redis)
        if self.redis:
            try:
                data = self.redis.get(self._hash_key(key))
                if data:
                    duration_ms = (time.time() - start_time) * 1000
                    self.metrics.record_hit("l2", duration_ms)
                    logger.debug(f"L2 cache HIT: {key} ({duration_ms:.2f}ms)")

                    # Deserialize and promote to L1
                    value = pickle.loads(data)
                    self._set_l1(key, value, ttl or self.l2_ttl)
                    return value
            except Exception as e:
                logger.warning(f"L2 cache error for key {key}: {e}")

        self.metrics.record_miss("l2")

        # Try L3 (disk)
        l3_path = self.l3_dir / f"{self._hash_key(key)}.cache"
        if l3_path.exists():
            try:
                with open(l3_path, "rb") as f:
                    data = pickle.load(f)

                duration_ms = (time.time() - start_time) * 1000
                self.metrics.record_hit("l3", duration_ms)
                logger.debug(f"L3 cache HIT: {key} ({duration_ms:.2f}ms)")

                # Promote to L1 and L2
                self._set_l1(key, data, ttl or self.l2_ttl)
                if self.redis:
                    self._set_l2(key, data, ttl or self.l2_ttl)

                return data
            except Exception as e:
                logger.warning(f"L3 cache error for key {key}: {e}")

        self.metrics.record_miss("l3")

        # Cache miss - compute value
        if miss_fn:
            logger.debug(f"Cache MISS: {key}, executing miss_fn")
            value = miss_fn()
            self.set(key, value, ttl=ttl or self.l2_ttl)
            return value

        return None

    def _set_l1(self, key: str, value: Any, ttl: int) -> None:
        """Set value in L1 cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        self.l1_cache[key] = value
        self.l1_ttl[key] = time.time() + ttl

    def _set_l2(self, key: str, value: Any, ttl: int) -> None:
        """Set value in L2 cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        if not self.redis:
            return

        try:
            serialized = pickle.dumps(value)
            self.redis.setex(self._hash_key(key), ttl, serialized)
        except Exception as e:
            logger.warning(f"L2 cache set error for key {key}: {e}")

    def _set_l3(self, key: str, value: Any) -> None:
        """Set value in L3 cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        l3_path = self.l3_dir / f"{self._hash_key(key)}.cache"
        try:
            with open(l3_path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"L3 cache set error for key {key}: {e}")

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        persist: bool = False,
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (default: use l2_ttl)
            persist: If True, also write to L3 disk cache (default: False)
        """
        cache_ttl = ttl or self.l2_ttl

        # Always set in L1
        self._set_l1(key, value, cache_ttl)

        # Set in L2 if available
        if self.redis:
            self._set_l2(key, value, cache_ttl)

        # Optionally persist to L3
        if persist:
            self._set_l3(key, value)

    def invalidate(self, pattern: str = "*") -> int:
        """Invalidate cache entries matching pattern.

        Args:
            pattern: Pattern to match keys (supports * wildcard)

        Returns:
            Number of keys invalidated
        """
        count = 0

        # Clear L1
        if pattern == "*":
            count = len(self.l1_cache)
            self.l1_cache.clear()
            self.l1_ttl.clear()
            logger.info(f"Cleared all L1 cache entries ({count})")
        else:
            keys_to_delete = [
                k for k in self.l1_cache
                if self._matches_pattern(k, pattern)
            ]
            for key in keys_to_delete:
                del self.l1_cache[key]
                if key in self.l1_ttl:
                    del self.l1_ttl[key]
            count = len(keys_to_delete)
            logger.info(f"Cleared {count} L1 cache entries matching '{pattern}'")

        # Clear L2
        if self.redis:
            try:
                if pattern == "*":
                    # Clear all keys in current DB
                    redis_count = self.redis.dbsize()
                    self.redis.flushdb()
                    count += redis_count
                    logger.info(f"Cleared {redis_count} L2 cache entries")
                else:
                    # Use SCAN to find matching keys
                    cursor = 0
                    redis_count = 0
                    while True:
                        cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
                        if keys:
                            redis_count += self.redis.delete(*keys)
                        if cursor == 0:
                            break
                    count += redis_count
                    logger.info(f"Cleared {redis_count} L2 cache entries matching '{pattern}'")
            except Exception as e:
                logger.warning(f"L2 invalidate error: {e}")

        # Clear L3
        try:
            if pattern == "*":
                l3_count = 0
                for cache_file in self.l3_dir.glob("*.cache"):
                    cache_file.unlink()
                    l3_count += 1
                count += l3_count
                logger.info(f"Cleared {l3_count} L3 cache entries")
            else:
                # For patterns, we'd need to decode each file to check the key
                # For now, only support full clear
                logger.warning("L3 pattern matching not fully supported, use '*' for full clear")
        except Exception as e:
            logger.warning(f"L3 invalidate error: {e}")

        return count

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern.

        Args:
            key: Cache key
            pattern: Pattern with * wildcard

        Returns:
            True if key matches pattern
        """
        if pattern == "*":
            return True

        # Simple wildcard matching
        if "*" in pattern:
            prefix, suffix = pattern.split("*", 1)
            return key.startswith(prefix) and key.endswith(suffix)

        return key == pattern

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics.

        Returns:
            Dictionary with cache statistics
        """
        stats = self.metrics.get_stats()

        # Add size information
        stats["l1_size"] = len(self.l1_cache)
        stats["l1_max_size"] = self.l1_max_size
        stats["l1_utilization"] = (
            len(self.l1_cache) / self.l1_max_size * 100
            if self.l1_max_size > 0 else 0.0
        )

        # Add L2 info
        stats["l2_available"] = self.redis is not None
        if self.redis:
            try:
                stats["l2_size"] = self.redis.dbsize()
            except Exception:
                stats["l2_size"] = 0
        else:
            stats["l2_size"] = 0

        # Add L3 info
        stats["l3_dir"] = str(self.l3_dir)
        stats["l3_size"] = len(list(self.l3_dir.glob("*.cache")))

        return stats

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics.reset()
        logger.info("Cache metrics reset")

    def health_check(self) -> Dict[str, Any]:
        """Check health of all cache layers.

        Returns:
            Health status for each layer
        """
        health = {
            "l1": {"status": "ok", "size": len(self.l1_cache)},
            "l2": {"status": "unavailable", "size": 0},
            "l3": {"status": "ok", "size": 0},
        }

        # Check L2
        if self.redis:
            try:
                self.redis.ping()
                health["l2"]["status"] = "ok"
                health["l2"]["size"] = self.redis.dbsize()
            except Exception as e:
                health["l2"]["status"] = "error"
                health["l2"]["error"] = str(e)

        # Check L3
        try:
            health["l3"]["size"] = len(list(self.l3_dir.glob("*.cache")))
            if not self.l3_dir.exists():
                health["l3"]["status"] = "error"
                health["l3"]["error"] = "Directory not found"
        except Exception as e:
            health["l3"]["status"] = "error"
            health["l3"]["error"] = str(e)

        return health


def make_cache_key(query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 20) -> str:
    """Generate deterministic cache key from search parameters.

    Args:
        query: Search query string
        filters: Optional filters dictionary
        limit: Result limit

    Returns:
        Cache key string
    """
    parts = [
        "search",
        query.lower().strip(),
        json.dumps(filters or {}, sort_keys=True),
        str(limit)
    ]
    key_str = "|".join(parts)
    return f"search:{hashlib.md5(key_str.encode()).hexdigest()}"
