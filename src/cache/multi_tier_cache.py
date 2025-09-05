"""
Multi-Tier Cache System
High-performance caching with L1 memory, L2 Redis, and intelligent eviction
"""

import json
import pickle
import hashlib
import time
from typing import Any, Optional, Callable, Dict, List, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
from functools import wraps
import logging
import threading
import redis
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    size: int
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: int = 300
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        return time.time() - self.created_at > self.ttl
    
    @property
    def age(self) -> float:
        """Get age in seconds"""
        return time.time() - self.created_at
    
    @property
    def last_access_age(self) -> float:
        """Get time since last access"""
        return time.time() - self.accessed_at


class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 100):
        """Initialize LRU cache"""
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry = self.cache[key]
                
                # Check expiration
                if entry.is_expired:
                    del self.cache[key]
                    self.misses += 1
                    return None
                
                # Update access metadata
                entry.accessed_at = time.time()
                entry.access_count += 1
                
                self.hits += 1
                return entry.value
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache"""
        with self.lock:
            # Calculate size
            try:
                size = len(pickle.dumps(value))
            except:
                size = 0
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl=ttl
            )
            
            # Check if we need to evict
            if key not in self.cache and len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            
            # Add to cache
            self.cache[key] = entry
            self.cache.move_to_end(key)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            total_size = sum(entry.size for entry in self.cache.values())
            
            return {
                'entries': len(self.cache),
                'max_size': self.max_size,
                'total_size_bytes': total_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }


class MultiTierCache:
    """Multi-level caching system with L1 memory and L2 Redis"""
    
    # Default TTLs for different data types (seconds)
    DEFAULT_TTLS = {
        'yarn_inventory': 300,      # 5 minutes
        'bom_data': 3600,          # 1 hour
        'production_orders': 60,    # 1 minute
        'ml_predictions': 1800,     # 30 minutes
        'static_data': 86400,       # 24 hours
        'default': 300              # 5 minutes
    }
    
    def __init__(self, 
                 l1_max_size: int = 100,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 enable_redis: bool = True):
        """
        Initialize multi-tier cache
        
        Args:
            l1_max_size: Maximum entries in L1 memory cache
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            enable_redis: Whether to enable Redis L2 cache
        """
        # L1: In-memory cache (microseconds access)
        self.l1_cache = LRUCache(max_size=l1_max_size)
        
        # L2: Redis cache (milliseconds access)
        self.enable_redis = enable_redis
        self.l2_cache = None
        
        if enable_redis:
            try:
                self.l2_cache = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False  # For pickle support
                )
                # Test connection
                self.l2_cache.ping()
                logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using L1 cache only.")
                self.l2_cache = None
                self.enable_redis = False
        
        # Cache statistics
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
        
        # Cache warming
        self.warm_cache_on_miss = True
        self.warming_callbacks = {}
    
    def _get_ttl(self, key: str) -> int:
        """Get TTL based on key pattern"""
        for data_type, ttl in self.DEFAULT_TTLS.items():
            if data_type in key:
                return ttl
        return self.DEFAULT_TTLS['default']
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str, 
            loader: Optional[Callable] = None,
            ttl: Optional[int] = None) -> Optional[Any]:
        """
        Get value from cache with fallback chain
        
        Args:
            key: Cache key
            loader: Optional function to load data if not cached
            ttl: Optional TTL override
            
        Returns:
            Cached value or None
        """
        # L1: Check memory cache
        value = self.l1_cache.get(key)
        if value is not None:
            self.l1_hits += 1
            return value
        
        # L2: Check Redis cache
        if self.l2_cache:
            try:
                serialized = self.l2_cache.get(key)
                if serialized:
                    value = pickle.loads(serialized)
                    
                    # Promote to L1
                    self.l1_cache.set(key, value, ttl or self._get_ttl(key))
                    
                    self.l2_hits += 1
                    return value
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Cache miss
        self.misses += 1
        
        # Load data if loader provided
        if loader:
            try:
                value = loader()
                if value is not None:
                    self.set(key, value, ttl)
                return value
            except Exception as e:
                logger.error(f"Loader function failed: {e}")
        
        # Check if we should warm cache
        if self.warm_cache_on_miss and key in self.warming_callbacks:
            self._warm_cache_entry(key)
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in all cache levels
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        ttl = ttl or self._get_ttl(key)
        
        # L1: Set in memory cache
        self.l1_cache.set(key, value, ttl)
        
        # L2: Set in Redis cache
        if self.l2_cache:
            try:
                serialized = pickle.dumps(value)
                self.l2_cache.setex(key, ttl, serialized)
            except Exception as e:
                logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str):
        """Delete key from all cache levels"""
        # L1: Delete from memory
        self.l1_cache.delete(key)
        
        # L2: Delete from Redis
        if self.l2_cache:
            try:
                self.l2_cache.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
    
    def invalidate(self, pattern: str):
        """
        Invalidate cache entries matching pattern
        
        Args:
            pattern: Pattern to match (e.g., "yarn_*", "*inventory*")
        """
        # L1: Clear matching entries
        keys_to_remove = []
        for key in self.l1_cache.cache.keys():
            if self._matches_pattern(key, pattern):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.l1_cache.delete(key)
        
        # L2: Clear from Redis
        if self.l2_cache:
            try:
                # Use SCAN to avoid blocking
                cursor = 0
                while True:
                    cursor, keys = self.l2_cache.scan(
                        cursor, 
                        match=pattern,
                        count=100
                    )
                    if keys:
                        self.l2_cache.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Redis invalidate error: {e}")
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern"""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    def clear_all(self):
        """Clear all cache entries"""
        # L1: Clear memory cache
        self.l1_cache.clear()
        
        # L2: Clear Redis cache
        if self.l2_cache:
            try:
                self.l2_cache.flushdb()
            except Exception as e:
                logger.error(f"Redis clear error: {e}")
        
        # Reset statistics
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.l1_hits + self.l2_hits + self.misses
        
        stats = {
            'l1_stats': self.l1_cache.get_stats(),
            'l1_hits': self.l1_hits,
            'l2_hits': self.l2_hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate': (self.l1_hits + self.l2_hits) / total_requests if total_requests > 0 else 0,
            'l1_hit_rate': self.l1_hits / total_requests if total_requests > 0 else 0,
            'l2_hit_rate': self.l2_hits / total_requests if total_requests > 0 else 0,
            'redis_enabled': self.enable_redis and self.l2_cache is not None
        }
        
        # Add Redis stats if available
        if self.l2_cache:
            try:
                info = self.l2_cache.info()
                stats['redis_info'] = {
                    'used_memory': info.get('used_memory_human', 'N/A'),
                    'connected_clients': info.get('connected_clients', 0),
                    'total_commands': info.get('total_commands_processed', 0)
                }
            except:
                pass
        
        return stats
    
    def register_warming_callback(self, key_pattern: str, callback: Callable):
        """
        Register callback for cache warming
        
        Args:
            key_pattern: Pattern of keys to warm
            callback: Function that returns data to cache
        """
        self.warming_callbacks[key_pattern] = callback
    
    def _warm_cache_entry(self, key: str):
        """Warm a specific cache entry"""
        for pattern, callback in self.warming_callbacks.items():
            if self._matches_pattern(key, pattern):
                try:
                    value = callback(key)
                    if value is not None:
                        self.set(key, value)
                        logger.info(f"Cache warmed for key: {key}")
                except Exception as e:
                    logger.error(f"Cache warming failed for {key}: {e}")
    
    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple keys at once
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs
        """
        results = {}
        
        # Check L1 first
        l1_misses = []
        for key in keys:
            value = self.l1_cache.get(key)
            if value is not None:
                results[key] = value
                self.l1_hits += 1
            else:
                l1_misses.append(key)
        
        # Check L2 for L1 misses
        if l1_misses and self.l2_cache:
            try:
                # Use pipeline for efficiency
                pipe = self.l2_cache.pipeline()
                for key in l1_misses:
                    pipe.get(key)
                
                l2_results = pipe.execute()
                
                for key, serialized in zip(l1_misses, l2_results):
                    if serialized:
                        value = pickle.loads(serialized)
                        results[key] = value
                        
                        # Promote to L1
                        self.l1_cache.set(key, value, self._get_ttl(key))
                        self.l2_hits += 1
                    else:
                        self.misses += 1
            except Exception as e:
                logger.error(f"Redis batch get error: {e}")
                self.misses += len(l1_misses)
        else:
            self.misses += len(l1_misses)
        
        return results
    
    def batch_set(self, items: Dict[str, Any], ttl: Optional[int] = None):
        """
        Set multiple key-value pairs at once
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Optional TTL for all items
        """
        # Set in L1
        for key, value in items.items():
            self.l1_cache.set(key, value, ttl or self._get_ttl(key))
        
        # Set in L2
        if self.l2_cache:
            try:
                pipe = self.l2_cache.pipeline()
                
                for key, value in items.items():
                    serialized = pickle.dumps(value)
                    pipe.setex(key, ttl or self._get_ttl(key), serialized)
                
                pipe.execute()
            except Exception as e:
                logger.error(f"Redis batch set error: {e}")


def cached(ttl: int = 300, 
          key_prefix: str = None,
          cache_instance: Optional[MultiTierCache] = None):
    """
    Decorator for automatic caching
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Optional prefix for cache keys
        cache_instance: Cache instance to use (creates new if None)
    """
    def decorator(func: Callable) -> Callable:
        # Create cache instance if not provided
        cache = cache_instance or MultiTierCache()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_prefix:
                cache_key = f"{key_prefix}:{func.__name__}"
            else:
                cache_key = func.__name__
            
            # Add arguments to key
            cache_key += f":{cache._generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            
            if result is not None:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            
            if result is not None:
                cache.set(cache_key, result, ttl)
            
            return result
        
        # Add cache control methods
        wrapper.cache = cache
        wrapper.invalidate = lambda: cache.invalidate(f"{key_prefix or ''}:{func.__name__}:*")
        
        return wrapper
    
    return decorator