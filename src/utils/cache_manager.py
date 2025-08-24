#!/usr/bin/env python3
"""
Cache Manager for Beverly Knits ERP System
Implements multi-level caching with TTL, persistence, and invalidation strategies.
"""

import pickle
import hashlib
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Union, Callable
from functools import lru_cache, wraps
import logging
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Comprehensive caching system with multiple storage backends and strategies.
    Supports in-memory, file-based caching with TTL and invalidation.
    """
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 default_ttl: int = 3600,
                 max_memory_items: int = 1000):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory for file-based cache storage
            default_ttl: Default time-to-live in seconds (1 hour)
            max_memory_items: Maximum items in memory cache
        """
        self.cache_dir = cache_dir or Path("./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_ttl = default_ttl
        self.max_memory_items = max_memory_items
        
        # In-memory cache with TTL
        self.memory_cache = OrderedDict()
        self.cache_metadata = {}
        
        # Statistics tracking
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0,
            'invalidations': 0,
            'errors': 0
        }
        
        # Thread lock for cache operations
        self.lock = threading.RLock()
        
        # Cache warming queue
        self.warming_queue = []
        
        logger.info(f"Cache manager initialized with TTL={default_ttl}s, max_items={max_memory_items}")
    
    def _generate_key(self, key: str, namespace: str = "default") -> str:
        """
        Generate a namespaced cache key.
        
        Args:
            key: Original key
            namespace: Cache namespace
            
        Returns:
            Namespaced key string
        """
        return f"{namespace}:{key}"
    
    def _hash_key(self, key: Any) -> str:
        """
        Generate a hash for complex keys.
        
        Args:
            key: Key to hash (can be dict, list, etc.)
            
        Returns:
            SHA256 hash of the key
        """
        if isinstance(key, (dict, list)):
            key_str = json.dumps(key, sort_keys=True)
        else:
            key_str = str(key)
        
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            
        Returns:
            Cached value or None if not found/expired
        """
        full_key = self._generate_key(key, namespace)
        
        with self.lock:
            # Check memory cache first
            if full_key in self.memory_cache:
                # Check TTL
                metadata = self.cache_metadata.get(full_key, {})
                if self._is_expired(metadata):
                    self._remove_from_memory(full_key)
                    self.stats['misses'] += 1
                    return None
                
                # Move to end (LRU)
                self.memory_cache.move_to_end(full_key)
                self.stats['hits'] += 1
                return self.memory_cache[full_key]
            
            # Check file cache
            file_path = self._get_file_path(full_key)
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Check TTL
                    if self._is_expired(data.get('metadata', {})):
                        file_path.unlink()
                        self.stats['misses'] += 1
                        return None
                    
                    value = data['value']
                    # Add to memory cache
                    self._add_to_memory(full_key, value, data.get('metadata'))
                    self.stats['hits'] += 1
                    return value
                except Exception as e:
                    logger.error(f"File cache read error: {e}")
                    self.stats['errors'] += 1
            
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, 
            ttl: Optional[int] = None,
            namespace: str = "default",
            persist: bool = True) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
            namespace: Cache namespace
            persist: Whether to persist to disk
            
        Returns:
            True if successful
        """
        full_key = self._generate_key(key, namespace)
        ttl = ttl or self.default_ttl
        
        metadata = {
            'created_at': time.time(),
            'ttl': ttl,
            'expires_at': time.time() + ttl if ttl else None
        }
        
        with self.lock:
            # Add to memory cache
            self._add_to_memory(full_key, value, metadata)
            
            if persist:
                # Save to file
                try:
                    file_path = self._get_file_path(full_key)
                    with open(file_path, 'wb') as f:
                        pickle.dump({'value': value, 'metadata': metadata}, f)
                except Exception as e:
                    logger.error(f"File cache write error: {e}")
                    self.stats['errors'] += 1
            
            self.stats['sets'] += 1
            return True
    
    def delete(self, key: str, namespace: str = "default") -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            
        Returns:
            True if deleted
        """
        full_key = self._generate_key(key, namespace)
        
        with self.lock:
            # Remove from memory
            deleted = self._remove_from_memory(full_key)
            
            # Remove file
            file_path = self._get_file_path(full_key)
            if file_path.exists():
                try:
                    file_path.unlink()
                    deleted = True
                except Exception as e:
                    logger.error(f"File delete error: {e}")
            
            if deleted:
                self.stats['invalidations'] += 1
            
            return deleted
    
    def invalidate_pattern(self, pattern: str, namespace: str = "default") -> int:
        """
        Invalidate all keys matching a pattern.
        
        Args:
            pattern: Pattern to match (supports * wildcard)
            namespace: Cache namespace
            
        Returns:
            Number of invalidated entries
        """
        count = 0
        pattern = self._generate_key(pattern, namespace)
        
        with self.lock:
            # Memory cache invalidation
            keys_to_delete = []
            for key in self.memory_cache:
                if self._matches_pattern(key, pattern):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                self._remove_from_memory(key)
                count += 1
            
            # File cache invalidation
            for file_path in self.cache_dir.glob("*.cache"):
                file_key = file_path.stem
                if self._matches_pattern(file_key, pattern):
                    try:
                        file_path.unlink()
                        count += 1
                    except Exception as e:
                        logger.error(f"File pattern delete error: {e}")
            
            self.stats['invalidations'] += count
            return count
    
    def clear_expired(self) -> int:
        """
        Clear all expired entries from cache.
        
        Returns:
            Number of cleared entries
        """
        count = 0
        
        with self.lock:
            # Clear expired from memory
            expired_keys = []
            for key, metadata in self.cache_metadata.items():
                if self._is_expired(metadata):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_from_memory(key)
                count += 1
            
            # Clear expired files
            for file_path in self.cache_dir.glob("*.cache"):
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    if self._is_expired(data.get('metadata', {})):
                        file_path.unlink()
                        count += 1
                except Exception as e:
                    logger.error(f"Error checking file expiry: {e}")
        
        logger.info(f"Cleared {count} expired cache entries")
        return count
    
    def warm_cache(self, items: List[Dict[str, Any]]) -> int:
        """
        Pre-populate cache with data.
        
        Args:
            items: List of dicts with 'key', 'value', 'ttl', 'namespace'
            
        Returns:
            Number of warmed entries
        """
        count = 0
        
        for item in items:
            try:
                self.set(
                    key=item['key'],
                    value=item['value'],
                    ttl=item.get('ttl'),
                    namespace=item.get('namespace', 'default')
                )
                count += 1
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
        
        logger.info(f"Warmed cache with {count} entries")
        return count
    
    def get_statistics(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'total_requests': total_requests,
                'hit_rate': hit_rate,
                'memory_items': len(self.memory_cache),
                'memory_size': sum(self._estimate_size(v) for v in self.memory_cache.values()),
                'file_items': len(list(self.cache_dir.glob("*.cache")))
            }
    
    def _add_to_memory(self, key: str, value: Any, metadata: Optional[Dict] = None) -> None:
        """Add item to memory cache with LRU eviction."""
        # Evict if at capacity
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove least recently used
            evicted_key = next(iter(self.memory_cache))
            self._remove_from_memory(evicted_key)
            self.stats['evictions'] += 1
        
        self.memory_cache[key] = value
        self.cache_metadata[key] = metadata or {
            'created_at': time.time(),
            'ttl': self.default_ttl
        }
    
    def _remove_from_memory(self, key: str) -> bool:
        """Remove item from memory cache."""
        if key in self.memory_cache:
            del self.memory_cache[key]
            if key in self.cache_metadata:
                del self.cache_metadata[key]
            return True
        return False
    
    def _is_expired(self, metadata: Dict) -> bool:
        """Check if cache entry is expired."""
        if not metadata:
            return True
        
        expires_at = metadata.get('expires_at')
        if expires_at and time.time() > expires_at:
            return True
        
        return False
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash for filename to avoid filesystem issues
        filename = self._hash_key(key) + ".cache"
        return self.cache_dir / filename
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern with wildcard support."""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            return 0


def cached(ttl: int = 3600, namespace: str = "default", 
          key_func: Optional[Callable] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds
        namespace: Cache namespace
        key_func: Custom key generation function
    
    Usage:
        @cached(ttl=300)
        def expensive_function(param):
            return compute_result(param)
    """
    def decorator(func):
        # Create a cache manager instance for this function
        cache = CacheManager()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            result = cache.get(cache_key, namespace)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl, namespace)
            
            return result
        
        # Add cache control methods
        wrapper.invalidate = lambda: cache.invalidate_pattern(f"{func.__name__}:*", namespace)
        wrapper.stats = lambda: cache.get_statistics()
        
        return wrapper
    
    return decorator


# Example usage for Beverly Knits ERP
class ERPCacheManager(CacheManager):
    """
    Specialized cache manager for Beverly Knits ERP with predefined namespaces.
    """
    
    NAMESPACES = {
        'yarn_inventory': 300,      # 5 minutes TTL
        'style_mappings': 3600,     # 1 hour TTL
        'bom_data': 1800,          # 30 minutes TTL
        'production_pipeline': 60,  # 1 minute TTL
        'forecasting': 900,        # 15 minutes TTL
        'kpi_metrics': 120        # 2 minutes TTL
    }
    
    def cache_yarn_inventory(self, data: Dict) -> bool:
        """Cache yarn inventory data."""
        return self.set('current', data, 
                       ttl=self.NAMESPACES['yarn_inventory'],
                       namespace='yarn_inventory')
    
    def get_yarn_inventory(self) -> Optional[Dict]:
        """Get cached yarn inventory."""
        return self.get('current', namespace='yarn_inventory')
    
    def cache_style_mapping(self, style: str, mapping: Dict) -> bool:
        """Cache individual style mapping."""
        return self.set(style, mapping,
                       ttl=self.NAMESPACES['style_mappings'],
                       namespace='style_mappings')
    
    def get_style_mapping(self, style: str) -> Optional[Dict]:
        """Get cached style mapping."""
        return self.get(style, namespace='style_mappings')
    
    def cache_bom(self, style: str, bom_data: List) -> bool:
        """Cache BOM data for a style."""
        return self.set(style, bom_data,
                       ttl=self.NAMESPACES['bom_data'],
                       namespace='bom_data')
    
    def get_bom(self, style: str) -> Optional[List]:
        """Get cached BOM data."""
        return self.get(style, namespace='bom_data')
    
    def invalidate_production_data(self) -> int:
        """Invalidate all production-related cache."""
        count = 0
        count += self.invalidate_pattern('*', namespace='production_pipeline')
        count += self.invalidate_pattern('*', namespace='kpi_metrics')
        return count


# Test functions
def test_cache_manager():
    """Test cache manager functionality."""
    
    print("=" * 80)
    print("CACHE MANAGER TEST")
    print("=" * 80)
    
    # Initialize cache manager
    cache = CacheManager(default_ttl=5)
    
    # Test basic operations
    print("\n1. Testing basic get/set:")
    cache.set("test_key", "test_value", ttl=10)
    value = cache.get("test_key")
    print(f"   Set and retrieved: {value}")
    
    # Test complex data
    print("\n2. Testing complex data:")
    complex_data = {
        'inventory': [1, 2, 3],
        'orders': {'order1': 100, 'order2': 200}
    }
    cache.set("complex", complex_data)
    retrieved = cache.get("complex")
    print(f"   Complex data retrieved: {retrieved == complex_data}")
    
    # Test expiration
    print("\n3. Testing TTL expiration:")
    cache.set("expire_test", "will expire", ttl=1)
    print(f"   Immediately after set: {cache.get('expire_test')}")
    time.sleep(2)
    print(f"   After 2 seconds: {cache.get('expire_test')}")
    
    # Test pattern invalidation
    print("\n4. Testing pattern invalidation:")
    cache.set("user:1", "Alice", namespace="users")
    cache.set("user:2", "Bob", namespace="users")
    cache.set("user:3", "Charlie", namespace="users")
    
    invalidated = cache.invalidate_pattern("user:*", namespace="users")
    print(f"   Invalidated {invalidated} entries")
    print(f"   After invalidation: {cache.get('user:1', namespace='users')}")
    
    # Test decorator
    print("\n5. Testing @cached decorator:")
    
    @cached(ttl=60)
    def expensive_calculation(n):
        print(f"   Computing factorial({n})...")
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
    
    # First call - computes
    result1 = expensive_calculation(5)
    print(f"   First call result: {result1}")
    
    # Second call - from cache
    result2 = expensive_calculation(5)
    print(f"   Second call (cached): {result2}")
    
    # Test ERP-specific cache
    print("\n6. Testing ERP Cache Manager:")
    erp_cache = ERPCacheManager()
    
    # Cache yarn inventory
    yarn_data = {'yarn_001': 1000, 'yarn_002': 2000}
    erp_cache.cache_yarn_inventory(yarn_data)
    cached_yarn = erp_cache.get_yarn_inventory()
    print(f"   Cached yarn inventory: {cached_yarn}")
    
    # Get statistics
    print("\n7. Cache Statistics:")
    stats = cache.get_statistics()
    for key, value in stats.items():
        if key != 'memory_size':
            print(f"   {key}: {value}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_cache_manager()