#!/usr/bin/env python3
"""
Beverly Knits ERP - Cache Optimization Module
Implements intelligent caching, cache warming, and cache statistics
Part of Phase 3: Performance Optimization
"""

import time
import logging
import json
import hashlib
import pickle
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from collections import OrderedDict, defaultdict
import pandas as pd
import threading
from pathlib import Path
import redis
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache optimization"""
    max_size_mb: int = 200
    default_ttl_seconds: int = 300  # 5 minutes
    warm_cache_on_startup: bool = True
    use_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    enable_statistics: bool = True
    eviction_policy: str = "LRU"  # LRU, LFU, or TTL


class CacheStatistics:
    """Track cache performance statistics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_requests = 0
        self.cache_sizes = []
        self.response_times = defaultdict(list)
        self.popular_keys = defaultdict(int)
        self.lock = threading.Lock()
    
    def record_hit(self, key: str, response_time: float):
        """Record a cache hit"""
        with self.lock:
            self.hits += 1
            self.total_requests += 1
            self.response_times['hit'].append(response_time)
            self.popular_keys[key] += 1
    
    def record_miss(self, key: str, response_time: float):
        """Record a cache miss"""
        with self.lock:
            self.misses += 1
            self.total_requests += 1
            self.response_times['miss'].append(response_time)
    
    def record_eviction(self):
        """Record a cache eviction"""
        with self.lock:
            self.evictions += 1
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        with self.lock:
            if self.total_requests == 0:
                return 0.0
            return (self.hits / self.total_requests) * 100
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self.lock:
            stats = {
                'hits': self.hits,
                'misses': self.misses,
                'total_requests': self.total_requests,
                'hit_rate': self.get_hit_rate(),
                'evictions': self.evictions,
                'avg_hit_time': sum(self.response_times['hit']) / len(self.response_times['hit']) 
                              if self.response_times['hit'] else 0,
                'avg_miss_time': sum(self.response_times['miss']) / len(self.response_times['miss'])
                               if self.response_times['miss'] else 0,
                'top_keys': sorted(self.popular_keys.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            return stats


class LRUCache:
    """Least Recently Used cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self.lock:
            if key in self.cache:
                # Update and move to end
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                self.cache[key] = value
                # Evict oldest if over capacity
                if len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
    
    def clear(self):
        """Clear the cache"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        with self.lock:
            return len(self.cache)


class CacheOptimizer:
    """
    Intelligent cache optimization for Beverly Knits ERP
    Supports multiple cache backends and optimization strategies
    """
    
    # TTL configurations for different data types
    TTL_CONFIG = {
        'static': 3600,      # 1 hour for static data
        'inventory': 300,    # 5 minutes for inventory
        'forecast': 900,     # 15 minutes for forecasts
        'planning': 600,     # 10 minutes for planning
        'yarn': 300,         # 5 minutes for yarn data
        'production': 180,   # 3 minutes for production
        'default': 300       # 5 minutes default
    }
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache optimizer
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.statistics = CacheStatistics()
        
        # Initialize cache backend
        if self.config.use_redis:
            self._init_redis_cache()
        else:
            self._init_memory_cache()
        
        # Cache warming queue
        self.warm_cache_queue = []
        self.warming_thread = None
        
        logger.info(f"CacheOptimizer initialized with {self.config.eviction_policy} policy")
        
        # Start cache warming if enabled (disabled for now to prevent hanging)
        # if self.config.warm_cache_on_startup:
        #     self.start_cache_warming()
    
    def _init_memory_cache(self):
        """Initialize in-memory cache"""
        self.cache = LRUCache(max_size=1000)
        self.ttl_store = {}  # Store TTL for each key
        logger.info("Using in-memory LRU cache")
    
    def _init_redis_cache(self):
        """Initialize Redis cache"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=False
            )
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to memory cache.")
            self.config.use_redis = False
            self._init_memory_cache()
    
    def _generate_cache_key(self, 
                           endpoint: str, 
                           params: Optional[Dict] = None,
                           data_type: str = 'default') -> str:
        """
        Generate a unique cache key
        
        Args:
            endpoint: API endpoint or function name
            params: Parameters for the request
            data_type: Type of data for TTL selection
            
        Returns:
            Cache key string
        """
        key_parts = [endpoint]
        
        if params:
            # Sort params for consistent key generation
            sorted_params = sorted(params.items())
            params_str = json.dumps(sorted_params, sort_keys=True)
            key_parts.append(hashlib.md5(params_str.encode()).hexdigest()[:8])
        
        key_parts.append(data_type)
        
        return ":".join(key_parts)
    
    def get(self, 
            key: str, 
            fetch_function: Optional[Callable] = None,
            ttl: Optional[int] = None) -> Any:
        """
        Get item from cache or fetch if not present
        
        Args:
            key: Cache key
            fetch_function: Function to call if cache miss
            ttl: Time to live in seconds
            
        Returns:
            Cached or fetched value
        """
        start_time = time.perf_counter()
        
        # Try to get from cache
        cached_value = self._get_from_cache(key)
        
        if cached_value is not None:
            # Cache hit
            response_time = (time.perf_counter() - start_time) * 1000
            self.statistics.record_hit(key, response_time)
            logger.debug(f"Cache hit for {key} ({response_time:.2f}ms)")
            return cached_value
        
        # Cache miss
        if fetch_function:
            logger.debug(f"Cache miss for {key}, fetching...")
            value = fetch_function()
            
            # Store in cache
            if value is not None:
                self.put(key, value, ttl)
            
            response_time = (time.perf_counter() - start_time) * 1000
            self.statistics.record_miss(key, response_time)
            
            return value
        
        response_time = (time.perf_counter() - start_time) * 1000
        self.statistics.record_miss(key, response_time)
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Put item in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        if ttl is None:
            # Determine TTL based on key pattern
            data_type = key.split(':')[-1] if ':' in key else 'default'
            ttl = self.TTL_CONFIG.get(data_type, self.config.default_ttl_seconds)
        
        self._put_in_cache(key, value, ttl)
        logger.debug(f"Cached {key} with TTL {ttl}s")
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get from underlying cache implementation"""
        if self.config.use_redis:
            try:
                value = self.redis_client.get(key)
                if value:
                    return pickle.loads(value)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        else:
            # Check TTL for memory cache
            if key in self.ttl_store:
                if datetime.now() > self.ttl_store[key]:
                    # Expired
                    del self.ttl_store[key]
                    return None
            
            return self.cache.get(key)
        
        return None
    
    def _put_in_cache(self, key: str, value: Any, ttl: int):
        """Put in underlying cache implementation"""
        if self.config.use_redis:
            try:
                serialized = pickle.dumps(value)
                self.redis_client.setex(key, ttl, serialized)
            except Exception as e:
                logger.error(f"Redis put error: {e}")
        else:
            self.cache.put(key, value)
            self.ttl_store[key] = datetime.now() + timedelta(seconds=ttl)
    
    def invalidate(self, pattern: Optional[str] = None):
        """
        Invalidate cache entries
        
        Args:
            pattern: Pattern to match keys (None = clear all)
        """
        if pattern is None:
            # Clear all
            if self.config.use_redis:
                self.redis_client.flushdb()
            else:
                self.cache.clear()
                self.ttl_store.clear()
            logger.info("Cache cleared")
        else:
            # Clear matching pattern
            if self.config.use_redis:
                for key in self.redis_client.scan_iter(match=pattern):
                    self.redis_client.delete(key)
            else:
                # Memory cache doesn't support pattern matching efficiently
                keys_to_delete = [k for k in self.cache.cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self.cache.cache[key]
                    self.ttl_store.pop(key, None)
            
            logger.info(f"Invalidated cache entries matching '{pattern}'")
    
    def warm_cache(self, endpoints: List[Tuple[str, Dict, str]]):
        """
        Warm cache with specified endpoints
        
        Args:
            endpoints: List of (endpoint, params, data_type) tuples
        """
        logger.info(f"Warming cache with {len(endpoints)} endpoints...")
        
        for endpoint, params, data_type in endpoints:
            key = self._generate_cache_key(endpoint, params, data_type)
            # Add to warming queue
            self.warm_cache_queue.append((key, endpoint, params, data_type))
        
        # Process warming queue
        self._process_warming_queue()
    
    def _process_warming_queue(self):
        """Process cache warming queue"""
        while self.warm_cache_queue:
            key, endpoint, params, data_type = self.warm_cache_queue.pop(0)
            
            # Skip if already cached
            if self._get_from_cache(key) is not None:
                continue
            
            # TODO: Implement actual data fetching based on endpoint
            # For now, we'll just log the warming request
            logger.debug(f"Would warm cache for {endpoint} with params {params}")
    
    def start_cache_warming(self):
        """Start automatic cache warming"""
        # Define critical endpoints to warm
        critical_endpoints = [
            ("/api/yarn-intelligence", {}, "yarn"),
            ("/api/inventory-intelligence-enhanced", {}, "inventory"),
            ("/api/production-pipeline", {}, "production"),
            ("/api/six-phase-planning", {}, "planning")
        ]
        
        # Start warming in background (disabled for testing)
        # threading.Thread(target=self.warm_cache, args=(critical_endpoints,), daemon=True).start()
        logger.info("Cache warming configured (disabled for testing)")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics
        
        Returns:
            Cache statistics dictionary
        """
        stats = self.statistics.get_statistics()
        
        # Add cache-specific stats
        if self.config.use_redis:
            info = self.redis_client.info('memory')
            stats['backend'] = 'redis'
            stats['memory_used_mb'] = info.get('used_memory', 0) / 1024 / 1024
            stats['keys'] = self.redis_client.dbsize()
        else:
            stats['backend'] = 'memory'
            stats['keys'] = self.cache.size()
            stats['memory_used_mb'] = sys.getsizeof(self.cache.cache) / 1024 / 1024
        
        # Add recommendations
        stats['recommendations'] = self._generate_recommendations(stats)
        
        return stats
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate cache optimization recommendations"""
        recommendations = []
        
        hit_rate = stats.get('hit_rate', 0)
        if hit_rate < 50:
            recommendations.append("Low hit rate - consider increasing TTL or cache size")
        elif hit_rate < 80:
            recommendations.append("Moderate hit rate - review cache warming strategy")
        
        if stats.get('evictions', 0) > stats.get('hits', 0) * 0.1:
            recommendations.append("High eviction rate - increase cache size")
        
        if stats.get('avg_miss_time', 0) > 1000:  # 1 second
            recommendations.append("Slow miss times - optimize fetch functions")
        
        return recommendations
    
    def optimize_ttl(self, 
                    endpoint: str,
                    access_pattern: List[datetime],
                    current_ttl: int) -> int:
        """
        Optimize TTL based on access patterns
        
        Args:
            endpoint: Endpoint to optimize
            access_pattern: List of access timestamps
            current_ttl: Current TTL value
            
        Returns:
            Optimized TTL value
        """
        if len(access_pattern) < 2:
            return current_ttl
        
        # Calculate average time between accesses
        intervals = []
        for i in range(1, len(access_pattern)):
            interval = (access_pattern[i] - access_pattern[i-1]).total_seconds()
            intervals.append(interval)
        
        avg_interval = sum(intervals) / len(intervals)
        
        # Set TTL to 2x average interval (with bounds)
        optimized_ttl = min(max(int(avg_interval * 2), 60), 3600)  # Between 1 min and 1 hour
        
        logger.info(f"Optimized TTL for {endpoint}: {current_ttl}s -> {optimized_ttl}s")
        
        return optimized_ttl


def cached(ttl: int = 300, data_type: str = 'default'):
    """
    Decorator for caching function results
    
    Args:
        ttl: Time to live in seconds
        data_type: Type of data for cache key generation
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create cache optimizer
            cache = get_cache_optimizer()
            
            # Generate cache key
            key = cache._generate_cache_key(
                func.__name__,
                {'args': str(args), 'kwargs': str(kwargs)},
                data_type
            )
            
            # Try to get from cache or compute
            return cache.get(
                key,
                lambda: func(*args, **kwargs),
                ttl
            )
        
        return wrapper
    return decorator


def invalidate_on_change(*patterns: str):
    """
    Decorator to invalidate cache when function is called
    
    Args:
        patterns: Cache key patterns to invalidate
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Run function
            result = func(*args, **kwargs)
            
            # Invalidate cache patterns
            cache = get_cache_optimizer()
            for pattern in patterns:
                cache.invalidate(pattern)
            
            return result
        
        return wrapper
    return decorator


# Global instance
_cache_instance = None

def get_cache_optimizer() -> CacheOptimizer:
    """Get singleton instance of CacheOptimizer"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheOptimizer()
    return _cache_instance


def test_cache_optimizer():
    """Test the cache optimizer"""
    print("=" * 80)
    print("Testing CacheOptimizer")
    print("=" * 80)
    
    # Create cache optimizer
    config = CacheConfig(
        max_size_mb=100,
        default_ttl_seconds=60,
        use_redis=False  # Use memory cache for testing
    )
    cache = CacheOptimizer(config)
    
    # Test 1: Basic caching
    print("\n1. Testing Basic Caching:")
    
    def expensive_function():
        time.sleep(0.1)  # Simulate expensive operation
        return "expensive_result"
    
    # First call - cache miss
    key = cache._generate_cache_key("/api/test", {"param": "value"}, "test")
    result1 = cache.get(key, expensive_function, ttl=60)
    print(f"  First call result: {result1}")
    
    # Second call - cache hit
    result2 = cache.get(key, expensive_function, ttl=60)
    print(f"  Second call result: {result2}")
    
    # Test 2: Cache statistics
    print("\n2. Cache Statistics:")
    stats = cache.get_cache_statistics()
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']:.1f}%")
    print(f"  Avg Hit Time: {stats['avg_hit_time']:.2f}ms")
    print(f"  Avg Miss Time: {stats['avg_miss_time']:.2f}ms")
    
    # Test 3: Cache invalidation
    print("\n3. Testing Cache Invalidation:")
    cache.put("test:key:1", "value1", ttl=60)
    cache.put("test:key:2", "value2", ttl=60)
    cache.put("other:key:1", "value3", ttl=60)
    
    print(f"  Cache size before invalidation: {cache.cache.size()}")
    cache.invalidate("test:*")
    print(f"  Cache size after invalidating 'test:*': {cache.cache.size()}")
    
    # Test 4: TTL optimization
    print("\n4. Testing TTL Optimization:")
    access_times = [
        datetime.now() - timedelta(minutes=10),
        datetime.now() - timedelta(minutes=8),
        datetime.now() - timedelta(minutes=6),
        datetime.now() - timedelta(minutes=4),
        datetime.now() - timedelta(minutes=2),
        datetime.now()
    ]
    
    optimized_ttl = cache.optimize_ttl("/api/endpoint", access_times, 300)
    print(f"  Original TTL: 300s")
    print(f"  Optimized TTL: {optimized_ttl}s")
    
    # Test 5: Decorator usage
    print("\n5. Testing Cache Decorator:")
    
    @cached(ttl=10, data_type='test')
    def decorated_function(x):
        time.sleep(0.05)
        return x * 2
    
    start = time.perf_counter()
    result1 = decorated_function(5)
    time1 = (time.perf_counter() - start) * 1000
    
    start = time.perf_counter()
    result2 = decorated_function(5)
    time2 = (time.perf_counter() - start) * 1000
    
    print(f"  First call: {result1} ({time1:.2f}ms)")
    print(f"  Second call: {result2} ({time2:.2f}ms)")
    print(f"  Speedup: {time1/time2:.1f}x")
    
    # Test 6: Recommendations
    print("\n6. Cache Recommendations:")
    final_stats = cache.get_cache_statistics()
    if final_stats['recommendations']:
        for rec in final_stats['recommendations']:
            print(f"  - {rec}")
    else:
        print("  No recommendations (cache performing well)")
    
    print("\n" + "=" * 80)
    print("✅ CacheOptimizer test complete")


if __name__ == "__main__":
    test_cache_optimizer()