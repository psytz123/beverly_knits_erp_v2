"""Multi-tier caching implementation for optimal performance."""

import json
import logging
import time
from typing import Any, Optional, Dict, List, Callable
from dataclasses import dataclass
from collections import OrderedDict
from datetime import datetime, timedelta
import hashlib
import pickle
import redis
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass
class CacheStrategy:
    """Cache strategy configuration."""
    ttl: int = 900  # Time to live in seconds
    warm_on_start: bool = False
    compression: bool = False
    max_size: int = 100  # Max items in memory


class LRUCache:
    """Least Recently Used cache implementation for L1 memory cache."""
    
    def __init__(self, maxsize: int = 100):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            value, expiry = self.cache[key]
            
            # Check expiry
            if expiry and time.time() > expiry:
                del self.cache[key]
                self.misses += 1
                return None
            
            return value
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any, ttl: int = None):
        """Put item in cache."""
        # Remove if already exists
        if key in self.cache:
            del self.cache[key]
        
        # Add to end
        expiry = time.time() + ttl if ttl else None
        self.cache[key] = (value, expiry)
        
        # Remove oldest if over capacity
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def __len__(self) -> int:
        return len(self.cache)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for complex types."""
    
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


class MultiTierCache:
    """
    Multi-level caching for optimal performance.
    L1: In-memory LRU cache (microseconds)
    L2: Redis cache (milliseconds)
    L3: Disk cache (optional, for large datasets)
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0, enable_disk: bool = False):
        """Initialize multi-tier cache."""
        self.logger = logging.getLogger(__name__)
        
        # L1: In-memory cache (microseconds)
        self.l1_memory = LRUCache(maxsize=100)
        self.l1_hits = 0
        self.l1_misses = 0
        
        # L2: Redis cache (milliseconds)
        try:
            self.l2_redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=False,  # Use binary for flexibility
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.l2_redis.ping()
            self.l2_enabled = True
            self.logger.info(f"Redis cache connected at {redis_host}:{redis_port}")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self.logger.warning(f"Redis not available: {e}. Using memory cache only.")
            self.l2_redis = None
            self.l2_enabled = False
        
        self.l2_hits = 0
        self.l2_misses = 0
        
        # Cache strategies by data type
        self.strategies = {
            'yarn_inventory': CacheStrategy(ttl=900, warm_on_start=True, max_size=200),
            'bom_data': CacheStrategy(ttl=3600, warm_on_start=True, max_size=500),
            'production_orders': CacheStrategy(ttl=60, warm_on_start=False, max_size=100),
            'ml_predictions': CacheStrategy(ttl=1800, warm_on_start=False, max_size=50),
            'api_responses': CacheStrategy(ttl=300, warm_on_start=False, max_size=100),
            'default': CacheStrategy(ttl=600, warm_on_start=False, max_size=100)
        }
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics
        self.stats = {
            'l1_size': 0,
            'l2_size': 0,
            'total_requests': 0,
            'cache_sets': 0,
            'cache_deletes': 0
        }
    
    def _generate_cache_key(self, key: str, namespace: str = None) -> str:
        """Generate cache key with optional namespace."""
        if namespace:
            return f"bki:{namespace}:{key}"
        return f"bki:{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Try JSON first (more portable)
            json_str = json.dumps(value, cls=CustomJSONEncoder)
            return json_str.encode('utf-8')
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if not data:
            return None
        
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            try:
                return pickle.loads(data)
            except pickle.UnpicklingError:
                return None
    
    async def get(self, key: str, data_type: str = 'default') -> Optional[Any]:
        """Get from cache with fallback chain."""
        self.stats['total_requests'] += 1
        
        # Generate full cache key
        cache_key = self._generate_cache_key(key, data_type)
        
        # L1: Check memory cache
        value = self.l1_memory.get(cache_key)
        if value is not None:
            self.l1_hits += 1
            return value
        self.l1_misses += 1
        
        # L2: Check Redis cache
        if self.l2_enabled:
            try:
                redis_value = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.l2_redis.get, cache_key
                )
                
                if redis_value:
                    self.l2_hits += 1
                    # Deserialize
                    value = self._deserialize(redis_value)
                    
                    # Promote to L1
                    strategy = self.strategies.get(data_type, self.strategies['default'])
                    self.l1_memory.put(cache_key, value, strategy.ttl)
                    
                    return value
            except redis.RedisError as e:
                self.logger.error(f"Redis get error: {e}")
        
        self.l2_misses += 1
        return None
    
    async def set(self, key: str, value: Any, data_type: str = 'default', ttl: int = None):
        """Set in all cache levels."""
        self.stats['cache_sets'] += 1
        
        # Get strategy
        strategy = self.strategies.get(data_type, self.strategies['default'])
        ttl = ttl or strategy.ttl
        
        # Generate full cache key
        cache_key = self._generate_cache_key(key, data_type)
        
        # L1: Memory cache
        self.l1_memory.put(cache_key, value, ttl)
        
        # L2: Redis cache (async)
        if self.l2_enabled:
            try:
                serialized = self._serialize(value)
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    self.l2_redis.setex,
                    cache_key,
                    ttl,
                    serialized
                )
            except redis.RedisError as e:
                self.logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str, data_type: str = None) -> bool:
        """Delete from all cache levels."""
        self.stats['cache_deletes'] += 1
        
        # Generate full cache key
        cache_key = self._generate_cache_key(key, data_type)
        
        # L1: Delete from memory
        l1_deleted = self.l1_memory.delete(cache_key)
        
        # L2: Delete from Redis
        l2_deleted = False
        if self.l2_enabled:
            try:
                l2_deleted = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.l2_redis.delete, cache_key
                ) > 0
            except redis.RedisError as e:
                self.logger.error(f"Redis delete error: {e}")
        
        return l1_deleted or l2_deleted
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        count = 0
        
        # L1: Clear matching from memory
        keys_to_delete = [k for k in self.l1_memory.cache.keys() if pattern in k]
        for key in keys_to_delete:
            if self.l1_memory.delete(key):
                count += 1
        
        # L2: Delete from Redis
        if self.l2_enabled:
            try:
                redis_pattern = self._generate_cache_key(pattern, '*')
                cursor = 0
                
                while True:
                    cursor, keys = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.l2_redis.scan,
                        cursor,
                        redis_pattern,
                        1000
                    )
                    
                    if keys:
                        deleted = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            self.l2_redis.delete,
                            *keys
                        )
                        count += deleted
                    
                    if cursor == 0:
                        break
                        
            except redis.RedisError as e:
                self.logger.error(f"Redis delete pattern error: {e}")
        
        return count
    
    def clear_l1(self):
        """Clear L1 memory cache."""
        self.l1_memory.clear()
        self.l1_hits = 0
        self.l1_misses = 0
    
    async def clear_l2(self):
        """Clear L2 Redis cache."""
        if self.l2_enabled:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.l2_redis.flushdb
                )
                self.l2_hits = 0
                self.l2_misses = 0
            except redis.RedisError as e:
                self.logger.error(f"Redis clear error: {e}")
    
    async def clear_all(self):
        """Clear all cache levels."""
        self.clear_l1()
        await self.clear_l2()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        l1_total = self.l1_hits + self.l1_misses
        l2_total = self.l2_hits + self.l2_misses
        
        stats = {
            'l1': {
                'hit_rate': (self.l1_hits / l1_total * 100) if l1_total > 0 else 0,
                'hits': self.l1_hits,
                'misses': self.l1_misses,
                'size': len(self.l1_memory),
                'max_size': self.l1_memory.maxsize
            },
            'l2': {
                'enabled': self.l2_enabled,
                'hit_rate': (self.l2_hits / l2_total * 100) if l2_total > 0 else 0,
                'hits': self.l2_hits,
                'misses': self.l2_misses
            },
            'total': {
                'requests': self.stats['total_requests'],
                'sets': self.stats['cache_sets'],
                'deletes': self.stats['cache_deletes']
            }
        }
        
        # Add Redis info if available
        if self.l2_enabled:
            try:
                info = self.l2_redis.info('memory')
                stats['l2']['memory_used'] = info.get('used_memory_human', 'N/A')
                stats['l2']['peak_memory'] = info.get('used_memory_peak_human', 'N/A')
            except redis.RedisError:
                pass
        
        return stats
    
    def close(self):
        """Clean up resources."""
        if self.l2_redis:
            self.l2_redis.close()
        self.executor.shutdown(wait=False)


class CacheWarmer:
    """Proactive cache warming for critical data."""
    
    def __init__(self, cache: MultiTierCache, data_loader):
        """Initialize cache warmer."""
        self.cache = cache
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)
        self.warm_tasks = []
    
    async def warm_on_startup(self):
        """Warm critical data on application startup."""
        self.logger.info("Starting cache warming...")
        
        critical_data = [
            ('yarn_inventory', self._warm_yarn_inventory),
            ('bom_data', self._warm_bom_data),
            ('work_centers', self._warm_work_centers)
        ]
        
        tasks = []
        for data_type, warm_func in critical_data:
            strategy = self.cache.strategies.get(data_type)
            if strategy and strategy.warm_on_start:
                tasks.append(warm_func())
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to warm {critical_data[i][0]}: {result}")
                else:
                    self.logger.info(f"Successfully warmed {critical_data[i][0]}")
        
        self.logger.info("Cache warming complete")
    
    async def _warm_yarn_inventory(self):
        """Warm yarn inventory data."""
        try:
            data = self.data_loader.load_yarn_inventory()
            await self.cache.set('yarn_inventory_full', data, 'yarn_inventory')
            return True
        except Exception as e:
            self.logger.error(f"Failed to warm yarn inventory: {e}")
            return False
    
    async def _warm_bom_data(self):
        """Warm BOM data."""
        try:
            data = self.data_loader.load_bom_data()
            await self.cache.set('bom_data_full', data, 'bom_data')
            return True
        except Exception as e:
            self.logger.error(f"Failed to warm BOM data: {e}")
            return False
    
    async def _warm_work_centers(self):
        """Warm work center data."""
        try:
            data = self.data_loader.load_work_centers()
            await self.cache.set('work_centers_full', data, 'production_orders')
            return True
        except Exception as e:
            self.logger.error(f"Failed to warm work centers: {e}")
            return False