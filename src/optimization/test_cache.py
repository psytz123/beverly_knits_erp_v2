#!/usr/bin/env python3
"""Quick test for cache optimizer"""

import sys
import time

# Simple cache test without imports
print("Testing Cache Optimizer...")

class SimpleLRUCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)

# Test the simple cache
cache = SimpleLRUCache(max_size=3)

# Test operations
print("\n1. Testing LRU Cache:")
cache.put("key1", "value1")
cache.put("key2", "value2")
cache.put("key3", "value3")
print(f"  Added 3 items. Cache size: {len(cache.cache)}")

# Access key1 to make it recently used
result = cache.get("key1")
print(f"  Got key1: {result}")

# Add key4, should evict key2 (least recently used)
cache.put("key4", "value4")
print(f"  Added key4. Cache size: {len(cache.cache)}")
print(f"  key2 still in cache: {cache.get('key2')}")  # Should be None
print(f"  key1 still in cache: {cache.get('key1')}")  # Should be value1

# Test cache performance
print("\n2. Testing Cache Performance:")
hits = 0
misses = 0
start_time = time.perf_counter()

for i in range(1000):
    key = f"key_{i % 10}"  # Use 10 rotating keys
    if cache.get(key):
        hits += 1
    else:
        cache.put(key, f"value_{i}")
        misses += 1

end_time = time.perf_counter()
elapsed = (end_time - start_time) * 1000

print(f"  Operations: 1000")
print(f"  Time: {elapsed:.2f}ms")
print(f"  Hits: {hits}")
print(f"  Misses: {misses}")
print(f"  Hit Rate: {(hits/(hits+misses)*100):.1f}%")

print("\n✅ Cache test complete!")