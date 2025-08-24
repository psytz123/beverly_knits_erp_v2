#!/usr/bin/env python3
"""
Beverly Knits ERP - Memory Optimization Module
Implements memory leak fixes, garbage collection, and memory monitoring
Part of Phase 3: Performance Optimization
"""

import gc
import logging
import psutil
import os
import sys
import tracemalloc
import weakref
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from functools import wraps
import pandas as pd
import numpy as np
from threading import Lock, Timer
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pandas performance warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class MemoryOptimizer:
    """
    Central memory optimization manager for Beverly Knits ERP
    Handles garbage collection, memory limits, and leak prevention
    """
    
    # Class-level settings
    MAX_DATAFRAME_SIZE = 100000  # Maximum rows per DataFrame
    MAX_CACHE_SIZE_MB = 200      # Maximum cache size in MB
    GC_THRESHOLD_MB = 500        # Trigger GC when memory exceeds this
    MEMORY_WARNING_PERCENT = 80  # Warn when memory usage exceeds this %
    
    def __init__(self):
        """Initialize memory optimizer"""
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        self.cache_refs = weakref.WeakValueDictionary()  # Weak references to cached objects
        self.cleanup_timer = None
        self.gc_lock = Lock()
        self.memory_snapshots = []
        
        # Configure garbage collection
        self._configure_gc()
        
        # Start memory monitoring
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()
        
        logger.info(f"MemoryOptimizer initialized. Initial memory: {self.initial_memory:.2f} MB")
    
    def _configure_gc(self):
        """Configure garbage collection for optimal performance"""
        # Set GC thresholds (generation 0, 1, 2)
        gc.set_threshold(700, 10, 10)
        
        # Enable automatic collection
        gc.enable()
        
        logger.info("Garbage collection configured with thresholds: (700, 10, 10)")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        memory_info = self.process.memory_info()
        
        stats = {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'gc_stats': gc.get_stats(),
            'gc_count': gc.get_count(),
            'tracked_objects': len(gc.get_objects()),
            'cache_size': len(self.cache_refs)
        }
        
        # Add tracemalloc snapshot if available
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:5]
            stats['top_memory_consumers'] = [
                {
                    'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'size_mb': stat.size / 1024 / 1024
                }
                for stat in top_stats
            ]
        
        return stats
    
    def optimize_dataframe(self, df: pd.DataFrame, name: str = "dataframe") -> pd.DataFrame:
        """
        Optimize DataFrame memory usage
        
        Args:
            df: DataFrame to optimize
            name: Name for logging
            
        Returns:
            Optimized DataFrame
        """
        if df is None or df.empty:
            return df
        
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Limit DataFrame size
        if len(df) > self.MAX_DATAFRAME_SIZE:
            logger.warning(f"DataFrame '{name}' exceeds size limit ({len(df)} > {self.MAX_DATAFRAME_SIZE}). Truncating...")
            df = df.head(self.MAX_DATAFRAME_SIZE)
        
        # Optimize data types
        for col in df.columns:
            col_type = df[col].dtype
            
            # Optimize numeric columns
            if col_type != 'object':
                try:
                    # Downcast integers
                    if 'int' in str(col_type):
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    # Downcast floats
                    elif 'float' in str(col_type):
                        df[col] = pd.to_numeric(df[col], downcast='float')
                except:
                    pass
            
            # Convert object columns to category if beneficial
            else:
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
                    try:
                        df[col] = df[col].astype('category')
                    except:
                        pass
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} duplicate rows from '{name}'")
        
        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (1 - final_memory / initial_memory) * 100 if initial_memory > 0 else 0
        
        if reduction > 10:  # Log significant reductions
            logger.info(f"Optimized DataFrame '{name}': {initial_memory:.2f} MB -> {final_memory:.2f} MB ({reduction:.1f}% reduction)")
        
        return df
    
    def limit_cache_size(self, cache_dict: Dict) -> Dict:
        """
        Limit cache dictionary size to prevent memory bloat
        
        Args:
            cache_dict: Cache dictionary to limit
            
        Returns:
            Limited cache dictionary
        """
        if not cache_dict:
            return cache_dict
        
        # Calculate cache size
        cache_size_mb = sys.getsizeof(cache_dict) / 1024 / 1024
        
        if cache_size_mb > self.MAX_CACHE_SIZE_MB:
            # Remove oldest entries (assuming dict maintains insertion order in Python 3.7+)
            items_to_remove = len(cache_dict) // 4  # Remove 25% of entries
            keys_to_remove = list(cache_dict.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                del cache_dict[key]
            
            logger.warning(f"Cache size exceeded limit ({cache_size_mb:.2f} MB > {self.MAX_CACHE_SIZE_MB} MB). "
                         f"Removed {items_to_remove} entries.")
        
        return cache_dict
    
    def collect_garbage(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform garbage collection
        
        Args:
            force: Force collection regardless of memory usage
            
        Returns:
            Collection statistics
        """
        with self.gc_lock:
            before_memory = self.get_memory_usage()
            before_objects = len(gc.get_objects())
            
            # Check if GC is needed
            if not force and before_memory < self.GC_THRESHOLD_MB:
                return {
                    'collected': False,
                    'reason': f'Memory below threshold ({before_memory:.2f} MB < {self.GC_THRESHOLD_MB} MB)'
                }
            
            # Perform garbage collection
            collected = gc.collect()
            
            # Force collection of higher generations
            if force or before_memory > self.GC_THRESHOLD_MB * 1.5:
                collected += gc.collect(1)  # Collect generation 1
                collected += gc.collect(2)  # Collect generation 2
            
            after_memory = self.get_memory_usage()
            after_objects = len(gc.get_objects())
            
            stats = {
                'collected': True,
                'objects_collected': collected,
                'memory_before_mb': before_memory,
                'memory_after_mb': after_memory,
                'memory_freed_mb': before_memory - after_memory,
                'objects_before': before_objects,
                'objects_after': after_objects,
                'timestamp': datetime.now().isoformat()
            }
            
            if stats['memory_freed_mb'] > 10:  # Log significant collections
                logger.info(f"Garbage collection freed {stats['memory_freed_mb']:.2f} MB "
                          f"({collected} objects collected)")
            
            return stats
    
    def monitor_memory_usage(self) -> Optional[Dict[str, Any]]:
        """
        Monitor current memory usage and trigger warnings/actions
        
        Returns:
            Warning information if threshold exceeded
        """
        current_memory = self.get_memory_usage()
        memory_percent = self.process.memory_percent()
        
        # Check for memory warning
        if memory_percent > self.MEMORY_WARNING_PERCENT:
            warning = {
                'level': 'CRITICAL' if memory_percent > 90 else 'WARNING',
                'memory_mb': current_memory,
                'memory_percent': memory_percent,
                'message': f'Memory usage high: {memory_percent:.1f}%',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.warning(f"High memory usage detected: {current_memory:.2f} MB ({memory_percent:.1f}%)")
            
            # Trigger automatic garbage collection
            self.collect_garbage(force=True)
            
            return warning
        
        # Store snapshot for trend analysis
        self.memory_snapshots.append({
            'timestamp': datetime.now(),
            'memory_mb': current_memory,
            'percent': memory_percent
        })
        
        # Keep only last hour of snapshots
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.memory_snapshots = [s for s in self.memory_snapshots if s['timestamp'] > cutoff_time]
        
        return None
    
    def cleanup_old_data(self, data_dict: Dict, max_age_minutes: int = 30) -> Dict:
        """
        Remove old data entries based on age
        
        Args:
            data_dict: Dictionary with timestamp keys or values
            max_age_minutes: Maximum age in minutes
            
        Returns:
            Cleaned dictionary
        """
        if not data_dict:
            return data_dict
        
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        initial_count = len(data_dict)
        
        # Try to clean based on timestamp values
        cleaned_dict = {}
        for key, value in data_dict.items():
            try:
                # Check if value has timestamp attribute
                if hasattr(value, 'timestamp'):
                    if value.timestamp > cutoff_time:
                        cleaned_dict[key] = value
                # Check if key is timestamp-like
                elif isinstance(key, (datetime, str)):
                    key_time = pd.to_datetime(key) if isinstance(key, str) else key
                    if key_time > cutoff_time:
                        cleaned_dict[key] = value
                else:
                    # Keep entries without timestamps
                    cleaned_dict[key] = value
            except:
                # Keep entries that can't be processed
                cleaned_dict[key] = value
        
        removed_count = initial_count - len(cleaned_dict)
        if removed_count > 0:
            logger.info(f"Cleaned {removed_count} old entries from data dictionary")
        
        return cleaned_dict
    
    def setup_automatic_cleanup(self, interval_minutes: int = 15):
        """
        Set up automatic memory cleanup timer
        
        Args:
            interval_minutes: Cleanup interval in minutes
        """
        def cleanup_task():
            logger.debug("Running automatic memory cleanup...")
            self.collect_garbage()
            self.monitor_memory_usage()
            
            # Reschedule
            self.cleanup_timer = Timer(interval_minutes * 60, cleanup_task)
            self.cleanup_timer.daemon = True
            self.cleanup_timer.start()
        
        # Cancel existing timer if any
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
        
        # Start new timer
        cleanup_task()
        logger.info(f"Automatic memory cleanup scheduled every {interval_minutes} minutes")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive memory report
        
        Returns:
            Memory usage report
        """
        stats = self.get_memory_stats()
        
        # Add trends if available
        if self.memory_snapshots:
            memory_values = [s['memory_mb'] for s in self.memory_snapshots]
            stats['trends'] = {
                'samples': len(memory_values),
                'min_mb': min(memory_values),
                'max_mb': max(memory_values),
                'avg_mb': sum(memory_values) / len(memory_values),
                'current_mb': memory_values[-1] if memory_values else 0
            }
        
        # Add recommendations
        recommendations = []
        if stats['percent'] > 80:
            recommendations.append("Consider increasing memory allocation")
        if stats['gc_count'][2] > 10:
            recommendations.append("High Gen-2 GC count indicates memory pressure")
        if stats.get('cache_size', 0) > 1000:
            recommendations.append("Cache size large, consider cache cleanup")
        
        stats['recommendations'] = recommendations
        stats['initial_memory_mb'] = self.initial_memory
        stats['memory_growth_mb'] = stats['rss_mb'] - self.initial_memory
        
        return stats
    
    def shutdown(self):
        """Clean shutdown of memory optimizer"""
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
        
        # Final garbage collection
        self.collect_garbage(force=True)
        
        # Stop tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        logger.info("MemoryOptimizer shutdown complete")


def memory_efficient(func):
    """
    Decorator to make functions memory efficient
    Performs garbage collection after execution
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Run function
        result = func(*args, **kwargs)
        
        # Collect garbage after execution
        gc.collect()
        
        return result
    
    return wrapper


def limit_dataframe_size(max_rows: int = 100000):
    """
    Decorator to limit DataFrame sizes in function returns
    
    Args:
        max_rows: Maximum number of rows allowed
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Check if result is a DataFrame
            if isinstance(result, pd.DataFrame) and len(result) > max_rows:
                logger.warning(f"Function {func.__name__} returned DataFrame with {len(result)} rows. "
                             f"Truncating to {max_rows} rows.")
                result = result.head(max_rows)
            
            # Check if result is a dict containing DataFrames
            elif isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, pd.DataFrame) and len(value) > max_rows:
                        logger.warning(f"Function {func.__name__} returned DataFrame '{key}' with {len(value)} rows. "
                                     f"Truncating to {max_rows} rows.")
                        result[key] = value.head(max_rows)
            
            return result
        
        return wrapper
    return decorator


# Global instance
_optimizer_instance = None

def get_memory_optimizer() -> MemoryOptimizer:
    """Get singleton instance of MemoryOptimizer"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = MemoryOptimizer()
    return _optimizer_instance


def test_memory_optimizer():
    """Test the memory optimizer"""
    print("=" * 80)
    print("Testing MemoryOptimizer")
    print("=" * 80)
    
    optimizer = MemoryOptimizer()
    
    # Test 1: Memory stats
    print("\n1. Current Memory Stats:")
    stats = optimizer.get_memory_stats()
    print(f"  RSS Memory: {stats['rss_mb']:.2f} MB")
    print(f"  Memory Percent: {stats['percent']:.2f}%")
    print(f"  Tracked Objects: {stats['tracked_objects']}")
    
    # Test 2: DataFrame optimization
    print("\n2. Testing DataFrame Optimization:")
    # Create a large DataFrame
    df = pd.DataFrame({
        'id': range(10000),
        'value': np.random.rand(10000),
        'category': ['A', 'B', 'C', 'D'] * 2500,
        'description': ['Test description'] * 10000
    })
    
    print(f"  Original size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    df_optimized = optimizer.optimize_dataframe(df, "test_df")
    print(f"  Optimized size: {df_optimized.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Test 3: Garbage collection
    print("\n3. Testing Garbage Collection:")
    gc_stats = optimizer.collect_garbage(force=True)
    print(f"  Objects collected: {gc_stats.get('objects_collected', 0)}")
    print(f"  Memory freed: {gc_stats.get('memory_freed_mb', 0):.2f} MB")
    
    # Test 4: Memory monitoring
    print("\n4. Testing Memory Monitoring:")
    warning = optimizer.monitor_memory_usage()
    if warning:
        print(f"  Warning: {warning['message']}")
    else:
        print("  Memory usage within normal limits")
    
    # Test 5: Memory report
    print("\n5. Memory Report:")
    report = optimizer.get_memory_report()
    print(f"  Initial Memory: {report['initial_memory_mb']:.2f} MB")
    print(f"  Current Memory: {report['rss_mb']:.2f} MB")
    print(f"  Memory Growth: {report['memory_growth_mb']:.2f} MB")
    if report['recommendations']:
        print("  Recommendations:")
        for rec in report['recommendations']:
            print(f"    - {rec}")
    
    # Cleanup
    optimizer.shutdown()
    
    print("\n" + "=" * 80)
    print("✅ MemoryOptimizer test complete")


if __name__ == "__main__":
    test_memory_optimizer()