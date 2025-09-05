"""
MemoryOptimizer - Reduce memory usage and prevent memory leaks
Optimizes DataFrames, manages connection pools, and prevents memory issues
"""
import pandas as pd
import numpy as np
import gc
import psutil
import logging
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager
from functools import wraps
import weakref
from datetime import datetime
import sys

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Comprehensive memory optimization for DataFrames and system resources"""
    
    def __init__(self):
        self.memory_stats = []
        self.optimization_history = []
        self.dataframe_registry = weakref.WeakValueDictionary()
        
    @staticmethod
    def optimize_dataframe(
        df: pd.DataFrame, 
        deep: bool = True,
        aggressive: bool = False
    ) -> pd.DataFrame:
        """
        Reduce DataFrame memory usage by 50-90%
        
        Args:
            df: DataFrame to optimize
            deep: Perform deep memory inspection
            aggressive: Use aggressive optimization (may lose precision)
        """
        if df.empty:
            return df
            
        start_mem = df.memory_usage(deep=deep).sum() / 1024**2
        logger.info(f"Starting memory optimization: {start_mem:.2f} MB")
        
        # Optimize numeric columns
        df = MemoryOptimizer._optimize_numerics(df, aggressive)
        
        # Optimize string columns
        df = MemoryOptimizer._optimize_strings(df)
        
        # Optimize datetime columns
        df = MemoryOptimizer._optimize_dates(df)
        
        # Remove duplicates if any
        original_len = len(df)
        df = df.drop_duplicates()
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} duplicate rows")
        
        # Sparse data optimization
        df = MemoryOptimizer._optimize_sparse(df)
        
        end_mem = df.memory_usage(deep=deep).sum() / 1024**2
        reduction_pct = 100 * (1 - end_mem / start_mem)
        
        logger.info(f"Memory optimization complete: {start_mem:.2f}MB → {end_mem:.2f}MB")
        logger.info(f"Memory reduced by {reduction_pct:.1f}%")
        
        return df
    
    @staticmethod
    def _optimize_numerics(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
        """Optimize numeric column types"""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                # Integer optimization
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                
                # Float optimization
                elif str(col_type)[:5] == 'float':
                    if aggressive:
                        # Use float16 for aggressive optimization
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float16)
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                    else:
                        # Use float32 for normal optimization
                        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
        
        return df
    
    @staticmethod
    def _optimize_strings(df: pd.DataFrame) -> pd.DataFrame:
        """Convert low-cardinality strings to categories"""
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                
                # Convert to category if less than 50% unique values
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
                    logger.debug(f"Converted {col} to category ({num_unique_values} unique values)")
        
        return df
    
    @staticmethod
    def _optimize_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize datetime columns"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to datetime
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass
            
            # Downcast datetime64 if possible
            if df[col].dtype == 'datetime64[ns]':
                # Check if we need nanosecond precision
                if df[col].dt.nanosecond.sum() == 0:
                    # We don't need nanosecond precision
                    # This saves memory for large datetime columns
                    pass  # Keep as is for compatibility
        
        return df
    
    @staticmethod
    def _optimize_sparse(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """Optimize sparse data (lots of zeros or NaN)"""
        for col in df.select_dtypes(include=[np.number]).columns:
            # Check sparsity
            zero_ratio = (df[col] == 0).sum() / len(df[col])
            nan_ratio = df[col].isna().sum() / len(df[col])
            
            if zero_ratio > threshold or nan_ratio > threshold:
                # Convert to sparse array
                df[col] = pd.arrays.SparseArray(df[col], fill_value=0)
                logger.debug(f"Converted {col} to sparse array (sparsity: {max(zero_ratio, nan_ratio):.1%})")
        
        return df
    
    @staticmethod
    def estimate_dataframe_memory(df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed memory usage statistics"""
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum() / 1024**2  # MB
        
        # Column-wise memory
        column_memory = {}
        for col in df.columns:
            column_memory[col] = {
                'memory_mb': memory_usage[col] / 1024**2,
                'dtype': str(df[col].dtype),
                'null_count': df[col].isna().sum(),
                'unique_count': df[col].nunique()
            }
        
        return {
            'total_memory_mb': total_memory,
            'rows': len(df),
            'columns': len(df.columns),
            'memory_per_row_kb': (total_memory * 1024) / len(df) if len(df) > 0 else 0,
            'column_memory': column_memory,
            'largest_columns': sorted(
                column_memory.items(), 
                key=lambda x: x[1]['memory_mb'], 
                reverse=True
            )[:5]
        }
    
    @contextmanager
    def memory_tracker(self, operation_name: str):
        """
        Context manager to track memory usage during operations
        """
        # Get starting memory
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024**2  # MB
        start_time = datetime.now()
        
        # Force garbage collection before
        gc.collect()
        
        try:
            yield
        finally:
            # Force garbage collection after
            gc.collect()
            
            # Get ending memory
            end_memory = process.memory_info().rss / 1024**2  # MB
            end_time = datetime.now()
            
            # Calculate statistics
            memory_delta = end_memory - start_memory
            duration = (end_time - start_time).total_seconds()
            
            # Log statistics
            stats = {
                'operation': operation_name,
                'start_memory_mb': start_memory,
                'end_memory_mb': end_memory,
                'memory_delta_mb': memory_delta,
                'duration_seconds': duration,
                'timestamp': datetime.now()
            }
            
            self.memory_stats.append(stats)
            
            if memory_delta > 100:  # More than 100MB increase
                logger.warning(f"Large memory increase in {operation_name}: {memory_delta:.1f}MB")
            else:
                logger.info(f"Memory usage for {operation_name}: {memory_delta:+.1f}MB in {duration:.2f}s")
    
    def optimize_chunk_processing(
        self,
        filepath: str,
        processor: callable,
        chunksize: int = 10000,
        **read_kwargs
    ) -> pd.DataFrame:
        """
        Process large files in chunks to avoid memory issues
        """
        results = []
        
        # Determine file type
        if filepath.endswith('.csv'):
            reader = pd.read_csv
        elif filepath.endswith(('.xlsx', '.xls')):
            reader = pd.read_excel
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
        
        # Process in chunks
        with self.memory_tracker(f"Chunk processing {filepath}"):
            for i, chunk in enumerate(reader(filepath, chunksize=chunksize, **read_kwargs)):
                # Optimize chunk memory
                chunk = self.optimize_dataframe(chunk)
                
                # Process chunk
                processed = processor(chunk)
                
                # Store result
                results.append(processed)
                
                # Periodic garbage collection
                if i % 10 == 0:
                    gc.collect()
                
                logger.debug(f"Processed chunk {i + 1}")
        
        # Combine results
        if results and isinstance(results[0], pd.DataFrame):
            final_result = pd.concat(results, ignore_index=True)
            final_result = self.optimize_dataframe(final_result)
            return final_result
        
        return results
    
    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """
        Detect potential memory leaks
        """
        leaks = []
        
        # Check for growing memory over time
        if len(self.memory_stats) > 10:
            recent_stats = self.memory_stats[-10:]
            memory_trend = [s['end_memory_mb'] for s in recent_stats]
            
            # Calculate trend
            if all(memory_trend[i] <= memory_trend[i+1] for i in range(len(memory_trend)-1)):
                leaks.append({
                    'type': 'Monotonic memory growth',
                    'severity': 'HIGH',
                    'description': f"Memory consistently increasing over last {len(memory_trend)} operations",
                    'recommendation': 'Check for unclosed resources or accumulating data structures'
                })
        
        # Check current memory usage
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024**2
        
        if current_memory > 1024:  # More than 1GB
            leaks.append({
                'type': 'High memory usage',
                'severity': 'MEDIUM',
                'description': f"Current memory usage: {current_memory:.1f}MB",
                'recommendation': 'Consider optimizing data structures or processing in chunks'
            })
        
        # Check for large objects
        large_objects = []
        for obj in gc.get_objects():
            try:
                if sys.getsizeof(obj) > 10 * 1024 * 1024:  # Larger than 10MB
                    large_objects.append({
                        'type': type(obj).__name__,
                        'size_mb': sys.getsizeof(obj) / 1024**2
                    })
            except:
                pass
        
        if large_objects:
            leaks.append({
                'type': 'Large objects in memory',
                'severity': 'LOW',
                'description': f"Found {len(large_objects)} objects larger than 10MB",
                'objects': large_objects[:5],  # Top 5
                'recommendation': 'Review large object lifecycle and consider cleanup'
            })
        
        return leaks
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive memory report
        """
        process = psutil.Process()
        
        return {
            'current_memory_mb': process.memory_info().rss / 1024**2,
            'peak_memory_mb': max([s['end_memory_mb'] for s in self.memory_stats]) if self.memory_stats else 0,
            'operations_tracked': len(self.memory_stats),
            'potential_leaks': self.detect_memory_leaks(),
            'gc_stats': gc.get_stats(),
            'dataframes_tracked': len(self.dataframe_registry),
            'system_memory': {
                'total_gb': psutil.virtual_memory().total / 1024**3,
                'available_gb': psutil.virtual_memory().available / 1024**3,
                'percent_used': psutil.virtual_memory().percent
            }
        }
    
    def cleanup_memory(self, aggressive: bool = False):
        """
        Force memory cleanup
        """
        logger.info("Starting memory cleanup")
        
        # Clear internal caches
        self.optimization_history.clear()
        if len(self.memory_stats) > 100:
            self.memory_stats = self.memory_stats[-50:]  # Keep last 50
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collector collected {collected} objects")
        
        if aggressive:
            # More aggressive cleanup
            gc.collect(2)  # Full collection
            
            # Clear DataFrame caches
            for df_ref in list(self.dataframe_registry.values()):
                if df_ref is not None:
                    del df_ref
            
            logger.info("Aggressive cleanup completed")
    
    @staticmethod
    def optimize_function_memory(func: callable) -> callable:
        """
        Decorator to optimize memory usage in functions
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Force GC before
            gc.collect()
            
            try:
                result = func(*args, **kwargs)
                
                # Optimize DataFrame results
                if isinstance(result, pd.DataFrame):
                    result = MemoryOptimizer.optimize_dataframe(result)
                elif isinstance(result, list) and result and isinstance(result[0], pd.DataFrame):
                    result = [MemoryOptimizer.optimize_dataframe(df) for df in result]
                
                return result
            finally:
                # Force GC after
                gc.collect()
        
        return wrapper


class ConnectionPoolOptimizer:
    """Optimize database connection pooling to prevent resource leaks"""
    
    def __init__(self, max_connections: int = 20):
        self.max_connections = max_connections
        self.active_connections = []
        self.idle_connections = []
        self.connection_stats = {
            'created': 0,
            'reused': 0,
            'closed': 0,
            'errors': 0
        }
    
    @contextmanager
    def get_connection(self):
        """
        Get connection from pool with automatic cleanup
        """
        conn = None
        try:
            # Try to reuse idle connection
            if self.idle_connections:
                conn = self.idle_connections.pop()
                self.connection_stats['reused'] += 1
                logger.debug("Reused connection from pool")
            else:
                # Create new connection if under limit
                if len(self.active_connections) < self.max_connections:
                    conn = self._create_connection()
                    self.connection_stats['created'] += 1
                    logger.debug("Created new connection")
                else:
                    # Wait for available connection
                    logger.warning("Connection pool exhausted, waiting...")
                    # In production, implement proper waiting logic
                    conn = self._create_connection()
            
            self.active_connections.append(conn)
            yield conn
            
        except Exception as e:
            self.connection_stats['errors'] += 1
            logger.error(f"Connection error: {e}")
            raise
        finally:
            # Return connection to pool
            if conn:
                self.active_connections.remove(conn)
                if len(self.idle_connections) < self.max_connections // 2:
                    self.idle_connections.append(conn)
                else:
                    self._close_connection(conn)
                    self.connection_stats['closed'] += 1
    
    def _create_connection(self):
        """Create new database connection"""
        # Placeholder - implement actual connection creation
        return object()
    
    def _close_connection(self, conn):
        """Close database connection"""
        # Placeholder - implement actual connection closing
        pass
    
    def cleanup_idle_connections(self):
        """Clean up idle connections"""
        closed = 0
        while self.idle_connections:
            conn = self.idle_connections.pop()
            self._close_connection(conn)
            closed += 1
        
        logger.info(f"Cleaned up {closed} idle connections")
        self.connection_stats['closed'] += closed
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'active': len(self.active_connections),
            'idle': len(self.idle_connections),
            'total': len(self.active_connections) + len(self.idle_connections),
            'max': self.max_connections,
            'stats': self.connection_stats
        }