"""
Batch Processor
High-performance batch processing for large datasets
"""

import pandas as pd
import numpy as np
from typing import Callable, Any, List, Dict, Optional, Iterator, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import logging
import time
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class BatchMetrics:
    """Metrics for batch processing"""
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    start_time: float = 0
    end_time: float = 0
    batch_times: List[float] = None
    
    def __post_init__(self):
        if self.batch_times is None:
            self.batch_times = []
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_items == 0:
            return 0
        return (self.processed_items - self.failed_items) / self.total_items
    
    @property
    def average_batch_time(self) -> float:
        """Calculate average batch processing time"""
        if not self.batch_times:
            return 0
        return sum(self.batch_times) / len(self.batch_times)
    
    @property
    def total_time(self) -> float:
        """Calculate total processing time"""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0
    
    @property
    def items_per_second(self) -> float:
        """Calculate processing throughput"""
        if self.total_time > 0:
            return self.processed_items / self.total_time
        return 0


class BatchProcessor:
    """Process large datasets in optimized batches"""
    
    def __init__(self, 
                 batch_size: int = 10000,
                 max_workers: Optional[int] = None,
                 use_multiprocessing: bool = False,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize batch processor
        
        Args:
            batch_size: Size of each batch
            max_workers: Maximum parallel workers (None for auto)
            use_multiprocessing: Use multiprocessing instead of threading
            progress_callback: Callback for progress updates
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or mp.cpu_count()
        self.use_multiprocessing = use_multiprocessing
        self.progress_callback = progress_callback
        self.metrics = BatchMetrics()
    
    def process_dataframe(self,
                         df: pd.DataFrame,
                         processor: Callable,
                         parallel: bool = True,
                         return_type: str = 'dataframe') -> Union[pd.DataFrame, List, Dict]:
        """
        Process DataFrame in batches
        
        Args:
            df: DataFrame to process
            processor: Processing function for each batch
            parallel: Whether to use parallel processing
            return_type: Type of return value ('dataframe', 'list', 'dict')
            
        Returns:
            Processed data in specified format
        """
        if df.empty:
            if return_type == 'dataframe':
                return pd.DataFrame()
            elif return_type == 'list':
                return []
            else:
                return {}
        
        # Initialize metrics
        self.metrics = BatchMetrics(
            total_items=len(df),
            start_time=time.time()
        )
        
        # Single batch optimization
        if len(df) <= self.batch_size:
            result = processor(df)
            self.metrics.processed_items = len(df)
            self.metrics.end_time = time.time()
            return self._format_result(result, return_type)
        
        # Process in batches
        batches = self._create_batches(df)
        
        if parallel:
            results = self._process_parallel(batches, processor)
        else:
            results = self._process_sequential(batches, processor)
        
        self.metrics.end_time = time.time()
        
        # Log performance metrics
        self._log_metrics()
        
        return self._combine_results(results, return_type)
    
    def process_iterable(self,
                        items: Iterator,
                        processor: Callable,
                        parallel: bool = True) -> List[Any]:
        """
        Process iterable in batches
        
        Args:
            items: Iterable to process
            processor: Processing function
            parallel: Whether to use parallel processing
            
        Returns:
            List of processed items
        """
        # Convert to list for batching
        items_list = list(items)
        
        if not items_list:
            return []
        
        self.metrics = BatchMetrics(
            total_items=len(items_list),
            start_time=time.time()
        )
        
        # Create batches
        batches = [
            items_list[i:i + self.batch_size]
            for i in range(0, len(items_list), self.batch_size)
        ]
        
        if parallel:
            results = self._process_parallel_list(batches, processor)
        else:
            results = self._process_sequential_list(batches, processor)
        
        self.metrics.end_time = time.time()
        self._log_metrics()
        
        # Flatten results
        return [item for batch in results for item in batch]
    
    def stream_process(self,
                      data_source: Callable,
                      processor: Callable,
                      batch_timeout: float = 1.0) -> Iterator[Any]:
        """
        Stream process data in batches
        
        Args:
            data_source: Callable that yields data
            processor: Processing function
            batch_timeout: Time to wait for batch to fill
            
        Yields:
            Processed results
        """
        batch = []
        batch_start_time = time.time()
        
        for item in data_source():
            batch.append(item)
            
            # Process batch when full or timeout reached
            if len(batch) >= self.batch_size or \
               (time.time() - batch_start_time) >= batch_timeout:
                
                # Process batch
                results = processor(batch)
                
                # Yield results
                for result in results:
                    yield result
                
                # Reset batch
                batch = []
                batch_start_time = time.time()
        
        # Process remaining items
        if batch:
            results = processor(batch)
            for result in results:
                yield result
    
    def _create_batches(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Create batches from DataFrame"""
        batches = []
        
        for start_idx in range(0, len(df), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(df))
            batch = df.iloc[start_idx:end_idx]
            batches.append(batch)
        
        return batches
    
    def _process_parallel(self,
                         batches: List[pd.DataFrame],
                         processor: Callable) -> List[Any]:
        """Process batches in parallel"""
        results = []
        
        # Choose executor based on configuration
        ExecutorClass = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        
        with ExecutorClass(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(processor, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results in order
            batch_results = [None] * len(batches)
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                batch_start = time.time()
                
                try:
                    result = future.result()
                    batch_results[batch_idx] = result
                    
                    # Update metrics
                    if isinstance(result, pd.DataFrame):
                        self.metrics.processed_items += len(result)
                    elif isinstance(result, list):
                        self.metrics.processed_items += len(result)
                    else:
                        self.metrics.processed_items += 1
                    
                    batch_time = time.time() - batch_start
                    self.metrics.batch_times.append(batch_time)
                    
                    # Progress callback
                    if self.progress_callback:
                        self.progress_callback(self.metrics)
                    
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    self.metrics.failed_items += self.batch_size
                    batch_results[batch_idx] = None
            
            # Filter out None results
            results = [r for r in batch_results if r is not None]
        
        return results
    
    def _process_sequential(self,
                          batches: List[pd.DataFrame],
                          processor: Callable) -> List[Any]:
        """Process batches sequentially"""
        results = []
        
        for i, batch in enumerate(batches):
            batch_start = time.time()
            
            try:
                result = processor(batch)
                results.append(result)
                
                # Update metrics
                if isinstance(result, pd.DataFrame):
                    self.metrics.processed_items += len(result)
                else:
                    self.metrics.processed_items += len(batch)
                
                batch_time = time.time() - batch_start
                self.metrics.batch_times.append(batch_time)
                
                # Progress callback
                if self.progress_callback:
                    self.progress_callback(self.metrics)
                    
            except Exception as e:
                logger.error(f"Batch {i} failed: {e}")
                self.metrics.failed_items += len(batch)
        
        return results
    
    def _process_parallel_list(self,
                              batches: List[List],
                              processor: Callable) -> List[List]:
        """Process list batches in parallel"""
        ExecutorClass = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        
        with ExecutorClass(max_workers=self.max_workers) as executor:
            futures = [executor.submit(processor, batch) for batch in batches]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    self.metrics.processed_items += len(result)
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    self.metrics.failed_items += self.batch_size
        
        return results
    
    def _process_sequential_list(self,
                                batches: List[List],
                                processor: Callable) -> List[List]:
        """Process list batches sequentially"""
        results = []
        
        for batch in batches:
            try:
                result = processor(batch)
                results.append(result)
                self.metrics.processed_items += len(result)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                self.metrics.failed_items += len(batch)
        
        return results
    
    def _combine_results(self,
                        results: List[Any],
                        return_type: str) -> Union[pd.DataFrame, List, Dict]:
        """Combine batch results based on return type"""
        if not results:
            if return_type == 'dataframe':
                return pd.DataFrame()
            elif return_type == 'list':
                return []
            else:
                return {}
        
        if return_type == 'dataframe':
            # Combine DataFrames
            if all(isinstance(r, pd.DataFrame) for r in results):
                return pd.concat(results, ignore_index=True)
            else:
                # Convert to DataFrames first
                dfs = [pd.DataFrame(r) if not isinstance(r, pd.DataFrame) else r 
                      for r in results]
                return pd.concat(dfs, ignore_index=True)
        
        elif return_type == 'list':
            # Flatten lists
            combined = []
            for result in results:
                if isinstance(result, list):
                    combined.extend(result)
                else:
                    combined.append(result)
            return combined
        
        elif return_type == 'dict':
            # Merge dictionaries
            combined = {}
            for result in results:
                if isinstance(result, dict):
                    combined.update(result)
            return combined
        
        else:
            return results
    
    def _format_result(self, result: Any, return_type: str) -> Any:
        """Format single result based on return type"""
        if return_type == 'dataframe':
            if isinstance(result, pd.DataFrame):
                return result
            else:
                return pd.DataFrame(result)
        elif return_type == 'list':
            if isinstance(result, list):
                return result
            else:
                return [result]
        elif return_type == 'dict':
            if isinstance(result, dict):
                return result
            else:
                return {'result': result}
        else:
            return result
    
    def _log_metrics(self):
        """Log processing metrics"""
        logger.info(f"Batch processing completed:")
        logger.info(f"  Total items: {self.metrics.total_items}")
        logger.info(f"  Processed: {self.metrics.processed_items}")
        logger.info(f"  Failed: {self.metrics.failed_items}")
        logger.info(f"  Success rate: {self.metrics.success_rate:.2%}")
        logger.info(f"  Total time: {self.metrics.total_time:.2f}s")
        logger.info(f"  Throughput: {self.metrics.items_per_second:.0f} items/s")
        logger.info(f"  Avg batch time: {self.metrics.average_batch_time:.3f}s")


class SmartBatchProcessor(BatchProcessor):
    """Smart batch processor with adaptive batch sizing and optimization"""
    
    def __init__(self, initial_batch_size: int = 10000, **kwargs):
        """Initialize smart batch processor"""
        super().__init__(batch_size=initial_batch_size, **kwargs)
        self.optimal_batch_size = initial_batch_size
        self.performance_history = []
    
    def process_dataframe(self,
                         df: pd.DataFrame,
                         processor: Callable,
                         **kwargs) -> Any:
        """Process with adaptive batch sizing"""
        # Test different batch sizes on small sample
        if len(df) > 100000:
            self.optimal_batch_size = self._find_optimal_batch_size(df, processor)
            self.batch_size = self.optimal_batch_size
            logger.info(f"Optimal batch size: {self.optimal_batch_size}")
        
        return super().process_dataframe(df, processor, **kwargs)
    
    def _find_optimal_batch_size(self,
                                df: pd.DataFrame,
                                processor: Callable) -> int:
        """Find optimal batch size through testing"""
        test_sizes = [1000, 5000, 10000, 20000, 50000]
        test_results = {}
        
        # Test each batch size on sample
        sample_size = min(50000, len(df))
        sample_df = df.sample(n=sample_size)
        
        for size in test_sizes:
            if size > len(sample_df):
                continue
            
            self.batch_size = size
            start_time = time.time()
            
            try:
                # Process sample
                super().process_dataframe(
                    sample_df,
                    processor,
                    parallel=True,
                    return_type='dataframe'
                )
                
                elapsed = time.time() - start_time
                throughput = sample_size / elapsed
                
                test_results[size] = {
                    'time': elapsed,
                    'throughput': throughput
                }
                
            except Exception as e:
                logger.warning(f"Batch size {size} failed: {e}")
        
        # Find best batch size
        if test_results:
            best_size = max(test_results.keys(), 
                          key=lambda k: test_results[k]['throughput'])
            return best_size
        
        return 10000  # Default


def batch_decorator(batch_size: int = 10000, 
                   parallel: bool = True):
    """
    Decorator for automatic batch processing
    
    Args:
        batch_size: Size of each batch
        parallel: Whether to use parallel processing
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            if not isinstance(df, pd.DataFrame):
                return func(df, *args, **kwargs)
            
            if len(df) <= batch_size:
                return func(df, *args, **kwargs)
            
            # Use batch processor
            processor = BatchProcessor(batch_size=batch_size)
            
            # Create processor function
            def process_batch(batch):
                return func(batch, *args, **kwargs)
            
            return processor.process_dataframe(
                df, 
                process_batch,
                parallel=parallel
            )
        
        return wrapper
    
    return decorator