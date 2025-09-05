"""
DataFrame Optimizer
Utilities for optimizing pandas DataFrame operations and eliminating iterrows()
"""

import pandas as pd
import numpy as np
from typing import Callable, Any, List, Dict, Optional, Union
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


class DataFrameOptimizer:
    """Optimize DataFrame operations for 10-100x performance improvements"""
    
    @staticmethod
    def optimize_iterrows(df: pd.DataFrame, 
                         operation: Callable,
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Replace iterrows with vectorized operations
        
        Args:
            df: DataFrame to process
            operation: Function to apply
            columns: Columns to process (None for all)
            
        Returns:
            Optimized DataFrame
        """
        if df.empty:
            return df
        
        result_df = df.copy()
        
        # Method 1: Try vectorization first (fastest)
        try:
            if columns:
                for col in columns:
                    if col in df.columns:
                        result_df[col] = np.vectorize(operation)(df[col].values)
            else:
                # Apply to entire DataFrame
                result_df = df.apply(operation)
            return result_df
        except:
            pass
        
        # Method 2: Use apply (faster than iterrows)
        try:
            if columns:
                for col in columns:
                    if col in df.columns:
                        result_df[col] = df[col].apply(operation)
            else:
                result_df = df.apply(lambda row: operation(row), axis=1)
            return result_df
        except:
            pass
        
        # Method 3: Fallback to itertuples (still faster than iterrows)
        try:
            results = []
            for row in df.itertuples(index=False):
                results.append(operation(row))
            
            if results and isinstance(results[0], dict):
                return pd.DataFrame(results)
            elif results and isinstance(results[0], (list, tuple)):
                return pd.DataFrame(results, columns=df.columns)
            else:
                result_df['result'] = results
                return result_df
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return df
    
    @staticmethod
    def vectorize_string_operations(df: pd.DataFrame, 
                                  column: str,
                                  operation: str,
                                  *args, **kwargs) -> pd.Series:
        """
        Vectorize string operations
        
        Args:
            df: DataFrame
            column: Column to process
            operation: String operation (e.g., 'contains', 'replace', 'lower')
            
        Returns:
            Processed Series
        """
        if column not in df.columns:
            return pd.Series()
        
        # Use pandas string methods (vectorized)
        str_accessor = df[column].astype(str).str
        
        if hasattr(str_accessor, operation):
            method = getattr(str_accessor, operation)
            return method(*args, **kwargs)
        else:
            logger.warning(f"String operation '{operation}' not found")
            return df[column]
    
    @staticmethod
    def batch_apply(df: pd.DataFrame,
                   func: Callable,
                   batch_size: int = 10000,
                   axis: int = 0) -> pd.DataFrame:
        """
        Apply function in batches for large DataFrames
        
        Args:
            df: DataFrame to process
            func: Function to apply
            batch_size: Size of each batch
            axis: 0 for rows, 1 for columns
            
        Returns:
            Processed DataFrame
        """
        if len(df) <= batch_size:
            return df.apply(func, axis=axis)
        
        results = []
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch = df.iloc[start_idx:end_idx]
            batch_result = batch.apply(func, axis=axis)
            results.append(batch_result)
        
        return pd.concat(results)
    
    @staticmethod
    def optimize_groupby(df: pd.DataFrame,
                        group_cols: Union[str, List[str]],
                        agg_dict: Dict[str, Union[str, List[str], Callable]]) -> pd.DataFrame:
        """
        Optimize groupby operations
        
        Args:
            df: DataFrame to group
            group_cols: Columns to group by
            agg_dict: Aggregation dictionary
            
        Returns:
            Grouped DataFrame
        """
        # Use categorical dtype for grouping columns (faster)
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        
        for col in group_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype('category')
        
        # Use named aggregation for clarity
        return df.groupby(group_cols, observed=True).agg(**agg_dict).reset_index()
    
    @staticmethod
    def optimize_merge(left: pd.DataFrame,
                      right: pd.DataFrame,
                      on: Union[str, List[str]],
                      how: str = 'inner') -> pd.DataFrame:
        """
        Optimize merge operations
        
        Args:
            left: Left DataFrame
            right: Right DataFrame
            on: Columns to merge on
            how: Merge type
            
        Returns:
            Merged DataFrame
        """
        # Sort both DataFrames by merge keys for faster merge
        if isinstance(on, str):
            on = [on]
        
        # Use categorical for merge keys if string
        for col in on:
            if col in left.columns and left[col].dtype == 'object':
                left[col] = left[col].astype('category')
            if col in right.columns and right[col].dtype == 'object':
                right[col] = right[col].astype('category')
        
        # Sort for faster merge (especially for large DataFrames)
        if len(left) > 10000 or len(right) > 10000:
            left = left.sort_values(on)
            right = right.sort_values(on)
        
        return pd.merge(left, right, on=on, how=how)
    
    @staticmethod
    def parallel_apply(df: pd.DataFrame,
                      func: Callable,
                      n_workers: int = 4) -> pd.DataFrame:
        """
        Apply function in parallel using multiple workers
        
        Args:
            df: DataFrame to process
            func: Function to apply
            n_workers: Number of parallel workers
            
        Returns:
            Processed DataFrame
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        
        # Don't use more workers than available CPUs
        n_workers = min(n_workers, mp.cpu_count())
        
        # Split DataFrame into chunks
        chunk_size = len(df) // n_workers
        chunks = []
        
        for i in range(n_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < n_workers - 1 else len(df)
            chunks.append(df.iloc[start_idx:end_idx])
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(func, chunk) for chunk in chunks]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel processing error: {e}")
        
        # Combine results
        return pd.concat(results, ignore_index=True)
    
    @staticmethod
    def eliminate_loops(df: pd.DataFrame, 
                       loop_code: str) -> str:
        """
        Convert loop code to vectorized operations
        
        Args:
            df: DataFrame being processed
            loop_code: String containing loop code
            
        Returns:
            Vectorized code string
        """
        optimizations = {
            # Pattern: for index, row in df.iterrows()
            r'for\s+\w+,\s*\w+\s+in\s+(\w+)\.iterrows\(\)':
                r'\1.apply(lambda row:',
            
            # Pattern: for row in df.itertuples()
            r'for\s+\w+\s+in\s+(\w+)\.itertuples\(\)':
                r'\1.apply(lambda row:',
            
            # Pattern: for i in range(len(df))
            r'for\s+\w+\s+in\s+range\(len\((\w+)\)\)':
                r'\1.apply(lambda row:',
            
            # Pattern: df.loc[i, col] = value
            r'(\w+)\.loc\[\w+,\s*["\'](\w+)["\']\]\s*=':
                r"\1['\2'] =",
            
            # Pattern: df.at[i, col] = value
            r'(\w+)\.at\[\w+,\s*["\'](\w+)["\']\]\s*=':
                r"\1['\2'] ="
        }
        
        import re
        optimized_code = loop_code
        
        for pattern, replacement in optimizations.items():
            optimized_code = re.sub(pattern, replacement, optimized_code)
        
        return optimized_code
    
    @staticmethod
    def optimize_memory(df: pd.DataFrame, 
                       deep: bool = True) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage
        
        Args:
            df: DataFrame to optimize
            deep: Whether to do deep optimization
            
        Returns:
            Memory-optimized DataFrame
        """
        optimized = df.copy()
        
        # Get initial memory usage
        initial_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            optimized[col] = pd.to_numeric(optimized[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float']).columns:
            optimized[col] = pd.to_numeric(optimized[col], downcast='float')
        
        # Convert strings to categories where beneficial
        if deep:
            for col in df.select_dtypes(include=['object']).columns:
                num_unique = df[col].nunique()
                num_total = len(df[col])
                
                # Convert to category if less than 50% unique values
                if num_unique / num_total < 0.5:
                    optimized[col] = optimized[col].astype('category')
        
        # Convert datetime columns
        for col in df.select_dtypes(include=['object']).columns:
            if col in optimized.columns and optimized[col].dtype == 'object':
                try:
                    optimized[col] = pd.to_datetime(optimized[col])
                except:
                    pass
        
        # Calculate memory savings
        final_mem = optimized.memory_usage(deep=True).sum() / 1024**2
        reduction = (1 - final_mem/initial_mem) * 100
        
        logger.info(f"Memory reduced by {reduction:.1f}% ({initial_mem:.1f}MB → {final_mem:.1f}MB)")
        
        return optimized
    
    @staticmethod
    def profile_operation(func: Callable) -> Callable:
        """
        Decorator to profile DataFrame operations
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function with profiling
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = 0
            
            # Get initial memory if DataFrame is passed
            if args and isinstance(args[0], pd.DataFrame):
                start_memory = args[0].memory_usage(deep=True).sum() / 1024**2
            
            # Run function
            result = func(*args, **kwargs)
            
            # Calculate metrics
            elapsed = time.time() - start_time
            
            # Log performance
            logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
            
            if isinstance(result, pd.DataFrame) and start_memory > 0:
                end_memory = result.memory_usage(deep=True).sum() / 1024**2
                logger.info(f"Memory: {start_memory:.1f}MB → {end_memory:.1f}MB")
            
            return result
        
        return wrapper


class DataFrameOptimizationReport:
    """Generate optimization reports for DataFrames"""
    
    @staticmethod
    def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze DataFrame for optimization opportunities
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Analysis report
        """
        report = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'optimization_opportunities': [],
            'recommendations': []
        }
        
        # Check for optimization opportunities
        
        # 1. String columns that could be categories
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:
                report['optimization_opportunities'].append({
                    'column': col,
                    'type': 'convert_to_category',
                    'unique_ratio': unique_ratio,
                    'potential_savings': f"{(1 - unique_ratio) * 100:.1f}%"
                })
        
        # 2. Numeric columns that could be downcast
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if df[col].dtype == 'int64':
                if df[col].min() >= -128 and df[col].max() <= 127:
                    report['optimization_opportunities'].append({
                        'column': col,
                        'type': 'downcast_to_int8',
                        'current': 'int64',
                        'recommended': 'int8'
                    })
                elif df[col].min() >= -32768 and df[col].max() <= 32767:
                    report['optimization_opportunities'].append({
                        'column': col,
                        'type': 'downcast_to_int16',
                        'current': 'int64',
                        'recommended': 'int16'
                    })
        
        # 3. Duplicate columns
        duplicate_cols = []
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                if df[col1].equals(df[col2]):
                    duplicate_cols.append((col1, col2))
        
        if duplicate_cols:
            report['optimization_opportunities'].append({
                'type': 'duplicate_columns',
                'columns': duplicate_cols
            })
        
        # Generate recommendations
        if report['optimization_opportunities']:
            report['recommendations'].append("Use DataFrameOptimizer.optimize_memory() to apply optimizations")
        
        if len(df) > 100000:
            report['recommendations'].append("Consider using batch processing for large operations")
        
        if df.select_dtypes(include=['object']).shape[1] > 5:
            report['recommendations'].append("Consider using categories for string columns")
        
        return report


def auto_optimize_file(filepath: str, output_path: Optional[str] = None):
    """
    Automatically optimize DataFrame operations in a Python file
    
    Args:
        filepath: Path to Python file
        output_path: Output path (None to overwrite)
    """
    import re
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Patterns to replace
    replacements = {
        # Replace iterrows
        r'for\s+(\w+),\s*(\w+)\s+in\s+(\w+)\.iterrows\(\):':
            r'# Optimized: vectorized operation\nfor \1, \2 in \3.itertuples():',
        
        # Replace iteritems (deprecated)
        r'\.iteritems\(\)': r'.items()',
        
        # Replace append in loop
        r'(\w+)\s*=\s*(\w+)\.append\(': r'\1 = pd.concat([\2, ',
        
        # Use loc for filtering
        r'(\w+)\[(\w+)\[(["\'][\w\s]+["\'])\]\s*==\s*([^\]]+)\]':
            r"\1.loc[\2[\3] == \4]",
    }
    
    optimized = content
    changes_made = []
    
    for pattern, replacement in replacements.items():
        if re.search(pattern, optimized):
            optimized = re.sub(pattern, replacement, optimized)
            changes_made.append(f"Replaced pattern: {pattern}")
    
    # Add import if needed
    if changes_made and 'import numpy as np' not in optimized:
        optimized = 'import numpy as np\n' + optimized
    
    # Write output
    output_file = output_path or filepath
    with open(output_file, 'w') as f:
        f.write(optimized)
    
    logger.info(f"Optimized {filepath}: {len(changes_made)} changes made")
    for change in changes_made:
        logger.info(f"  - {change}")