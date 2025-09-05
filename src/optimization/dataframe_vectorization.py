"""
DataFrame Vectorization Optimizer
Replaces inefficient iterrows() with vectorized operations for 10-100x speedup
Created: 2025-09-05
"""

import pandas as pd
import numpy as np
from typing import Callable, Any, Dict, List, Tuple
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


class DataFrameVectorizer:
    """
    Provides vectorized alternatives to common DataFrame.iterrows() patterns
    """
    
    @staticmethod
    def benchmark(func: Callable) -> Callable:
        """Decorator to benchmark function execution time"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"{func.__name__} took {elapsed:.4f} seconds")
            return result
        return wrapper
    
    @staticmethod
    def replace_iterrows_with_apply(df: pd.DataFrame, 
                                   operation: Callable,
                                   axis: int = 1) -> pd.Series:
        """
        Replace iterrows with apply (5-10x faster)
        
        BEFORE:
        for idx, row in df.iterrows():
            result = operation(row)
            
        AFTER:
        result = df.apply(operation, axis=1)
        """
        return df.apply(operation, axis=axis)
    
    @staticmethod
    def replace_iterrows_with_vectorize(df: pd.DataFrame, 
                                       column: str,
                                       operation: Callable) -> pd.Series:
        """
        Replace iterrows with numpy vectorize (10-50x faster)
        
        BEFORE:
        for idx, row in df.iterrows():
            df.at[idx, 'result'] = operation(row[column])
            
        AFTER:
        df['result'] = np.vectorize(operation)(df[column])
        """
        return np.vectorize(operation)(df[column].values)
    
    @staticmethod
    def replace_iterrows_with_broadcasting(df: pd.DataFrame, 
                                          condition_col: str,
                                          value_col: str,
                                          threshold: float) -> pd.Series:
        """
        Replace conditional iterrows with boolean indexing (50-100x faster)
        
        BEFORE:
        for idx, row in df.iterrows():
            if row[condition_col] > threshold:
                df.at[idx, value_col] = new_value
                
        AFTER:
        df.loc[df[condition_col] > threshold, value_col] = new_value
        """
        mask = df[condition_col] > threshold
        return mask
    
    @staticmethod
    def replace_iterrows_accumulation(df: pd.DataFrame,
                                     columns: List[str]) -> pd.Series:
        """
        Replace iterrows for accumulation with vectorized sum (100x faster)
        
        BEFORE:
        total = 0
        for idx, row in df.iterrows():
            total += row['value'] * row['quantity']
            
        AFTER:
        total = (df['value'] * df['quantity']).sum()
        """
        if len(columns) == 1:
            return df[columns[0]].sum()
        else:
            result = df[columns[0]]
            for col in columns[1:]:
                result = result * df[col]
            return result.sum()
    
    @staticmethod
    def replace_iterrows_groupby(df: pd.DataFrame,
                                group_col: str,
                                agg_col: str,
                                agg_func: str = 'sum') -> pd.DataFrame:
        """
        Replace iterrows with groupby for aggregation (50x faster)
        
        BEFORE:
        results = {}
        for idx, row in df.iterrows():
            key = row[group_col]
            if key not in results:
                results[key] = 0
            results[key] += row[agg_col]
            
        AFTER:
        results = df.groupby(group_col)[agg_col].agg(agg_func)
        """
        return df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
    
    @staticmethod
    def replace_iterrows_merge(df1: pd.DataFrame,
                              df2: pd.DataFrame,
                              on: str) -> pd.DataFrame:
        """
        Replace iterrows lookup with merge (100x faster)
        
        BEFORE:
        for idx, row in df1.iterrows():
            matching = df2[df2['id'] == row['id']]
            if not matching.empty:
                df1.at[idx, 'value'] = matching.iloc[0]['value']
                
        AFTER:
        df1 = df1.merge(df2[['id', 'value']], on='id', how='left')
        """
        return df1.merge(df2, on=on, how='left')
    
    @staticmethod
    def vectorize_planning_balance(df: pd.DataFrame) -> pd.Series:
        """
        Vectorized planning balance calculation
        Planning Balance = Theoretical Balance + Allocated + On Order
        Note: Allocated is already negative
        """
        return df['theoretical_balance'] + df['allocated'] + df['on_order']
    
    @staticmethod
    def vectorize_shortage_detection(df: pd.DataFrame, 
                                    threshold: float = 0) -> pd.DataFrame:
        """
        Vectorized shortage detection
        """
        df['planning_balance'] = DataFrameVectorizer.vectorize_planning_balance(df)
        return df[df['planning_balance'] < threshold].copy()
    
    @staticmethod
    def vectorize_yarn_requirements(bom_df: pd.DataFrame,
                                  orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized yarn requirement calculation
        """
        # Merge orders with BOM
        requirements = orders_df.merge(
            bom_df[['style_id', 'yarn_id', 'quantity_per_unit']], 
            on='style_id', 
            how='left'
        )
        
        # Vectorized calculation
        requirements['yarn_required'] = (
            requirements['order_quantity'] * requirements['quantity_per_unit']
        )
        
        # Group by yarn_id and sum
        return requirements.groupby('yarn_id')['yarn_required'].sum().reset_index()
    
    @staticmethod
    def optimize_iterrows_in_code(code: str) -> Tuple[str, List[str]]:
        """
        Analyze code and suggest vectorized replacements
        Returns: (optimized_code, list_of_changes)
        """
        changes = []
        lines = code.split('\n')
        optimized_lines = []
        
        for i, line in enumerate(lines):
            if 'for' in line and 'iterrows()' in line:
                # Detect pattern and suggest replacement
                if 'df.iterrows()' in line or '.iterrows()' in line:
                    changes.append(f"Line {i+1}: Replace iterrows() with vectorized operation")
                    
                    # Try to detect the pattern
                    next_lines = '\n'.join(lines[i:min(i+5, len(lines))])
                    
                    if 'df.at[' in next_lines or 'df.loc[' in next_lines:
                        # Pattern: Setting values
                        optimized_lines.append(f"# OPTIMIZED: Use vectorized operation instead of iterrows")
                        optimized_lines.append(f"# df['column'] = df.apply(lambda row: operation(row), axis=1)")
                    elif 'if row[' in next_lines:
                        # Pattern: Conditional logic
                        optimized_lines.append(f"# OPTIMIZED: Use boolean indexing")
                        optimized_lines.append(f"# mask = df['column'] > threshold")
                        optimized_lines.append(f"# df.loc[mask, 'target'] = value")
                    else:
                        # General pattern
                        optimized_lines.append(f"# OPTIMIZED: Consider using df.apply() or vectorized operations")
                else:
                    optimized_lines.append(line)
            else:
                optimized_lines.append(line)
        
        return '\n'.join(optimized_lines), changes


class PerformanceBenchmark:
    """
    Benchmarks to demonstrate performance improvements
    """
    
    @staticmethod
    def create_test_dataframe(size: int = 10000) -> pd.DataFrame:
        """Create a test DataFrame for benchmarking"""
        return pd.DataFrame({
            'yarn_id': [f'Y{i:04d}' for i in range(size)],
            'theoretical_balance': np.random.randn(size) * 100,
            'allocated': -np.abs(np.random.randn(size) * 50),  # Negative values
            'on_order': np.random.randn(size) * 30,
            'quantity': np.random.randint(1, 100, size),
            'price': np.random.randn(size) * 10
        })
    
    @staticmethod
    def benchmark_iterrows_vs_vectorized(df: pd.DataFrame):
        """Compare iterrows vs vectorized performance"""
        
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK: iterrows() vs Vectorized")
        print("="*60)
        print(f"DataFrame size: {len(df)} rows")
        
        # Method 1: iterrows (SLOW)
        start = time.perf_counter()
        result_iterrows = []
        for idx, row in df.iterrows():
            balance = row['theoretical_balance'] + row['allocated'] + row['on_order']
            result_iterrows.append(balance)
        time_iterrows = time.perf_counter() - start
        
        # Method 2: apply (FASTER)
        start = time.perf_counter()
        result_apply = df.apply(
            lambda row: row['theoretical_balance'] + row['allocated'] + row['on_order'], 
            axis=1
        )
        time_apply = time.perf_counter() - start
        
        # Method 3: vectorized (FASTEST)
        start = time.perf_counter()
        result_vectorized = df['theoretical_balance'] + df['allocated'] + df['on_order']
        time_vectorized = time.perf_counter() - start
        
        print(f"\n1. iterrows():    {time_iterrows:.4f} seconds (baseline)")
        print(f"2. apply():       {time_apply:.4f} seconds ({time_iterrows/time_apply:.1f}x faster)")
        print(f"3. vectorized:    {time_vectorized:.4f} seconds ({time_iterrows/time_vectorized:.1f}x faster)")
        
        print(f"\n✨ Vectorization is {time_iterrows/time_vectorized:.1f}x faster than iterrows!")
        print("="*60)
        
        return {
            'iterrows': time_iterrows,
            'apply': time_apply,
            'vectorized': time_vectorized,
            'speedup': time_iterrows/time_vectorized
        }


def demonstrate_optimizations():
    """
    Demonstrate various optimization techniques
    """
    print("\n" + "="*60)
    print("DataFrame Vectorization Examples")
    print("="*60)
    
    # Create sample data
    df = pd.DataFrame({
        'yarn_id': ['Y001', 'Y002', 'Y003', 'Y004', 'Y005'],
        'theoretical_balance': [100, 200, 150, 300, 250],
        'allocated': [-20, -50, -30, -100, -80],  # Already negative
        'on_order': [50, 0, 25, 75, 100],
        'min_stock': [50, 100, 75, 150, 125]
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    vectorizer = DataFrameVectorizer()
    
    # Example 1: Planning Balance Calculation
    print("\n1. Vectorized Planning Balance Calculation:")
    df['planning_balance'] = vectorizer.vectorize_planning_balance(df)
    print(df[['yarn_id', 'planning_balance']])
    
    # Example 2: Shortage Detection
    print("\n2. Vectorized Shortage Detection (balance < min_stock):")
    shortages = df[df['planning_balance'] < df['min_stock']]
    print(shortages[['yarn_id', 'planning_balance', 'min_stock']])
    
    # Example 3: Conditional Update
    print("\n3. Vectorized Conditional Update:")
    df.loc[df['planning_balance'] < 0, 'status'] = 'CRITICAL'
    df.loc[df['planning_balance'] >= 0, 'status'] = 'OK'
    print(df[['yarn_id', 'planning_balance', 'status']])
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_optimizations()
    
    # Run benchmark
    benchmark = PerformanceBenchmark()
    test_df = benchmark.create_test_dataframe(10000)
    results = benchmark.benchmark_iterrows_vs_vectorized(test_df)
    
    print(f"\n🚀 Expected improvement for 157 iterrows instances:")
    print(f"   Current time: ~{157 * results['iterrows']:.2f} seconds")
    print(f"   Optimized time: ~{157 * results['vectorized']:.2f} seconds")
    print(f"   Time saved: ~{157 * (results['iterrows'] - results['vectorized']):.2f} seconds")