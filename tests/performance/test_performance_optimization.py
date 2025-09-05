"""
Performance Optimization Tests
Tests for Phase 3 performance optimizations
"""
import pytest
import pandas as pd
import numpy as np
import asyncio
import time
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.optimization.performance.dataframe_optimizer import DataFrameOptimizer
from src.optimization.performance.query_optimizer import QueryOptimizer
from src.optimization.performance.async_processor import AsyncProcessor, BackgroundScheduler
from src.optimization.performance.memory_optimizer import MemoryOptimizer, ConnectionPoolOptimizer
from src.optimization.performance.performance_integration import (
    PerformanceIntegration, 
    optimize_dataframe_operation,
    benchmark_optimization
)


class TestDataFrameOptimizer:
    """Test DataFrame optimization functions"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'yarn_id': [f'Y{i:03d}' for i in range(1000)],
            'theoretical_balance': np.random.uniform(0, 1000, 1000),
            'allocated': np.random.uniform(-500, 0, 1000),
            'on_order': np.random.uniform(0, 500, 1000),
            'min_stock': np.random.uniform(50, 200, 1000)
        })
    
    def test_planning_balance_calculation(self, sample_df):
        """Test vectorized planning balance calculation"""
        # Apply optimization
        result = DataFrameOptimizer.optimize_planning_balance_calculation(sample_df)
        
        # Check that planning_balance column was added
        assert 'planning_balance' in result.columns
        
        # Verify calculation correctness
        expected = (
            sample_df['theoretical_balance'] + 
            sample_df['allocated'] + 
            sample_df['on_order']
        )
        np.testing.assert_array_almost_equal(
            result['planning_balance'].values,
            expected.values
        )
    
    def test_shortage_detection(self, sample_df):
        """Test vectorized shortage detection"""
        # Add planning balance first
        sample_df = DataFrameOptimizer.optimize_planning_balance_calculation(sample_df)
        
        # Detect shortages
        shortages = DataFrameOptimizer.optimize_shortage_detection(
            sample_df, 'min_stock', 'planning_balance'
        )
        
        # Verify all detected items are actually shortages
        assert all(shortages['planning_balance'] < shortages['min_stock'])
        
        # Check shortage_amount calculation
        assert 'shortage_amount' in shortages.columns
        expected_shortage = shortages['min_stock'] - shortages['planning_balance']
        np.testing.assert_array_almost_equal(
            shortages['shortage_amount'].values,
            expected_shortage.values
        )
    
    def test_bom_explosion(self):
        """Test BOM explosion optimization"""
        bom_df = pd.DataFrame({
            'style_id': ['S001', 'S001', 'S002'],
            'yarn_id': ['Y001', 'Y002', 'Y001'],
            'quantity_per': [0.5, 0.3, 0.7]
        })
        
        result = DataFrameOptimizer.optimize_bom_explosion(bom_df, 100, 'S001')
        
        # Check calculations
        assert len(result) == 2  # Only S001 items
        assert all(result['style_id'] == 'S001')
        assert result[result['yarn_id'] == 'Y001']['required_quantity'].iloc[0] == 50
        assert result[result['yarn_id'] == 'Y002']['required_quantity'].iloc[0] == 30
    
    def test_memory_optimization(self, sample_df):
        """Test memory usage reduction"""
        # Get initial memory
        initial_memory = sample_df.memory_usage(deep=True).sum()
        
        # Optimize memory
        optimized = DataFrameOptimizer.optimize_memory_usage(sample_df)
        
        # Get final memory
        final_memory = optimized.memory_usage(deep=True).sum()
        
        # Should reduce memory
        assert final_memory < initial_memory
        
        # Data should remain the same
        pd.testing.assert_frame_equal(
            sample_df.reset_index(drop=True),
            optimized.reset_index(drop=True),
            check_dtype=False  # Allow dtype changes
        )
    
    def test_performance_improvement(self, sample_df):
        """Test actual performance improvement"""
        import timeit
        
        # Define functions to test
        def slow_method(df):
            for index, row in df.iterrows():
                df.at[index, 'planning_balance'] = (
                    row['theoretical_balance'] + 
                    row['allocated'] + 
                    row['on_order']
                )
            return df
        
        def fast_method(df):
            return DataFrameOptimizer.optimize_planning_balance_calculation(df)
        
        # Time both methods
        slow_time = timeit.timeit(
            lambda: slow_method(sample_df.copy()),
            number=10
        )
        
        fast_time = timeit.timeit(
            lambda: fast_method(sample_df.copy()),
            number=10
        )
        
        # Fast method should be at least 10x faster
        assert fast_time < slow_time / 10


class TestQueryOptimizer:
    """Test query optimization"""
    
    @pytest.fixture
    def optimizer(self):
        return QueryOptimizer()
    
    def test_yarn_query_optimization(self, optimizer):
        """Test yarn query optimization"""
        query, params = optimizer.optimize_yarn_query(
            conditions={'status': 'active', 'yarn_id': ['Y001', 'Y002']},
            limit=100
        )
        
        # Check query structure
        assert 'SELECT' in query
        assert 'yarn_id' in query
        assert 'WHERE' in query
        assert 'LIMIT 100' in query
        assert 'USE INDEX' in query
        
        # Check params
        assert 'Y001' in params
        assert 'Y002' in params
    
    def test_batch_fetch(self, optimizer):
        """Test batch fetching"""
        # Mock database connection
        optimizer.db = Mock()
        optimizer.db.fetch_df = Mock(return_value=pd.DataFrame({'id': [1, 2, 3]}))
        
        # Test batch fetch
        result = optimizer.batch_fetch(
            'test_table',
            list(range(2500)),  # More than batch size
            batch_size=1000
        )
        
        # Should make 3 calls (2500 / 1000 = 3 batches)
        assert optimizer.db.fetch_df.call_count == 3
    
    def test_query_caching(self, optimizer):
        """Test query result caching"""
        # Create test data
        test_df = pd.DataFrame({'col1': [1, 2, 3]})
        
        # Cache result
        query_hash = optimizer.cache_query_result(
            "SELECT * FROM test",
            [],
            test_df
        )
        
        # Retrieve from cache
        cached = optimizer.get_cached_query_result(query_hash, ttl_seconds=300)
        
        # Should return cached data
        pd.testing.assert_frame_equal(cached, test_df)
        
        # Check statistics
        assert query_hash in optimizer.query_stats
        assert optimizer.query_stats[query_hash]['count'] == 1


class TestAsyncProcessor:
    """Test async processing capabilities"""
    
    @pytest.fixture
    async def processor(self):
        async with AsyncProcessor() as proc:
            yield proc
    
    @pytest.mark.asyncio
    async def test_replace_blocking_sleep(self, processor):
        """Test non-blocking sleep"""
        start = time.time()
        await processor.replace_blocking_sleep(0.1)
        elapsed = time.time() - start
        
        # Should complete in approximately 0.1 seconds
        assert 0.09 < elapsed < 0.15
    
    @pytest.mark.asyncio
    async def test_batch_process_async(self, processor):
        """Test concurrent batch processing"""
        items = list(range(20))
        
        async def process_item(item):
            await asyncio.sleep(0.01)  # Simulate work
            return item * 2
        
        results = await processor.batch_process_async(
            items,
            process_item,
            max_concurrent=5
        )
        
        # Check all items processed
        assert len(results) == 20
        assert results == [i * 2 for i in range(20)]
    
    @pytest.mark.asyncio
    async def test_heavy_calculation(self, processor):
        """Test CPU-intensive task processing"""
        def heavy_calc(n):
            return sum(i ** 2 for i in range(n))
        
        result = await processor.process_heavy_calculation(heavy_calc, 1000)
        expected = sum(i ** 2 for i in range(1000))
        
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_retry_async(self, processor):
        """Test retry with exponential backoff"""
        attempt_count = 0
        
        async def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await processor.retry_async(
            flaky_function,
            max_retries=3,
            backoff_factor=0.1
        )
        
        assert result == "success"
        assert attempt_count == 3


class TestMemoryOptimizer:
    """Test memory optimization"""
    
    @pytest.fixture
    def optimizer(self):
        return MemoryOptimizer()
    
    def test_dataframe_memory_reduction(self, optimizer):
        """Test DataFrame memory optimization"""
        # Create large DataFrame
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 10000),
            'float_col': np.random.random(10000),
            'str_col': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        # Get initial memory
        initial_memory = df.memory_usage(deep=True).sum()
        
        # Optimize
        optimized = MemoryOptimizer.optimize_dataframe(df)
        
        # Get optimized memory
        final_memory = optimized.memory_usage(deep=True).sum()
        
        # Should reduce memory significantly
        assert final_memory < initial_memory * 0.7  # At least 30% reduction
        
        # Check data integrity
        assert len(optimized) == len(df)
        assert list(optimized.columns) == list(df.columns)
    
    def test_memory_tracking(self, optimizer):
        """Test memory usage tracking"""
        with optimizer.memory_tracker("test_operation"):
            # Simulate memory usage
            data = pd.DataFrame(np.random.random((1000, 10)))
        
        # Check stats were recorded
        assert len(optimizer.memory_stats) == 1
        assert optimizer.memory_stats[0]['operation'] == "test_operation"
        assert 'memory_delta_mb' in optimizer.memory_stats[0]
    
    def test_memory_leak_detection(self, optimizer):
        """Test memory leak detection"""
        # Simulate growing memory usage
        for i in range(15):
            optimizer.memory_stats.append({
                'end_memory_mb': 100 + i * 10,
                'operation': f'op_{i}'
            })
        
        leaks = optimizer.detect_memory_leaks()
        
        # Should detect monotonic growth
        assert len(leaks) > 0
        assert any('Monotonic memory growth' in leak.get('type', '') for leak in leaks)


class TestPerformanceIntegration:
    """Test integrated performance optimizations"""
    
    @pytest.fixture
    def integration(self):
        return PerformanceIntegration()
    
    def test_inventory_service_optimization(self, integration):
        """Test inventory service optimization"""
        # Create mock service
        service = Mock()
        service.calculate_planning_balance = lambda df: df
        service.detect_shortages = lambda df, t: df
        
        # Apply optimizations
        integration.optimize_inventory_service(service)
        
        # Check methods were replaced
        assert hasattr(service, '_original_methods')
        assert 'calculate_planning_balance' in service._original_methods
        
        # Check stats updated
        df = pd.DataFrame({'test': [1, 2, 3]})
        service.calculate_planning_balance(df)
        assert integration.optimization_stats['dataframe_optimizations'] > 0
    
    def test_benchmark_optimization(self):
        """Test optimization benchmarking"""
        def slow_function(df):
            # Simulate slow operation
            for index, row in df.iterrows():
                df.at[index, 'doubled'] = row['value'] * 2
            return df
        
        # Create test data
        df = pd.DataFrame({'value': range(100)})
        
        # Benchmark
        results = benchmark_optimization(slow_function, df.copy())
        
        # Check benchmark results
        assert 'original_time_ms' in results
        assert 'optimized_time_ms' in results
        assert 'speed_improvement_pct' in results
        assert results['optimized_time_ms'] < results['original_time_ms']


class TestBackgroundScheduler:
    """Test background task scheduling"""
    
    @pytest.fixture
    def scheduler(self):
        return BackgroundScheduler()
    
    @pytest.mark.asyncio
    async def test_periodic_task(self, scheduler):
        """Test periodic task execution"""
        call_count = 0
        
        async def test_task():
            nonlocal call_count
            call_count += 1
        
        # Add task with short interval
        scheduler.add_periodic_task(
            'test_task',
            test_task,
            interval=0.1,
            run_immediately=True
        )
        
        # Start scheduler
        await scheduler.start()
        
        # Wait for tasks to run
        await asyncio.sleep(0.35)
        
        # Stop scheduler
        await scheduler.stop()
        
        # Should have run at least 3 times
        assert call_count >= 3
    
    def test_task_status(self, scheduler):
        """Test task status reporting"""
        scheduler.add_periodic_task(
            'test_task',
            lambda: None,
            interval=60
        )
        
        status = scheduler.get_task_status('test_task')
        
        assert status is not None
        assert status['name'] == 'test_task'
        assert status['interval'] == 60
        assert status['run_count'] == 0


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarks"""
    
    def test_iterrows_vs_vectorized(self):
        """Benchmark iterrows vs vectorized operations"""
        size = 5000
        df = pd.DataFrame({
            'a': np.random.random(size),
            'b': np.random.random(size),
            'c': np.random.random(size)
        })
        
        # Iterrows approach
        def with_iterrows(data):
            for idx, row in data.iterrows():
                data.at[idx, 'sum'] = row['a'] + row['b'] + row['c']
            return data
        
        # Vectorized approach
        def vectorized(data):
            data['sum'] = data['a'] + data['b'] + data['c']
            return data
        
        # Time both
        import timeit
        
        iterrows_time = timeit.timeit(
            lambda: with_iterrows(df.copy()),
            number=10
        )
        
        vectorized_time = timeit.timeit(
            lambda: vectorized(df.copy()),
            number=10
        )
        
        speedup = iterrows_time / vectorized_time
        print(f"Vectorized is {speedup:.1f}x faster than iterrows")
        
        # Vectorized should be at least 50x faster
        assert speedup > 50
    
    def test_memory_optimization_impact(self):
        """Test memory optimization impact on large DataFrames"""
        # Create large DataFrame
        size = 100000
        df = pd.DataFrame({
            'id': range(size),
            'category': np.random.choice(['A', 'B', 'C', 'D'], size),
            'value': np.random.random(size) * 1000,
            'flag': np.random.choice([True, False], size),
            'text': [f'item_{i}' for i in range(size)]
        })
        
        # Original memory
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize
        optimized = MemoryOptimizer.optimize_dataframe(df.copy())
        optimized_memory = optimized.memory_usage(deep=True).sum() / 1024**2
        
        reduction = (1 - optimized_memory / original_memory) * 100
        print(f"Memory reduced by {reduction:.1f}% ({original_memory:.1f}MB → {optimized_memory:.1f}MB)")
        
        # Should achieve at least 40% reduction
        assert reduction > 40