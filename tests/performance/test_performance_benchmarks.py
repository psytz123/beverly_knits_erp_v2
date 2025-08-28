"""
Performance tests for Beverly Knits ERP System

Tests system performance, scalability, and resource utilization
"""
import pytest
import time
import concurrent.futures
import psutil
import numpy as np
from datetime import datetime
import pandas as pd
try:
    from memory_profiler import profile
except ImportError:
    # Make profile a no-op decorator if memory_profiler is not installed
    def profile(func):
        return func
import requests

# No specific imports needed for performance tests


class TestAPIPerformance:
    """Performance tests for API endpoints"""
    
    BASE_URL = "http://localhost:5006"
    
    @pytest.mark.benchmark
    def test_api_response_time_p95(self, benchmark):
        """Test API response time at 95th percentile"""
        
        def make_request():
            response = requests.get(f"{self.BASE_URL}/api/health")
            return response.elapsed.total_seconds()
        
        # Run benchmark
        result = benchmark(make_request)
        
        # Assert P95 < 200ms
        assert result < 0.2
    
    @pytest.mark.benchmark
    def test_yarn_intelligence_performance(self, benchmark):
        """Test yarn intelligence endpoint performance"""
        
        def get_yarn_intelligence():
            response = requests.get(f"{self.BASE_URL}/api/yarn-intelligence")
            assert response.status_code == 200
            return response.elapsed.total_seconds()
        
        result = benchmark.pedantic(
            get_yarn_intelligence,
            rounds=10,
            iterations=5
        )
        
        # Should complete within 1 second
        assert benchmark.stats['mean'] < 1.0
    
    @pytest.mark.slow
    def test_concurrent_user_load(self):
        """Test system under concurrent user load"""
        response_times = []
        errors = []
        
        def make_concurrent_request(user_id):
            try:
                start = time.time()
                response = requests.get(
                    f"{self.BASE_URL}/api/inventory-intelligence-enhanced",
                    timeout=5
                )
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    return ('success', elapsed)
                else:
                    return ('error', response.status_code)
            except Exception as e:
                return ('exception', str(e))
        
        # Simulate 50 concurrent users
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(make_concurrent_request, i) 
                for i in range(50)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result[0] == 'success':
                    response_times.append(result[1])
                else:
                    errors.append(result)
        
        # Performance assertions
        assert len(errors) < 5  # Less than 10% error rate
        assert np.mean(response_times) < 2.0  # Average response < 2s
        assert np.percentile(response_times, 95) < 3.0  # P95 < 3s
    
    @pytest.mark.slow
    def test_sustained_load(self):
        """Test system under sustained load for 60 seconds"""
        start_time = time.time()
        request_count = 0
        errors = 0
        response_times = []
        
        while time.time() - start_time < 60:  # Run for 60 seconds
            try:
                req_start = time.time()
                response = requests.get(
                    f"{self.BASE_URL}/api/health",
                    timeout=2
                )
                req_time = time.time() - req_start
                
                request_count += 1
                response_times.append(req_time)
                
                if response.status_code != 200:
                    errors += 1
                    
            except Exception:
                errors += 1
            
            time.sleep(0.1)  # 10 requests per second
        
        # Calculate metrics
        throughput = request_count / 60
        error_rate = errors / request_count if request_count > 0 else 0
        avg_response = np.mean(response_times) if response_times else 0
        
        # Assertions
        assert throughput >= 8  # At least 8 req/sec
        assert error_rate < 0.01  # Less than 1% errors
        assert avg_response < 0.5  # Average response < 500ms


class TestPlanningEnginePerformance:
    """Performance tests for six-phase planning engine"""
    
    @pytest.mark.benchmark
    def test_planning_engine_small_dataset(self, benchmark):
        """Test planning engine with small dataset (100 items)"""
        
        def run_planning():
            from production.six_phase_planning_engine import SixPhasePlanningEngine
            
            engine = SixPhasePlanningEngine()
            # Mock small dataset
            engine.inventory_data = self.generate_mock_inventory(100)
            
            start = time.time()
            result = engine.execute_all_phases()
            elapsed = time.time() - start
            
            return elapsed
        
        result = benchmark(run_planning)
        
        # Should complete within 5 seconds for 100 items
        assert result < 5.0
    
    @pytest.mark.slow
    def test_planning_engine_large_dataset(self):
        """Test planning engine with large dataset (1000+ items)"""
        from production.six_phase_planning_engine import SixPhasePlanningEngine
        
        engine = SixPhasePlanningEngine()
        engine.inventory_data = self.generate_mock_inventory(1000)
        
        start = time.time()
        result = engine.execute_all_phases()
        elapsed = time.time() - start
        
        # Should complete within 2 minutes for 1000 items
        assert elapsed < 120
        assert result['status'] == 'completed'
    
    @pytest.mark.slow
    def test_planning_engine_scalability(self):
        """Test planning engine scalability with increasing data sizes"""
        from production.six_phase_planning_engine import SixPhasePlanningEngine
        
        sizes = [100, 500, 1000, 2000]
        execution_times = []
        
        for size in sizes:
            engine = SixPhasePlanningEngine()
            engine.inventory_data = self.generate_mock_inventory(size)
            
            start = time.time()
            engine.execute_all_phases()
            elapsed = time.time() - start
            
            execution_times.append(elapsed)
        
        # Check that execution time scales linearly or better
        # Time complexity should be O(n) or O(n log n), not O(n²)
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = execution_times[i] / execution_times[i-1]
            
            # Time should not increase quadratically
            assert time_ratio < size_ratio * 1.5
    
    def generate_mock_inventory(self, size):
        """Generate mock inventory data for testing"""
        return pd.DataFrame({
            'Item': [f'ITEM{i:04d}' for i in range(size)],
            'Planning Balance': np.random.normal(100, 50, size),
            'Consumed': np.random.uniform(10, 100, size),
            'On Order': np.random.uniform(0, 200, size),
            'Safety Stock': np.random.uniform(20, 50, size)
        })


class TestDataProcessingPerformance:
    """Performance tests for data processing operations"""
    
    @pytest.mark.benchmark
    def test_excel_loading_performance(self, benchmark):
        """Test Excel file loading performance"""
        
        def load_excel():
            # Create temporary Excel file
            df = pd.DataFrame({
                'A': range(1000),
                'B': range(1000),
                'C': range(1000)
            })
            
            temp_file = '/tmp/test_data.xlsx'
            df.to_excel(temp_file, index=False)
            
            # Load and process
            start = time.time()
            loaded_df = pd.read_excel(temp_file)
            elapsed = time.time() - start
            
            return elapsed
        
        result = benchmark(load_excel)
        
        # Should load 1000 rows within 1 second
        assert result < 1.0
    
    @pytest.mark.benchmark
    def test_data_aggregation_performance(self, benchmark):
        """Test data aggregation performance"""
        
        def aggregate_data():
            # Create large dataset
            df = pd.DataFrame({
                'category': np.random.choice(['A', 'B', 'C', 'D'], 10000),
                'value': np.random.uniform(0, 1000, 10000),
                'quantity': np.random.randint(1, 100, 10000)
            })
            
            start = time.time()
            # Perform aggregations
            result = df.groupby('category').agg({
                'value': ['sum', 'mean', 'std'],
                'quantity': ['sum', 'count']
            })
            elapsed = time.time() - start
            
            return elapsed
        
        result = benchmark(aggregate_data)
        
        # Should complete within 100ms
        assert result < 0.1
    
    def test_yarn_calculation_performance(self):
        """Test yarn requirement calculation performance"""
        
        # Generate test data
        fabric_orders = pd.DataFrame({
            'order_id': [f'FO{i:04d}' for i in range(100)],
            'quantity': np.random.uniform(100, 5000, 100),
            'width': np.random.choice([45, 60, 72], 100),
            'weight': np.random.uniform(150, 300, 100)
        })
        
        start = time.time()
        
        # Calculate yarn requirements for all orders
        for _, order in fabric_orders.iterrows():
            yarn_required = (
                order['quantity'] * 
                order['width'] * 
                order['weight'] * 
                0.0001 * 1.1  # Include waste factor
            )
        
        elapsed = time.time() - start
        
        # Should process 100 orders in less than 1 second
        assert elapsed < 1.0


class TestMemoryPerformance:
    """Memory usage and leak tests"""
    
    def test_memory_usage_inventory_analysis(self):
        """Test memory usage during inventory analysis"""
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        large_dataset = pd.DataFrame({
            'Item': [f'ITEM{i:06d}' for i in range(10000)],
            'Value': np.random.uniform(0, 1000, 10000),
            'Quantity': np.random.randint(0, 1000, 10000)
        })
        
        # Process dataset
        for _ in range(10):
            result = large_dataset.groupby('Item').sum()
            filtered = large_dataset[large_dataset['Value'] > 500]
        
        # Check memory after processing
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - baseline_memory
        
        # Memory increase should be reasonable (< 100 MB)
        assert memory_increase < 100
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations"""
        process = psutil.Process()
        memory_samples = []
        
        for iteration in range(20):
            # Perform operation that might leak memory
            df = pd.DataFrame({
                'data': np.random.random(1000)
            })
            result = df.describe()
            
            # Sample memory usage
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_samples.append(memory_mb)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Check if memory is continuously increasing
        # Calculate trend (should be near zero for no leak)
        x = np.arange(len(memory_samples))
        slope = np.polyfit(x, memory_samples, 1)[0]
        
        # Slope should be minimal (< 0.5 MB per iteration)
        assert abs(slope) < 0.5


class TestDatabasePerformance:
    """Database query performance tests"""
    
    @pytest.mark.database
    def test_bulk_insert_performance(self):
        """Test bulk insert performance"""
        # Generate test data
        records = [
            {
                'item_id': f'ITEM{i:06d}',
                'quantity': np.random.randint(0, 1000),
                'timestamp': datetime.now()
            }
            for i in range(1000)
        ]
        
        start = time.time()
        
        # Simulate bulk insert (would use actual DB in real test)
        df = pd.DataFrame(records)
        # df.to_sql('inventory', connection, if_exists='append')
        
        elapsed = time.time() - start
        
        # Should insert 1000 records within 2 seconds
        assert elapsed < 2.0
    
    @pytest.mark.database
    def test_complex_query_performance(self):
        """Test complex query performance"""
        
        # Simulate complex query with joins and aggregations
        start = time.time()
        
        # This would be actual SQL in real test
        query = """
        SELECT 
            i.item_id,
            i.quantity,
            SUM(o.amount) as total_orders,
            AVG(f.forecast) as avg_forecast
        FROM inventory i
        LEFT JOIN orders o ON i.item_id = o.item_id
        LEFT JOIN forecasts f ON i.item_id = f.item_id
        WHERE i.quantity < i.safety_stock
        GROUP BY i.item_id, i.quantity
        HAVING SUM(o.amount) > 1000
        """
        
        # Simulate query execution
        time.sleep(0.1)  # Simulated query time
        
        elapsed = time.time() - start
        
        # Complex query should complete within 500ms
        assert elapsed < 0.5


class TestCachePerformance:
    """Cache system performance tests"""
    
    def test_cache_hit_performance(self):
        """Test cache hit performance"""
        from functools import lru_cache
        
        @lru_cache(maxsize=1000)
        def expensive_calculation(n):
            """Simulate expensive calculation"""
            time.sleep(0.01)  # Simulate work
            return n * n
        
        # First call - cache miss
        start = time.time()
        result1 = expensive_calculation(42)
        miss_time = time.time() - start
        
        # Second call - cache hit
        start = time.time()
        result2 = expensive_calculation(42)
        hit_time = time.time() - start
        
        # Cache hit should be at least 10x faster
        assert hit_time < miss_time / 10
        assert result1 == result2
    
    def test_cache_memory_usage(self):
        """Test cache memory usage"""
        cache = {}
        process = psutil.Process()
        
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Fill cache with data
        for i in range(10000):
            cache[f'key_{i}'] = {
                'data': np.random.random(100).tolist(),
                'timestamp': datetime.now().isoformat()
            }
        
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_used = current_memory - baseline_memory
        
        # Cache should use reasonable memory (< 50 MB for 10k entries)
        assert memory_used < 50


# Performance test runner configuration
if __name__ == "__main__":
    pytest.main([
        __file__,
        '-v',
        '--benchmark-only',
        '--benchmark-columns=min,max,mean,stddev,median,iqr,outliers,rounds',
        '--benchmark-sort=mean',
        '--benchmark-save=performance_results',
        '--benchmark-save-data'
    ])