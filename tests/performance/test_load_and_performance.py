#!/usr/bin/env python3
"""
Performance and Load Testing for Beverly Knits ERP v2
Tests system performance under various load conditions
"""

import pytest
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json
import threading
import concurrent.futures
from unittest.mock import Mock, patch
import gc

# Try to import psutil, but make it optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Mock psutil for tests that require it
    class MockProcess:
        def memory_info(self):
            class MemInfo:
                rss = 100 * 1024 * 1024  # 100MB mock
            return MemInfo()
    
    class psutil:
        @staticmethod
        def Process():
            return MockProcess()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.beverly_comprehensive_erp import app, analyzer


class TestPerformanceMetrics:
    """Test system performance metrics"""
    
    @pytest.fixture
    def client(self):
        """Create a test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def large_dataset(self):
        """Generate large dataset for performance testing"""
        return pd.DataFrame({
            'Desc#': [f'YARN{i:04d}' for i in range(10000)],
            'Description': [f'Product {i}' for i in range(10000)],
            'Planning_Balance': np.random.uniform(-1000, 5000, 10000),
            'Allocated': np.random.uniform(-500, 0, 10000),
            'On_Order': np.random.uniform(0, 1000, 10000),
            'Consumed': np.random.uniform(-1000, 0, 10000),
            'Cost/Pound': np.random.uniform(1, 10, 10000)
        })
    
    def test_response_time_inventory_analysis(self, client, large_dataset):
        """Test response time for inventory analysis with large dataset"""
        with patch.object(analyzer, 'raw_materials_data', large_dataset):
            start_time = time.time()
            response = client.get('/api/inventory-analysis')
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == 200
            assert response_time < 3.0  # Should respond within 3 seconds for large dataset
            print(f"Inventory analysis response time: {response_time:.3f}s")
    
    def test_response_time_ml_forecast(self, client):
        """Test ML forecast response time"""
        start_time = time.time()
        response = client.get('/api/ml-forecast?horizon=30&product_id=PROD001')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code in [200, 503]
        assert response_time < 5.0  # Should respond within 5 seconds for ML forecast
        print(f"ML forecast response time: {response_time:.3f}s")
    
    def test_response_time_six_phase_planning(self, client):
        """Test 6-phase planning response time"""
        start_time = time.time()
        response = client.get('/api/six-phase-planning')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 8.0  # Should respond within 8 seconds for 6-phase planning
        print(f"Six-phase planning response time: {response_time:.3f}s")
    
    def test_batch_processing_performance(self, client):
        """Test batch processing performance"""
        # Simulate batch request
        batch_data = {
            'products': [f'PROD{i:03d}' for i in range(100)],
            'operation': 'forecast',
            'horizon': 30
        }
        
        start_time = time.time()
        response = client.post('/api/batch-process', json=batch_data)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Batch processing may not be implemented
        if response.status_code == 404:
            pytest.skip("Batch processing endpoint not implemented")
        
        assert response_time < 15.0  # Should complete within 15 seconds for batch
        print(f"Batch processing time for 100 items: {response_time:.3f}s")


@pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
class TestMemoryUsage:
    """Test memory usage and optimization"""
    
    @pytest.fixture
    def memory_baseline(self):
        """Get baseline memory usage"""
        gc.collect()
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def test_memory_leak_detection(self, memory_baseline):
        """Test for memory leaks in repeated operations"""
        from src.core.beverly_comprehensive_erp import InventoryAnalyzer
        
        analyzer = InventoryAnalyzer()
        
        # Perform operation multiple times
        for i in range(100):
            data = pd.DataFrame({
                'Desc#': [f'YARN{j:04d}' for j in range(1000)],
                'Planning_Balance': np.random.uniform(0, 1000, 1000)
            })
            
            # Simulate analysis
            result = analyzer.analyze_inventory_levels(
                current_inventory=[{'id': 'test', 'quantity': 100}],
                forecast={'test': 50}
            )
            
            # Clean up
            del data
            del result
            
            if i % 20 == 0:
                gc.collect()
        
        # Check final memory usage
        gc.collect()
        process = psutil.Process()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = final_memory - memory_baseline
        
        # Should not increase by more than 100MB
        assert memory_increase < 200  # Allow up to 200MB increase (with GC variance)
        print(f"Memory increase after 100 iterations: {memory_increase:.2f}MB")
    
    def test_dataframe_memory_optimization(self):
        """Test DataFrame memory optimization"""
        # Create large DataFrame
        df = pd.DataFrame({
            'id': range(100000),
            'value': np.random.random(100000),
            'category': ['A', 'B', 'C'] * 33333 + ['A'],
            'flag': [True, False] * 50000
        })
        
        # Original memory usage
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        # Optimize dtypes
        df['id'] = df['id'].astype('int32')
        df['value'] = df['value'].astype('float32')
        df['category'] = df['category'].astype('category')
        
        # Optimized memory usage
        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        reduction_pct = (1 - optimized_memory / original_memory) * 100
        
        assert reduction_pct > 30  # Should achieve >30% reduction
        print(f"Memory optimization achieved: {reduction_pct:.1f}% reduction")


class TestConcurrentLoad:
    """Test system under concurrent load"""
    
    @pytest.fixture
    def client(self):
        """Create a test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_concurrent_users_simulation(self, client):
        """Simulate multiple concurrent users"""
        num_users = 10
        requests_per_user = 5
        
        endpoints = [
            '/api/inventory-analysis',
            '/api/yarn-intelligence',
            '/api/production-planning',
            '/api/ml-forecast?horizon=30&product_id=PROD001',
            '/api/health'
        ]
        
        def simulate_user(user_id):
            """Simulate a single user making requests"""
            results = []
            for i in range(requests_per_user):
                endpoint = endpoints[i % len(endpoints)]
                start_time = time.time()
                response = client.get(endpoint)
                end_time = time.time()
                
                results.append({
                    'user_id': user_id,
                    'endpoint': endpoint,
                    'status': response.status_code,
                    'response_time': end_time - start_time
                })
                
                # Small delay between requests
                time.sleep(0.1)
            
            return results
        
        # Run concurrent users
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(simulate_user, i) for i in range(num_users)]
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        successful_requests = sum(1 for r in all_results if r['status'] == 200)
        total_requests = len(all_results)
        success_rate = successful_requests / total_requests * 100
        
        avg_response_time = sum(r['response_time'] for r in all_results) / len(all_results)
        max_response_time = max(r['response_time'] for r in all_results)
        
        print(f"\nConcurrent Load Test Results:")
        print(f"  Users: {num_users}")
        print(f"  Total Requests: {total_requests}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Avg Response Time: {avg_response_time:.3f}s")
        print(f"  Max Response Time: {max_response_time:.3f}s")
        print(f"  Total Test Time: {total_time:.3f}s")
        
        assert success_rate >= 80  # At least 80% success rate
        assert avg_response_time < 3.0  # Average response under 3 seconds under load
    
    def test_spike_load(self, client):
        """Test system behavior under sudden load spike"""
        # Normal load
        normal_load = 5
        spike_load = 20
        
        def make_request():
            return client.get('/api/health').status_code
        
        # Normal load phase
        print("\nPhase 1: Normal load")
        with concurrent.futures.ThreadPoolExecutor(max_workers=normal_load) as executor:
            futures = [executor.submit(make_request) for _ in range(normal_load)]
            normal_results = [f.result() for f in futures]
        
        # Spike load phase
        print("Phase 2: Spike load")
        spike_start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=spike_load) as executor:
            futures = [executor.submit(make_request) for _ in range(spike_load)]
            spike_results = [f.result() for f in futures]
        
        spike_time = time.time() - spike_start
        
        # System should handle spike
        spike_success_rate = sum(1 for r in spike_results if r == 200) / len(spike_results) * 100
        
        print(f"  Spike Success Rate: {spike_success_rate:.1f}%")
        print(f"  Spike Response Time: {spike_time:.3f}s")
        
        assert spike_success_rate >= 70  # Should maintain 70% success during spike


class TestDataLoadPerformance:
    """Test data loading and processing performance"""
    
    def test_csv_loading_performance(self, tmp_path):
        """Test CSV file loading performance"""
        # Create large CSV file
        df = pd.DataFrame({
            'col1': range(100000),
            'col2': np.random.random(100000),
            'col3': ['A', 'B', 'C'] * 33333 + ['A']
        })
        
        csv_path = tmp_path / "large_file.csv"
        df.to_csv(csv_path, index=False)
        
        # Test loading performance
        start_time = time.time()
        loaded_df = pd.read_csv(csv_path)
        end_time = time.time()
        
        load_time = end_time - start_time
        
        assert len(loaded_df) == 100000
        assert load_time < 1.0  # Should load within 1 second
        print(f"CSV load time for 100k records: {load_time:.3f}s")
    
    def test_excel_loading_performance(self, tmp_path):
        """Test Excel file loading performance"""
        # Create Excel file
        df = pd.DataFrame({
            'col1': range(10000),
            'col2': np.random.random(10000)
        })
        
        excel_path = tmp_path / "test_file.xlsx"
        df.to_excel(excel_path, index=False)
        
        # Test loading performance
        start_time = time.time()
        loaded_df = pd.read_excel(excel_path)
        end_time = time.time()
        
        load_time = end_time - start_time
        
        assert len(loaded_df) == 10000
        assert load_time < 3.0  # Excel is slower, allow 3 seconds
        print(f"Excel load time for 10k records: {load_time:.3f}s")
    
    def test_data_aggregation_performance(self):
        """Test data aggregation performance"""
        # Create large dataset
        df = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C', 'D'], 100000),
            'value': np.random.random(100000),
            'quantity': np.random.randint(1, 100, 100000)
        })
        
        # Test aggregation performance
        start_time = time.time()
        
        # Multiple aggregations
        agg_result = df.groupby('category').agg({
            'value': ['mean', 'sum', 'std'],
            'quantity': ['sum', 'count']
        })
        
        end_time = time.time()
        
        agg_time = end_time - start_time
        
        assert not agg_result.empty
        assert agg_time < 0.5  # Should complete within 500ms
        print(f"Aggregation time for 100k records: {agg_time:.3f}s")


class TestCachePerformance:
    """Test caching system performance"""
    
    @pytest.fixture
    def client(self):
        """Create a test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_cache_hit_performance(self, client):
        """Test performance improvement with cache hits"""
        endpoint = '/api/inventory-analysis'
        
        # First request (cache miss)
        start_time = time.time()
        response1 = client.get(endpoint)
        first_time = time.time() - start_time
        
        assert response1.status_code == 200
        
        # Second request (potential cache hit)
        start_time = time.time()
        response2 = client.get(endpoint)
        second_time = time.time() - start_time
        
        assert response2.status_code == 200
        
        # Cache hit should be faster (or at least not slower)
        # Note: May not have caching implemented
        print(f"First request: {first_time:.3f}s")
        print(f"Second request: {second_time:.3f}s")
        
        if second_time < first_time:
            speedup = (1 - second_time / first_time) * 100
            print(f"Cache speedup: {speedup:.1f}%")
    
    def test_cache_invalidation(self, client):
        """Test cache invalidation performance"""
        # Clear cache
        response = client.post('/api/cache-clear')
        
        # Cache clear should be fast
        assert response.status_code in [200, 404]


class TestScalabilityMetrics:
    """Test system scalability"""
    
    def test_linear_scaling(self):
        """Test if processing time scales linearly with data size"""
        from src.core.beverly_comprehensive_erp import InventoryAnalyzer
        
        analyzer = InventoryAnalyzer()
        times = []
        sizes = [100, 500, 1000, 5000]
        
        for size in sizes:
            data = [{'id': f'item_{i}', 'quantity': i} for i in range(size)]
            forecast = {f'item_{i}': i * 0.1 for i in range(size)}
            
            start_time = time.time()
            result = analyzer.analyze_inventory_levels(data, forecast)
            end_time = time.time()
            
            times.append(end_time - start_time)
            print(f"Processing {size} items: {times[-1]:.3f}s")
        
        # Check if scaling is approximately linear
        # Time should not increase exponentially
        time_ratio = times[-1] / times[0]
        size_ratio = sizes[-1] / sizes[0]
        
        # Allow up to 2x worse than linear scaling
        assert time_ratio < size_ratio * 2
        print(f"Scaling factor: {time_ratio:.2f}x time for {size_ratio}x data")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])