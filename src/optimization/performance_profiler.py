#!/usr/bin/env python3
"""
Beverly Knits ERP - Performance Profiler
Profiles API endpoints, identifies bottlenecks, and provides optimization recommendations
Part of Phase 3: Performance Optimization
"""

import time
import logging
import json
import cProfile
import pstats
import io
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import wraps
import requests
import statistics
import concurrent.futures
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EndpointProfile:
    """Profile data for a single endpoint"""
    endpoint: str
    method: str
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    success_rate: float
    samples: int
    timestamp: str
    bottlenecks: List[Dict[str, Any]] = None


class PerformanceProfiler:
    """
    Comprehensive performance profiler for Beverly Knits ERP
    Identifies bottlenecks and provides optimization recommendations
    """
    
    # Performance thresholds
    EXCELLENT_THRESHOLD_MS = 50
    GOOD_THRESHOLD_MS = 200
    ACCEPTABLE_THRESHOLD_MS = 500
    
    def __init__(self, base_url: str = "http://localhost:5005"):
        """
        Initialize performance profiler
        
        Args:
            base_url: Base URL of the API to profile
        """
        self.base_url = base_url
        self.profiles = {}
        self.recommendations = []
        self.session = requests.Session()
        
        logger.info(f"PerformanceProfiler initialized for {base_url}")
    
    def profile_endpoint(self, 
                        endpoint: str, 
                        method: str = "GET", 
                        samples: int = 10,
                        payload: Optional[Dict] = None) -> EndpointProfile:
        """
        Profile a single API endpoint
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            samples: Number of requests to sample
            payload: Optional request payload
            
        Returns:
            EndpointProfile with performance metrics
        """
        url = f"{self.base_url}{endpoint}"
        response_times = []
        errors = 0
        
        logger.info(f"Profiling {method} {endpoint} with {samples} samples...")
        
        for i in range(samples):
            try:
                start_time = time.perf_counter()
                
                if method == "GET":
                    response = self.session.get(url, timeout=30)
                elif method == "POST":
                    response = self.session.post(url, json=payload, timeout=30)
                else:
                    response = self.session.request(method, url, json=payload, timeout=30)
                
                end_time = time.perf_counter()
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                if response.status_code < 400:
                    response_times.append(response_time)
                else:
                    errors += 1
                    logger.warning(f"Error response from {endpoint}: {response.status_code}")
                
                # Small delay between requests
                if i < samples - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                errors += 1
                logger.error(f"Error profiling {endpoint}: {e}")
        
        if not response_times:
            logger.error(f"No successful responses from {endpoint}")
            return None
        
        # Calculate statistics
        response_times.sort()
        profile = EndpointProfile(
            endpoint=endpoint,
            method=method,
            avg_response_time=statistics.mean(response_times),
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            p50_response_time=response_times[len(response_times) // 2],
            p95_response_time=response_times[int(len(response_times) * 0.95)] if len(response_times) > 1 else response_times[0],
            p99_response_time=response_times[int(len(response_times) * 0.99)] if len(response_times) > 1 else response_times[0],
            success_rate=(samples - errors) / samples * 100,
            samples=samples,
            timestamp=datetime.now().isoformat()
        )
        
        # Store profile
        self.profiles[f"{method}_{endpoint}"] = profile
        
        # Generate recommendations
        self._generate_recommendations(profile)
        
        return profile
    
    def profile_all_endpoints(self) -> Dict[str, EndpointProfile]:
        """
        Profile all known API endpoints
        
        Returns:
            Dictionary of endpoint profiles
        """
        endpoints = [
            # Core endpoints
            ("/api/health", "GET"),
            ("/api/debug-data", "GET"),
            ("/api/reload-data", "GET"),
            
            # Yarn intelligence endpoints
            ("/api/yarn-intelligence", "GET"),
            ("/api/inventory-intelligence-enhanced", "GET"),
            ("/api/yarn-aggregation", "GET"),
            ("/api/yarn-substitution-intelligent", "GET"),
            ("/api/yarn-forecast-shortages", "GET"),
            
            # Planning endpoints
            ("/api/six-phase-planning", "GET"),
            ("/api/production-pipeline", "GET"),
            ("/api/production-planning", "GET"),
            
            # ML endpoints
            ("/api/ml-forecast-report", "GET"),
            ("/api/ml-forecast-detailed", "GET"),
            
            # Cache endpoints
            ("/api/cache-stats", "GET"),
        ]
        
        logger.info(f"Profiling {len(endpoints)} endpoints...")
        
        for endpoint, method in endpoints:
            try:
                self.profile_endpoint(endpoint, method, samples=5)
            except Exception as e:
                logger.error(f"Failed to profile {endpoint}: {e}")
        
        return self.profiles
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks across all profiled endpoints
        
        Returns:
            List of bottleneck descriptions
        """
        bottlenecks = []
        
        for key, profile in self.profiles.items():
            # Check response time thresholds
            if profile.p95_response_time > self.ACCEPTABLE_THRESHOLD_MS:
                bottlenecks.append({
                    'endpoint': profile.endpoint,
                    'type': 'SLOW_RESPONSE',
                    'severity': 'CRITICAL',
                    'p95_time': profile.p95_response_time,
                    'threshold': self.ACCEPTABLE_THRESHOLD_MS,
                    'recommendation': 'Optimize query/processing logic'
                })
            elif profile.p95_response_time > self.GOOD_THRESHOLD_MS:
                bottlenecks.append({
                    'endpoint': profile.endpoint,
                    'type': 'MODERATE_RESPONSE',
                    'severity': 'WARNING',
                    'p95_time': profile.p95_response_time,
                    'threshold': self.GOOD_THRESHOLD_MS,
                    'recommendation': 'Consider caching or optimization'
                })
            
            # Check success rate
            if profile.success_rate < 95:
                bottlenecks.append({
                    'endpoint': profile.endpoint,
                    'type': 'LOW_SUCCESS_RATE',
                    'severity': 'CRITICAL',
                    'success_rate': profile.success_rate,
                    'recommendation': 'Fix errors and improve error handling'
                })
            
            # Check variance
            variance = profile.max_response_time - profile.min_response_time
            if variance > profile.avg_response_time * 2:
                bottlenecks.append({
                    'endpoint': profile.endpoint,
                    'type': 'HIGH_VARIANCE',
                    'severity': 'WARNING',
                    'variance': variance,
                    'recommendation': 'Investigate inconsistent performance'
                })
        
        return bottlenecks
    
    def _generate_recommendations(self, profile: EndpointProfile):
        """
        Generate optimization recommendations for an endpoint
        
        Args:
            profile: EndpointProfile to analyze
        """
        recommendations = []
        
        # Response time recommendations
        if profile.avg_response_time > self.ACCEPTABLE_THRESHOLD_MS:
            recommendations.append({
                'endpoint': profile.endpoint,
                'priority': 'HIGH',
                'issue': f'Average response time {profile.avg_response_time:.0f}ms exceeds acceptable threshold',
                'suggestions': [
                    'Add caching layer',
                    'Optimize database queries',
                    'Implement pagination',
                    'Use async processing'
                ]
            })
        elif profile.avg_response_time > self.GOOD_THRESHOLD_MS:
            recommendations.append({
                'endpoint': profile.endpoint,
                'priority': 'MEDIUM',
                'issue': f'Average response time {profile.avg_response_time:.0f}ms could be improved',
                'suggestions': [
                    'Consider response caching',
                    'Review query optimization',
                    'Implement partial responses'
                ]
            })
        
        # Success rate recommendations
        if profile.success_rate < 100:
            recommendations.append({
                'endpoint': profile.endpoint,
                'priority': 'HIGH',
                'issue': f'Success rate {profile.success_rate:.1f}% indicates errors',
                'suggestions': [
                    'Add error handling',
                    'Implement retry logic',
                    'Review timeout settings',
                    'Add circuit breaker pattern'
                ]
            })
        
        # Variance recommendations
        if profile.max_response_time > profile.min_response_time * 5:
            recommendations.append({
                'endpoint': profile.endpoint,
                'priority': 'MEDIUM',
                'issue': 'High response time variance detected',
                'suggestions': [
                    'Investigate cache misses',
                    'Check for resource contention',
                    'Review garbage collection impact',
                    'Consider connection pooling'
                ]
            })
        
        self.recommendations.extend(recommendations)
    
    def run_load_test(self, 
                      endpoint: str, 
                      concurrent_users: int = 10,
                      duration_seconds: int = 30) -> Dict[str, Any]:
        """
        Run a load test on an endpoint
        
        Args:
            endpoint: API endpoint to test
            concurrent_users: Number of concurrent users
            duration_seconds: Test duration
            
        Returns:
            Load test results
        """
        url = f"{self.base_url}{endpoint}"
        results = {
            'endpoint': endpoint,
            'concurrent_users': concurrent_users,
            'duration': duration_seconds,
            'requests': [],
            'errors': 0,
            'start_time': datetime.now().isoformat()
        }
        
        logger.info(f"Running load test on {endpoint} with {concurrent_users} users for {duration_seconds}s...")
        
        def make_request():
            try:
                start = time.perf_counter()
                response = requests.get(url, timeout=10)
                duration = (time.perf_counter() - start) * 1000
                return {
                    'duration_ms': duration,
                    'status_code': response.status_code,
                    'success': response.status_code < 400
                }
            except Exception as e:
                return {
                    'duration_ms': 10000,  # Timeout
                    'error': str(e),
                    'success': False
                }
        
        # Run load test
        start_time = time.time()
        request_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            
            while time.time() - start_time < duration_seconds:
                futures.append(executor.submit(make_request))
                request_count += 1
                time.sleep(0.1)  # Small delay between spawning requests
            
            # Wait for all requests to complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results['requests'].append(result)
                if not result['success']:
                    results['errors'] += 1
        
        # Calculate statistics
        successful_requests = [r for r in results['requests'] if r['success']]
        if successful_requests:
            response_times = [r['duration_ms'] for r in successful_requests]
            results['stats'] = {
                'total_requests': len(results['requests']),
                'successful_requests': len(successful_requests),
                'failed_requests': results['errors'],
                'avg_response_time': statistics.mean(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'p95_response_time': sorted(response_times)[int(len(response_times) * 0.95)],
                'requests_per_second': len(results['requests']) / duration_seconds,
                'success_rate': len(successful_requests) / len(results['requests']) * 100
            }
        
        logger.info(f"Load test complete. Processed {len(results['requests'])} requests.")
        
        return results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Returns:
            Performance report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'base_url': self.base_url,
            'endpoints_profiled': len(self.profiles),
            'profiles': {},
            'bottlenecks': self.identify_bottlenecks(),
            'recommendations': self.recommendations,
            'summary': {}
        }
        
        # Add profile data
        for key, profile in self.profiles.items():
            report['profiles'][key] = asdict(profile)
        
        # Calculate summary statistics
        if self.profiles:
            avg_times = [p.avg_response_time for p in self.profiles.values()]
            p95_times = [p.p95_response_time for p in self.profiles.values()]
            success_rates = [p.success_rate for p in self.profiles.values()]
            
            report['summary'] = {
                'avg_response_time': statistics.mean(avg_times),
                'avg_p95_time': statistics.mean(p95_times),
                'avg_success_rate': statistics.mean(success_rates),
                'slowest_endpoint': max(self.profiles.values(), key=lambda p: p.avg_response_time).endpoint,
                'fastest_endpoint': min(self.profiles.values(), key=lambda p: p.avg_response_time).endpoint,
                'total_bottlenecks': len(report['bottlenecks']),
                'critical_bottlenecks': sum(1 for b in report['bottlenecks'] if b['severity'] == 'CRITICAL')
            }
        
        # Performance rating
        avg_time = report['summary'].get('avg_response_time', 0)
        if avg_time < self.EXCELLENT_THRESHOLD_MS:
            report['summary']['rating'] = 'EXCELLENT'
        elif avg_time < self.GOOD_THRESHOLD_MS:
            report['summary']['rating'] = 'GOOD'
        elif avg_time < self.ACCEPTABLE_THRESHOLD_MS:
            report['summary']['rating'] = 'ACCEPTABLE'
        else:
            report['summary']['rating'] = 'NEEDS_IMPROVEMENT'
        
        return report
    
    def save_report(self, filename: str = None):
        """
        Save performance report to file
        
        Args:
            filename: Output filename (default: performance_report_<timestamp>.json)
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        report = self.generate_performance_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {filename}")
        
        return filename


def profile_function(func):
    """
    Decorator to profile function execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        logger.debug(f"{func.__name__} executed in {execution_time:.2f}ms")
        
        return result
    
    return wrapper


def detailed_profile(func):
    """
    Decorator for detailed profiling using cProfile
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        # Print profiling results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        
        logger.info(f"Detailed profile for {func.__name__}:\n{s.getvalue()}")
        
        return result
    
    return wrapper


def test_performance_profiler():
    """Test the performance profiler"""
    print("=" * 80)
    print("Testing PerformanceProfiler")
    print("=" * 80)
    
    profiler = PerformanceProfiler()
    
    # Test 1: Profile health endpoint
    print("\n1. Profiling Health Endpoint:")
    try:
        profile = profiler.profile_endpoint("/api/health", samples=3)
        if profile:
            print(f"  Endpoint: {profile.endpoint}")
            print(f"  Avg Response Time: {profile.avg_response_time:.2f}ms")
            print(f"  P95 Response Time: {profile.p95_response_time:.2f}ms")
            print(f"  Success Rate: {profile.success_rate:.1f}%")
        else:
            print("  Failed to profile endpoint (server may not be running)")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Note: Server must be running on port 5005 for live testing")
    
    # Test 2: Identify bottlenecks
    print("\n2. Bottleneck Analysis:")
    # Create mock profile for testing
    mock_profile = EndpointProfile(
        endpoint="/api/slow-endpoint",
        method="GET",
        avg_response_time=600,
        min_response_time=500,
        max_response_time=800,
        p50_response_time=600,
        p95_response_time=750,
        p99_response_time=800,
        success_rate=90,
        samples=10,
        timestamp=datetime.now().isoformat()
    )
    profiler.profiles["GET_/api/slow-endpoint"] = mock_profile
    
    bottlenecks = profiler.identify_bottlenecks()
    print(f"  Found {len(bottlenecks)} bottlenecks")
    for bottleneck in bottlenecks:
        print(f"    - {bottleneck['type']}: {bottleneck['endpoint']} ({bottleneck['severity']})")
    
    # Test 3: Generate report
    print("\n3. Performance Report:")
    report = profiler.generate_performance_report()
    print(f"  Endpoints Profiled: {report['endpoints_profiled']}")
    print(f"  Total Bottlenecks: {len(report['bottlenecks'])}")
    if report['summary']:
        print(f"  Performance Rating: {report['summary'].get('rating', 'N/A')}")
    
    # Test 4: Save report
    print("\n4. Saving Report:")
    filename = profiler.save_report("test_performance_report.json")
    print(f"  Report saved to: {filename}")
    
    # Clean up test file
    Path(filename).unlink(missing_ok=True)
    
    print("\n" + "=" * 80)
    print("✅ PerformanceProfiler test complete")


if __name__ == "__main__":
    test_performance_profiler()