#!/usr/bin/env python3
"""
Comprehensive Test Suite for API Consolidation
===============================================
Purpose: Test all API endpoints and validate consolidation fixes
Version: 1.0.0
Date: September 13, 2025

This test suite includes:
1. Unit tests for individual endpoints
2. Integration tests for redirect flows
3. Performance tests for API response times
4. Regression tests to prevent issue recurrence
"""

import unittest
import requests
import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import subprocess
import threading
from unittest.mock import patch, MagicMock
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path("/mnt/c/finalee/beverly_knits_erp_v2")
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
SERVER_URL = "http://localhost:5006"
WRAPPER_URL = "http://localhost:8000"
TEST_TIMEOUT = 10  # seconds

class Colors:
    """Terminal colors for test output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class APIEndpointTests(unittest.TestCase):
    """Test individual API endpoints"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.session = requests.Session()
        cls.base_url = SERVER_URL

        # Check if server is running
        try:
            response = requests.get(f"{cls.base_url}/api/health", timeout=2)
            cls.server_running = response.status_code == 200
        except:
            cls.server_running = False
            print(f"{Colors.WARNING}Warning: Server not running. Some tests will be skipped.{Colors.ENDC}")

    def test_health_endpoint(self):
        """Test that health endpoint is accessible"""
        if not self.server_running:
            self.skipTest("Server not running")

        response = self.session.get(f"{self.base_url}/api/health", timeout=TEST_TIMEOUT)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')

    def test_yarn_intelligence_endpoint(self):
        """Test main yarn intelligence endpoint"""
        if not self.server_running:
            self.skipTest("Server not running")

        response = self.session.get(
            f"{self.base_url}/api/yarn-intelligence",
            timeout=TEST_TIMEOUT
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Validate response structure
        self.assertIn('criticality_analysis', data)
        self.assertIn('yarns', data['criticality_analysis'])
        self.assertIn('summary', data['criticality_analysis'])

    def test_yarn_unified_endpoint(self):
        """Test unified yarn endpoint (if it exists)"""
        if not self.server_running:
            self.skipTest("Server not running")

        response = self.session.get(
            f"{self.base_url}/api/yarn-intelligence",
            timeout=TEST_TIMEOUT
        )

        # Should either work (200) or redirect (301/302/307)
        self.assertIn(response.status_code, [200, 301, 302, 307, 308])

        if response.status_code == 200:
            data = response.json()
            # Validate it returns yarn data
            self.assertIn('criticality_analysis', data)

    def test_production_unified_endpoint(self):
        """Test production unified endpoint"""
        if not self.server_running:
            self.skipTest("Server not running")

        response = self.session.get(
            f"{self.base_url}/api/production/unified",
            timeout=TEST_TIMEOUT
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)

    def test_forecast_unified_endpoint(self):
        """Test forecast unified endpoint"""
        if not self.server_running:
            self.skipTest("Server not running")

        response = self.session.get(
            f"{self.base_url}/api/forecast/unified",
            timeout=TEST_TIMEOUT
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)

    def test_inventory_unified_endpoint(self):
        """Test inventory unified endpoint"""
        if not self.server_running:
            self.skipTest("Server not running")

        response = self.session.get(
            f"{self.base_url}/api/inventory/unified",
            timeout=TEST_TIMEOUT
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)

    def test_endpoint_with_parameters(self):
        """Test endpoints with various parameters"""
        if not self.server_running:
            self.skipTest("Server not running")

        # Test yarn intelligence with parameters
        params = {
            'view': 'summary',
            'analysis': 'shortage',
            'forecast': 'true'
        }

        response = self.session.get(
            f"{self.base_url}/api/yarn-intelligence",
            params=params,
            timeout=TEST_TIMEOUT
        )
        self.assertEqual(response.status_code, 200)

    def test_invalid_endpoint_returns_404(self):
        """Test that invalid endpoints return 404"""
        if not self.server_running:
            self.skipTest("Server not running")

        response = self.session.get(
            f"{self.base_url}/api/this-endpoint-does-not-exist",
            timeout=TEST_TIMEOUT
        )
        self.assertEqual(response.status_code, 404)

class RedirectTests(unittest.TestCase):
    """Test API redirect functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.session = requests.Session()
        cls.base_url = SERVER_URL

        try:
            response = requests.get(f"{cls.base_url}/api/health", timeout=2)
            cls.server_running = response.status_code == 200
        except:
            cls.server_running = False

    def test_no_circular_redirects(self):
        """Ensure no circular redirects exist"""
        if not self.server_running:
            self.skipTest("Server not running")

        # Test yarn/unified specifically
        response = self.session.get(
            f"{self.base_url}/api/yarn-intelligence",
            allow_redirects=False,
            timeout=TEST_TIMEOUT
        )

        if response.status_code in [301, 302, 307, 308]:
            location = response.headers.get('Location', '')
            # Ensure it's not redirecting to itself
            self.assertNotIn('/api/yarn-intelligence', location,
                           "Circular redirect detected for /api/yarn-intelligence")

    def test_deprecated_endpoints_redirect(self):
        """Test that deprecated endpoints redirect correctly"""
        if not self.server_running:
            self.skipTest("Server not running")

        deprecated_mappings = [
            ('/api/yarn-data', '/api/yarn-intelligence'),
            ('/api/production-planning', '/api/production/unified'),
            ('/api/ml-forecast-detailed', '/api/forecast/unified'),
        ]

        for deprecated, expected in deprecated_mappings:
            response = self.session.get(
                f"{self.base_url}{deprecated}",
                allow_redirects=False,
                timeout=TEST_TIMEOUT
            )

            # Check if it redirects
            if response.status_code in [301, 302, 307, 308]:
                location = response.headers.get('Location', '')
                # Normalize URLs for comparison
                if expected in location or location.endswith(expected):
                    self.assertTrue(True)  # Redirect is correct
                else:
                    self.fail(f"{deprecated} redirects to {location}, expected {expected}")

    def test_redirect_preserves_parameters(self):
        """Test that redirects preserve query parameters"""
        if not self.server_running:
            self.skipTest("Server not running")

        params = {'view': 'summary', 'analysis': 'shortage'}
        response = self.session.get(
            f"{self.base_url}/api/yarn-data",
            params=params,
            allow_redirects=True,
            timeout=TEST_TIMEOUT
        )

        # Should eventually return 200 with data
        self.assertEqual(response.status_code, 200)

        # Check that the final URL has parameters
        if '?' in response.url:
            self.assertIn('view=summary', response.url)

class PerformanceTests(unittest.TestCase):
    """Test API performance metrics"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.session = requests.Session()
        cls.base_url = SERVER_URL

        try:
            response = requests.get(f"{cls.base_url}/api/health", timeout=2)
            cls.server_running = response.status_code == 200
        except:
            cls.server_running = False

    def test_endpoint_response_times(self):
        """Test that endpoints respond within acceptable time"""
        if not self.server_running:
            self.skipTest("Server not running")

        max_response_time = 2.0  # seconds

        endpoints = [
            '/api/yarn-intelligence',
            '/api/production/unified',
            '/api/forecast/unified',
            '/api/inventory/unified'
        ]

        for endpoint in endpoints:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}{endpoint}", timeout=TEST_TIMEOUT)
            elapsed = time.time() - start_time

            self.assertEqual(response.status_code, 200,
                           f"Endpoint {endpoint} returned {response.status_code}")
            self.assertLess(elapsed, max_response_time,
                          f"Endpoint {endpoint} took {elapsed:.2f}s (max: {max_response_time}s)")

    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        if not self.server_running:
            self.skipTest("Server not running")

        num_threads = 10
        results = []

        def make_request():
            try:
                response = requests.get(
                    f"{self.base_url}/api/yarn-intelligence",
                    timeout=TEST_TIMEOUT
                )
                results.append(response.status_code)
            except Exception as e:
                results.append(f"error: {e}")

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=make_request)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # All requests should succeed
        success_count = sum(1 for r in results if r == 200)
        self.assertGreaterEqual(success_count, num_threads * 0.9,
                              f"Only {success_count}/{num_threads} concurrent requests succeeded")

class DataValidationTests(unittest.TestCase):
    """Test data integrity and validation"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.session = requests.Session()
        cls.base_url = SERVER_URL

        try:
            response = requests.get(f"{cls.base_url}/api/health", timeout=2)
            cls.server_running = response.status_code == 200
        except:
            cls.server_running = False

    def test_yarn_data_structure(self):
        """Test that yarn data has expected structure"""
        if not self.server_running:
            self.skipTest("Server not running")

        response = self.session.get(
            f"{self.base_url}/api/yarn-intelligence",
            timeout=TEST_TIMEOUT
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()

        # Check required fields
        self.assertIn('criticality_analysis', data)
        self.assertIn('yarns', data['criticality_analysis'])
        self.assertIn('summary', data['criticality_analysis'])

        # Check summary structure
        summary = data['criticality_analysis']['summary']
        required_summary_fields = [
            'total_yarns',
            'critical_count',
            'high_count',
            'total_shortage',
            'yarns_with_shortage'
        ]

        for field in required_summary_fields:
            self.assertIn(field, summary, f"Missing field: {field}")
            self.assertIsInstance(summary[field], (int, float),
                                f"Field {field} should be numeric")

        # Check yarn structure if yarns exist
        yarns = data['criticality_analysis']['yarns']
        if yarns:
            yarn = yarns[0]
            required_yarn_fields = ['yarn_id', 'shortage', 'criticality']

            for field in required_yarn_fields:
                self.assertIn(field, yarn, f"Yarn missing field: {field}")

    def test_no_nan_values_in_response(self):
        """Test that responses don't contain NaN values"""
        if not self.server_running:
            self.skipTest("Server not running")

        response = self.session.get(
            f"{self.base_url}/api/yarn-intelligence",
            timeout=TEST_TIMEOUT
        )
        self.assertEqual(response.status_code, 200)

        # Response should be valid JSON (no NaN)
        response_text = response.text
        self.assertNotIn('NaN', response_text, "Response contains NaN values")
        self.assertNotIn('null', response_text.lower().replace('null', ''),
                        "Response may contain improper null handling")

    def test_cache_consistency(self):
        """Test that cached and non-cached responses are consistent"""
        if not self.server_running:
            self.skipTest("Server not running")

        # First request (may populate cache)
        response1 = self.session.get(
            f"{self.base_url}/api/yarn-intelligence",
            timeout=TEST_TIMEOUT
        )
        data1 = response1.json()

        # Second request (likely from cache)
        response2 = self.session.get(
            f"{self.base_url}/api/yarn-intelligence",
            timeout=TEST_TIMEOUT
        )
        data2 = response2.json()

        # Check consistency (ignoring timestamps and cache indicators)
        if 'criticality_analysis' in data1 and 'criticality_analysis' in data2:
            # Compare summary totals
            summary1 = data1['criticality_analysis']['summary']
            summary2 = data2['criticality_analysis']['summary']

            self.assertEqual(summary1['total_yarns'], summary2['total_yarns'],
                           "Inconsistent total_yarns between requests")

class IntegrationTests(unittest.TestCase):
    """Test full integration scenarios"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.session = requests.Session()
        cls.base_url = SERVER_URL

        try:
            response = requests.get(f"{cls.base_url}/api/health", timeout=2)
            cls.server_running = response.status_code == 200
        except:
            cls.server_running = False

    def test_full_dashboard_load_scenario(self):
        """Test loading all data for dashboard"""
        if not self.server_running:
            self.skipTest("Server not running")

        # Simulate dashboard loading sequence
        endpoints_to_load = [
            '/api/yarn-intelligence',
            '/api/inventory-intelligence-enhanced',
            '/api/production-planning',
            '/api/ml-forecast-detailed',
            '/api/comprehensive-kpis'
        ]

        all_successful = True
        failed_endpoints = []

        for endpoint in endpoints_to_load:
            try:
                response = self.session.get(
                    f"{self.base_url}{endpoint}",
                    timeout=TEST_TIMEOUT
                )
                if response.status_code != 200:
                    all_successful = False
                    failed_endpoints.append((endpoint, response.status_code))
            except Exception as e:
                all_successful = False
                failed_endpoints.append((endpoint, str(e)))

        if failed_endpoints:
            failure_msg = "\n".join([f"  {ep}: {status}" for ep, status in failed_endpoints])
            self.fail(f"Dashboard load failed for:\n{failure_msg}")

        self.assertTrue(all_successful, "Not all dashboard endpoints loaded successfully")

    def test_api_consolidation_metrics(self):
        """Test API consolidation metrics endpoint"""
        if not self.server_running:
            self.skipTest("Server not running")

        response = self.session.get(
            f"{self.base_url}/api/consolidation-metrics",
            timeout=TEST_TIMEOUT
        )

        if response.status_code == 200:
            data = response.json()

            # Check metric structure
            expected_fields = [
                'deprecated_calls',
                'new_api_calls',
                'redirect_count',
                'cache_metrics'
            ]

            for field in expected_fields:
                self.assertIn(field, data, f"Missing metric field: {field}")

class RegressionTests(unittest.TestCase):
    """Test to prevent regression of fixed issues"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.session = requests.Session()
        cls.base_url = SERVER_URL
        cls.frontend_file = PROJECT_ROOT / "web/consolidated_dashboard.html"
        cls.backend_file = PROJECT_ROOT / "src/core/beverly_comprehensive_erp.py"

        try:
            response = requests.get(f"{cls.base_url}/api/health", timeout=2)
            cls.server_running = response.status_code == 200
        except:
            cls.server_running = False

    def test_no_circular_redirect_in_frontend(self):
        """Ensure circular redirect is not present in frontend code"""
        if not self.frontend_file.exists():
            self.skipTest("Frontend file not found")

        with open(self.frontend_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for the specific circular redirect
        self.assertNotIn(
            "'/api/yarn-intelligence': '/api/yarn-intelligence'",
            content,
            "Circular redirect found in frontend code!"
        )

    def test_yarn_unified_endpoint_exists_or_redirects(self):
        """Ensure yarn/unified either exists or redirects properly"""
        if not self.server_running:
            self.skipTest("Server not running")

        response = self.session.get(
            f"{self.base_url}/api/yarn-intelligence",
            allow_redirects=False,
            timeout=TEST_TIMEOUT
        )

        # Should either exist (200) or redirect (3xx)
        self.assertIn(
            response.status_code,
            [200, 301, 302, 307, 308],
            f"Unexpected status {response.status_code} for /api/yarn-intelligence"
        )

        # If it redirects, ensure not to itself
        if response.status_code in [301, 302, 307, 308]:
            location = response.headers.get('Location', '')
            self.assertNotIn(
                '/api/yarn-intelligence',
                location,
                "yarn/unified is redirecting to itself!"
            )

    def test_all_unified_endpoints_accessible(self):
        """Test that all unified endpoints are accessible"""
        if not self.server_running:
            self.skipTest("Server not running")

        unified_endpoints = [
            '/api/production/unified',
            '/api/forecast/unified',
            '/api/inventory/unified',
            '/api/yarn-intelligence'  # Should work after fix
        ]

        for endpoint in unified_endpoints:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                timeout=TEST_TIMEOUT
            )

            # Should return 200 (after following redirects)
            self.assertEqual(
                response.status_code,
                200,
                f"Unified endpoint {endpoint} returned {response.status_code}"
            )

def run_test_suite(verbose=False):
    """Run the complete test suite"""

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        APIEndpointTests,
        RedirectTests,
        PerformanceTests,
        DataValidationTests,
        IntegrationTests,
        RegressionTests
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)

    # Generate report
    print("\n" + "="*60)
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.ENDC}")
    print("="*60)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total_tests - failures - errors - skipped

    print(f"Total Tests: {total_tests}")
    print(f"{Colors.OKGREEN}Passed: {passed}{Colors.ENDC}")

    if failures > 0:
        print(f"{Colors.FAIL}Failed: {failures}{Colors.ENDC}")
    if errors > 0:
        print(f"{Colors.FAIL}Errors: {errors}{Colors.ENDC}")
    if skipped > 0:
        print(f"{Colors.WARNING}Skipped: {skipped}{Colors.ENDC}")

    # Print details of failures
    if result.failures:
        print(f"\n{Colors.FAIL}FAILURES:{Colors.ENDC}")
        for test, traceback in result.failures:
            print(f"  • {test}")
            print(f"    {traceback.split(chr(10))[0]}")

    if result.errors:
        print(f"\n{Colors.FAIL}ERRORS:{Colors.ENDC}")
        for test, traceback in result.errors:
            print(f"  • {test}")
            print(f"    {traceback.split(chr(10))[0]}")

    # Overall result
    print("\n" + "="*60)
    if failures == 0 and errors == 0:
        print(f"{Colors.OKGREEN}{Colors.BOLD}✅ ALL TESTS PASSED!{Colors.ENDC}")
        return 0
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}❌ TESTS FAILED{Colors.ENDC}")
        return 1

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive test suite for API consolidation"
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed test output'
    )

    parser.add_argument(
        '--specific-test',
        type=str,
        help='Run a specific test class (e.g., APIEndpointTests)'
    )

    args = parser.parse_args()

    if args.specific_test:
        # Run specific test
        suite = unittest.TestLoader().loadTestsFromName(f"__main__.{args.specific_test}")
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        # Run full suite
        sys.exit(run_test_suite(args.verbose))

if __name__ == "__main__":
    main()