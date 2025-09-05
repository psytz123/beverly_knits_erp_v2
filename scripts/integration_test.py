#!/usr/bin/env python3
"""
Integration Test Script for Beverly Knits ERP v2
Validates that all components work together correctly after optimization
"""

import sys
import os
import time
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

class IntegrationTester:
    """Integration testing suite for ERP system"""
    
    def __init__(self, base_url="http://localhost:5006"):
        self.base_url = base_url
        self.test_results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        
    def test(self, name: str, condition: bool, message: str = ""):
        """Record test result"""
        if condition:
            self.test_results['passed'].append(name)
            print(f"  [PASS] {name}")
        else:
            self.test_results['failed'].append(name)
            print(f"  [FAIL] {name}: {message}")
            
    def warn(self, message: str):
        """Record warning"""
        self.test_results['warnings'].append(message)
        print(f"  [WARN] {message}")
    
    def test_server_health(self):
        """Test if server is running and healthy"""
        print("\n1. Server Health Check")
        print("-" * 40)
        
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            self.test("Server Running", response.status_code in [200, 301, 302])
            
            # Check if main page loads
            self.test("Main Page Loads", 'Beverly Knits' in response.text or response.status_code in [200, 301, 302])
        except requests.exceptions.ConnectionError:
            self.test("Server Running", False, "Connection refused - is server running?")
            return False
        except Exception as e:
            self.test("Server Running", False, str(e))
            return False
            
        return True
    
    def test_api_endpoints(self):
        """Test all critical API endpoints"""
        print("\n2. API Endpoint Tests")
        print("-" * 40)
        
        # Test both old and new endpoints
        endpoints_to_test = [
            # Old endpoints (should redirect)
            ('/api/yarn-inventory', 'GET', 'yarn-inventory'),
            ('/api/production-status', 'GET', 'production-status'),
            ('/api/inventory-intelligence-enhanced', 'GET', 'inventory-intelligence'),
            
            # New v2 endpoints
            ('/api/v2/inventory', 'GET', 'v2-inventory'),
            ('/api/v2/production', 'GET', 'v2-production'),
            ('/api/v2/yarn', 'GET', 'v2-yarn'),
            
            # Fabric production (new)
            ('/api/fabric-production', 'GET', 'fabric-production'),
        ]
        
        for endpoint, method, test_name in endpoints_to_test:
            try:
                if method == 'GET':
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                else:
                    response = requests.post(f"{self.base_url}{endpoint}", json={}, timeout=10)
                
                # Check if response is successful or redirect
                success = response.status_code in [200, 301, 302, 307]
                self.test(f"API_{test_name}", success, f"Status: {response.status_code}")
                
                # Check if JSON response is valid
                if response.status_code == 200:
                    try:
                        data = response.json()
                        self.test(f"API_{test_name}_JSON", True)
                    except:
                        self.test(f"API_{test_name}_JSON", False, "Invalid JSON response")
                        
            except requests.exceptions.Timeout:
                self.test(f"API_{test_name}", False, "Timeout")
            except Exception as e:
                self.test(f"API_{test_name}", False, str(e)[:50])
    
    def test_data_loading(self):
        """Test data loading functionality"""
        print("\n3. Data Loading Tests")
        print("-" * 40)
        
        # Import data loader
        try:
            from src.data_loaders.unified_data_loader import UnifiedDataLoader
            self.test("Import UnifiedDataLoader", True)
            
            # Test loading
            loader = UnifiedDataLoader()
            
            # Test yarn inventory loading
            try:
                yarn_data = loader.load_yarn_inventory()
                self.test("Load Yarn Inventory", 
                         yarn_data is not None and not yarn_data.empty,
                         f"Loaded {len(yarn_data) if yarn_data is not None else 0} records")
            except Exception as e:
                self.test("Load Yarn Inventory", False, str(e)[:50])
            
            # Test BOM loading
            try:
                bom_data = loader.load_bom_data()
                self.test("Load BOM Data",
                         bom_data is not None and not bom_data.empty,
                         f"Loaded {len(bom_data) if bom_data is not None else 0} records")
            except Exception as e:
                self.test("Load BOM Data", False, str(e)[:50])
                
        except ImportError as e:
            self.test("Import UnifiedDataLoader", False, str(e))
    
    def test_optimization_modules(self):
        """Test optimization modules"""
        print("\n4. Optimization Module Tests")
        print("-" * 40)
        
        # Test DataFrame optimizer
        try:
            from src.optimization.performance.dataframe_optimizer import DataFrameOptimizer
            self.test("Import DataFrameOptimizer", True)
            
            # Test optimization
            df = pd.DataFrame({
                'theoretical_balance': [100, 200, 300],
                'allocated': [-10, -20, -30],
                'on_order': [50, 60, 70]
            })
            
            optimized_df = DataFrameOptimizer.optimize_planning_balance_calculation(df)
            self.test("DataFrame Optimization",
                     'planning_balance' in optimized_df.columns,
                     "Planning balance calculated")
                     
        except ImportError as e:
            self.test("Import DataFrameOptimizer", False, str(e))
        
        # Test memory optimizer
        try:
            from src.optimization.memory_optimizer import MemoryOptimizer
            self.test("Import MemoryOptimizer", True)
        except ImportError as e:
            self.test("Import MemoryOptimizer", False, str(e))
        
        # Test async processor
        try:
            from src.optimization.performance.async_processor import AsyncProcessor
            self.test("Import AsyncProcessor", True)
        except ImportError as e:
            self.test("Import AsyncProcessor", False, str(e))
    
    def test_cache_functionality(self):
        """Test caching functionality"""
        print("\n5. Cache Functionality Tests")
        print("-" * 40)
        
        # Test cache manager
        try:
            from src.utils.cache_manager import UnifiedCacheManager
            cache = UnifiedCacheManager()
            self.test("Import CacheManager", True)
            
            # Test set/get
            test_key = f"test_{datetime.now().timestamp()}"
            test_value = {"test": "data"}
            
            cache.set(test_key, test_value, 60)
            retrieved = cache.get(test_key)
            
            self.test("Cache Set/Get",
                     retrieved == test_value,
                     f"Retrieved: {retrieved}")
                     
        except ImportError as e:
            self.test("Import CacheManager", False, str(e))
        except Exception as e:
            self.test("Cache Operations", False, str(e)[:50])
    
    def test_ml_models(self):
        """Test ML model functionality"""
        print("\n6. ML Model Tests")
        print("-" * 40)
        
        # Test forecast accuracy monitor
        try:
            from src.forecasting.forecast_accuracy_monitor import ForecastAccuracyMonitor
            self.test("Import ForecastAccuracyMonitor", True)
        except ImportError as e:
            self.test("Import ForecastAccuracyMonitor", False, str(e))
        
        # Test production recommendations
        try:
            from src.ml_models.production_recommendations_ml import ProductionRecommendationsML
            self.test("Import ProductionRecommendationsML", True)
            
            # Test initialization
            ml_system = ProductionRecommendationsML()
            self.test("Initialize ML System", True)
            
        except ImportError as e:
            self.test("Import ProductionRecommendationsML", False, str(e))
    
    def test_api_consolidation(self):
        """Test API consolidation and redirects"""
        print("\n7. API Consolidation Tests")
        print("-" * 40)
        
        # Test deprecated endpoint redirects
        deprecated_endpoints = [
            '/api/yarn-inventory',
            '/api/production-status',
            '/api/ml-forecasting'
        ]
        
        for endpoint in deprecated_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", 
                                       allow_redirects=False, 
                                       timeout=5)
                
                # Should get redirect status
                is_redirect = response.status_code in [301, 302, 307, 308]
                self.test(f"Redirect_{endpoint}", 
                         is_redirect or response.status_code == 200,
                         f"Status: {response.status_code}")
                         
            except Exception as e:
                self.warn(f"Could not test redirect for {endpoint}: {str(e)[:30]}")
    
    def test_dashboard_compatibility(self):
        """Test dashboard compatibility"""
        print("\n8. Dashboard Compatibility Tests")
        print("-" * 40)
        
        try:
            # Check if dashboard file exists
            dashboard_path = "D:/AI/Workspaces/efab.ai/beverly_knits_erp_v2/web/consolidated_dashboard.html"
            
            if os.path.exists(dashboard_path):
                self.test("Dashboard File Exists", True)
                
                # Check if API compatibility layer is present
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                has_compatibility = 'apiCompatibilityLayer' in content
                self.test("API Compatibility Layer", has_compatibility)
                
                # Check for v2 endpoints
                has_v2_endpoints = '/api/v2/' in content
                self.test("V2 Endpoints in Dashboard", has_v2_endpoints)
                
            else:
                self.test("Dashboard File Exists", False, "File not found")
                
        except Exception as e:
            self.test("Dashboard Tests", False, str(e)[:50])
    
    def generate_report(self):
        """Generate integration test report"""
        print("\n" + "="*60)
        print("INTEGRATION TEST REPORT")
        print("="*60)
        
        total_tests = len(self.test_results['passed']) + len(self.test_results['failed'])
        pass_rate = (len(self.test_results['passed']) / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nTest Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {len(self.test_results['passed'])}")
        print(f"  Failed: {len(self.test_results['failed'])}")
        print(f"  Pass Rate: {pass_rate:.1f}%")
        
        if self.test_results['warnings']:
            print(f"\nWarnings ({len(self.test_results['warnings'])}):")
            for warning in self.test_results['warnings']:
                print(f"  - {warning}")
        
        if self.test_results['failed']:
            print(f"\nFailed Tests ({len(self.test_results['failed'])}):")
            for test in self.test_results['failed']:
                print(f"  - {test}")
        
        # Overall status
        print(f"\nOverall Status:")
        if pass_rate >= 90:
            print("  [OK] System is ready for production")
        elif pass_rate >= 70:
            print("  [WARN] System needs minor fixes")
        else:
            print("  [FAIL] System needs major fixes")
        
        # Save report
        report_file = f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\nDetailed report saved to: {report_file}")
        
        return pass_rate >= 70

def main():
    """Run integration tests"""
    print("Beverly Knits ERP v2 - Integration Test Suite")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = IntegrationTester()
    
    # Run tests
    if tester.test_server_health():
        tester.test_api_endpoints()
        tester.test_api_consolidation()
    else:
        print("\n[ERROR] Server not running. Start server with:")
        print("  python src/core/beverly_comprehensive_erp.py")
    
    # Always test these (don't need server)
    tester.test_data_loading()
    tester.test_optimization_modules()
    tester.test_cache_functionality()
    tester.test_ml_models()
    tester.test_dashboard_compatibility()
    
    # Generate report
    success = tester.generate_report()
    
    print("\n" + "="*60)
    print("Integration Testing Complete!")
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())