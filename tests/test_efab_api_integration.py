#!/usr/bin/env python3
"""
eFab API Integration Test Script
Test connectivity and basic functionality of eFab API integration
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import required modules
try:
    from src.api_clients.efab_api_client import EFabAPIClient
    from src.api_clients.efab_auth_manager import EFabAuthManager
    from src.api_clients.efab_transformers import EFabDataTransformer
    from src.data_loaders.efab_api_loader import EFabAPIDataLoader
    from src.config.secure_api_config import EFabAPIConfig
    from src.config.feature_flags import (
        is_efab_api_enabled,
        enable_efab_api,
        disable_efab_api,
        set_efab_rollout,
        get_efab_rollout_percentage
    )
    API_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing API modules: {e}")
    API_MODULES_AVAILABLE = False

class EFabAPITester:
    """Test suite for eFab API integration"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }
        self.client = None
        self.transformer = None
        self.loader = None
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍"
        }.get(level, "•")
        print(f"[{timestamp}] {prefix} {message}")
        
    def record_test(self, test_name: str, success: bool, details: str = "", data: Any = None):
        """Record test result"""
        self.results["tests"][test_name] = {
            "success": success,
            "details": details,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        self.results["summary"]["total"] += 1
        if success:
            self.results["summary"]["passed"] += 1
        else:
            self.results["summary"]["failed"] += 1
            
    async def test_environment_setup(self) -> bool:
        """Test 1: Verify environment configuration"""
        self.log("Testing environment setup...", "INFO")
        
        required_vars = [
            "EFAB_BASE_URL",
            "EFAB_USERNAME",
            "EFAB_PASSWORD"
        ]
        
        missing = []
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing.append(var)
            else:
                self.log(f"  {var}: {'*' * 8} (set)", "DEBUG")
                
        if missing:
            self.record_test("Environment Setup", False, f"Missing: {', '.join(missing)}")
            self.log(f"Missing environment variables: {', '.join(missing)}", "ERROR")
            return False
        else:
            self.record_test("Environment Setup", True, "All required variables present")
            self.log("Environment setup complete", "SUCCESS")
            return True
            
    async def test_api_modules(self) -> bool:
        """Test 2: Verify API modules are available"""
        self.log("Testing API module availability...", "INFO")
        
        if not API_MODULES_AVAILABLE:
            self.record_test("API Modules", False, "Import failed")
            self.log("API modules not available", "ERROR")
            return False
            
        modules = {
            "EFabAPIClient": EFabAPIClient,
            "EFabAuthManager": EFabAuthManager,
            "EFabDataTransformer": EFabDataTransformer,
            "EFabAPIDataLoader": EFabAPIDataLoader,
            "EFabAPIConfig": EFabAPIConfig
        }
        
        for name, module in modules.items():
            if module:
                self.log(f"  ✓ {name}", "DEBUG")
            else:
                self.log(f"  ✗ {name} missing", "ERROR")
                self.record_test("API Modules", False, f"{name} not available")
                return False
                
        self.record_test("API Modules", True, "All modules available")
        self.log("All API modules available", "SUCCESS")
        return True
        
    async def test_feature_flags(self) -> bool:
        """Test 3: Test feature flag configuration"""
        self.log("Testing feature flag configuration...", "INFO")
        
        # Check current state
        api_enabled = is_efab_api_enabled()
        rollout_pct = get_efab_rollout_percentage()
        
        self.log(f"  API Enabled: {api_enabled}", "DEBUG")
        self.log(f"  Rollout %: {rollout_pct}", "DEBUG")
        
        # Test enable/disable
        original_state = api_enabled
        
        # Test enable
        enable_efab_api()
        if not is_efab_api_enabled():
            self.record_test("Feature Flags", False, "Enable failed")
            return False
            
        # Test disable
        disable_efab_api()
        if is_efab_api_enabled():
            self.record_test("Feature Flags", False, "Disable failed")
            return False
            
        # Test rollout percentage
        set_efab_rollout(50)
        if get_efab_rollout_percentage() != 50:
            self.record_test("Feature Flags", False, "Rollout percentage failed")
            return False
            
        # Restore original state
        if original_state:
            enable_efab_api()
        else:
            disable_efab_api()
            
        self.record_test("Feature Flags", True, "Feature flags working correctly")
        self.log("Feature flags configured successfully", "SUCCESS")
        return True
        
    async def test_api_client_initialization(self) -> bool:
        """Test 4: Initialize API client"""
        self.log("Initializing API client...", "INFO")
        
        try:
            config = EFabAPIConfig.get_credentials()
            if not config or not config.get('base_url'):
                self.record_test("API Client Init", False, "Invalid configuration")
                self.log("Invalid API configuration", "ERROR")
                return False
                
            self.client = EFabAPIClient(config)
            self.transformer = EFabDataTransformer()
            
            self.record_test("API Client Init", True, "Client initialized")
            self.log("API client initialized successfully", "SUCCESS")
            return True
            
        except Exception as e:
            self.record_test("API Client Init", False, str(e))
            self.log(f"Failed to initialize API client: {e}", "ERROR")
            return False
            
    async def test_authentication(self) -> bool:
        """Test 5: Test authentication with eFab API"""
        self.log("Testing API authentication...", "INFO")
        
        if not self.client:
            self.record_test("Authentication", False, "No client available")
            return False
            
        try:
            # Attempt to authenticate
            authenticated = await self.client.auth_manager.authenticate()
            
            if authenticated:
                self.record_test("Authentication", True, "Successfully authenticated")
                self.log("Authentication successful", "SUCCESS")
                self.log(f"  Session valid: {self.client.auth_manager.is_session_valid()}", "DEBUG")
                return True
            else:
                self.record_test("Authentication", False, "Authentication failed")
                self.log("Authentication failed", "ERROR")
                return False
                
        except Exception as e:
            self.record_test("Authentication", False, str(e))
            self.log(f"Authentication error: {e}", "ERROR")
            return False
            
    async def test_health_check(self) -> bool:
        """Test 6: Test API health check endpoint"""
        self.log("Testing API health check...", "INFO")
        
        if not self.client:
            self.record_test("Health Check", False, "No client available")
            return False
            
        try:
            health_status = await self.client.health_check()
            
            if health_status:
                self.record_test("Health Check", True, "API is healthy")
                self.log("API health check passed", "SUCCESS")
                return True
            else:
                self.record_test("Health Check", False, "API unhealthy")
                self.log("API health check failed", "WARNING")
                return False
                
        except Exception as e:
            self.record_test("Health Check", False, str(e))
            self.log(f"Health check error: {e}", "ERROR")
            return False
            
    async def test_yarn_inventory_endpoint(self) -> bool:
        """Test 7: Test yarn inventory endpoint"""
        self.log("Testing yarn inventory endpoint...", "INFO")
        
        if not self.client:
            self.record_test("Yarn Inventory", False, "No client available")
            return False
            
        try:
            # Fetch yarn inventory data
            yarn_data = await self.client.get_yarn_active()
            
            if yarn_data is not None and not yarn_data.empty:
                # Transform the data
                transformed = self.transformer.transform_yarn_active(yarn_data)
                
                self.log(f"  Records retrieved: {len(yarn_data)}", "DEBUG")
                self.log(f"  Columns: {list(transformed.columns)[:5]}...", "DEBUG")
                
                # Verify Planning Balance calculation
                if 'Planning Balance' in transformed.columns:
                    self.log("  Planning Balance calculated", "DEBUG")
                    
                self.record_test("Yarn Inventory", True, 
                               f"Retrieved {len(yarn_data)} records",
                               {"record_count": len(yarn_data), 
                                "sample_columns": list(transformed.columns)[:10]})
                self.log("Yarn inventory endpoint working", "SUCCESS")
                return True
            else:
                self.record_test("Yarn Inventory", False, "No data returned")
                self.log("No yarn inventory data returned", "WARNING")
                return False
                
        except Exception as e:
            self.record_test("Yarn Inventory", False, str(e))
            self.log(f"Yarn inventory error: {e}", "ERROR")
            return False
            
    async def test_data_loader_integration(self) -> bool:
        """Test 8: Test EFabAPIDataLoader integration"""
        self.log("Testing data loader integration...", "INFO")
        
        try:
            # Initialize the loader
            self.loader = EFabAPIDataLoader()
            
            # Test API availability check
            if self.loader.api_available:
                self.log("  API marked as available", "DEBUG")
            else:
                self.log("  API marked as unavailable", "WARNING")
                
            # Try loading yarn inventory through the loader
            yarn_df = await self.loader.load_yarn_inventory_async()
            
            if yarn_df is not None and not yarn_df.empty:
                self.record_test("Data Loader", True, 
                               f"Loaded {len(yarn_df)} records",
                               {"api_available": self.loader.api_available,
                                "record_count": len(yarn_df)})
                self.log("Data loader integration successful", "SUCCESS")
                return True
            else:
                self.record_test("Data Loader", False, "No data loaded")
                self.log("Data loader returned no data", "WARNING")
                return False
                
        except Exception as e:
            self.record_test("Data Loader", False, str(e))
            self.log(f"Data loader error: {e}", "ERROR")
            return False
            
    async def test_fallback_mechanism(self) -> bool:
        """Test 9: Test fallback to file loading"""
        self.log("Testing fallback mechanism...", "INFO")
        
        if not self.loader:
            self.loader = EFabAPIDataLoader()
            
        try:
            # Temporarily disable API to test fallback
            original_available = self.loader.api_available
            self.loader.api_available = False
            
            # Try loading data (should fall back to files)
            yarn_df = await self.loader.load_yarn_inventory_async()
            
            # Restore original state
            self.loader.api_available = original_available
            
            if yarn_df is not None and not yarn_df.empty:
                self.record_test("Fallback Mechanism", True, 
                               "Successfully fell back to file loading",
                               {"record_count": len(yarn_df)})
                self.log("Fallback mechanism working", "SUCCESS")
                return True
            else:
                self.record_test("Fallback Mechanism", False, "Fallback failed")
                self.log("Fallback mechanism failed", "ERROR")
                return False
                
        except Exception as e:
            self.record_test("Fallback Mechanism", False, str(e))
            self.log(f"Fallback test error: {e}", "ERROR")
            return False
            
    async def test_performance_comparison(self) -> bool:
        """Test 10: Compare API vs file loading performance"""
        self.log("Testing performance comparison...", "INFO")
        
        if not self.loader:
            self.loader = EFabAPIDataLoader()
            
        try:
            import time
            
            # Test API loading time
            if self.loader.api_available:
                start_api = time.time()
                yarn_api = await self.loader.load_yarn_inventory_async()
                api_time = time.time() - start_api
                self.log(f"  API load time: {api_time:.2f}s", "DEBUG")
            else:
                api_time = None
                self.log("  API not available for comparison", "WARNING")
                
            # Test file loading time
            self.loader.api_available = False
            start_file = time.time()
            yarn_file = await self.loader.load_yarn_inventory_async()
            file_time = time.time() - start_file
            self.log(f"  File load time: {file_time:.2f}s", "DEBUG")
            
            # Restore API availability
            self.loader.api_available = True
            
            performance_data = {
                "api_time": api_time,
                "file_time": file_time
            }
            
            if api_time and file_time:
                improvement = ((file_time - api_time) / file_time) * 100
                self.log(f"  Performance improvement: {improvement:.1f}%", "DEBUG")
                performance_data["improvement_pct"] = improvement
                
            self.record_test("Performance", True, 
                           "Performance comparison completed",
                           performance_data)
            self.log("Performance comparison completed", "SUCCESS")
            return True
            
        except Exception as e:
            self.record_test("Performance", False, str(e))
            self.log(f"Performance test error: {e}", "ERROR")
            return False
            
    async def run_all_tests(self):
        """Run all integration tests"""
        self.log("=" * 60)
        self.log("Starting eFab API Integration Tests", "INFO")
        self.log("=" * 60)
        
        # Define test sequence
        tests = [
            ("Environment Setup", self.test_environment_setup),
            ("API Modules", self.test_api_modules),
            ("Feature Flags", self.test_feature_flags),
            ("Client Initialization", self.test_api_client_initialization),
            ("Authentication", self.test_authentication),
            ("Health Check", self.test_health_check),
            ("Yarn Inventory", self.test_yarn_inventory_endpoint),
            ("Data Loader", self.test_data_loader_integration),
            ("Fallback Mechanism", self.test_fallback_mechanism),
            ("Performance", self.test_performance_comparison)
        ]
        
        # Run tests
        for test_name, test_func in tests:
            self.log(f"\n📋 Test: {test_name}")
            self.log("-" * 40)
            
            try:
                success = await test_func()
                if not success and test_name in ["Environment Setup", "API Modules"]:
                    self.log("Critical test failed, stopping test suite", "ERROR")
                    break
            except Exception as e:
                self.log(f"Unexpected error in {test_name}: {e}", "ERROR")
                self.record_test(test_name, False, f"Unexpected error: {e}")
                
        # Print summary
        self.log("\n" + "=" * 60)
        self.log("Test Summary", "INFO")
        self.log("=" * 60)
        
        summary = self.results["summary"]
        self.log(f"Total Tests: {summary['total']}")
        self.log(f"✅ Passed: {summary['passed']}")
        self.log(f"❌ Failed: {summary['failed']}")
        
        if summary['failed'] == 0:
            self.log("\n🎉 All tests passed! eFab API integration is ready.", "SUCCESS")
        else:
            self.log(f"\n⚠️ {summary['failed']} test(s) failed. Review the results above.", "WARNING")
            
        # Save results to file
        results_file = project_root / "test_results" / f"efab_api_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        self.log(f"\nResults saved to: {results_file}", "INFO")
        
        # Clean up
        if self.client:
            await self.client.close()
            
        return summary['failed'] == 0

async def main():
    """Main entry point for testing"""
    tester = EFabAPITester()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    # Check if running in dry-run mode
    if "--dry-run" in sys.argv:
        os.environ["EFAB_API_DRY_RUN"] = "true"
        print("🔍 Running in DRY-RUN mode (no actual API calls)")
        
    # Run the tests
    asyncio.run(main())