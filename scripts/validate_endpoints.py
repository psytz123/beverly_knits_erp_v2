#!/usr/bin/env python3
"""
Beverly Knits ERP Endpoint Validation Suite
Phase 1 Day 9-10: Create validation suite for all endpoints
Tests that all critical endpoints still work after fixes
"""

import requests
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EndpointValidator:
    """Validate all critical Beverly Knits ERP endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:5006"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "endpoint_results": {}
        }
    
    def validate_endpoint(self, endpoint: str, method: str = "GET", 
                         data: Dict = None, expected_status: int = 200,
                         required_fields: List[str] = None) -> Tuple[bool, str]:
        """Validate a single endpoint"""
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            # Make request
            if method == "GET":
                response = requests.get(url, timeout=30)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=30)
            else:
                response = requests.request(method, url, json=data, timeout=30)
            
            # Check status code
            if response.status_code != expected_status:
                return False, f"Expected status {expected_status}, got {response.status_code}"
            
            # Check response is valid JSON (for API endpoints)
            if endpoint.startswith("/api/"):
                try:
                    json_data = response.json()
                    
                    # Check required fields if specified
                    if required_fields:
                        missing_fields = []
                        for field in required_fields:
                            if field not in json_data:
                                missing_fields.append(field)
                        
                        if missing_fields:
                            return False, f"Missing required fields: {missing_fields}"
                    
                except json.JSONDecodeError:
                    return False, "Response is not valid JSON"
            
            return True, "OK"
            
        except requests.exceptions.Timeout:
            return False, "Request timeout (>30s)"
        except requests.exceptions.ConnectionError:
            return False, "Connection error - server may not be running"
        except Exception as e:
            return False, str(e)
    
    def run_all_validations(self):
        """Run validation tests for all critical endpoints"""
        
        logger.info("Starting endpoint validation suite...")
        logger.info("=" * 60)
        
        # Define validation tests
        validations = [
            # Core Dashboard APIs
            {
                "endpoint": "/api/production-planning",
                "name": "Production Planning API",
                "required_fields": ["production_plan", "capacity"]
            },
            {
                "endpoint": "/api/inventory-intelligence-enhanced",
                "name": "Inventory Intelligence API",
                "required_fields": ["inventory_analysis"]
            },
            {
                "endpoint": "/api/ml-forecast-detailed",
                "name": "ML Forecast API",
                "required_fields": ["forecast"]
            },
            {
                "endpoint": "/api/inventory-netting",
                "name": "Inventory Netting API",
                "required_fields": ["netting_results"]
            },
            {
                "endpoint": "/api/comprehensive-kpis",
                "name": "KPIs API",
                "required_fields": ["kpis"]
            },
            {
                "endpoint": "/api/yarn-intelligence",
                "name": "Yarn Intelligence API",
                "required_fields": ["yarn_analysis"]
            },
            {
                "endpoint": "/api/production-suggestions",
                "name": "Production Suggestions API",
                "required_fields": ["suggestions"]
            },
            {
                "endpoint": "/api/po-risk-analysis",
                "name": "PO Risk Analysis API",
                "required_fields": ["risk_analysis"]
            },
            {
                "endpoint": "/api/production-pipeline",
                "name": "Production Pipeline API",
                "required_fields": ["pipeline"]
            },
            {
                "endpoint": "/api/yarn-substitution-intelligent",
                "name": "Yarn Substitution API",
                "required_fields": ["substitutions"]
            },
            {
                "endpoint": "/api/knit-orders",
                "name": "Knit Orders API",
                "required_fields": ["orders"]
            },
            {
                "endpoint": "/api/health",
                "name": "Health Check API",
                "required_fields": ["status"]
            },
            
            # Dashboard pages
            {
                "endpoint": "/consolidated",
                "name": "Consolidated Dashboard",
                "required_fields": None  # HTML page
            }
        ]
        
        # Run validations
        for validation in validations:
            self.results["total_tests"] += 1
            
            logger.info(f"Testing: {validation['name']}...")
            
            passed, message = self.validate_endpoint(
                validation["endpoint"],
                required_fields=validation.get("required_fields")
            )
            
            if passed:
                self.results["passed"] += 1
                logger.info(f"  ✓ {validation['name']} - PASSED")
            else:
                self.results["failed"] += 1
                logger.error(f"  ✗ {validation['name']} - FAILED: {message}")
                self.results["errors"].append({
                    "endpoint": validation["endpoint"],
                    "name": validation["name"],
                    "error": message
                })
            
            self.results["endpoint_results"][validation["endpoint"]] = {
                "name": validation["name"],
                "passed": passed,
                "message": message
            }
        
        # Business logic validations
        logger.info("\nRunning business logic validations...")
        self.validate_business_logic()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {self.results['total_tests']}")
        logger.info(f"Passed: {self.results['passed']}")
        logger.info(f"Failed: {self.results['failed']}")
        
        if self.results["failed"] > 0:
            logger.warning("\nFailed Tests:")
            for error in self.results["errors"]:
                logger.warning(f"  - {error['name']}: {error['error']}")
        
        # Save results
        self.save_results()
        
        return self.results["failed"] == 0
    
    def validate_business_logic(self):
        """Validate critical business logic calculations"""
        
        test_cases = [
            {
                "name": "Planning Balance Formula",
                "description": "Verify Planning Balance = Theoretical + Allocated + On_Order",
                "test": self.test_planning_balance_formula
            },
            {
                "name": "Negative Allocated Values",
                "description": "Verify Allocated values are negative",
                "test": self.test_negative_allocated
            },
            {
                "name": "Yarn Shortage Detection",
                "description": "Verify yarn shortage detection logic",
                "test": self.test_yarn_shortage_detection
            }
        ]
        
        for test_case in test_cases:
            self.results["total_tests"] += 1
            
            logger.info(f"Testing: {test_case['name']}...")
            
            try:
                passed, message = test_case["test"]()
                
                if passed:
                    self.results["passed"] += 1
                    logger.info(f"  ✓ {test_case['name']} - PASSED")
                else:
                    self.results["failed"] += 1
                    logger.error(f"  ✗ {test_case['name']} - FAILED: {message}")
                    self.results["errors"].append({
                        "endpoint": "business_logic",
                        "name": test_case["name"],
                        "error": message
                    })
                    
            except Exception as e:
                self.results["failed"] += 1
                logger.error(f"  ✗ {test_case['name']} - ERROR: {e}")
                self.results["errors"].append({
                    "endpoint": "business_logic",
                    "name": test_case["name"],
                    "error": str(e)
                })
    
    def test_planning_balance_formula(self) -> Tuple[bool, str]:
        """Test Planning Balance calculation"""
        
        try:
            # Get yarn intelligence data
            response = requests.get(f"{self.base_url}/api/yarn-intelligence", timeout=30)
            
            if response.status_code != 200:
                return False, f"Failed to get yarn data: status {response.status_code}"
            
            data = response.json()
            
            # Check if we have yarn analysis
            if "yarn_analysis" not in data:
                return False, "No yarn_analysis in response"
            
            # Verify formula for first few items
            yarn_items = data["yarn_analysis"].get("yarns", [])[:5]
            
            for yarn in yarn_items:
                theoretical = yarn.get("theoretical_balance", 0)
                allocated = yarn.get("allocated", 0)
                on_order = yarn.get("on_order", 0)
                planning = yarn.get("planning_balance", 0)
                
                # Calculate expected planning balance
                expected = theoretical + allocated + on_order
                
                # Allow small floating point difference
                if abs(planning - expected) > 0.01:
                    return False, f"Formula mismatch for yarn {yarn.get('yarn_id')}: {planning} != {expected}"
            
            return True, "Planning Balance formula verified"
            
        except Exception as e:
            return False, str(e)
    
    def test_negative_allocated(self) -> Tuple[bool, str]:
        """Test that Allocated values are negative"""
        
        try:
            # Get yarn intelligence data
            response = requests.get(f"{self.base_url}/api/yarn-intelligence", timeout=30)
            
            if response.status_code != 200:
                return False, f"Failed to get yarn data: status {response.status_code}"
            
            data = response.json()
            
            # Check allocated values
            yarn_items = data["yarn_analysis"].get("yarns", [])
            positive_count = 0
            
            for yarn in yarn_items:
                allocated = yarn.get("allocated", 0)
                if allocated > 0:
                    positive_count += 1
            
            if positive_count > 0:
                return False, f"Found {positive_count} yarns with positive allocated values"
            
            return True, "All allocated values are negative or zero"
            
        except Exception as e:
            return False, str(e)
    
    def test_yarn_shortage_detection(self) -> Tuple[bool, str]:
        """Test yarn shortage detection logic"""
        
        try:
            # Get yarn intelligence data
            response = requests.get(f"{self.base_url}/api/yarn-intelligence", timeout=30)
            
            if response.status_code != 200:
                return False, f"Failed to get yarn data: status {response.status_code}"
            
            data = response.json()
            
            # Check if shortage detection is working
            if "shortages" not in data["yarn_analysis"]:
                return False, "No shortage detection in response"
            
            shortages = data["yarn_analysis"]["shortages"]
            
            # Verify shortage logic
            for shortage in shortages[:5]:
                planning_balance = shortage.get("planning_balance", 0)
                required = shortage.get("required_quantity", 0)
                
                # Shortage should exist when planning_balance < required
                if planning_balance >= required:
                    return False, f"Invalid shortage detected for yarn {shortage.get('yarn_id')}"
            
            return True, "Yarn shortage detection verified"
            
        except Exception as e:
            return False, str(e)
    
    def save_results(self):
        """Save validation results"""
        
        report_path = Path("docs/reports/validation_results.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to: {report_path}")
        
        # Create summary report
        summary_path = Path("docs/reports/validation_summary.md")
        
        with open(summary_path, 'w') as f:
            f.write("# Beverly Knits ERP Validation Summary\n\n")
            f.write(f"Generated: {self.results['timestamp']}\n\n")
            
            f.write("## Overall Results\n\n")
            f.write(f"- Total Tests: {self.results['total_tests']}\n")
            f.write(f"- Passed: {self.results['passed']}\n")
            f.write(f"- Failed: {self.results['failed']}\n")
            f.write(f"- Success Rate: {self.results['passed']/self.results['total_tests']*100:.1f}%\n\n")
            
            if self.results["failed"] > 0:
                f.write("## Failed Tests\n\n")
                for error in self.results["errors"]:
                    f.write(f"- **{error['name']}**: {error['error']}\n")
                f.write("\n")
            
            f.write("## Endpoint Results\n\n")
            f.write("| Endpoint | Status | Message |\n")
            f.write("|----------|--------|----------|\n")
            
            for endpoint, result in self.results["endpoint_results"].items():
                status = "✓ PASS" if result["passed"] else "✗ FAIL"
                f.write(f"| {endpoint} | {status} | {result['message']} |\n")
        
        logger.info(f"Summary saved to: {summary_path}")

def main():
    """Run the validation suite"""
    
    logger.info("Beverly Knits ERP Validation Suite")
    logger.info("Phase 1 Completion Validation")
    logger.info("")
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:5006/api/health", timeout=5)
    except:
        logger.error("Server not running on port 5006!")
        logger.info("Please start the server first:")
        logger.info("  python src/core/beverly_comprehensive_erp.py")
        return 1
    
    # Run validation
    validator = EndpointValidator()
    success = validator.run_all_validations()
    
    if success:
        logger.info("\n✓ ALL VALIDATIONS PASSED - Phase 1 Complete!")
        return 0
    else:
        logger.error("\n✗ Some validations failed - review and fix issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())