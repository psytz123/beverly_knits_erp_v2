#!/usr/bin/env python3
"""
Phase 2 Service Validation Script (Simple version without unicode)
Tests the modularized services extracted from beverly_comprehensive_erp.py
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_section(title):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def main():
    """Run validation tests"""
    print_section("PHASE 2 SERVICE VALIDATION")
    
    passed = 0
    failed = 0
    
    # Test 1: Import all services
    print("TEST 1: Importing Services")
    services = [
        'inventory_analyzer_service',
        'sales_forecasting_service', 
        'capacity_planning_service',
        'business_rules',
        'erp_service_manager'
    ]
    
    for service in services:
        try:
            module = __import__(f'src.services.{service}')
            print(f"  [PASS] Imported {service}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] Could not import {service}: {e}")
            failed += 1
    
    # Test 2: Instantiate key classes
    print("\nTEST 2: Instantiating Classes")
    
    try:
        from src.services.inventory_analyzer_service import InventoryAnalyzer
        analyzer = InventoryAnalyzer()
        print("  [PASS] InventoryAnalyzer instantiated")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] InventoryAnalyzer: {e}")
        failed += 1
    
    try:
        from src.services.sales_forecasting_service import SalesForecastingEngine
        forecasting = SalesForecastingEngine()
        print("  [PASS] SalesForecastingEngine instantiated")
        print(f"        ML Available: {forecasting.ML_AVAILABLE}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] SalesForecastingEngine: {e}")
        failed += 1
    
    try:
        from src.services.capacity_planning_service import CapacityPlanningEngine
        capacity = CapacityPlanningEngine()
        print("  [PASS] CapacityPlanningEngine instantiated")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] CapacityPlanningEngine: {e}")
        failed += 1
    
    try:
        from src.services.business_rules import BusinessRules
        rules = BusinessRules()
        print("  [PASS] BusinessRules instantiated")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] BusinessRules: {e}")
        failed += 1
    
    try:
        from src.services.erp_service_manager import ERPServiceManager
        manager = ERPServiceManager()
        print("  [PASS] ERPServiceManager instantiated")
        status = manager.get_service_status()
        print(f"        Active services: {status['summary']['active']}/{status['summary']['total']}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] ERPServiceManager: {e}")
        failed += 1
    
    # Test 3: Critical business logic
    print("\nTEST 3: Critical Business Logic")
    
    try:
        from src.services.business_rules import BusinessRules
        
        # Test Planning Balance
        result = BusinessRules.calculate_planning_balance(1000, -200, 300)
        expected = 1100
        if abs(result - expected) < 0.01:
            print(f"  [PASS] Planning Balance: {result} == {expected}")
            passed += 1
        else:
            print(f"  [FAIL] Planning Balance: {result} != {expected}")
            failed += 1
        
        # Test Weekly Demand
        weekly = BusinessRules.calculate_weekly_demand(
            consumed_data=True, 
            monthly_consumed=430
        )
        expected = 100  # 430/4.3
        if abs(weekly - expected) < 1:
            print(f"  [PASS] Weekly Demand: {weekly:.1f} ~= {expected}")
            passed += 1
        else:
            print(f"  [FAIL] Weekly Demand: {weekly} != {expected}")
            failed += 1
        
        # Test Yarn Substitution Score
        score = BusinessRules.calculate_yarn_substitution_score(0.9, 0.8, 0.7)
        expected = 0.8  # (0.9*0.3 + 0.8*0.4 + 0.7*0.3)
        if abs(score - expected) < 0.01:
            print(f"  [PASS] Yarn Substitution Score: {score:.2f} == {expected}")
            passed += 1
        else:
            print(f"  [FAIL] Yarn Substitution Score: {score} != {expected}")
            failed += 1
            
        # Test Risk Levels
        risk_tests = [(5, 'CRITICAL'), (10, 'HIGH'), (20, 'MEDIUM'), (40, 'LOW')]
        all_pass = True
        for days, expected_risk in risk_tests:
            risk = BusinessRules.calculate_shortage_risk(days)
            if risk != expected_risk:
                all_pass = False
                break
        
        if all_pass:
            print(f"  [PASS] Shortage Risk Levels")
            passed += 1
        else:
            print(f"  [FAIL] Shortage Risk Levels")
            failed += 1
            
    except Exception as e:
        print(f"  [FAIL] Business logic tests: {e}")
        failed += 1
    
    # Test 4: Service methods work
    print("\nTEST 4: Service Methods")
    
    try:
        from src.services.inventory_analyzer_service import InventoryAnalyzer
        analyzer = InventoryAnalyzer()
        result = analyzer.analyze_inventory(None)
        if result['total_items'] == 0:
            print("  [PASS] InventoryAnalyzer handles empty data")
            passed += 1
        else:
            print("  [FAIL] InventoryAnalyzer empty data handling")
            failed += 1
    except Exception as e:
        print(f"  [FAIL] InventoryAnalyzer method test: {e}")
        failed += 1
    
    try:
        from src.services.capacity_planning_service import CapacityPlanningEngine
        capacity = CapacityPlanningEngine()
        production_plan = {'product1': 100}
        requirements = capacity.calculate_finite_capacity_requirements(production_plan)
        if 'product1' in requirements:
            print("  [PASS] Capacity requirements calculation")
            passed += 1
        else:
            print("  [FAIL] Capacity requirements calculation")
            failed += 1
    except Exception as e:
        print(f"  [FAIL] Capacity method test: {e}")
        failed += 1
    
    # Summary
    print_section("RESULTS SUMMARY")
    total = passed + failed
    print(f"Tests Run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if total > 0:
        success_rate = (passed / total) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("\nPHASE 2 VALIDATION: PASSED")
            print("The modularization is successful!")
        elif success_rate >= 70:
            print("\nPHASE 2 VALIDATION: PARTIAL SUCCESS")
            print("Most services work but some issues need attention.")
        else:
            print("\nPHASE 2 VALIDATION: NEEDS WORK")
            print("Significant issues detected.")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())