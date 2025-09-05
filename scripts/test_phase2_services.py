#!/usr/bin/env python3
"""
Phase 2 Service Validation Script
Tests the modularized services extracted from beverly_comprehensive_erp.py
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test results tracking
test_results = {
    'passed': [],
    'failed': [],
    'errors': []
}

def print_section(title):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_service_imports():
    """Test that all service modules can be imported"""
    print_section("Testing Service Imports")
    
    services_to_test = [
        ('inventory_analyzer_service', 'InventoryAnalyzer', 'InventoryManagementPipeline'),
        ('sales_forecasting_service', 'SalesForecastingEngine',),
        ('capacity_planning_service', 'CapacityPlanningEngine',),
        ('business_rules', 'BusinessRules', 'ValidationRules'),
        ('erp_service_manager', 'ERPServiceManager',)
    ]
    
    for module_info in services_to_test:
        module_name = module_info[0]
        classes = module_info[1:]
        
        try:
            # Try to import the module
            module = __import__(f'src.services.{module_name}', fromlist=classes)
            print(f"✅ Successfully imported {module_name}")
            
            # Check that expected classes exist
            for class_name in classes:
                if hasattr(module, class_name):
                    print(f"   ✅ Found class {class_name}")
                    test_results['passed'].append(f"Import {module_name}.{class_name}")
                else:
                    print(f"   ❌ Missing class {class_name}")
                    test_results['failed'].append(f"Missing {module_name}.{class_name}")
                    
        except Exception as e:
            print(f"❌ Failed to import {module_name}: {e}")
            test_results['errors'].append(f"Import error {module_name}: {str(e)}")
    
    return len(test_results['errors']) == 0

def test_service_instantiation():
    """Test that services can be instantiated"""
    print_section("Testing Service Instantiation")
    
    # Test InventoryAnalyzer
    try:
        from src.services.inventory_analyzer_service import InventoryAnalyzer, InventoryManagementPipeline
        
        analyzer = InventoryAnalyzer()
        print("✅ InventoryAnalyzer instantiated successfully")
        test_results['passed'].append("InventoryAnalyzer instantiation")
        
        pipeline = InventoryManagementPipeline()
        print("✅ InventoryManagementPipeline instantiated successfully")
        test_results['passed'].append("InventoryManagementPipeline instantiation")
        
    except Exception as e:
        print(f"❌ Failed to instantiate inventory services: {e}")
        test_results['failed'].append(f"Inventory instantiation: {str(e)}")
    
    # Test SalesForecastingEngine
    try:
        from src.services.sales_forecasting_service import SalesForecastingEngine
        
        forecasting = SalesForecastingEngine()
        print("✅ SalesForecastingEngine instantiated successfully")
        test_results['passed'].append("SalesForecastingEngine instantiation")
        
        # Check ML engines initialization
        if hasattr(forecasting, 'ML_AVAILABLE'):
            print(f"   ML Available: {forecasting.ML_AVAILABLE}")
            print(f"   ML Engines: {list(forecasting.ml_engines.keys()) if forecasting.ml_engines else 'None'}")
        
    except Exception as e:
        print(f"❌ Failed to instantiate forecasting service: {e}")
        test_results['failed'].append(f"Forecasting instantiation: {str(e)}")
    
    # Test CapacityPlanningEngine
    try:
        from src.services.capacity_planning_service import CapacityPlanningEngine
        
        capacity = CapacityPlanningEngine()
        print("✅ CapacityPlanningEngine instantiated successfully")
        test_results['passed'].append("CapacityPlanningEngine instantiation")
        
        # Check shift patterns
        if hasattr(capacity, 'shift_patterns'):
            print(f"   Shift patterns: {list(capacity.shift_patterns.keys())}")
        
    except Exception as e:
        print(f"❌ Failed to instantiate capacity service: {e}")
        test_results['failed'].append(f"Capacity instantiation: {str(e)}")
    
    # Test BusinessRules
    try:
        from src.services.business_rules import BusinessRules, ValidationRules
        
        rules = BusinessRules()
        print("✅ BusinessRules instantiated successfully")
        test_results['passed'].append("BusinessRules instantiation")
        
        validation = ValidationRules()
        print("✅ ValidationRules instantiated successfully")
        test_results['passed'].append("ValidationRules instantiation")
        
    except Exception as e:
        print(f"❌ Failed to instantiate business rules: {e}")
        test_results['failed'].append(f"BusinessRules instantiation: {str(e)}")
    
    # Test ERPServiceManager
    try:
        from src.services.erp_service_manager import ERPServiceManager
        
        manager = ERPServiceManager()
        print("✅ ERPServiceManager instantiated successfully")
        test_results['passed'].append("ERPServiceManager instantiation")
        
        # Check service status
        status = manager.get_service_status()
        print(f"   Services Status: {status['summary']}")
        for service, is_active in status['services'].items():
            print(f"      - {service}: {'✅ Active' if is_active else '❌ Inactive'}")
        
    except Exception as e:
        print(f"❌ Failed to instantiate service manager: {e}")
        test_results['failed'].append(f"ServiceManager instantiation: {str(e)}")

def test_critical_business_logic():
    """Test critical business calculations are preserved"""
    print_section("Testing Critical Business Logic")
    
    try:
        from src.services.business_rules import BusinessRules
        
        # Test Planning Balance calculation
        # Formula: Planning_Balance = Theoretical_Balance + Allocated + On_Order
        # CRITICAL: Allocated is already negative in the data
        
        theoretical = 1000
        allocated = -200  # Already negative
        on_order = 300
        
        result = BusinessRules.calculate_planning_balance(theoretical, allocated, on_order)
        expected = 1100  # 1000 + (-200) + 300
        
        if abs(result - expected) < 0.01:
            print(f"✅ Planning Balance calculation correct: {result} == {expected}")
            test_results['passed'].append("Planning Balance formula")
        else:
            print(f"❌ Planning Balance calculation wrong: {result} != {expected}")
            test_results['failed'].append(f"Planning Balance: got {result}, expected {expected}")
        
        # Test Weekly Demand calculation
        weekly_demand = BusinessRules.calculate_weekly_demand(
            consumed_data=True,
            monthly_consumed=430  # Should give 100 weekly
        )
        expected = 100  # 430 / 4.3
        
        if abs(weekly_demand - expected) < 1:
            print(f"✅ Weekly demand calculation correct: {weekly_demand:.1f} ≈ {expected}")
            test_results['passed'].append("Weekly demand formula")
        else:
            print(f"❌ Weekly demand calculation wrong: {weekly_demand} != {expected}")
            test_results['failed'].append(f"Weekly demand: got {weekly_demand}, expected {expected}")
        
        # Test Yarn Substitution Score
        score = BusinessRules.calculate_yarn_substitution_score(
            color_match=0.9,
            composition_match=0.8,
            weight_match=0.7
        )
        expected = 0.9 * 0.3 + 0.8 * 0.4 + 0.7 * 0.3  # 0.8
        
        if abs(score - expected) < 0.01:
            print(f"✅ Yarn substitution score correct: {score:.2f} == {expected:.2f}")
            test_results['passed'].append("Yarn substitution formula")
        else:
            print(f"❌ Yarn substitution score wrong: {score} != {expected}")
            test_results['failed'].append(f"Yarn substitution: got {score}, expected {expected}")
        
        # Test Shortage Risk Levels
        risk_levels = [
            (5, 'CRITICAL'),
            (10, 'HIGH'),
            (20, 'MEDIUM'),
            (40, 'LOW')
        ]
        
        all_correct = True
        for days, expected_risk in risk_levels:
            risk = BusinessRules.calculate_shortage_risk(days)
            if risk == expected_risk:
                print(f"   ✅ Risk level for {days} days: {risk}")
            else:
                print(f"   ❌ Risk level wrong for {days} days: {risk} != {expected_risk}")
                all_correct = False
        
        if all_correct:
            test_results['passed'].append("Shortage risk levels")
        else:
            test_results['failed'].append("Some shortage risk levels incorrect")
            
    except Exception as e:
        print(f"❌ Failed to test business logic: {e}")
        test_results['errors'].append(f"Business logic test: {str(e)}")

def test_service_methods():
    """Test that key service methods work"""
    print_section("Testing Service Methods")
    
    # Test InventoryAnalyzer methods
    try:
        from src.services.inventory_analyzer_service import InventoryAnalyzer
        import pandas as pd
        
        analyzer = InventoryAnalyzer()
        
        # Test with empty data
        result = analyzer.analyze_inventory(None)
        if 'total_items' in result and result['total_items'] == 0:
            print("✅ InventoryAnalyzer handles empty data correctly")
            test_results['passed'].append("InventoryAnalyzer empty data handling")
        else:
            print("❌ InventoryAnalyzer empty data handling failed")
            test_results['failed'].append("InventoryAnalyzer empty data handling")
        
        # Test with sample data
        sample_data = pd.DataFrame({
            'Desc#': ['YARN001', 'YARN002', 'YARN003'],
            'Planning Balance': [100, -50, 200],
            'quantity': [100, -50, 200]
        })
        
        result = analyzer.analyze_inventory(sample_data)
        if result['total_items'] == 3:
            print(f"✅ InventoryAnalyzer processes data: {result['total_items']} items")
            print(f"   Critical: {result['summary']['critical_count']}, "
                  f"High Risk: {result['summary']['high_risk_count']}, "
                  f"Healthy: {result['summary']['healthy_count']}")
            test_results['passed'].append("InventoryAnalyzer data processing")
        else:
            print("❌ InventoryAnalyzer data processing failed")
            test_results['failed'].append("InventoryAnalyzer data processing")
            
    except Exception as e:
        print(f"❌ InventoryAnalyzer method test failed: {e}")
        test_results['errors'].append(f"InventoryAnalyzer methods: {str(e)}")
    
    # Test SalesForecastingEngine methods
    try:
        from src.services.sales_forecasting_service import SalesForecastingEngine
        import pandas as pd
        import numpy as np
        
        forecasting = SalesForecastingEngine()
        
        # Test consistency score calculation
        sample_history = pd.Series([100, 105, 98, 102, 101])  # Consistent data
        consistency = forecasting.calculate_consistency_score(sample_history)
        
        if 'consistency_score' in consistency:
            print(f"✅ Consistency score calculation works: {consistency['consistency_score']:.2f}")
            print(f"   CV: {consistency['cv']:.2f}, Recommendation: {consistency['recommendation']}")
            test_results['passed'].append("Forecasting consistency calculation")
        else:
            print("❌ Consistency score calculation failed")
            test_results['failed'].append("Forecasting consistency calculation")
            
    except Exception as e:
        print(f"❌ SalesForecastingEngine method test failed: {e}")
        test_results['errors'].append(f"SalesForecastingEngine methods: {str(e)}")
    
    # Test CapacityPlanningEngine methods
    try:
        from src.services.capacity_planning_service import CapacityPlanningEngine
        
        capacity = CapacityPlanningEngine()
        
        # Test capacity requirements calculation
        production_plan = {'product1': 100, 'product2': 200}
        requirements = capacity.calculate_finite_capacity_requirements(production_plan)
        
        if 'product1' in requirements and 'machine_hours' in requirements['product1']:
            print(f"✅ Capacity requirements calculation works")
            for product, req in requirements.items():
                print(f"   {product}: {req['machine_hours']:.1f} machine hours")
            test_results['passed'].append("Capacity requirements calculation")
        else:
            print("❌ Capacity requirements calculation failed")
            test_results['failed'].append("Capacity requirements calculation")
            
    except Exception as e:
        print(f"❌ CapacityPlanningEngine method test failed: {e}")
        test_results['errors'].append(f"CapacityPlanningEngine methods: {str(e)}")

def test_service_integration():
    """Test that ERPServiceManager integrates all services"""
    print_section("Testing Service Integration")
    
    try:
        from src.services.erp_service_manager import ERPServiceManager
        import pandas as pd
        
        manager = ERPServiceManager()
        
        # Test integrated analysis
        sample_inventory = pd.DataFrame({
            'Desc#': ['YARN001', 'YARN002'],
            'Planning Balance': [100, -50],
            'Theoretical_Balance': [150, 50],
            'Allocated': [-50, -100],
            'On_Order': [0, 0]
        })
        
        result = manager.run_integrated_analysis(
            inventory_data=sample_inventory
        )
        
        if result['status'] == 'success':
            print("✅ Integrated analysis runs successfully")
            print(f"   Services available: {result['services_available']}")
            test_results['passed'].append("Integrated analysis")
        else:
            print("❌ Integrated analysis failed")
            test_results['failed'].append("Integrated analysis")
        
        # Test planning balance calculation through manager
        balance = manager.calculate_planning_balance(1000, -200, 300)
        if balance == 1100:
            print(f"✅ Manager correctly delegates to business rules: {balance}")
            test_results['passed'].append("Manager delegation")
        else:
            print(f"❌ Manager delegation failed: {balance} != 1100")
            test_results['failed'].append("Manager delegation")
            
    except Exception as e:
        print(f"❌ Service integration test failed: {e}")
        test_results['errors'].append(f"Service integration: {str(e)}")

def generate_report():
    """Generate final test report"""
    print_section("TEST RESULTS SUMMARY")
    
    total_tests = len(test_results['passed']) + len(test_results['failed']) + len(test_results['errors'])
    
    print(f"Total Tests Run: {total_tests}")
    print(f"✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    print(f"🔥 Errors: {len(test_results['errors'])}")
    
    if test_results['failed']:
        print("\n❌ Failed Tests:")
        for failed in test_results['failed']:
            print(f"   - {failed}")
    
    if test_results['errors']:
        print("\n🔥 Test Errors:")
        for error in test_results['errors']:
            print(f"   - {error}")
    
    success_rate = (len(test_results['passed']) / total_tests * 100) if total_tests > 0 else 0
    print(f"\n{'='*60}")
    print(f"  SUCCESS RATE: {success_rate:.1f}%")
    print(f"{'='*60}")
    
    # Overall assessment
    if success_rate >= 90:
        print("\n✅ PHASE 2 VALIDATION: PASSED")
        print("The modularization appears successful!")
    elif success_rate >= 70:
        print("\n⚠️  PHASE 2 VALIDATION: PARTIAL SUCCESS")
        print("Most services work but some issues need attention.")
    else:
        print("\n❌ PHASE 2 VALIDATION: NEEDS WORK")
        print("Significant issues detected in the modularization.")
    
    return success_rate

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  PHASE 2 SERVICE VALIDATION TEST SUITE")
    print("  Testing Modularized Services")
    print("="*60)
    
    try:
        # Run test suites
        test_service_imports()
        test_service_instantiation()
        test_critical_business_logic()
        test_service_methods()
        test_service_integration()
        
        # Generate report
        success_rate = generate_report()
        
        # Exit with appropriate code
        if success_rate >= 90:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"\n🔥 CRITICAL TEST FAILURE: {e}")
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()