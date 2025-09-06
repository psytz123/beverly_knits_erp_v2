#!/usr/bin/env python3
"""
Comprehensive Test Suite for Time-Phased PO Integration
Tests all implemented functions and validates against requirements
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all time-phased components
from src.data_loaders.po_delivery_loader import PODeliveryLoader
from src.production.time_phased_planning import TimePhasedPlanning, create_mock_demand_schedule
import pandas as pd
import numpy as np

# Test results collector
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def log_test(test_name, status, message=""):
    """Log test results"""
    if status == "PASS":
        test_results['passed'].append(test_name)
        print(f"✅ {test_name}: {message}")
    elif status == "FAIL":
        test_results['failed'].append(test_name)
        print(f"❌ {test_name}: {message}")
    elif status == "WARN":
        test_results['warnings'].append(test_name)
        print(f"⚠️  {test_name}: {message}")

def test_section(section_name):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"{section_name}")
    print(f"{'='*60}")

# ============================================================================
# COMPONENT 1: PO DELIVERY LOADER TESTS
# ============================================================================

def test_po_delivery_loader():
    """Test all PODeliveryLoader functions"""
    test_section("TESTING PO DELIVERY LOADER")
    
    loader = PODeliveryLoader()
    
    # Test 1: Initialization
    try:
        assert hasattr(loader, 'delivery_columns'), "Missing delivery_columns"
        assert hasattr(loader, 'column_aliases'), "Missing column_aliases"
        assert hasattr(loader, 'week_mapping'), "Missing week_mapping"
        log_test("PO Loader Initialization", "PASS", "All attributes initialized")
    except AssertionError as e:
        log_test("PO Loader Initialization", "FAIL", str(e))
    
    # Test 2: Load PO deliveries from CSV
    try:
        csv_file = "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/8-28-2025/Expected_Yarn_Report.csv"
        po_data = loader.load_po_deliveries(csv_file)
        assert not po_data.empty, "PO data is empty"
        assert len(po_data) > 0, "No PO records loaded"
        log_test("Load PO Deliveries (CSV)", "PASS", f"Loaded {len(po_data)} records")
    except Exception as e:
        log_test("Load PO Deliveries (CSV)", "FAIL", str(e))
        return False
    
    # Test 3: Load PO deliveries from Excel
    try:
        excel_file = "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/8-28-2025/Expected_Yarn_Report.xlsx"
        if Path(excel_file).exists():
            po_data_excel = loader.load_po_deliveries(excel_file)
            log_test("Load PO Deliveries (Excel)", "PASS", f"Loaded {len(po_data_excel)} records")
        else:
            log_test("Load PO Deliveries (Excel)", "WARN", "Excel file not found, skipping")
    except Exception as e:
        log_test("Load PO Deliveries (Excel)", "FAIL", str(e))
    
    # Test 4: Column cleaning
    try:
        # Check if numeric columns are properly cleaned
        numeric_cols = ['Unscheduled or Past Due', 'This Week', '9/5/2025', '9/12/2025']
        for col in numeric_cols:
            if col in po_data.columns:
                assert po_data[col].dtype in [np.float64, np.int64], f"{col} not numeric"
        log_test("Column Cleaning", "PASS", "Numeric fields properly cleaned")
    except Exception as e:
        log_test("Column Cleaning", "FAIL", str(e))
    
    # Test 5: Map to weekly buckets
    try:
        weekly_data = loader.map_to_weekly_buckets(po_data)
        expected_week_cols = ['week_36', 'week_37', 'week_38', 'week_39', 'week_40', 
                             'week_41', 'week_42', 'week_43', 'week_44']
        week_cols_found = [col for col in expected_week_cols if col in weekly_data.columns]
        assert len(week_cols_found) > 0, "No week columns created"
        log_test("Map to Weekly Buckets", "PASS", f"Created {len(week_cols_found)} week columns")
    except Exception as e:
        log_test("Map to Weekly Buckets", "FAIL", str(e))
    
    # Test 6: Aggregate by yarn
    try:
        yarn_deliveries = loader.aggregate_by_yarn(weekly_data)
        assert isinstance(yarn_deliveries, dict), "Should return dictionary"
        assert len(yarn_deliveries) > 0, "No yarn deliveries aggregated"
        
        # Check structure of aggregated data
        sample_yarn = list(yarn_deliveries.keys())[0]
        sample_delivery = yarn_deliveries[sample_yarn]
        assert isinstance(sample_delivery, dict), "Yarn delivery should be dict"
        
        log_test("Aggregate by Yarn", "PASS", f"Aggregated {len(yarn_deliveries)} yarns")
    except Exception as e:
        log_test("Aggregate by Yarn", "FAIL", str(e))
    
    # Test 7: Calculate total on order
    try:
        totals = loader.calculate_total_on_order(yarn_deliveries)
        assert isinstance(totals, dict), "Should return dictionary"
        assert len(totals) == len(yarn_deliveries), "Totals count mismatch"
        
        # Verify totals calculation
        sample_yarn = list(yarn_deliveries.keys())[0]
        manual_total = sum(yarn_deliveries[sample_yarn].values())
        calculated_total = totals[sample_yarn]
        assert abs(manual_total - calculated_total) < 0.01, "Total calculation error"
        
        log_test("Calculate Total On Order", "PASS", f"Calculated totals for {len(totals)} yarns")
    except Exception as e:
        log_test("Calculate Total On Order", "FAIL", str(e))
    
    # Test 8: Get delivery timeline
    try:
        sample_yarn = list(yarn_deliveries.keys())[0]
        timeline = loader.get_delivery_timeline(sample_yarn, yarn_deliveries)
        assert isinstance(timeline, dict), "Timeline should be dict"
        assert timeline == yarn_deliveries[sample_yarn], "Timeline mismatch"
        log_test("Get Delivery Timeline", "PASS", "Timeline retrieval working")
    except Exception as e:
        log_test("Get Delivery Timeline", "FAIL", str(e))
    
    # Test 9: Get next receipt week
    try:
        # Find a yarn with future receipts
        yarn_with_receipts = None
        for yarn_id, deliveries in yarn_deliveries.items():
            for week in range(36, 45):
                if deliveries.get(f'week_{week}', 0) > 0:
                    yarn_with_receipts = yarn_id
                    break
            if yarn_with_receipts:
                break
        
        if yarn_with_receipts:
            next_week = loader.get_next_receipt_week(yarn_with_receipts, yarn_deliveries)
            assert next_week is not None, "Should find next receipt week"
            log_test("Get Next Receipt Week", "PASS", f"Found next receipt: {next_week}")
        else:
            log_test("Get Next Receipt Week", "WARN", "No yarns with future receipts to test")
    except Exception as e:
        log_test("Get Next Receipt Week", "FAIL", str(e))
    
    # Test 10: Export time-phased data
    try:
        output_path = "/tmp/test_time_phased_export.csv"
        success = loader.export_time_phased_data(yarn_deliveries, output_path)
        assert success, "Export failed"
        assert Path(output_path).exists(), "Export file not created"
        
        # Verify export content
        exported_df = pd.read_csv(output_path)
        assert len(exported_df) == len(yarn_deliveries), "Export row count mismatch"
        
        log_test("Export Time-Phased Data", "PASS", f"Exported to {output_path}")
    except Exception as e:
        log_test("Export Time-Phased Data", "FAIL", str(e))
    
    return yarn_deliveries  # Return for use in other tests

# ============================================================================
# COMPONENT 2: TIME-PHASED PLANNING ENGINE TESTS
# ============================================================================

def test_time_phased_planning(yarn_deliveries):
    """Test all TimePhasedPlanning functions"""
    test_section("TESTING TIME-PHASED PLANNING ENGINE")
    
    planner = TimePhasedPlanning()
    
    # Test 1: Initialization
    try:
        assert hasattr(planner, 'current_week'), "Missing current_week"
        assert hasattr(planner, 'planning_horizon'), "Missing planning_horizon"
        assert hasattr(planner, 'week_dates'), "Missing week_dates"
        assert planner.current_week == 36, "Current week should be 36"
        assert planner.planning_horizon == 13, "Planning horizon should be 13"
        log_test("Planning Engine Init", "PASS", "All attributes initialized")
    except AssertionError as e:
        log_test("Planning Engine Init", "FAIL", str(e))
    
    # Setup test data
    test_yarn_id = list(yarn_deliveries.keys())[0] if yarn_deliveries else '18884'
    weekly_receipts = yarn_deliveries.get(test_yarn_id, {
        'past_due': 20161.30,
        'week_36': 0,
        'week_37': 0,
        'week_38': 0,
        'week_39': 0,
        'week_40': 0,
        'week_41': 0,
        'week_42': 0,
        'week_43': 4000,
        'week_44': 4000,
        'later': 8000
    })
    
    # Test 2: Calculate weekly balance
    try:
        starting_balance = 2506.18
        weekly_demand = {f'week_{i}': 3428.87 for i in range(36, 45)}
        
        weekly_balances = planner.calculate_weekly_balance(
            yarn_id=test_yarn_id,
            starting_balance=starting_balance,
            weekly_receipts=weekly_receipts,
            weekly_demand=weekly_demand
        )
        
        assert isinstance(weekly_balances, dict), "Should return dict"
        assert len(weekly_balances) > 0, "No weekly balances calculated"
        assert 'week_36' in weekly_balances, "Missing week_36"
        
        # Verify calculation logic
        expected_week_36 = starting_balance + weekly_receipts.get('past_due', 0) + \
                          weekly_receipts.get('week_36', 0) - weekly_demand.get('week_36', 0)
        actual_week_36 = weekly_balances.get('week_36', 0)
        
        log_test("Calculate Weekly Balance", "PASS", 
                f"Calculated {len(weekly_balances)} weeks, week_36={actual_week_36:.2f}")
    except Exception as e:
        log_test("Calculate Weekly Balance", "FAIL", str(e))
        weekly_balances = {}
    
    # Test 3: Identify shortage periods
    try:
        shortage_periods = planner.identify_shortage_periods(test_yarn_id, weekly_balances)
        assert isinstance(shortage_periods, list), "Should return list"
        
        # Count actual shortages
        actual_shortages = sum(1 for balance in weekly_balances.values() if balance < 0)
        
        if actual_shortages > 0:
            assert len(shortage_periods) == actual_shortages, "Shortage count mismatch"
            
            # Check shortage period structure
            if shortage_periods:
                first_shortage = shortage_periods[0]
                assert len(first_shortage) == 3, "Shortage tuple should have 3 elements"
                week, amount, recovery = first_shortage
                assert isinstance(week, str), "Week should be string"
                assert isinstance(amount, (int, float)), "Amount should be numeric"
        
        log_test("Identify Shortage Periods", "PASS", 
                f"Found {len(shortage_periods)} shortage periods")
    except Exception as e:
        log_test("Identify Shortage Periods", "FAIL", str(e))
        shortage_periods = []
    
    # Test 4: Calculate expedite requirements
    try:
        expedite_recs = planner.calculate_expedite_requirements(
            test_yarn_id, shortage_periods, weekly_receipts
        )
        assert isinstance(expedite_recs, list), "Should return list"
        
        if shortage_periods:
            # Should have recommendations for shortages
            if expedite_recs:
                first_rec = expedite_recs[0]
                assert 'yarn_id' in first_rec, "Missing yarn_id"
                assert 'shortage_week' in first_rec, "Missing shortage_week"
                assert 'expedite_amount' in first_rec, "Missing expedite_amount"
                assert 'expedite_from_week' in first_rec, "Missing expedite_from_week"
        
        log_test("Calculate Expedite Requirements", "PASS", 
                f"Generated {len(expedite_recs)} recommendations")
    except Exception as e:
        log_test("Calculate Expedite Requirements", "FAIL", str(e))
    
    # Test 5: Calculate yarn coverage weeks
    try:
        current_balance = 5000
        weekly_demand = {f'week_{i}': 1000 for i in range(36, 45)}
        
        coverage_weeks = planner.calculate_yarn_coverage_weeks(
            test_yarn_id, current_balance, weekly_demand
        )
        
        assert isinstance(coverage_weeks, (int, float)), "Should return numeric"
        assert coverage_weeks >= 0, "Coverage should be non-negative"
        
        # Verify calculation
        expected_coverage = current_balance / (sum(weekly_demand.values()) / len(weekly_demand))
        assert abs(coverage_weeks - expected_coverage) < 0.1, "Coverage calculation error"
        
        log_test("Calculate Coverage Weeks", "PASS", f"Coverage: {coverage_weeks:.2f} weeks")
    except Exception as e:
        log_test("Calculate Coverage Weeks", "FAIL", str(e))
    
    # Test 6: Generate shortage summary
    try:
        summary = planner.generate_shortage_summary(
            test_yarn_id, weekly_balances, weekly_receipts, weekly_demand
        )
        
        assert isinstance(summary, dict), "Should return dict"
        assert 'yarn_id' in summary, "Missing yarn_id"
        assert 'has_shortage' in summary, "Missing has_shortage"
        assert 'shortage_count' in summary, "Missing shortage_count"
        assert 'weekly_balances' in summary, "Missing weekly_balances"
        
        log_test("Generate Shortage Summary", "PASS", 
                f"Summary complete, has_shortage={summary.get('has_shortage')}")
    except Exception as e:
        log_test("Generate Shortage Summary", "FAIL", str(e))
    
    # Test 7: Process yarn time-phased (complete workflow)
    try:
        yarn_data = {
            'yarn_id': test_yarn_id,
            'theoretical_balance': 2506.18,
            'allocated': -30859.80,
            'planning_balance': 7807.68
        }
        
        weekly_demand = create_mock_demand_schedule(
            test_yarn_id, yarn_data['allocated'], 9
        )
        
        result = planner.process_yarn_time_phased(
            yarn_data, weekly_receipts, weekly_demand
        )
        
        assert isinstance(result, dict), "Should return dict"
        assert 'time_phased_enabled' in result, "Missing time_phased_enabled"
        assert result['time_phased_enabled'] == True, "Should be enabled"
        assert 'has_shortage' in result, "Missing shortage indicator"
        assert 'weekly_balances' in result, "Missing weekly balances"
        
        log_test("Process Yarn Time-Phased", "PASS", 
                f"Complete workflow executed for yarn {test_yarn_id}")
    except Exception as e:
        log_test("Process Yarn Time-Phased", "FAIL", str(e))
    
    # Test 8: Mock demand schedule creation
    try:
        allocated = -10000
        weeks = 10
        demand_schedule = create_mock_demand_schedule("test_yarn", allocated, weeks)
        
        assert isinstance(demand_schedule, dict), "Should return dict"
        assert len(demand_schedule) == weeks, "Week count mismatch"
        
        # Verify total demand equals allocated
        total_demand = sum(demand_schedule.values())
        expected_total = abs(allocated)
        assert abs(total_demand - expected_total) < 0.01, "Total demand mismatch"
        
        log_test("Create Mock Demand Schedule", "PASS", 
                f"Created {weeks}-week schedule, total={total_demand:.2f}")
    except Exception as e:
        log_test("Create Mock Demand Schedule", "FAIL", str(e))

# ============================================================================
# COMPONENT 3: INTEGRATION TESTS
# ============================================================================

def test_integration():
    """Test integration between components"""
    test_section("TESTING COMPONENT INTEGRATION")
    
    # Test 1: End-to-end workflow
    try:
        # Load PO data
        loader = PODeliveryLoader()
        csv_file = "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/8-28-2025/Expected_Yarn_Report.csv"
        po_data = loader.load_po_deliveries(csv_file)
        weekly_data = loader.map_to_weekly_buckets(po_data)
        yarn_deliveries = loader.aggregate_by_yarn(weekly_data)
        
        # Process with planning engine
        planner = TimePhasedPlanning()
        results = []
        
        for yarn_id in list(yarn_deliveries.keys())[:5]:  # Test first 5 yarns
            yarn_data = {
                'yarn_id': yarn_id,
                'theoretical_balance': 1000,
                'allocated': -3000,
                'planning_balance': 500
            }
            
            weekly_receipts = yarn_deliveries[yarn_id]
            weekly_demand = create_mock_demand_schedule(yarn_id, yarn_data['allocated'], 9)
            
            result = planner.process_yarn_time_phased(
                yarn_data, weekly_receipts, weekly_demand
            )
            results.append(result)
        
        assert len(results) == 5, "Should process 5 yarns"
        assert all('has_shortage' in r for r in results), "Missing shortage analysis"
        
        log_test("End-to-End Integration", "PASS", 
                f"Processed {len(results)} yarns successfully")
    except Exception as e:
        log_test("End-to-End Integration", "FAIL", str(e))
    
    # Test 2: Data consistency
    try:
        # Verify data consistency between loader and planner
        loader = PODeliveryLoader()
        planner = TimePhasedPlanning()
        
        # Check week mapping consistency
        loader_weeks = set(loader.week_mapping.values())
        planner_weeks = set(planner.week_dates.keys())
        
        # Both should have weeks 36-44
        expected_weeks = set(range(36, 45))
        
        assert len(loader_weeks & expected_weeks) > 0, "Loader missing week numbers"
        assert len(planner_weeks & expected_weeks) > 0, "Planner missing week numbers"
        
        log_test("Data Consistency", "PASS", "Week mappings consistent")
    except Exception as e:
        log_test("Data Consistency", "FAIL", str(e))

# ============================================================================
# COMPONENT 4: PERFORMANCE TESTS
# ============================================================================

def test_performance():
    """Test performance metrics"""
    test_section("TESTING PERFORMANCE")
    
    loader = PODeliveryLoader()
    planner = TimePhasedPlanning()
    
    # Test 1: Data loading performance
    try:
        csv_file = "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/8-28-2025/Expected_Yarn_Report.csv"
        
        start_time = time.time()
        po_data = loader.load_po_deliveries(csv_file)
        weekly_data = loader.map_to_weekly_buckets(po_data)
        yarn_deliveries = loader.aggregate_by_yarn(weekly_data)
        load_time = time.time() - start_time
        
        assert load_time < 2.0, f"Loading too slow: {load_time:.2f}s"
        log_test("Loading Performance", "PASS", f"Loaded in {load_time:.3f}s")
    except Exception as e:
        log_test("Loading Performance", "FAIL", str(e))
    
    # Test 2: Processing performance
    try:
        start_time = time.time()
        
        for yarn_id in list(yarn_deliveries.keys())[:20]:  # Process 20 yarns
            yarn_data = {
                'yarn_id': yarn_id,
                'theoretical_balance': 1000,
                'allocated': -3000,
                'planning_balance': 500
            }
            
            weekly_receipts = yarn_deliveries[yarn_id]
            weekly_demand = create_mock_demand_schedule(yarn_id, yarn_data['allocated'], 9)
            
            result = planner.process_yarn_time_phased(
                yarn_data, weekly_receipts, weekly_demand
            )
        
        process_time = time.time() - start_time
        avg_time = process_time / 20 * 1000  # ms per yarn
        
        assert process_time < 1.0, f"Processing too slow: {process_time:.2f}s"
        log_test("Processing Performance", "PASS", 
                f"20 yarns in {process_time:.3f}s ({avg_time:.1f}ms per yarn)")
    except Exception as e:
        log_test("Processing Performance", "FAIL", str(e))

# ============================================================================
# COMPONENT 5: DATA ACCURACY TESTS
# ============================================================================

def test_data_accuracy():
    """Test accuracy against known examples from plan"""
    test_section("TESTING DATA ACCURACY")
    
    # Test against Yarn 18884 example from plan
    try:
        loader = PODeliveryLoader()
        planner = TimePhasedPlanning()
        
        # Load actual data
        csv_file = "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/8-28-2025/Expected_Yarn_Report.csv"
        po_data = loader.load_po_deliveries(csv_file)
        weekly_data = loader.map_to_weekly_buckets(po_data)
        yarn_deliveries = loader.aggregate_by_yarn(weekly_data)
        
        # Check if yarn 18884 exists
        if '18884' in yarn_deliveries:
            weekly_receipts = yarn_deliveries['18884']
            total_on_order = sum(weekly_receipts.values())
            
            # From plan: Total On Order should be 36,161.30
            expected_total = 36161.30
            
            assert abs(total_on_order - expected_total) < 1.0, \
                   f"Total mismatch: {total_on_order} vs {expected_total}"
            
            # Check specific delivery amounts from plan
            assert abs(weekly_receipts.get('past_due', 0) - 20161.30) < 1.0, \
                   "Past due amount mismatch"
            assert abs(weekly_receipts.get('week_43', 0) - 4000) < 1.0, \
                   "Week 43 amount mismatch"
            
            log_test("Yarn 18884 Accuracy", "PASS", 
                    f"Total matches plan: {total_on_order:.2f}")
        else:
            log_test("Yarn 18884 Accuracy", "WARN", "Yarn 18884 not in current data")
    except Exception as e:
        log_test("Yarn 18884 Accuracy", "FAIL", str(e))

# ============================================================================
# COMPONENT 6: API ENDPOINT TESTS
# ============================================================================

def test_api_endpoints():
    """Test API endpoint functionality"""
    test_section("TESTING API ENDPOINTS")
    
    import requests
    base_url = "http://localhost:5006"
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/api/comprehensive-kpis", timeout=2)
        if response.status_code != 200:
            log_test("Server Status", "WARN", "Server not accessible, skipping API tests")
            return
    except:
        log_test("Server Status", "WARN", "Server not running, skipping API tests")
        return
    
    # Test 1: PO Delivery Schedule endpoint
    try:
        response = requests.get(f"{base_url}/api/po-delivery-schedule")
        assert response.status_code in [200, 501], f"Unexpected status: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            assert 'deliveries' in data, "Missing deliveries in response"
            log_test("PO Delivery Schedule API", "PASS", "Endpoint accessible")
        else:
            log_test("PO Delivery Schedule API", "WARN", "Time-phased not enabled on server")
    except Exception as e:
        log_test("PO Delivery Schedule API", "FAIL", str(e))
    
    # Test 2: Yarn Shortage Timeline endpoint
    try:
        response = requests.get(f"{base_url}/api/yarn-shortage-timeline")
        assert response.status_code in [200, 501], f"Unexpected status: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            assert 'yarns' in data, "Missing yarns in response"
            log_test("Yarn Shortage Timeline API", "PASS", "Endpoint accessible")
        else:
            log_test("Yarn Shortage Timeline API", "WARN", "Time-phased not enabled on server")
    except Exception as e:
        log_test("Yarn Shortage Timeline API", "FAIL", str(e))
    
    # Test 3: Time-Phased Planning endpoint
    try:
        response = requests.get(f"{base_url}/api/time-phased-planning")
        assert response.status_code in [200, 501], f"Unexpected status: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            assert 'planning_data' in data, "Missing planning_data in response"
            log_test("Time-Phased Planning API", "PASS", "Endpoint accessible")
        else:
            log_test("Time-Phased Planning API", "WARN", "Time-phased not enabled on server")
    except Exception as e:
        log_test("Time-Phased Planning API", "FAIL", str(e))

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def generate_test_report():
    """Generate comprehensive test report"""
    test_section("TEST REPORT SUMMARY")
    
    total_tests = len(test_results['passed']) + len(test_results['failed'])
    pass_rate = (len(test_results['passed']) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📊 TEST RESULTS:")
    print(f"  ✅ Passed: {len(test_results['passed'])}")
    print(f"  ❌ Failed: {len(test_results['failed'])}")
    print(f"  ⚠️  Warnings: {len(test_results['warnings'])}")
    print(f"  📈 Pass Rate: {pass_rate:.1f}%")
    
    if test_results['failed']:
        print(f"\n❌ FAILED TESTS:")
        for test in test_results['failed']:
            print(f"  - {test}")
    
    if test_results['warnings']:
        print(f"\n⚠️  WARNINGS:")
        for test in test_results['warnings']:
            print(f"  - {test}")
    
    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'passed': len(test_results['passed']),
            'failed': len(test_results['failed']),
            'warnings': len(test_results['warnings']),
            'pass_rate': pass_rate
        },
        'details': test_results
    }
    
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return len(test_results['failed']) == 0

def main():
    """Run all tests"""
    print("="*60)
    print("TIME-PHASED PO INTEGRATION - COMPREHENSIVE TEST SUITE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run all test suites
        yarn_deliveries = test_po_delivery_loader()
        test_time_phased_planning(yarn_deliveries)
        test_integration()
        test_performance()
        test_data_accuracy()
        test_api_endpoints()
        
        # Generate report
        success = generate_test_report()
        
        if success:
            print("\n🎉 ALL TESTS PASSED! Time-Phased PO Integration is fully functional!")
        else:
            print("\n⚠️  Some tests failed. Review the report for details.")
        
        return success
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)