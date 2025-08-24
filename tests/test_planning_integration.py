#!/usr/bin/env python3
"""
Comprehensive Test Suite for 6-Phase Planning Integration
Tests database, APIs, frontend integration, and refresh mechanism
"""

import unittest
import json
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Note: planning_integration_db and planning_data_api modules not found in codebase
# from planning_integration_db import PlanningIntegrationDB
# from planning_data_api import PlanningDataAPI
# This test file may need to be updated or removed

class TestPlanningIntegrationDB(unittest.TestCase):
    """Test database operations"""
    
    def setUp(self):
        """Set up test database"""
        self.db = PlanningIntegrationDB('test_planning.db')
    
    def tearDown(self):
        """Clean up test database"""
        self.db.close()
        # Remove test database
        if Path('test_planning.db').exists():
            Path('test_planning.db').unlink()
    
    def test_create_execution(self):
        """Test creating a planning execution"""
        execution_id = self.db.create_planning_execution('test', 'test_user')
        self.assertIsNotNone(execution_id)
        
        # Verify it was created
        latest = self.db.get_latest_execution(status=None)
        self.assertEqual(latest['execution_id'], execution_id)
        self.assertEqual(latest['triggered_by'], 'test')
    
    def test_update_execution_status(self):
        """Test updating execution status"""
        execution_id = self.db.create_planning_execution('test', 'test_user')
        
        # Update to running
        self.db.update_execution_status(execution_id, 'running', phases_completed=3)
        
        # Verify update
        latest = self.db.get_latest_execution(status=None)
        self.assertEqual(latest['execution_status'], 'running')
        self.assertEqual(latest['total_phases_completed'], 3)
    
    def test_store_phase1_data(self):
        """Test storing Phase 1 demand data"""
        execution_id = self.db.create_planning_execution('test', 'test_user')
        
        demand_data = [
            {
                'style_number': 'TEST-001',
                'quantity_yards': 1000,
                'delivery_date': '2025-09-01',
                'confidence_score': 0.85,
                'priority_rank': 1
            }
        ]
        
        self.db.store_phase1_data(execution_id, demand_data)
        
        # Verify data was stored
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as count FROM phase1_demand_consolidation
            WHERE execution_id = ?
        """, (execution_id,))
        
        result = cursor.fetchone()
        self.assertEqual(result['count'], 1)
    
    def test_store_phase2_data(self):
        """Test storing Phase 2 inventory data"""
        execution_id = self.db.create_planning_execution('test', 'test_user')
        
        inventory_data = [
            {
                'inventory_type': 'yarn',
                'item_id': 'YARN-001',
                'current_quantity': 500,
                'available_quantity': 300,
                'planning_balance': -200,
                'days_of_supply': 5
            }
        ]
        
        self.db.store_phase2_data(execution_id, inventory_data)
        
        # Verify data was stored
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT * FROM phase2_inventory_assessment
            WHERE execution_id = ?
        """, (execution_id,))
        
        result = cursor.fetchone()
        self.assertEqual(result['item_id'], 'YARN-001')
        self.assertEqual(result['planning_balance'], -200)
    
    def test_get_phase_data_for_tab(self):
        """Test retrieving phase data for specific tab"""
        execution_id = self.db.create_planning_execution('test', 'test_user')
        
        # Store some test data
        inventory_data = [
            {
                'inventory_type': 'yarn',
                'item_id': f'YARN-{i:03d}',
                'current_quantity': 500 - i*50,
                'available_quantity': 300 - i*30,
                'planning_balance': 100 - i*100,
                'days_of_supply': 10 - i
            }
            for i in range(5)
        ]
        
        self.db.store_phase2_data(execution_id, inventory_data)
        
        # Get data for inventory tab
        data = self.db.get_phase_data_for_tab(execution_id, 2, 'inventory')
        
        self.assertIsNotNone(data)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
    
    def test_cleanup_old_executions(self):
        """Test cleaning up old executions"""
        # Create an old execution
        execution_id = self.db.create_planning_execution('test', 'test_user')
        
        # Manually set timestamp to 100 days ago
        cursor = self.db.conn.cursor()
        old_date = datetime.now() - timedelta(days=100)
        cursor.execute("""
            UPDATE planning_executions 
            SET execution_timestamp = ?
            WHERE execution_id = ?
        """, (old_date, execution_id))
        self.db.conn.commit()
        
        # Clean up executions older than 90 days
        self.db.cleanup_old_executions(90)
        
        # Verify it was deleted
        latest = self.db.get_latest_execution(status=None)
        self.assertIsNone(latest)


class TestPlanningDataAPI(unittest.TestCase):
    """Test API functionality"""
    
    def setUp(self):
        """Set up test API handler"""
        # Use test database
        self.api_handler = PlanningDataAPI()
        self.api_handler.db = PlanningIntegrationDB('test_planning_api.db')
    
    def tearDown(self):
        """Clean up"""
        self.api_handler.db.close()
        if Path('test_planning_api.db').exists():
            Path('test_planning_api.db').unlink()
    
    def test_get_execution_summary(self):
        """Test getting execution summary"""
        # Create test execution with data
        execution_id = self.api_handler.db.create_planning_execution('test', 'test_user')
        self.api_handler.db.update_execution_status(execution_id, 'completed', phases_completed=6)
        
        # Add some phase data
        self.api_handler.db.store_phase1_data(execution_id, [
            {'style_number': 'TEST-001', 'quantity_yards': 1000, 'delivery_date': '2025-09-01'}
        ])
        
        # Get summary
        summary = self.api_handler.get_execution_summary(execution_id)
        
        self.assertTrue(summary['success'])
        self.assertIn('execution', summary)
        self.assertIn('summaries', summary)
        self.assertEqual(summary['execution']['execution_id'], execution_id)
    
    def test_get_tab_data(self):
        """Test getting data for a specific tab"""
        # Create test execution with data
        execution_id = self.api_handler.db.create_planning_execution('test', 'test_user')
        
        # Add inventory data
        self.api_handler.db.store_phase2_data(execution_id, [
            {
                'inventory_type': 'yarn',
                'item_id': 'YARN-001',
                'current_quantity': 500,
                'available_quantity': 300,
                'planning_balance': -200
            }
        ])
        
        # Get inventory tab data
        result = self.api_handler.get_tab_data('inventory', execution_id)
        
        self.assertTrue(result['success'])
        self.assertIn('data', result)
        self.assertIn('phase2', result['data'])
    
    def test_compare_executions(self):
        """Test comparing two executions"""
        # Create first execution
        exec1 = self.api_handler.db.create_planning_execution('test', 'test_user')
        self.api_handler.db.store_phase1_data(exec1, [
            {'style_number': 'TEST-001', 'quantity_yards': 1000, 'delivery_date': '2025-09-01'}
        ])
        
        # Create second execution with different data
        exec2 = self.api_handler.db.create_planning_execution('test', 'test_user')
        self.api_handler.db.store_phase1_data(exec2, [
            {'style_number': 'TEST-001', 'quantity_yards': 1200, 'delivery_date': '2025-09-01'},
            {'style_number': 'TEST-002', 'quantity_yards': 800, 'delivery_date': '2025-09-15'}
        ])
        
        # Compare
        result = self.api_handler.compare_executions(exec1, exec2)
        
        self.assertTrue(result['success'])
        self.assertIn('comparison', result)
        self.assertIn('differences', result['comparison'])


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.db = PlanningIntegrationDB('test_integration.db')
        self.api_handler = PlanningDataAPI()
        self.api_handler.db = self.db
    
    def tearDown(self):
        """Clean up"""
        self.db.close()
        if Path('test_integration.db').exists():
            Path('test_integration.db').unlink()
    
    def test_full_planning_flow(self):
        """Test complete planning execution flow"""
        # Create execution
        execution_id = self.db.create_planning_execution('integration_test', 'test_user')
        
        # Simulate phase data storage
        phases_data = {
            'phase1': [
                {'style_number': f'STYLE-{i:03d}', 'quantity_yards': 1000 + i*100, 
                 'delivery_date': '2025-09-01', 'confidence_score': 0.8 + i*0.02}
                for i in range(5)
            ],
            'phase2': [
                {'inventory_type': 'yarn', 'item_id': f'YARN-{i:03d}',
                 'current_quantity': 500 + i*50, 'available_quantity': 300 + i*30,
                 'planning_balance': 100 - i*50}
                for i in range(10)
            ],
            'phase3': [
                {'style_number': f'STYLE-{i:03d}', 'time_period': '2025-09-01',
                 'gross_demand_yards': 1000 + i*100, 'net_requirement_yards': 500 + i*50,
                 'stockout_risk_level': 'MEDIUM', 'requirement_date': '2025-09-01'}
                for i in range(5)
            ],
            'phase4': [
                {'style_number': f'STYLE-{i:03d}', 'yarn_id': f'YARN-{j:03d}',
                 'bom_percentage': 20, 'base_requirement_lbs': 100 + i*10 + j*5,
                 'total_requirement_lbs': 105 + i*10 + j*5, 'shortage_lbs': max(0, i*10 - 20)}
                for i in range(3) for j in range(5)
            ],
            'phase5_procurement': [
                {'yarn_id': f'YARN-{i:03d}', 'supplier_name': f'Supplier {i}',
                 'order_quantity_lbs': 200 + i*50, 'need_by_date': '2025-08-15',
                 'priority_level': 'HIGH' if i < 3 else 'MEDIUM'}
                for i in range(5)
            ],
            'phase5_production': [
                {'style_number': f'STYLE-{i:03d}', 'quantity_lbs': 500 + i*100,
                 'suggested_start_date': '2025-08-01', 'priority_level': 'HIGH'}
                for i in range(3)
            ],
            'phase6': [
                {'metric_name': f'KPI_{i}', 'metric_value': 85 + i*2,
                 'metric_category': 'Operational', 'is_kpi': True,
                 'target_value': 90, 'trend_direction': 'improving'}
                for i in range(5)
            ]
        }
        
        # Store all phase data
        self.db.store_phase1_data(execution_id, phases_data['phase1'])
        self.db.store_phase2_data(execution_id, phases_data['phase2'])
        self.db.store_phase3_data(execution_id, phases_data['phase3'])
        self.db.store_phase4_data(execution_id, phases_data['phase4'])
        self.db.store_phase5_procurement(execution_id, phases_data['phase5_procurement'])
        self.db.store_phase5_production(execution_id, phases_data['phase5_production'])
        self.db.store_phase6_data(execution_id, phases_data['phase6'])
        
        # Update execution status
        self.db.update_execution_status(execution_id, 'completed', 
                                       phases_completed=6, duration_seconds=120.5)
        
        # Test retrieving summary
        summary = self.api_handler.get_execution_summary(execution_id)
        self.assertTrue(summary['success'])
        self.assertEqual(summary['execution']['total_phases_completed'], 6)
        
        # Test tab data retrieval
        tabs = ['overview', 'production', 'inventory', 'planning', 
                'forecasting', 'analytics', 'suppliers', 'knit-orders']
        
        for tab in tabs:
            result = self.api_handler.get_tab_data(tab, execution_id)
            self.assertTrue(result['success'], f"Failed to get data for {tab} tab")
            self.assertIn('data', result)
    
    def test_performance_benchmarks(self):
        """Test performance meets requirements"""
        # Create execution with large dataset
        execution_id = self.db.create_planning_execution('performance_test', 'test_user')
        
        # Generate large dataset (1000+ items as per requirements)
        start_time = time.time()
        
        large_inventory = [
            {'inventory_type': 'yarn', 'item_id': f'YARN-{i:04d}',
             'current_quantity': 500 + i, 'available_quantity': 300 + i,
             'planning_balance': 100 - i}
            for i in range(1200)  # 1200 yarns as per actual system
        ]
        
        self.db.store_phase2_data(execution_id, large_inventory)
        
        storage_time = time.time() - start_time
        
        # Storage should be fast (under 5 seconds for 1200 items)
        self.assertLess(storage_time, 5, f"Storage took {storage_time:.2f}s, should be < 5s")
        
        # Test retrieval performance
        start_time = time.time()
        data = self.db.get_phase_data_for_tab(execution_id, 2, 'inventory')
        retrieval_time = time.time() - start_time
        
        # Retrieval should be under 200ms as per requirements
        self.assertLess(retrieval_time, 0.2, f"Retrieval took {retrieval_time:.3f}s, should be < 0.2s")
        
        # Verify data integrity
        self.assertEqual(len(data), 1200)


class TestDataValidation(unittest.TestCase):
    """Test data validation and business rules"""
    
    def setUp(self):
        """Set up validation tests"""
        self.db = PlanningIntegrationDB('test_validation.db')
    
    def tearDown(self):
        """Clean up"""
        self.db.close()
        if Path('test_validation.db').exists():
            Path('test_validation.db').unlink()
    
    def test_planning_balance_calculation(self):
        """Test Planning Balance calculation with negative Allocated"""
        execution_id = self.db.create_planning_execution('test', 'test_user')
        
        # Test data with ALREADY NEGATIVE allocated (as per actual data)
        inventory_data = [
            {
                'inventory_type': 'yarn',
                'item_id': '18868',  # Critical yarn from actual system
                'current_quantity': 1000,
                'allocated_quantity': -494,  # Already negative!
                'on_order_quantity': 0,
                'available_quantity': 506,
                'planning_balance': -494,  # Should be negative (shortage)
                'days_of_supply': 0
            }
        ]
        
        self.db.store_phase2_data(execution_id, inventory_data)
        
        # Verify correct storage
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT * FROM phase2_inventory_assessment
            WHERE execution_id = ? AND item_id = '18868'
        """, (execution_id,))
        
        result = cursor.fetchone()
        self.assertEqual(result['planning_balance'], -494)
        self.assertEqual(result['allocated_quantity'], -494)
    
    def test_risk_level_validation(self):
        """Test stockout risk level validation"""
        execution_id = self.db.create_planning_execution('test', 'test_user')
        
        requirements_data = [
            {
                'style_number': 'TEST-001',
                'time_period': '2025-09-01',
                'gross_demand_yards': 1000,
                'net_requirement_yards': 1000,
                'stockout_risk_level': 'CRITICAL',  # Valid value
                'requirement_date': '2025-09-01'
            }
        ]
        
        # Should succeed
        self.db.store_phase3_data(execution_id, requirements_data)
        
        # Test invalid risk level
        bad_data = requirements_data.copy()
        bad_data[0]['stockout_risk_level'] = 'INVALID'
        
        # Should handle gracefully (default to MEDIUM)
        self.db.store_phase3_data(execution_id, bad_data)
    
    def test_priority_level_validation(self):
        """Test priority level validation"""
        execution_id = self.db.create_planning_execution('test', 'test_user')
        
        procurement_data = [
            {
                'yarn_id': 'YARN-001',
                'supplier_name': 'Test Supplier',
                'order_quantity_lbs': 500,
                'need_by_date': '2025-08-15',
                'priority_level': 'CRITICAL'  # Valid value
            }
        ]
        
        # Should succeed
        self.db.store_phase5_procurement(execution_id, procurement_data)


def run_all_tests():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPlanningIntegrationDB))
    suite.addTests(loader.loadTestsFromTestCase(TestPlanningDataAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailed Tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nTests with Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)