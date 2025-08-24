#!/usr/bin/env python3
"""
Integration Tests for Critical Business Workflows
Tests end-to-end scenarios that span multiple services
"""

import pytest
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Test configuration
BASE_URL = os.environ.get('TEST_BASE_URL', 'http://localhost:5005')
API_KEY = os.environ.get('TEST_API_KEY', 'test_api_key_123')


class TestYarnProcurementWorkflow:
    """Test the complete yarn procurement workflow"""
    
    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers for requests"""
        return {
            'X-API-Key': API_KEY,
            'Content-Type': 'application/json'
        }
    
    def test_complete_yarn_procurement_cycle(self, auth_headers):
        """
        Test complete workflow:
        1. Check yarn inventory
        2. Identify shortages
        3. Generate procurement plan
        4. Create purchase orders
        5. Update inventory
        """
        
        # Step 1: Check current yarn inventory
        response = requests.get(
            f"{BASE_URL}/api/yarn-intelligence",
            headers=auth_headers
        )
        assert response.status_code == 200
        yarn_data = response.json()
        
        # Verify response structure
        assert 'criticality_analysis' in yarn_data  # Fixed: API returns criticality_analysis
        assert 'recommendations' in yarn_data
        assert 'procurement_plan' in yarn_data
        
        # Step 2: Get shortage details
        if yarn_data.get('criticality_analysis', {}).get('yarns'):  # Fixed: Use correct structure
            critical_yarn = yarn_data['criticality_analysis']['yarns'][0]
            yarn_id = critical_yarn.get('Desc#') or critical_yarn.get('yarn_id')
            
            # Get detailed shortage analysis
            response = requests.get(
                f"{BASE_URL}/api/yarn-shortage-analysis",
                params={'yarn_id': yarn_id},
                headers=auth_headers
            )
            
            if response.status_code == 200:
                shortage_data = response.json()
                assert 'shortage_quantity' in shortage_data or 'status' in shortage_data
        
        # Step 3: Generate procurement recommendations
        response = requests.post(
            f"{BASE_URL}/api/procurement-recommendations",
            json={'urgency': 'high', 'budget_limit': 100000},
            headers=auth_headers
        )
        
        if response.status_code == 200:
            procurement = response.json()
            assert 'recommendations' in procurement or 'items' in procurement
        
        # Step 4: Simulate purchase order creation
        # Check recommendations instead of procurement_plan (which doesn't exist in API)
        if yarn_data.get('recommendations'):
            # Use first critical yarn as urgent item for testing
            urgent_item = {}
            if yarn_data.get('criticality_analysis', {}).get('yarns'):
                urgent_item = yarn_data['criticality_analysis']['yarns'][0]
            
            po_data = {
                'supplier': urgent_item.get('Supplier', 'Default Supplier'),
                'items': [{
                    'yarn_id': urgent_item.get('Desc#', 'TEST001'),
                    'quantity': urgent_item.get('recommended_order', 100),
                    'unit_price': urgent_item.get('Cost/Pound', 5.0)
                }],
                'delivery_date': (datetime.now() + timedelta(days=14)).isoformat()
            }
            
            # Create purchase order (if endpoint exists)
            response = requests.post(
                f"{BASE_URL}/api/purchase-orders",
                json=po_data,
                headers=auth_headers
            )
            
            # Check if endpoint exists
            if response.status_code != 404:
                assert response.status_code in [200, 201]
                if response.status_code in [200, 201]:
                    po_response = response.json()
                    assert 'order_id' in po_response or 'id' in po_response
    
    def test_yarn_substitution_workflow(self, auth_headers):
        """Test yarn substitution recommendation workflow"""
        
        # Step 1: Get yarn alternatives
        response = requests.get(
            f"{BASE_URL}/api/yarn-alternatives",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        alternatives = response.json()
        
        # Verify structure
        assert isinstance(alternatives, (dict, list))
        
        # Step 2: If alternatives exist, test substitution validation
        if isinstance(alternatives, dict) and 'substitutions' in alternatives:
            if alternatives['substitutions']:
                substitution = alternatives['substitutions'][0]
                
                # Validate substitution
                validation_data = {
                    'original_yarn': substitution.get('original'),
                    'substitute_yarn': substitution.get('substitute'),
                    'quantity_needed': 100
                }
                
                response = requests.post(
                    f"{BASE_URL}/api/validate-substitution",
                    json=validation_data,
                    headers=auth_headers
                )
                
                if response.status_code != 404:
                    assert response.status_code == 200
                    validation = response.json()
                    assert 'is_valid' in validation or 'valid' in validation


class TestProductionPlanningWorkflow:
    """Test production planning and scheduling workflows"""
    
    @pytest.fixture
    def sample_production_order(self):
        """Sample production order for testing"""
        return {
            'order_id': 'TEST_ORDER_001',
            'products': [
                {
                    'product_id': 'KNIT_001',
                    'quantity': 500,
                    'due_date': (datetime.now() + timedelta(days=30)).isoformat()
                },
                {
                    'product_id': 'KNIT_002',
                    'quantity': 300,
                    'due_date': (datetime.now() + timedelta(days=45)).isoformat()
                }
            ],
            'priority': 'high'
        }
    
    def test_six_phase_planning_workflow(self):
        """Test the complete 6-phase planning process"""
        
        # Execute 6-phase planning
        response = requests.get(f"{BASE_URL}/api/six-phase-planning")
        
        assert response.status_code == 200
        planning_result = response.json()
        
        # Verify all 6 phases are present
        expected_phases = [
            'demand_analysis',
            'bom_explosion',
            'capacity_planning',
            'supply_chain',
            'production_scheduling',
            'execution_monitoring'
        ]
        
        for phase in expected_phases:
            assert phase in planning_result or f'phase_{phase}' in planning_result
    
    def test_capacity_planning_workflow(self, sample_production_order):
        """Test capacity planning and bottleneck identification"""
        
        # Step 1: Submit production requirements
        response = requests.post(
            f"{BASE_URL}/api/capacity-requirements",
            json=sample_production_order
        )
        
        if response.status_code != 404:
            assert response.status_code == 200
            capacity_req = response.json()
            
            # Step 2: Check for bottlenecks
            response = requests.get(f"{BASE_URL}/api/capacity-bottlenecks")
            
            if response.status_code == 200:
                bottlenecks = response.json()
                assert isinstance(bottlenecks, (dict, list))
                
                # Step 3: Get optimization recommendations
                if bottlenecks:
                    response = requests.post(
                        f"{BASE_URL}/api/optimize-capacity",
                        json={'bottlenecks': bottlenecks}
                    )
                    
                    if response.status_code != 404:
                        assert response.status_code == 200
    
    def test_production_scheduling_workflow(self, sample_production_order):
        """Test production scheduling optimization"""
        
        # Step 1: Create production schedule
        response = requests.post(
            f"{BASE_URL}/api/production-schedule",
            json=sample_production_order
        )
        
        if response.status_code != 404:
            assert response.status_code in [200, 201]
            schedule = response.json()
            
            # Verify schedule structure
            assert 'schedule' in schedule or 'production_schedule' in schedule
            
            # Step 2: Check schedule conflicts
            response = requests.get(f"{BASE_URL}/api/schedule-conflicts")
            
            if response.status_code == 200:
                conflicts = response.json()
                assert isinstance(conflicts, (dict, list))


class TestForecastingWorkflow:
    """Test ML forecasting and demand planning workflows"""
    
    def test_demand_forecasting_workflow(self):
        """Test complete demand forecasting workflow"""
        
        # Step 1: Get current forecast
        response = requests.get(
            f"{BASE_URL}/api/ml-forecast",
            params={'horizon': 30, 'product_id': 'all'}
        )
        
        assert response.status_code == 200
        forecast = response.json()
        
        # Verify forecast structure
        assert 'forecast' in forecast or 'predictions' in forecast
        assert 'confidence' in forecast or 'accuracy' in forecast
        
        # Step 2: Get forecast accuracy metrics
        response = requests.get(f"{BASE_URL}/api/forecast-accuracy")
        
        if response.status_code == 200:
            accuracy = response.json()
            assert 'mape' in accuracy or 'accuracy' in accuracy
        
        # Step 3: Trigger model retraining if accuracy is low
        if response.status_code == 200:
            accuracy_value = accuracy.get('accuracy', accuracy.get('mape', 100))
            
            if accuracy_value < 85:  # If accuracy below 85%
                response = requests.post(
                    f"{BASE_URL}/api/retrain-ml",
                    json={'model': 'ensemble', 'force': True}
                )
                
                if response.status_code != 404:
                    assert response.status_code in [200, 202]
    
    def test_seasonal_adjustment_workflow(self):
        """Test seasonal demand adjustment workflow"""
        
        # Step 1: Get seasonal patterns
        response = requests.get(
            f"{BASE_URL}/api/seasonal-patterns",
            params={'product_category': 'knit'}
        )
        
        if response.status_code == 200:
            patterns = response.json()
            assert isinstance(patterns, dict)
            
            # Step 2: Apply seasonal adjustments
            if patterns:
                response = requests.post(
                    f"{BASE_URL}/api/apply-seasonal-adjustment",
                    json={'patterns': patterns, 'forecast_horizon': 90}
                )
                
                if response.status_code != 404:
                    assert response.status_code == 200


class TestInventoryManagementWorkflow:
    """Test inventory management and optimization workflows"""
    
    def test_inventory_optimization_workflow(self):
        """Test complete inventory optimization cycle"""
        
        # Step 1: Get current inventory status
        response = requests.get(f"{BASE_URL}/api/inventory-analysis")
        assert response.status_code == 200
        inventory = response.json()
        
        # Step 2: Identify optimization opportunities
        response = requests.get(f"{BASE_URL}/api/inventory-optimization")
        
        if response.status_code == 200:
            optimization = response.json()
            
            # Step 3: Calculate EOQ for items
            if 'items_to_reorder' in optimization:
                for item in optimization['items_to_reorder'][:5]:  # Test first 5
                    response = requests.post(
                        f"{BASE_URL}/api/calculate-eoq",
                        json={
                            'item_id': item.get('id'),
                            'demand': item.get('monthly_demand', 100)
                        }
                    )
                    
                    if response.status_code != 404:
                        assert response.status_code == 200
                        eoq = response.json()
                        assert 'eoq' in eoq or 'quantity' in eoq
    
    def test_stock_alert_workflow(self):
        """Test stock alert and emergency procurement workflow"""
        
        # Step 1: Get emergency items
        response = requests.get(f"{BASE_URL}/api/emergency-yarns")
        assert response.status_code == 200
        emergency = response.json()
        
        if 'emergency_items' in emergency and emergency['emergency_items']:
            # Step 2: Create emergency purchase order
            emergency_item = emergency['emergency_items'][0]
            
            po_data = {
                'type': 'emergency',
                'items': [{
                    'yarn_id': emergency_item.get('Desc#', 'EMERGENCY_001'),
                    'quantity': emergency_item.get('shortage', 100),
                    'expedited': True
                }]
            }
            
            response = requests.post(
                f"{BASE_URL}/api/emergency-procurement",
                json=po_data
            )
            
            if response.status_code != 404:
                assert response.status_code in [200, 201]


class TestDataSynchronizationWorkflow:
    """Test data synchronization and validation workflows"""
    
    def test_data_sync_workflow(self):
        """Test complete data synchronization process"""
        
        # Step 1: Check sync status
        response = requests.get(f"{BASE_URL}/api/sync-status")
        
        if response.status_code == 200:
            sync_status = response.json()
            assert 'last_sync' in sync_status or 'status' in sync_status
            
            # Step 2: Trigger sync if needed
            if sync_status.get('needs_sync', False):
                response = requests.post(
                    f"{BASE_URL}/api/trigger-sync",
                    json={'source': 'sharepoint', 'force': False}
                )
                
                if response.status_code != 404:
                    assert response.status_code in [200, 202]
                    
                    # Step 3: Monitor sync progress
                    max_attempts = 10
                    for _ in range(max_attempts):
                        time.sleep(2)  # Wait 2 seconds
                        
                        response = requests.get(f"{BASE_URL}/api/sync-progress")
                        if response.status_code == 200:
                            progress = response.json()
                            if progress.get('status') == 'completed':
                                break
    
    def test_data_validation_workflow(self):
        """Test data validation and cleansing workflow"""
        
        # Step 1: Validate current data
        response = requests.get(f"{BASE_URL}/api/validate-data")
        
        if response.status_code == 200:
            validation = response.json()
            assert 'is_valid' in validation or 'valid' in validation
            
            # Step 2: If invalid, get detailed errors
            if not validation.get('is_valid', validation.get('valid', True)):
                response = requests.get(f"{BASE_URL}/api/data-errors")
                
                if response.status_code == 200:
                    errors = response.json()
                    assert isinstance(errors, (dict, list))
                    
                    # Step 3: Attempt auto-fix
                    response = requests.post(
                        f"{BASE_URL}/api/auto-fix-data",
                        json={'errors': errors[:10]}  # Fix first 10 errors
                    )
                    
                    if response.status_code != 404:
                        assert response.status_code == 200


class TestEndToEndOrderFulfillment:
    """Test complete order fulfillment workflow"""
    
    def test_order_to_delivery_workflow(self):
        """
        Test complete workflow from order to delivery:
        1. Receive customer order
        2. Check inventory availability
        3. Plan production if needed
        4. Allocate materials
        5. Schedule production
        6. Track progress
        7. Complete order
        """
        
        # Step 1: Create customer order
        order_data = {
            'customer_id': 'CUST_001',
            'items': [
                {
                    'style_id': 'STYLE_001',
                    'quantity': 100,
                    'due_date': (datetime.now() + timedelta(days=30)).isoformat()
                }
            ],
            'priority': 'normal'
        }
        
        response = requests.post(
            f"{BASE_URL}/api/sales-orders",
            json=order_data
        )
        
        if response.status_code in [200, 201]:
            order = response.json()
            order_id = order.get('order_id', order.get('id'))
            
            # Step 2: Check material availability
            response = requests.get(
                f"{BASE_URL}/api/check-availability",
                params={'order_id': order_id}
            )
            
            if response.status_code == 200:
                availability = response.json()
                
                # Step 3: If materials not available, plan procurement
                if not availability.get('all_available', True):
                    response = requests.post(
                        f"{BASE_URL}/api/plan-procurement",
                        json={'order_id': order_id}
                    )
                    
                    if response.status_code != 404:
                        assert response.status_code == 200
                
                # Step 4: Schedule production
                response = requests.post(
                    f"{BASE_URL}/api/schedule-order",
                    json={'order_id': order_id}
                )
                
                if response.status_code != 404:
                    assert response.status_code == 200
                    
                    # Step 5: Track order status
                    response = requests.get(
                        f"{BASE_URL}/api/order-status",
                        params={'order_id': order_id}
                    )
                    
                    if response.status_code == 200:
                        status = response.json()
                        assert 'status' in status


class TestPerformanceAndLoad:
    """Test system performance under load"""
    
    def test_concurrent_api_calls(self):
        """Test system handles concurrent API calls"""
        import concurrent.futures
        
        def make_api_call(endpoint):
            """Make a single API call"""
            try:
                response = requests.get(f"{BASE_URL}{endpoint}")
                return response.status_code == 200
            except:
                return False
        
        # Test endpoints
        endpoints = [
            '/api/inventory-analysis',
            '/api/yarn-intelligence',
            '/api/production-pipeline',
            '/api/ml-forecast',
            '/api/six-phase-planning'
        ]
        
        # Make concurrent calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(5):  # 5 rounds
                for endpoint in endpoints:
                    futures.append(executor.submit(make_api_call, endpoint))
            
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # At least 80% should succeed
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8
    
    def test_large_data_handling(self):
        """Test system handles large data requests"""
        
        # Request large dataset
        response = requests.get(
            f"{BASE_URL}/api/inventory-analysis",
            params={'limit': 10000}
        )
        
        assert response.status_code == 200
        
        # Check response time (should be under 5 seconds)
        start = time.time()
        response = requests.get(
            f"{BASE_URL}/api/yarn-intelligence",
            params={'include_all': True}
        )
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 5.0  # Should respond within 5 seconds


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])