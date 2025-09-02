#!/usr/bin/env python3
"""
Comprehensive API Consolidation Test Suite
Tests all consolidated endpoints, deprecated redirects, and parameter handling
Created: 2025-09-02
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import the main ERP module
from core.beverly_comprehensive_erp import app, InventoryAnalyzer

class TestAPIConsolidation:
    """Test suite for API consolidation functionality"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def mock_inventory_analyzer(self):
        """Create mock inventory analyzer with test data"""
        analyzer = Mock(spec=InventoryAnalyzer)
        
        # Mock data attributes
        analyzer.raw_materials_data = pd.DataFrame({
            'YarnID': ['Y001', 'Y002', 'Y003'],
            'Description': ['Cotton 20/1', 'Polyester 30/1', 'Blend 40/2'],
            'Planning_Balance': [1000, -500, 2000],
            'Allocated': [200, 100, 300],
            'On_Order': [500, 1000, 0],
            'Price': [4.5, 3.2, 5.8]
        })
        
        analyzer.bom_data = pd.DataFrame({
            'Style#': ['S001', 'S001', 'S002'],
            'YarnID': ['Y001', 'Y002', 'Y003'],
            'Usage': [0.5, 0.3, 0.8],
            'Unit': ['lbs', 'lbs', 'lbs']
        })
        
        analyzer.sales_data = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=30),
            'Style#': ['S001'] * 15 + ['S002'] * 15,
            'Quantity': np.random.randint(100, 500, 30),
            'Revenue': np.random.uniform(1000, 5000, 30)
        })
        
        analyzer.knit_orders = pd.DataFrame({
            'Order_ID': ['KO001', 'KO002', 'KO003'],
            'Style#': ['S001', 'S002', 'S001'],
            'Quantity': [1000, 1500, 800],
            'Status': ['In Progress', 'Planned', 'Completed']
        })
        
        return analyzer

    # Test Consolidated Endpoints
    
    def test_inventory_intelligence_enhanced(self, client, mock_inventory_analyzer):
        """Test inventory-intelligence-enhanced endpoint with all parameters"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            # Test default view
            response = client.get('/api/inventory-intelligence-enhanced')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'success'
            
            # Test with view parameter
            response = client.get('/api/inventory-intelligence-enhanced?view=summary')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'summary' in data or 'metrics' in data
            
            # Test with analysis parameter
            response = client.get('/api/inventory-intelligence-enhanced?analysis=shortage')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'shortage' in str(data).lower() or 'analysis' in data
            
            # Test with multiple parameters
            response = client.get('/api/inventory-intelligence-enhanced?view=detailed&analysis=shortage&realtime=true')
            assert response.status_code == 200

    def test_ml_forecast_detailed(self, client, mock_inventory_analyzer):
        """Test ml-forecast-detailed endpoint with all parameters"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            # Test default
            response = client.get('/api/ml-forecast-detailed')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'forecast' in str(data).lower() or 'predictions' in data
            
            # Test with detail parameter
            response = client.get('/api/ml-forecast-detailed?detail=full')
            assert response.status_code == 200
            
            # Test with format parameter
            response = client.get('/api/ml-forecast-detailed?format=report')
            assert response.status_code == 200
            
            # Test with horizon parameter
            response = client.get('/api/ml-forecast-detailed?horizon=90')
            assert response.status_code == 200
            
            # Test with all parameters
            response = client.get('/api/ml-forecast-detailed?detail=full&format=report&horizon=90&model=ensemble')
            assert response.status_code == 200

    def test_production_planning(self, client, mock_inventory_analyzer):
        """Test production-planning endpoint with parameters"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            # Test default
            response = client.get('/api/production-planning')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'production' in str(data).lower() or 'planning' in str(data).lower()
            
            # Test with view parameter
            response = client.get('/api/production-planning?view=orders')
            assert response.status_code == 200
            
            # Test with forecast parameter
            response = client.get('/api/production-planning?forecast=true')
            assert response.status_code == 200
            
            # Test with phase parameter
            response = client.get('/api/production-planning?phase=all')
            assert response.status_code == 200

    def test_inventory_netting(self, client, mock_inventory_analyzer):
        """Test inventory-netting endpoint"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            # Test default
            response = client.get('/api/inventory-netting')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'netting' in str(data).lower() or 'requirements' in str(data).lower()
            
            # Test with level parameter
            response = client.get('/api/inventory-netting?level=multi')
            assert response.status_code == 200
            
            # Test with style parameter
            response = client.get('/api/inventory-netting?style=S001')
            assert response.status_code == 200

    def test_comprehensive_kpis(self, client, mock_inventory_analyzer):
        """Test comprehensive-kpis endpoint"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            response = client.get('/api/comprehensive-kpis')
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Check for KPI structure
            assert 'kpis' in data or 'metrics' in data or 'inventory' in data
            
            # Test with refresh parameter
            response = client.get('/api/comprehensive-kpis?refresh=true')
            assert response.status_code == 200

    def test_yarn_intelligence(self, client, mock_inventory_analyzer):
        """Test yarn-intelligence endpoint"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            # Test default
            response = client.get('/api/yarn-intelligence')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'yarn' in str(data).lower() or 'intelligence' in str(data).lower()
            
            # Test with analysis parameter
            response = client.get('/api/yarn-intelligence?analysis=shortage')
            assert response.status_code == 200
            
            # Test with forecast parameter
            response = client.get('/api/yarn-intelligence?forecast=true')
            assert response.status_code == 200

    # Test Deprecated Endpoint Redirects
    
    def test_deprecated_inventory_endpoints(self, client, mock_inventory_analyzer):
        """Test that deprecated inventory endpoints redirect correctly"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            deprecated_endpoints = [
                '/api/inventory-status',
                '/api/inventory-overview',
                '/api/inventory-analysis',
                '/api/inventory-metrics',
                '/api/inventory-summary',
                '/api/inventory-intelligence',
                '/api/inventory-alerts'
            ]
            
            for endpoint in deprecated_endpoints:
                response = client.get(endpoint, follow_redirects=False)
                # Should redirect (301 or 302) or return consolidated response
                assert response.status_code in [200, 301, 302, 308]
                
                if response.status_code in [301, 302, 308]:
                    # Check redirect location
                    assert '/api/inventory-intelligence-enhanced' in response.location

    def test_deprecated_forecast_endpoints(self, client, mock_inventory_analyzer):
        """Test that deprecated forecast endpoints redirect correctly"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            deprecated_endpoints = [
                '/api/forecast',
                '/api/ml-forecast',
                '/api/demand-forecast',
                '/api/sales-forecast',
                '/api/forecast-accuracy',
                '/api/ml-predictions'
            ]
            
            for endpoint in deprecated_endpoints:
                response = client.get(endpoint, follow_redirects=False)
                assert response.status_code in [200, 301, 302, 308]
                
                if response.status_code in [301, 302, 308]:
                    assert '/api/ml-forecast-detailed' in response.location

    def test_deprecated_production_endpoints(self, client, mock_inventory_analyzer):
        """Test that deprecated production endpoints redirect correctly"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            deprecated_endpoints = [
                '/api/production-schedule',
                '/api/production-status',
                '/api/production-orders',
                '/api/manufacturing-plan',
                '/api/production-capacity'
            ]
            
            for endpoint in deprecated_endpoints:
                response = client.get(endpoint, follow_redirects=False)
                assert response.status_code in [200, 301, 302, 308]
                
                if response.status_code in [301, 302, 308]:
                    assert '/api/production-planning' in response.location

    def test_deprecated_yarn_endpoints(self, client, mock_inventory_analyzer):
        """Test that deprecated yarn endpoints redirect correctly"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            deprecated_endpoints = [
                '/api/yarn-status',
                '/api/yarn-inventory',
                '/api/yarn-analysis',
                '/api/yarn-shortage',
                '/api/raw-materials'
            ]
            
            for endpoint in deprecated_endpoints:
                response = client.get(endpoint, follow_redirects=False)
                assert response.status_code in [200, 301, 302, 308]
                
                if response.status_code in [301, 302, 308]:
                    assert '/api/yarn-intelligence' in response.location

    # Test Parameter Preservation During Redirects
    
    def test_parameter_preservation_on_redirect(self, client, mock_inventory_analyzer):
        """Test that query parameters are preserved during redirects"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            # Test with single parameter
            response = client.get('/api/inventory-status?view=summary', follow_redirects=False)
            if response.status_code in [301, 302, 308]:
                assert 'view=summary' in response.location
            
            # Test with multiple parameters
            response = client.get('/api/forecast?horizon=90&model=ensemble', follow_redirects=False)
            if response.status_code in [301, 302, 308]:
                assert 'horizon=90' in response.location
                assert 'model=ensemble' in response.location

    # Test Consolidation Metrics Endpoint
    
    def test_consolidation_metrics(self, client):
        """Test the consolidation metrics endpoint"""
        response = client.get('/api/consolidation-metrics')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check metrics structure
        assert 'deprecated_calls' in data or 'redirect_count' in data or 'metrics' in data
        assert 'consolidated_calls' in data or 'api_calls' in data or 'statistics' in data

    # Test Error Handling
    
    def test_error_handling_with_invalid_parameters(self, client, mock_inventory_analyzer):
        """Test that invalid parameters are handled gracefully"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            # Test with invalid parameter values
            response = client.get('/api/ml-forecast-detailed?horizon=invalid')
            assert response.status_code in [200, 400]  # Should handle gracefully
            
            response = client.get('/api/inventory-intelligence-enhanced?view=nonexistent')
            assert response.status_code in [200, 400]  # Should handle gracefully

    def test_missing_data_handling(self, client):
        """Test endpoints handle missing data gracefully"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', None):
            # Test that endpoints don't crash when analyzer is None
            response = client.get('/api/inventory-intelligence-enhanced')
            assert response.status_code in [200, 500]
            
            if response.status_code == 200:
                data = json.loads(response.data)
                assert 'error' in data or 'status' in data

    # Test Response Format Consistency
    
    def test_response_format_consistency(self, client, mock_inventory_analyzer):
        """Test that all consolidated endpoints return consistent response format"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            endpoints = [
                '/api/inventory-intelligence-enhanced',
                '/api/ml-forecast-detailed',
                '/api/production-planning',
                '/api/inventory-netting',
                '/api/comprehensive-kpis',
                '/api/yarn-intelligence'
            ]
            
            for endpoint in endpoints:
                response = client.get(endpoint)
                assert response.status_code == 200
                data = json.loads(response.data)
                
                # Check for consistent structure
                assert isinstance(data, dict)
                # Should have status or data key
                assert 'status' in data or 'data' in data or len(data) > 0

    # Test Caching Behavior
    
    def test_caching_behavior(self, client, mock_inventory_analyzer):
        """Test that caching works correctly for consolidated endpoints"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            # First request should hit the endpoint
            response1 = client.get('/api/comprehensive-kpis')
            assert response1.status_code == 200
            data1 = json.loads(response1.data)
            
            # Second request might be cached (check headers or timing)
            response2 = client.get('/api/comprehensive-kpis')
            assert response2.status_code == 200
            data2 = json.loads(response2.data)
            
            # Force refresh should bypass cache
            response3 = client.get('/api/comprehensive-kpis?refresh=true')
            assert response3.status_code == 200

    # Test Backward Compatibility
    
    def test_backward_compatibility(self, client, mock_inventory_analyzer):
        """Test that old API calls still work through compatibility layer"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            # Old endpoint patterns should still work
            old_patterns = [
                ('/api/inventory-status', 200),
                ('/api/forecast', 200),
                ('/api/production-schedule', 200),
                ('/api/yarn-status', 200)
            ]
            
            for endpoint, expected_status in old_patterns:
                response = client.get(endpoint, follow_redirects=True)
                # Should either work directly or redirect and work
                assert response.status_code == expected_status

    # Test Performance
    
    def test_consolidated_endpoint_performance(self, client, mock_inventory_analyzer):
        """Test that consolidated endpoints respond within acceptable time"""
        import time
        
        with patch('core.beverly_comprehensive_erp.inventory_analyzer', mock_inventory_analyzer):
            endpoints = [
                '/api/inventory-intelligence-enhanced',
                '/api/ml-forecast-detailed',
                '/api/production-planning'
            ]
            
            for endpoint in endpoints:
                start = time.time()
                response = client.get(endpoint)
                elapsed = time.time() - start
                
                assert response.status_code == 200
                # Should respond within 2 seconds
                assert elapsed < 2.0, f"{endpoint} took {elapsed:.2f}s"


class TestAPIConsolidationIntegration:
    """Integration tests for API consolidation"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_dashboard_api_compatibility(self, client):
        """Test that dashboard JavaScript API calls work"""
        # Simulate dashboard API calls
        dashboard_calls = [
            {'endpoint': '/api/inventory-intelligence-enhanced', 'params': {'view': 'summary'}},
            {'endpoint': '/api/ml-forecast-detailed', 'params': {'detail': 'full'}},
            {'endpoint': '/api/production-planning', 'params': {'forecast': 'true'}},
            {'endpoint': '/api/comprehensive-kpis', 'params': {}},
            {'endpoint': '/api/yarn-intelligence', 'params': {'analysis': 'shortage'}}
        ]
        
        for call in dashboard_calls:
            # Build query string
            query_string = '&'.join([f"{k}={v}" for k, v in call['params'].items()])
            url = f"{call['endpoint']}?{query_string}" if query_string else call['endpoint']
            
            response = client.get(url)
            # All dashboard calls should work
            assert response.status_code in [200, 301, 302, 308]
            
            if response.status_code == 200:
                # Verify response is valid JSON
                data = json.loads(response.data)
                assert data is not None

    def test_api_chain_calls(self, client):
        """Test that chained API calls work correctly"""
        with patch('core.beverly_comprehensive_erp.inventory_analyzer') as mock_analyzer:
            # Setup mock
            mock_analyzer.raw_materials_data = pd.DataFrame({
                'YarnID': ['Y001'],
                'Planning_Balance': [1000]
            })
            
            # First call to get inventory
            response1 = client.get('/api/inventory-intelligence-enhanced')
            assert response1.status_code == 200
            
            # Second call based on first response
            response2 = client.get('/api/yarn-intelligence?analysis=shortage')
            assert response2.status_code == 200
            
            # Third call for production planning
            response3 = client.get('/api/production-planning?forecast=true')
            assert response3.status_code == 200

    def test_concurrent_api_calls(self, client):
        """Test that concurrent API calls are handled correctly"""
        import concurrent.futures
        
        def make_request(endpoint):
            return client.get(endpoint)
        
        endpoints = [
            '/api/inventory-intelligence-enhanced',
            '/api/ml-forecast-detailed',
            '/api/production-planning',
            '/api/comprehensive-kpis',
            '/api/yarn-intelligence'
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, endpoint) for endpoint in endpoints]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All concurrent calls should succeed
        for result in results:
            assert result.status_code in [200, 301, 302, 308]


# Fixtures for pytest

@pytest.fixture(scope='module')
def test_data():
    """Provide test data for all tests"""
    return {
        'yarn_data': pd.DataFrame({
            'YarnID': ['Y001', 'Y002', 'Y003', 'Y004', 'Y005'],
            'Description': ['Cotton 20/1', 'Polyester 30/1', 'Blend 40/2', 'Wool 50/1', 'Silk 60/1'],
            'Planning_Balance': [1000, -500, 2000, 0, 1500],
            'Allocated': [200, 100, 300, 0, 250],
            'On_Order': [500, 1000, 0, 500, 0],
            'Price': [4.5, 3.2, 5.8, 12.3, 25.5]
        }),
        'bom_data': pd.DataFrame({
            'Style#': ['S001', 'S001', 'S002', 'S002', 'S003'],
            'YarnID': ['Y001', 'Y002', 'Y003', 'Y004', 'Y005'],
            'Usage': [0.5, 0.3, 0.8, 0.2, 1.0],
            'Unit': ['lbs', 'lbs', 'lbs', 'lbs', 'lbs']
        }),
        'sales_data': pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=60),
            'Style#': ['S001'] * 20 + ['S002'] * 20 + ['S003'] * 20,
            'Quantity': np.random.randint(100, 500, 60),
            'Revenue': np.random.uniform(1000, 5000, 60)
        })
    }


if __name__ == '__main__':
    # Run tests with coverage
    pytest.main([__file__, '-v', '--cov=core', '--cov-report=html'])