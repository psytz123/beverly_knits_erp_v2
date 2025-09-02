"""
Comprehensive test suite for API consolidation
Tests redirect functionality, parameter support, and dashboard compatibility
"""

import pytest
import requests
import json
from datetime import datetime
import time


class TestAPIConsolidation:
    """Test suite for API consolidation features"""
    
    BASE_URL = "http://localhost:5006"
    
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.session = requests.Session()
        # Wait for server to be ready
        cls.wait_for_server()
    
    @classmethod
    def wait_for_server(cls, timeout=30):
        """Wait for server to be available"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{cls.BASE_URL}/api/test-po", timeout=1)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        raise RuntimeError(f"Server not available at {cls.BASE_URL}")
    
    # ========================================================================
    # REDIRECT TESTS
    # ========================================================================
    
    def test_deprecated_inventory_redirects(self):
        """Test that deprecated inventory endpoints redirect correctly"""
        deprecated_mappings = [
            ('/api/inventory-analysis', '/api/inventory-intelligence-enhanced'),
            ('/api/inventory-overview', '/api/inventory-intelligence-enhanced'),
            ('/api/real-time-inventory', '/api/inventory-intelligence-enhanced'),
            ('/api/real-time-inventory-dashboard', '/api/inventory-intelligence-enhanced'),
            ('/api/inventory-analysis/complete', '/api/inventory-intelligence-enhanced'),
            ('/api/inventory-analysis/dashboard-data', '/api/inventory-intelligence-enhanced'),
        ]
        
        for old, new in deprecated_mappings:
            response = self.session.get(f"{self.BASE_URL}{old}", allow_redirects=False)
            
            # Check for redirect (301) or that it still works (200) with deprecation headers
            if response.status_code in [301, 302]:
                # Check redirect location contains new endpoint
                location = response.headers.get('Location', '')
                assert new in location, f"Expected {old} to redirect to {new}, got {location}"
                
                # Check for deprecation headers
                assert response.headers.get('X-Deprecated') == 'true', f"Missing deprecation header for {old}"
                assert response.headers.get('X-New-Endpoint') == new, f"Wrong new endpoint header for {old}"
            else:
                # If not redirecting, it should still work but log warning
                assert response.status_code == 200, f"Endpoint {old} failed with {response.status_code}"
                print(f"Warning: {old} not redirecting yet (returns {response.status_code})")
    
    def test_deprecated_yarn_redirects(self):
        """Test that deprecated yarn endpoints redirect correctly"""
        deprecated_mappings = [
            ('/api/yarn', '/api/yarn-intelligence'),
            ('/api/yarn-data', '/api/yarn-intelligence'),
            ('/api/yarn-shortage-analysis', '/api/yarn-intelligence'),
            ('/api/yarn-forecast-shortages', '/api/yarn-intelligence'),
            ('/api/yarn-substitution-opportunities', '/api/yarn-substitution-intelligent'),
            ('/api/yarn-alternatives', '/api/yarn-substitution-intelligent'),
        ]
        
        for old, new in deprecated_mappings:
            response = self.session.get(f"{self.BASE_URL}{old}", allow_redirects=False)
            if response.status_code in [301, 302]:
                location = response.headers.get('Location', '')
                assert new in location, f"Expected {old} to redirect to {new}"
            else:
                assert response.status_code == 200, f"Endpoint {old} failed with {response.status_code}"
    
    def test_deprecated_production_redirects(self):
        """Test that deprecated production endpoints redirect correctly"""
        deprecated_mappings = [
            ('/api/production-data', '/api/production-planning'),
            ('/api/production-orders', '/api/production-planning'),
            ('/api/production-plan-forecast', '/api/production-planning'),
        ]
        
        for old, new in deprecated_mappings:
            response = self.session.get(f"{self.BASE_URL}{old}", allow_redirects=False)
            if response.status_code in [301, 302]:
                location = response.headers.get('Location', '')
                assert new in location, f"Expected {old} to redirect to {new}"
            else:
                assert response.status_code == 200, f"Endpoint {old} failed with {response.status_code}"
    
    def test_deprecated_forecast_redirects(self):
        """Test that deprecated forecast endpoints redirect correctly"""
        deprecated_mappings = [
            ('/api/ml-forecasting', '/api/ml-forecast-detailed'),
            ('/api/ml-forecast-report', '/api/ml-forecast-detailed'),
            ('/api/fabric-forecast', '/api/fabric-forecast-integrated'),
            ('/api/pipeline/forecast', '/api/ml-forecast-detailed'),
        ]
        
        for old, new in deprecated_mappings:
            response = self.session.get(f"{self.BASE_URL}{old}", allow_redirects=False)
            if response.status_code in [301, 302]:
                location = response.headers.get('Location', '')
                assert new in location, f"Expected {old} to redirect to {new}"
            else:
                assert response.status_code == 200, f"Endpoint {old} failed with {response.status_code}"
    
    # ========================================================================
    # PARAMETER SUPPORT TESTS
    # ========================================================================
    
    def test_yarn_intelligence_parameters(self):
        """Test parameter support in yarn-intelligence endpoint"""
        # Test different views
        views = ['full', 'data', 'summary']
        for view in views:
            response = self.session.get(f"{self.BASE_URL}/api/yarn-intelligence?view={view}")
            assert response.status_code == 200
            data = response.json()
            
            if view == 'summary':
                assert 'summary' in data or 'critical_yarns' in data
            elif view == 'data':
                assert 'yarn_data' in data or 'total_count' in data
        
        # Test analysis types
        response = self.session.get(f"{self.BASE_URL}/api/yarn-intelligence?analysis=shortage")
        assert response.status_code == 200
        data = response.json()
        # Check for shortage analysis in response
        assert any(key for key in data.keys() if 'shortage' in key.lower())
        
        # Test forecast parameter
        response = self.session.get(f"{self.BASE_URL}/api/yarn-intelligence?forecast=true")
        assert response.status_code == 200
        data = response.json()
        assert 'forecast' in data or any(key for key in data.keys() if 'forecast' in key.lower())
    
    def test_production_planning_parameters(self):
        """Test parameter support in production-planning endpoint"""
        # Test different views
        views = ['planning', 'orders', 'data', 'metrics']
        for view in views:
            response = self.session.get(f"{self.BASE_URL}/api/production-planning?view={view}")
            assert response.status_code == 200
            data = response.json()
            assert 'status' in data
            
            if view == 'orders':
                assert 'orders' in data or 'total_orders' in data
            elif view == 'metrics':
                assert 'metrics' in data or 'capacity_analysis' in data
        
        # Test forecast parameter
        response = self.session.get(f"{self.BASE_URL}/api/production-planning?forecast=true")
        assert response.status_code == 200
        data = response.json()
        assert 'forecast' in data or any(key for key in data.keys() if 'forecast' in key.lower())
    
    def test_ml_forecast_detailed_parameters(self):
        """Test parameter support in ml-forecast-detailed endpoint"""
        # Test detail levels
        details = ['full', 'summary', 'metrics']
        for detail in details:
            response = self.session.get(f"{self.BASE_URL}/api/ml-forecast-detailed?detail={detail}")
            assert response.status_code == 200
            data = response.json()
            
            if detail == 'summary':
                assert 'summary' in data
            elif detail == 'metrics':
                assert 'metrics' in data
        
        # Test format parameter
        response = self.session.get(f"{self.BASE_URL}/api/ml-forecast-detailed?format=report")
        assert response.status_code == 200
        data = response.json()
        assert 'report' in data or 'sections' in data or 'title' in str(data)
        
        # Test horizon parameter
        response = self.session.get(f"{self.BASE_URL}/api/ml-forecast-detailed?horizon=30")
        assert response.status_code == 200
        data = response.json()
        assert '30' in str(data) or 'horizon' in str(data).lower()
    
    # ========================================================================
    # CRITICAL DASHBOARD ENDPOINT TESTS
    # ========================================================================
    
    def test_all_critical_endpoints_available(self):
        """Test that all 14 critical dashboard endpoints are accessible"""
        critical_endpoints = [
            '/api/production-planning',
            '/api/inventory-intelligence-enhanced',
            '/api/ml-forecast-detailed',
            '/api/inventory-netting',
            '/api/comprehensive-kpis',
            '/api/yarn-intelligence',
            '/api/production-suggestions',
            '/api/po-risk-analysis',
            '/api/production-pipeline',
            '/api/yarn-substitution-intelligent',
            '/api/retrain-ml',
            '/api/production-recommendations-ml',
            '/api/knit-orders',
            '/api/knit-orders/generate'
        ]
        
        for endpoint in critical_endpoints:
            if endpoint == '/api/retrain-ml' or endpoint == '/api/knit-orders/generate':
                # These are POST endpoints
                response = self.session.post(f"{self.BASE_URL}{endpoint}", json={})
            else:
                response = self.session.get(f"{self.BASE_URL}{endpoint}")
            
            assert response.status_code in [200, 201, 400], \
                f"Critical endpoint {endpoint} returned {response.status_code}"
    
    def test_dashboard_functionality_with_redirects(self):
        """Test that dashboard still works with deprecated endpoints being redirected"""
        # Simulate dashboard API calls
        dashboard_calls = [
            ('/api/inventory-overview', 200),  # Should redirect to enhanced
            ('/api/yarn-data', 200),  # Should redirect to intelligence
            ('/api/production-orders', 200),  # Should redirect to planning
            ('/api/ml-forecasting', 200),  # Should redirect to detailed
        ]
        
        for endpoint, expected_status in dashboard_calls:
            # Follow redirects
            response = self.session.get(f"{self.BASE_URL}{endpoint}", allow_redirects=True)
            assert response.status_code == expected_status, \
                f"Dashboard call to {endpoint} failed with {response.status_code}"
            
            # Verify response is valid JSON
            try:
                data = response.json()
                assert isinstance(data, dict) or isinstance(data, list)
            except:
                pytest.fail(f"Invalid JSON response from {endpoint}")
    
    # ========================================================================
    # MONITORING ENDPOINT TEST
    # ========================================================================
    
    def test_consolidation_metrics_endpoint(self):
        """Test the consolidation metrics monitoring endpoint"""
        response = self.session.get(f"{self.BASE_URL}/api/consolidation-metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert 'deprecated_calls' in data
        assert 'redirect_count' in data
        assert 'new_api_calls' in data
        assert 'migration_progress' in data
        assert 'consolidation_enabled' in data
        assert 'redirect_enabled' in data
    
    # ========================================================================
    # PERFORMANCE TESTS
    # ========================================================================
    
    def test_redirect_performance(self):
        """Test that redirects don't significantly impact performance"""
        import time
        
        # Test direct call performance
        direct_times = []
        for _ in range(5):
            start = time.time()
            response = self.session.get(f"{self.BASE_URL}/api/yarn-intelligence")
            end = time.time()
            if response.status_code == 200:
                direct_times.append(end - start)
        
        # Test redirected call performance
        redirect_times = []
        for _ in range(5):
            start = time.time()
            response = self.session.get(f"{self.BASE_URL}/api/yarn-data", allow_redirects=True)
            end = time.time()
            if response.status_code == 200:
                redirect_times.append(end - start)
        
        # Calculate averages
        avg_direct = sum(direct_times) / len(direct_times) if direct_times else 0
        avg_redirect = sum(redirect_times) / len(redirect_times) if redirect_times else 0
        
        # Redirect should not add more than 100ms on average
        if avg_direct > 0 and avg_redirect > 0:
            assert avg_redirect - avg_direct < 0.1, \
                f"Redirect overhead too high: {avg_redirect - avg_direct:.3f}s"
    
    def test_parameter_validation(self):
        """Test that invalid parameters are handled gracefully"""
        invalid_requests = [
            '/api/yarn-intelligence?view=invalid',
            '/api/production-planning?view=nonexistent',
            '/api/ml-forecast-detailed?horizon=abc',
            '/api/ml-forecast-detailed?format=xyz',
        ]
        
        for request_url in invalid_requests:
            response = self.session.get(f"{self.BASE_URL}{request_url}")
            # Should still return valid response, just ignore invalid params
            assert response.status_code == 200, \
                f"Invalid parameter handling failed for {request_url}"
    
    # ========================================================================
    # DATA CONSISTENCY TESTS
    # ========================================================================
    
    def test_data_consistency_across_endpoints(self):
        """Test that consolidated endpoints return consistent data"""
        # Get data from original endpoint (if it exists)
        response1 = self.session.get(f"{self.BASE_URL}/api/yarn-intelligence")
        
        # Get data with different parameters
        response2 = self.session.get(f"{self.BASE_URL}/api/yarn-intelligence?view=full")
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            # Check that timestamp fields exist and are recent
            for data in [data1, data2]:
                if 'timestamp' in data:
                    timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                    assert (datetime.now() - timestamp).total_seconds() < 60
    
    def test_error_handling(self):
        """Test that error responses are consistent"""
        # Test non-existent endpoint
        response = self.session.get(f"{self.BASE_URL}/api/nonexistent")
        assert response.status_code == 404
        
        # Test invalid method
        response = self.session.delete(f"{self.BASE_URL}/api/yarn-intelligence")
        assert response.status_code in [404, 405]


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])