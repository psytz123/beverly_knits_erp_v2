#!/usr/bin/env python3
"""
API v2 Consolidation Test Suite - Phase 4
Comprehensive tests for consolidated API endpoints
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from flask import Flask
from typing import Dict, Any

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.v2.consolidated_routes import api_v2, standardize_response, handle_nan_values
from src.api.v2.blueprint_integration import (
    register_blueprints, 
    DEPRECATED_ENDPOINT_MAPPINGS,
    create_redirect_handler
)
from src.api.v2.parameter_mapper import ParameterMapper, RequestTransformer, ResponseTransformer
from src.api.v2.dependency_injection import ServiceContainer, inject


class TestAPIv2Endpoints:
    """Test consolidated API v2 endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create test Flask app with v2 blueprint"""
        app = Flask(__name__)
        app.config['TESTING'] = True
        app.register_blueprint(api_v2)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    def test_inventory_endpoint_basic(self, client):
        """Test basic inventory endpoint"""
        with patch('src.api.v2.consolidated_routes.inventory_analyzer') as mock_analyzer:
            mock_analyzer.get_inventory_summary.return_value = {
                'items': [],
                'total': 0
            }
            
            response = client.get('/api/v2/inventory')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'success' in data
            assert data['success'] is True
            assert 'data' in data
    
    def test_inventory_endpoint_with_params(self, client):
        """Test inventory endpoint with query parameters"""
        with patch('src.api.v2.consolidated_routes.inventory_analyzer') as mock_analyzer:
            mock_analyzer.get_yarn_inventory.return_value = {
                'yarns': [{'id': 'Y001', 'balance': 100}]
            }
            
            response = client.get('/api/v2/inventory?view=yarn&shortage_only=false')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'data' in data
            assert 'metadata' in data['data']
            assert data['data']['metadata']['view'] == 'yarn'
    
    def test_production_endpoint_get(self, client):
        """Test production GET endpoint"""
        with patch('src.api.v2.consolidated_routes.get_production_status') as mock_status:
            mock_status.return_value = {
                'orders': [],
                'total': 0
            }
            
            response = client.get('/api/v2/production')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert data['success'] is True
    
    def test_production_endpoint_post(self, client):
        """Test production POST endpoint"""
        with patch('src.api.v2.consolidated_routes.create_production_order') as mock_create:
            mock_create.return_value = {
                'id': 'PO-12345',
                'status': 'created'
            }
            
            order_data = {
                'style_id': 'S001',
                'quantity': 100,
                'deadline': '2025-02-01'
            }
            
            response = client.post('/api/v2/production',
                                  json=order_data,
                                  content_type='application/json')
            assert response.status_code == 201
            
            data = json.loads(response.data)
            assert data['success'] is True
    
    def test_forecast_endpoint(self, client):
        """Test forecast endpoint"""
        with patch('src.api.v2.consolidated_routes.get_forecast') as mock_forecast:
            mock_forecast.return_value = {
                'forecast': [100, 110, 120],
                'model': 'ensemble'
            }
            
            response = client.get('/api/v2/forecast?model=ensemble&horizon=90')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'data' in data
    
    def test_analytics_endpoint(self, client):
        """Test analytics endpoint"""
        with patch('src.api.v2.consolidated_routes.get_comprehensive_kpis') as mock_kpis:
            mock_kpis.return_value = {
                'inventory_turnover': 4.2,
                'efficiency': 0.88
            }
            
            response = client.get('/api/v2/analytics?category=kpi')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert data['success'] is True
    
    def test_yarn_endpoint(self, client):
        """Test yarn endpoint"""
        with patch('src.api.v2.consolidated_routes.inventory_analyzer') as mock_analyzer:
            mock_analyzer.get_yarn_inventory.return_value = {
                'yarns': [{'id': 'Y001', 'description': 'Test Yarn'}]
            }
            
            response = client.get('/api/v2/yarn?action=inventory')
            assert response.status_code == 200
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/api/v2/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['data']['status'] == 'healthy'
    
    def test_docs_endpoint(self, client):
        """Test API documentation endpoint"""
        response = client.get('/api/v2/docs')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'data' in data
        assert 'endpoints' in data['data']
        assert len(data['data']['endpoints']) > 0


class TestDeprecatedEndpointRedirects:
    """Test deprecated endpoint redirect functionality"""
    
    @pytest.fixture
    def app(self):
        """Create test app with redirects"""
        app = Flask(__name__)
        app.config['TESTING'] = True
        register_blueprints(app, enable_deprecation_redirects=True)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    def test_deprecated_redirect(self, client):
        """Test that deprecated endpoints redirect correctly"""
        # Mock the v2 endpoint
        with patch('src.api.v2.consolidated_routes.inventory_analyzer'):
            response = client.get('/api/yarn-inventory', follow_redirects=False)
            
            # Should redirect with 308 status
            assert response.status_code == 308
            assert '/api/v2/inventory' in response.location
    
    def test_redirect_preserves_params(self, client):
        """Test that redirects preserve query parameters"""
        with patch('src.api.v2.consolidated_routes.inventory_analyzer'):
            response = client.get('/api/yarn-inventory?format=csv', follow_redirects=False)
            
            assert response.status_code == 308
            assert 'format=csv' in response.location or 'format' in response.location


class TestParameterMapping:
    """Test parameter mapping functionality"""
    
    def test_basic_parameter_mapping(self):
        """Test basic parameter mapping"""
        old_params = {
            'include_shortages': 'true',
            'format_type': 'json'
        }
        
        mapped = ParameterMapper.map_parameters('yarn-inventory', old_params)
        
        assert 'shortage_only' in mapped
        assert mapped['shortage_only'] == 'true'
        assert 'format' in mapped
        assert mapped['format'] == 'json'
    
    def test_value_transformation(self):
        """Test parameter value transformation"""
        old_params = {
            'model_type': 'time_series',
            'kpi_type': 'operational'
        }
        
        mapped = ParameterMapper.map_parameters('ml-forecasting', old_params)
        
        assert mapped.get('model') == 'arima'
        
        mapped2 = ParameterMapper.map_parameters('comprehensive-kpis', old_params)
        assert mapped2.get('category') == 'performance'
    
    def test_get_v2_url(self):
        """Test URL conversion"""
        old_url = '/api/yarn-inventory?include_shortages=true'
        new_url = ParameterMapper.get_v2_url(old_url)
        
        assert '/api/v2/inventory' in new_url
        assert 'view=yarn' in new_url
    
    def test_request_transformation(self):
        """Test request body transformation"""
        old_body = {
            'order_id': 'ORD-001',
            'product': 'STYLE-123',
            'quantity': 100
        }
        
        new_body = RequestTransformer.transform_request_body('production-order', old_body)
        
        assert new_body['id'] == 'ORD-001'
        assert new_body['style_id'] == 'STYLE-123'
        assert new_body['quantity'] == 100
    
    def test_response_transformation(self):
        """Test response transformation"""
        v2_response = {
            'data': {
                'items': [
                    {'id': 'Y001', 'planning_balance': -100},
                    {'id': 'Y002', 'planning_balance': 200}
                ]
            }
        }
        
        legacy_response = ResponseTransformer.transform_response('yarn-inventory', v2_response)
        
        assert 'yarns' in legacy_response
        assert legacy_response['total_count'] == 2
        assert legacy_response['shortage_count'] == 1


class TestDependencyInjection:
    """Test dependency injection functionality"""
    
    def test_service_container(self):
        """Test service container registration and retrieval"""
        container = ServiceContainer()
        
        # Register a test service
        def create_test_service():
            return {'name': 'test_service'}
        
        container.register('test_service', create_test_service, singleton=True)
        
        # Get service
        service1 = container.get('test_service')
        service2 = container.get('test_service')
        
        # Should be same instance (singleton)
        assert service1 is service2
        assert service1['name'] == 'test_service'
    
    def test_inject_decorator(self):
        """Test injection decorator"""
        container = ServiceContainer()
        
        def create_mock_service():
            return Mock(get_data=lambda: {'test': 'data'})
        
        container.register('mock_service', create_mock_service)
        
        @inject(service='mock_service')
        def test_function(service):
            return service.get_data()
        
        result = test_function()
        assert result == {'test': 'data'}
    
    def test_service_reset(self):
        """Test service container reset"""
        container = ServiceContainer()
        
        counter = {'value': 0}
        
        def create_counting_service():
            counter['value'] += 1
            return {'count': counter['value']}
        
        container.register('counting_service', create_counting_service, singleton=True)
        
        service1 = container.get('counting_service')
        assert service1['count'] == 1
        
        container.reset()
        
        service2 = container.get('counting_service')
        assert service2['count'] == 2  # New instance created


class TestHelperFunctions:
    """Test helper functions"""
    
    def test_handle_nan_values(self):
        """Test NaN value handling"""
        import numpy as np
        
        data = {
            'value': 100,
            'nan_value': np.nan,
            'list': [1, np.nan, 3],
            'nested': {
                'inner_nan': np.nan,
                'inner_value': 'test'
            }
        }
        
        cleaned = handle_nan_values(data)
        
        assert cleaned['value'] == 100
        assert cleaned['nan_value'] is None
        assert cleaned['list'][1] is None
        assert cleaned['nested']['inner_nan'] is None
        assert cleaned['nested']['inner_value'] == 'test'
    
    def test_standardize_response(self):
        """Test response standardization"""
        # Success case
        response, status = standardize_response({'test': 'data'}, 200, 'Success')
        data = json.loads(response.data)
        
        assert status == 200
        assert data['success'] is True
        assert data['message'] == 'Success'
        assert data['data']['test'] == 'data'
        
        # Error case
        response, status = standardize_response('Error occurred', 400)
        data = json.loads(response.data)
        
        assert status == 400
        assert data['success'] is False
        assert data['error'] == 'Error occurred'
        assert data['data'] is None


# Performance Tests
class TestPerformance:
    """Performance tests for API consolidation"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        app = Flask(__name__)
        app.config['TESTING'] = True
        register_blueprints(app)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    def test_endpoint_consolidation_reduces_code(self):
        """Verify that consolidation reduces endpoint count"""
        # Count deprecated endpoints
        deprecated_count = len(DEPRECATED_ENDPOINT_MAPPINGS)
        
        # Count v2 endpoints (should be much fewer)
        v2_endpoints = [
            '/inventory', '/production', '/forecast',
            '/analytics', '/yarn', '/health', '/docs'
        ]
        v2_count = len(v2_endpoints)
        
        # Calculate reduction
        reduction_percentage = (deprecated_count - v2_count) / deprecated_count * 100
        
        assert v2_count < deprecated_count
        assert reduction_percentage > 70  # Should achieve >70% reduction
    
    def test_response_time(self, client):
        """Test that v2 endpoints respond quickly"""
        import time
        
        with patch('src.api.v2.consolidated_routes.inventory_analyzer'):
            start = time.time()
            response = client.get('/api/v2/inventory')
            end = time.time()
            
            response_time = (end - start) * 1000  # Convert to ms
            
            assert response.status_code == 200
            assert response_time < 100  # Should respond within 100ms


# Integration Tests
class TestIntegration:
    """Integration tests for full API flow"""
    
    @pytest.fixture
    def app(self):
        """Create fully configured app"""
        app = Flask(__name__)
        app.config['TESTING'] = True
        
        # Initialize all components
        from src.api.v2.blueprint_integration import init_api_v2
        init_api_v2(app)
        
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    def test_full_migration_flow(self, client):
        """Test complete migration from deprecated to v2"""
        with patch('src.api.v2.consolidated_routes.inventory_analyzer'):
            # Call deprecated endpoint
            response = client.get('/api/yarn-inventory', follow_redirects=True)
            
            # Should get successful response from v2 endpoint
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'success' in data
    
    def test_consolidation_metrics(self, client):
        """Test consolidation metrics endpoint"""
        response = client.get('/api/consolidation-metrics')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'consolidated_endpoints' in data
        assert 'deprecated_endpoints' in data
        assert 'reduction_percentage' in data
        assert data['reduction_percentage'] > 70


if __name__ == "__main__":
    pytest.main([__file__, '-v'])