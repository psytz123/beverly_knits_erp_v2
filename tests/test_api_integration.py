"""
Comprehensive tests for eFab.ai API integration
Tests authentication, data loading, transformation, and fallback mechanisms
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import json
import aiohttp
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.api_clients.efab_api_client import EFabAPIClient, CircuitBreaker, CircuitOpenError, APICache
from src.api_clients.efab_auth_manager import EFabAuthManager, AuthenticationError, AuthStatus, SessionInfo
from src.api_clients.efab_transformers import EFabDataTransformer
from src.config.secure_api_config import EFabAPIConfig
from src.data_loaders.efab_api_loader import EFabAPIDataLoader


# ===========================
# Fixtures
# ===========================

@pytest.fixture
def mock_config():
    """Mock API configuration"""
    return {
        'base_url': 'https://efab.test.com',
        'username': 'test_user',
        'password': 'test_password',
        'session_timeout': 3600,
        'retry_count': 3,
        'circuit_breaker_threshold': 5,
        'circuit_breaker_timeout': 60,
        'enable_cache': True,
        'enable_fallback': True,
        'api_enabled': True,
        'endpoints': {
            'yarn_active': 'https://efab.test.com/api/yarn/active',
            'health': 'https://efab.test.com/api/health'
        }
    }


@pytest.fixture
def sample_yarn_data():
    """Sample yarn inventory data"""
    return {
        'data': [
            {
                'yarn_id': '18884',
                'description': '100% COTTON 30/1 ROYAL BLUE',
                'theoretical_balance': 2506.18,
                'allocated': -30859.80,
                'on_order': 36161.30,
                'cost_per_pound': 2.85
            },
            {
                'yarn_id': '18885',
                'description': '100% COTTON 30/1 RED',
                'theoretical_balance': 1000.00,
                'allocated': -5000.00,
                'on_order': 3000.00,
                'cost_per_pound': 2.90
            }
        ]
    }


@pytest.fixture
def sample_knit_orders():
    """Sample knit orders data"""
    return {
        'data': [
            {
                'ko_number': 'KO-2024-001',
                'style': 'ST-1001',
                'qty_ordered_lbs': 5000,
                'machine': '161',
                'status': 'In Progress',
                'due_date': '2025-10-15'
            },
            {
                'ko_number': 'KO-2024-002',
                'style': 'ST-1002',
                'qty_ordered_lbs': 3000,
                'machine': None,
                'status': 'Pending',
                'due_date': '2025-10-20'
            }
        ]
    }


@pytest.fixture
def sample_po_deliveries():
    """Sample PO delivery data"""
    return {
        'data': {
            '18884': {
                'deliveries': {
                    'past_due': 20161.30,
                    '2025-10-10': 4000,
                    '2025-10-17': 4000,
                    'later': 8000
                }
            },
            '18885': {
                'deliveries': {
                    'past_due': 0,
                    '2025-10-10': 2000,
                    '2025-10-17': 1000,
                    'later': 0
                }
            }
        }
    }


# ===========================
# Authentication Manager Tests
# ===========================

class TestEFabAuthManager:
    """Test authentication manager functionality"""
    
    @pytest.mark.asyncio
    async def test_successful_authentication(self, mock_config):
        """Test successful authentication flow"""
        auth_manager = EFabAuthManager(mock_config)
        
        # Mock HTTP session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"token": "test_token_123"}')
        mock_response.cookies = {'dancer.session': 'test_token_123'}
        
        with patch.object(auth_manager, '_http_session') as mock_session:
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
            
            token = await auth_manager.authenticate()
            
            assert token == 'test_token_123'
            assert auth_manager.auth_status == AuthStatus.AUTHENTICATED
            assert auth_manager.session_info is not None
            assert auth_manager.session_info.token == 'test_token_123'
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, mock_config):
        """Test authentication failure handling"""
        auth_manager = EFabAuthManager(mock_config)
        
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value='Invalid credentials')
        
        with patch.object(auth_manager, '_http_session') as mock_session:
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
            
            with pytest.raises(AuthenticationError):
                await auth_manager.authenticate()
            
            assert auth_manager.auth_status == AuthStatus.FAILED
    
    def test_session_validity_check(self, mock_config):
        """Test session validity checking"""
        auth_manager = EFabAuthManager(mock_config)
        
        # No session
        assert not auth_manager.is_session_valid()
        
        # Create valid session
        auth_manager.session_info = SessionInfo(
            token='test_token',
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            username='test_user'
        )
        auth_manager.auth_status = AuthStatus.AUTHENTICATED
        
        assert auth_manager.is_session_valid()
        
        # Expired session
        auth_manager.session_info.expires_at = datetime.now() - timedelta(hours=1)
        assert not auth_manager.is_session_valid()
    
    @pytest.mark.asyncio
    async def test_session_refresh(self, mock_config):
        """Test session refresh functionality"""
        auth_manager = EFabAuthManager(mock_config)
        
        # Setup existing session
        auth_manager.session_info = SessionInfo(
            token='old_token',
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=5),
            username='test_user',
            refresh_token='refresh_123'
        )
        auth_manager.auth_status = AuthStatus.AUTHENTICATED
        
        # Mock refresh response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'token': 'new_token', 'refresh_token': 'refresh_456'})
        
        with patch.object(auth_manager, '_http_session') as mock_session:
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
            
            success = await auth_manager.refresh_session()
            
            assert success
            assert auth_manager.session_info.token == 'new_token'
            assert auth_manager.session_info.refresh_token == 'refresh_456'


# ===========================
# API Client Tests
# ===========================

class TestEFabAPIClient:
    """Test API client functionality"""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, mock_config):
        """Test client initialization"""
        client = EFabAPIClient(mock_config)
        
        assert client.base_url == 'https://efab.test.com'
        assert client.circuit_breaker is not None
        assert client.cache is not None
        assert client.auth_manager is not None
    
    @pytest.mark.asyncio
    async def test_successful_api_request(self, mock_config, sample_yarn_data):
        """Test successful API request"""
        client = EFabAPIClient(mock_config)
        
        # Mock auth manager
        client.auth_manager.ensure_authenticated = AsyncMock(return_value='test_token')
        client.auth_manager.get_auth_headers = Mock(return_value={'Authorization': 'Bearer test_token'})
        
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps(sample_yarn_data))
        
        with patch.object(client, '_session') as mock_session:
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.request.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await client._make_request('GET', '/api/yarn/active')
            
            assert 'data' in result
            assert len(result['data']) == 2
            assert result['data'][0]['yarn_id'] == '18884'
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, mock_config):
        """Test circuit breaker activation after failures"""
        client = EFabAPIClient(mock_config)
        client.circuit_breaker.failure_threshold = 2  # Lower threshold for testing
        
        # Mock auth manager
        client.auth_manager.ensure_authenticated = AsyncMock(return_value='test_token')
        client.auth_manager.get_auth_headers = Mock(return_value={'Authorization': 'Bearer test_token'})
        
        # Mock failing responses
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value='Server error')
        
        with patch.object(client, '_session') as mock_session:
            mock_session.request = AsyncMock(side_effect=aiohttp.ClientError('Connection failed'))
            
            # First failure
            with pytest.raises(aiohttp.ClientError):
                await client._make_request('GET', '/api/test', use_cache=False)
            
            # Second failure - should open circuit
            with pytest.raises(aiohttp.ClientError):
                await client._make_request('GET', '/api/test', use_cache=False)
            
            # Third attempt - circuit should be open
            with pytest.raises(CircuitOpenError):
                await client._make_request('GET', '/api/test', use_cache=False)
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, mock_config, sample_yarn_data):
        """Test caching of API responses"""
        client = EFabAPIClient(mock_config)
        
        # Mock auth manager
        client.auth_manager.ensure_authenticated = AsyncMock(return_value='test_token')
        client.auth_manager.get_auth_headers = Mock(return_value={'Authorization': 'Bearer test_token'})
        
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps(sample_yarn_data))
        
        with patch.object(client, '_session') as mock_session:
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.request.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # First request - should hit API
            result1 = await client._make_request('GET', '/api/yarn/active')
            assert mock_session.request.call_count == 1
            
            # Second request - should use cache
            result2 = await client._make_request('GET', '/api/yarn/active')
            assert mock_session.request.call_count == 1  # No additional call
            
            assert result1 == result2
    
    @pytest.mark.asyncio
    async def test_parallel_data_loading(self, mock_config):
        """Test parallel data loading"""
        client = EFabAPIClient(mock_config)
        
        # Mock all API methods to return DataFrames
        client.get_yarn_active = AsyncMock(return_value=pd.DataFrame({'yarn_id': ['18884']}))
        client.get_knit_orders = AsyncMock(return_value=pd.DataFrame({'ko_number': ['KO-001']}))
        client.get_yarn_po = AsyncMock(return_value=pd.DataFrame({'po_number': ['PO-001']}))
        
        # Execute parallel loading
        with patch.object(client, 'get_greige_inventory', AsyncMock(return_value=pd.DataFrame())):
            with patch.object(client, 'get_finished_inventory', AsyncMock(return_value=pd.DataFrame())):
                results = await client.get_all_data_parallel()
        
        assert 'yarn_inventory' in results
        assert 'knit_orders' in results
        assert 'yarn_po' in results
        assert len(results['yarn_inventory']) > 0
        assert len(results['knit_orders']) > 0


# ===========================
# Data Transformer Tests
# ===========================

class TestEFabDataTransformer:
    """Test data transformation functionality"""
    
    def test_yarn_active_transformation(self, sample_yarn_data):
        """Test yarn inventory transformation"""
        transformer = EFabDataTransformer()
        
        df = transformer.transform_yarn_active(sample_yarn_data)
        
        assert not df.empty
        assert 'Desc#' in df.columns
        assert 'Planning Balance' in df.columns
        assert 'Yarn Description' in df.columns
        
        # Check Planning Balance calculation
        first_row = df.iloc[0]
        expected_balance = 2506.18 + (-30859.80) + 36161.30
        assert abs(first_row['Planning Balance'] - expected_balance) < 0.01
    
    def test_knit_orders_transformation(self, sample_knit_orders):
        """Test knit orders transformation"""
        transformer = EFabDataTransformer()
        
        df = transformer.transform_knit_orders(sample_knit_orders)
        
        assert not df.empty
        assert 'KO#' in df.columns
        assert 'Style#' in df.columns
        assert 'Qty Ordered (lbs)' in df.columns
        assert 'Machine' in df.columns
        
        # Check assignment status
        assert df.iloc[0]['Assigned'] == True  # Has machine
        assert df.iloc[1]['Assigned'] == False  # No machine
    
    def test_po_deliveries_transformation(self, sample_po_deliveries):
        """Test PO deliveries transformation"""
        transformer = EFabDataTransformer()
        
        df = transformer.transform_yarn_expected(sample_po_deliveries)
        
        assert not df.empty
        assert 'yarn_id' in df.columns
        assert 'week_past_due' in df.columns
        assert 'week_later' in df.columns
        
        # Check weekly bucket columns exist
        for week in range(36, 45):
            assert f'week_{week}' in df.columns
        
        # Check past due amount
        yarn_18884 = df[df['yarn_id'] == '18884'].iloc[0]
        assert yarn_18884['week_past_due'] == 20161.30
    
    def test_numeric_field_cleaning(self):
        """Test cleaning of numeric fields"""
        transformer = EFabDataTransformer()
        
        assert transformer._clean_numeric_field('1,234.56') == 1234.56
        assert transformer._clean_numeric_field('$100.00') == 100.00
        assert transformer._clean_numeric_field(None) == 0.0
        assert transformer._clean_numeric_field('') == 0.0
        assert transformer._clean_numeric_field(42) == 42.0
    
    def test_date_field_cleaning(self):
        """Test cleaning of date fields"""
        transformer = EFabDataTransformer()
        
        # Test various date formats
        assert transformer._clean_date_field('2025-10-15') is not None
        assert transformer._clean_date_field('10/15/2025') is not None
        assert transformer._clean_date_field(None) is None
        assert transformer._clean_date_field('invalid') is None
    
    def test_validation(self, sample_yarn_data):
        """Test transformation validation"""
        transformer = EFabDataTransformer()
        
        original_df = pd.DataFrame(sample_yarn_data['data'])
        transformed_df = transformer.transform_yarn_active(sample_yarn_data)
        
        assert transformer.validate_transformation(original_df, transformed_df, 'yarn_inventory')


# ===========================
# API Data Loader Tests
# ===========================

class TestEFabAPIDataLoader:
    """Test API data loader with fallback"""
    
    @pytest.mark.asyncio
    async def test_api_first_loading(self, mock_config, sample_yarn_data):
        """Test API-first data loading"""
        loader = EFabAPIDataLoader()
        
        # Mock API client
        with patch.object(loader, 'api_client') as mock_client:
            mock_client.get_yarn_active = AsyncMock(return_value=pd.DataFrame(sample_yarn_data['data']))
            
            with patch.object(loader, '_check_api_availability', AsyncMock(return_value=True)):
                df = await loader._load_yarn_inventory_async()
        
        assert not df.empty
        assert 'Planning Balance' in df.columns
    
    @pytest.mark.asyncio
    async def test_fallback_to_file(self, mock_config):
        """Test fallback to file loading when API fails"""
        loader = EFabAPIDataLoader()
        loader.enable_fallback = True
        
        # Mock API failure
        with patch.object(loader, '_check_api_availability', AsyncMock(return_value=False)):
            # Mock file loading
            with patch.object(loader, '_load_yarn_inventory_from_file') as mock_file_load:
                mock_file_load.return_value = pd.DataFrame({'Desc#': ['18884']})
                
                df = await loader._load_yarn_inventory_async()
        
        assert not df.empty
        mock_file_load.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_health_check(self, mock_config):
        """Test API availability check"""
        loader = EFabAPIDataLoader()
        
        # Mock successful health check
        with patch.object(loader, 'api_client') as mock_client:
            mock_client.health_check = AsyncMock(return_value=True)
            mock_client.initialize = AsyncMock()
            
            is_available = await loader._check_api_availability()
            
            assert is_available
            assert loader.api_failure_count == 0
    
    def test_loader_status(self, mock_config):
        """Test loader status reporting"""
        loader = EFabAPIDataLoader()
        
        status = loader.get_loader_status()
        
        assert 'api_enabled' in status
        assert 'api_available' in status
        assert 'fallback_enabled' in status
        assert 'cached_datasets' in status


# ===========================
# Circuit Breaker Tests
# ===========================

class TestCircuitBreaker:
    """Test circuit breaker pattern"""
    
    @pytest.mark.asyncio
    async def test_circuit_closed_state(self):
        """Test circuit breaker in closed state"""
        cb = CircuitBreaker(failure_threshold=3)
        
        async with cb:
            pass  # Success
        
        assert cb.state == cb.state.CLOSED
        assert cb.stats.success_count == 1
        assert cb.stats.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures"""
        cb = CircuitBreaker(failure_threshold=2)
        
        # First failure
        try:
            async with cb:
                raise Exception("Test failure")
        except Exception:
            pass
        
        assert cb.state == cb.state.CLOSED
        assert cb.stats.failure_count == 1
        
        # Second failure - should open
        try:
            async with cb:
                raise Exception("Test failure")
        except Exception:
            pass
        
        assert cb.state == cb.state.OPEN
        assert cb.stats.failure_count == 2
        
        # Next attempt should fail immediately
        with pytest.raises(CircuitOpenError):
            async with cb:
                pass
    
    @pytest.mark.asyncio
    async def test_circuit_half_open_recovery(self):
        """Test circuit recovery through half-open state"""
        cb = CircuitBreaker(failure_threshold=1, timeout_duration=0.1)
        
        # Open the circuit
        try:
            async with cb:
                raise Exception("Test failure")
        except Exception:
            pass
        
        assert cb.state == cb.state.OPEN
        
        # Wait for timeout
        await asyncio.sleep(0.2)
        
        # Should enter half-open state
        async with cb:
            pass  # Success
        
        # Should be closed again
        assert cb.state == cb.state.CLOSED
        assert cb.stats.failure_count == 0


# ===========================
# Cache Tests
# ===========================

class TestAPICache:
    """Test API caching functionality"""
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test cache set and get operations"""
        cache = APICache()
        
        # Set value
        await cache.set('/api/test', {'data': 'test'}, 60)
        
        # Get value
        result = await cache.get('/api/test')
        assert result == {'data': 'test'}
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache expiration"""
        cache = APICache()
        
        # Set with very short TTL
        await cache.set('/api/test', {'data': 'test'}, 0.1)
        
        # Should be available immediately
        assert await cache.get('/api/test') is not None
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be expired
        assert await cache.get('/api/test') is None
    
    @pytest.mark.asyncio
    async def test_cache_with_params(self):
        """Test caching with different parameters"""
        cache = APICache()
        
        # Set with different params
        await cache.set('/api/test', {'data': 'a'}, 60, {'id': 1})
        await cache.set('/api/test', {'data': 'b'}, 60, {'id': 2})
        
        # Get with specific params
        result1 = await cache.get('/api/test', {'id': 1})
        result2 = await cache.get('/api/test', {'id': 2})
        
        assert result1 == {'data': 'a'}
        assert result2 == {'data': 'b'}


# ===========================
# Integration Tests
# ===========================

class TestIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_data_pipeline(self, mock_config, sample_yarn_data):
        """Test complete data pipeline from API to transformed data"""
        # Create loader
        loader = EFabAPIDataLoader()
        
        # Mock API responses
        with patch.object(loader, 'api_client') as mock_client:
            mock_client.get_yarn_active = AsyncMock(return_value=pd.DataFrame(sample_yarn_data['data']))
            mock_client.health_check = AsyncMock(return_value=True)
            mock_client.initialize = AsyncMock()
            
            # Load data
            with patch.object(loader, 'api_enabled', True):
                df = await loader._load_yarn_inventory_async()
        
        # Verify transformation
        assert not df.empty
        assert 'Desc#' in df.columns
        assert 'Planning Balance' in df.columns
        assert df['Desc#'].iloc[0] == '18884'
    
    @pytest.mark.asyncio
    async def test_api_failure_recovery(self, mock_config):
        """Test system recovery from API failures"""
        loader = EFabAPIDataLoader()
        
        # Simulate API failures then recovery
        with patch.object(loader, 'api_client') as mock_client:
            # First attempts fail
            mock_client.health_check = AsyncMock(side_effect=[False, False, True])
            mock_client.get_yarn_active = AsyncMock(return_value=pd.DataFrame({'yarn_id': ['18884']}))
            mock_client.initialize = AsyncMock()
            
            # First check - fails
            is_available = await loader._check_api_availability()
            assert not is_available
            assert loader.api_failure_count == 1
            
            # Second check - fails
            is_available = await loader._check_api_availability()
            assert not is_available
            assert loader.api_failure_count == 2
            
            # Third check - recovers
            is_available = await loader._check_api_availability()
            assert is_available
            assert loader.api_failure_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])