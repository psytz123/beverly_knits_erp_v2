"""
eFab.ai API Client
Main API client with resilience patterns, caching, and parallel loading capabilities
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
import json
from enum import Enum
from dataclasses import dataclass
import hashlib
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from .efab_auth_manager import EFabAuthManager, AuthenticationError

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changed_at: Optional[datetime] = None


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    Prevents cascading failures by failing fast
    """
    
    def __init__(self, failure_threshold: int = 5, timeout_duration: int = 60):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_duration: Seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        """Context manager entry"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                # Check if timeout has passed
                if self.stats.state_changed_at:
                    time_since_open = (datetime.now() - self.stats.state_changed_at).total_seconds()
                    if time_since_open >= self.timeout_duration:
                        logger.info("Circuit breaker attempting recovery (half-open)")
                        self.state = CircuitState.HALF_OPEN
                        self.stats.state_changed_at = datetime.now()
                    else:
                        raise CircuitOpenError(f"Circuit open for {self.timeout_duration - time_since_open:.0f} more seconds")
                else:
                    raise CircuitOpenError("Circuit is open")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        async with self._lock:
            if exc_type is None:
                # Success
                self.stats.success_count += 1
                self.stats.last_success_time = datetime.now()
                
                if self.state == CircuitState.HALF_OPEN:
                    logger.info("Circuit breaker recovered (closed)")
                    self.state = CircuitState.CLOSED
                    self.stats.failure_count = 0
                    self.stats.state_changed_at = datetime.now()
            else:
                # Failure
                self.stats.failure_count += 1
                self.stats.last_failure_time = datetime.now()
                
                if self.state == CircuitState.HALF_OPEN:
                    logger.warning("Circuit breaker recovery failed (open)")
                    self.state = CircuitState.OPEN
                    self.stats.state_changed_at = datetime.now()
                elif self.state == CircuitState.CLOSED:
                    if self.stats.failure_count >= self.failure_threshold:
                        logger.error(f"Circuit breaker opened after {self.stats.failure_count} failures")
                        self.state = CircuitState.OPEN
                        self.stats.state_changed_at = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'state': self.state.value,
            'failure_count': self.stats.failure_count,
            'success_count': self.stats.success_count,
            'last_failure': self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            'last_success': self.stats.last_success_time.isoformat() if self.stats.last_success_time else None
        }


class CircuitOpenError(Exception):
    """Exception raised when circuit is open"""
    pass


class APICache:
    """Simple in-memory cache for API responses"""
    
    def __init__(self):
        self.cache: Dict[str, tuple] = {}
        self._lock = asyncio.Lock()
    
    def _generate_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from endpoint and parameters"""
        key_str = endpoint
        if params:
            # Sort params for consistent key generation
            sorted_params = sorted(params.items())
            key_str += str(sorted_params)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Get cached value if not expired"""
        async with self._lock:
            key = self._generate_key(endpoint, params)
            if key in self.cache:
                value, expires_at = self.cache[key]
                if datetime.now() < expires_at:
                    logger.debug(f"Cache hit for {endpoint}")
                    return value
                else:
                    del self.cache[key]
        return None
    
    async def set(self, endpoint: str, value: Any, ttl_seconds: int, params: Optional[Dict] = None):
        """Set cache value with TTL"""
        async with self._lock:
            key = self._generate_key(endpoint, params)
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            self.cache[key] = (value, expires_at)
            logger.debug(f"Cached {endpoint} for {ttl_seconds} seconds")
    
    async def clear(self, pattern: Optional[str] = None):
        """Clear cache, optionally by pattern"""
        async with self._lock:
            if pattern:
                keys_to_delete = [k for k in self.cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self.cache[key]
            else:
                self.cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache),
            'expired_entries': sum(1 for _, (_, exp) in self.cache.items() if datetime.now() >= exp)
        }


class EFabAPIClient:
    """
    eFab.ai API client with resilience patterns
    Implements retry logic, circuit breaking, caching, and parallel loading
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize API client
        
        Args:
            config: Configuration dictionary from secure_api_config
        """
        self.config = config
        self.base_url = config.get('base_url', 'https://efab.bkiapps.com')
        self.endpoints = config.get('endpoints', {})
        
        # Initialize components
        self.auth_manager = EFabAuthManager(config)
        self.cache = APICache() if config.get('enable_cache', True) else None
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('circuit_breaker_threshold', 5),
            timeout_duration=config.get('circuit_breaker_timeout', 60)
        )
        
        # Configuration
        self.max_parallel = config.get('max_parallel_requests', 5)
        self.retry_config = {
            'stop': stop_after_attempt(config.get('retry_count', 3)),
            'wait': wait_exponential(
                multiplier=config.get('retry_delay', 1),
                max=30
            ),
            'retry': retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
        }
        
        # Cache TTLs
        self.cache_ttls = config.get('cache_ttls', {})
        
        # Session
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(self.max_parallel)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize client and authenticate"""
        # Create HTTP session
        if not self._session:
            timeout = aiohttp.ClientTimeout(
                total=90,
                connect=self.config.get('connection_timeout', 30),
                sock_read=self.config.get('read_timeout', 60)
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        
        # Initialize auth manager
        await self.auth_manager.initialize()
    
    async def cleanup(self):
        """Clean up resources"""
        if self._session:
            await self._session.close()
            self._session = None
        
        await self.auth_manager.cleanup()
    
    @retry(**{'stop': stop_after_attempt(3), 'wait': wait_exponential(multiplier=1, max=10)})
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and circuit breaking
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            use_cache: Whether to use caching
            
        Returns:
            Response data as dictionary
            
        Raises:
            Various API and network errors
        """
        # Check cache first for GET requests
        if method == 'GET' and use_cache and self.cache:
            cached = await self.cache.get(endpoint, params)
            if cached is not None:
                return cached
        
        # Build full URL
        if endpoint.startswith('http'):
            url = endpoint
        else:
            url = f"{self.base_url}{endpoint}"
        
        # Get authentication headers
        await self.auth_manager.ensure_authenticated()
        headers = self.auth_manager.get_auth_headers()
        
        # Make request with circuit breaker
        async with self.circuit_breaker:
            async with self._semaphore:  # Limit parallel requests
                try:
                    logger.debug(f"{method} {url}")
                    
                    async with self._session.request(
                        method,
                        url,
                        params=params,
                        json=data,
                        headers=headers,
                        ssl=True
                    ) as response:
                        response_text = await response.text()
                        
                        if response.status == 200:
                            try:
                                result = json.loads(response_text)
                            except json.JSONDecodeError:
                                # Return text response wrapped in dict
                                result = {'data': response_text}
                            
                            # Cache successful GET responses
                            if method == 'GET' and use_cache and self.cache:
                                # Determine TTL based on endpoint
                                ttl = self._get_cache_ttl(endpoint)
                                await self.cache.set(endpoint, result, ttl, params)
                            
                            return result
                        
                        elif response.status == 401:
                            # Try to refresh authentication once
                            logger.warning("Got 401, attempting to re-authenticate")
                            await self.auth_manager.authenticate()
                            # Retry the request (will be handled by @retry decorator)
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status
                            )
                        
                        else:
                            logger.error(f"API request failed: {response.status} - {response_text}")
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status
                            )
                            
                except aiohttp.ClientError as e:
                    logger.error(f"Network error: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    raise
    
    def _get_cache_ttl(self, endpoint: str) -> int:
        """Get cache TTL for endpoint"""
        # Check specific endpoint TTLs
        for key, ttl in self.cache_ttls.items():
            if key in endpoint:
                return ttl
        
        # Default TTL
        return 300  # 5 minutes
    
    # API Methods
    
    async def get_yarn_active(self) -> pd.DataFrame:
        """Get active yarn inventory"""
        response = await self._make_request('GET', '/api/yarn/active')
        if 'data' in response:
            return pd.DataFrame(response['data'])
        return pd.DataFrame()
    
    async def get_greige_inventory(self, stage: str = 'g00') -> pd.DataFrame:
        """Get greige inventory by stage (g00 or g02)"""
        endpoint = f'/api/greige/{stage}'
        response = await self._make_request('GET', endpoint)
        if 'data' in response:
            return pd.DataFrame(response['data'])
        return pd.DataFrame()
    
    async def get_finished_inventory(self, stage: str = 'f01') -> pd.DataFrame:
        """Get finished inventory by stage (i01 or f01)"""
        endpoint = f'/api/finished/{stage}'
        response = await self._make_request('GET', endpoint)
        if 'data' in response:
            return pd.DataFrame(response['data'])
        return pd.DataFrame()
    
    async def get_yarn_po(self) -> pd.DataFrame:
        """Get yarn purchase orders"""
        response = await self._make_request('GET', '/api/yarn-po')
        if 'data' in response:
            return pd.DataFrame(response['data'])
        return pd.DataFrame()
    
    async def get_knit_orders(self) -> pd.DataFrame:
        """Get knit orders"""
        response = await self._make_request('GET', '/fabric/knitorder/list')
        if 'data' in response:
            return pd.DataFrame(response['data'])
        return pd.DataFrame()
    
    async def get_styles(self) -> pd.DataFrame:
        """Get style master data"""
        response = await self._make_request('GET', '/api/styles')
        if 'data' in response:
            return pd.DataFrame(response['data'])
        return pd.DataFrame()
    
    async def get_yarn_expected(self) -> pd.DataFrame:
        """Get expected yarn deliveries (time-phased)"""
        response = await self._make_request('GET', '/api/report/yarn_expected')
        if 'data' in response:
            return pd.DataFrame(response['data'])
        return pd.DataFrame()
    
    async def get_sales_activity(self) -> pd.DataFrame:
        """Get sales activity report"""
        response = await self._make_request('GET', '/api/report/sales_activity')
        if 'data' in response:
            return pd.DataFrame(response['data'])
        return pd.DataFrame()
    
    async def get_yarn_demand(self) -> pd.DataFrame:
        """Get yarn demand report"""
        response = await self._make_request('GET', '/api/report/yarn_demand')
        if 'data' in response:
            return pd.DataFrame(response['data'])
        return pd.DataFrame()
    
    async def get_sales_order_plan(self) -> pd.DataFrame:
        """Get sales order planning data"""
        response = await self._make_request('GET', '/api/sales-order/plan/list')
        if 'data' in response:
            return pd.DataFrame(response['data'])
        return pd.DataFrame()
    
    async def get_yarn_demand_ko(self) -> pd.DataFrame:
        """Get yarn demand by knit order"""
        response = await self._make_request('GET', '/api/report/yarn_demand_ko')
        if 'data' in response:
            return pd.DataFrame(response['data'])
        return pd.DataFrame()
    
    async def health_check(self) -> bool:
        """
        Check API health
        
        Returns:
            True if API is healthy
        """
        try:
            response = await self._make_request(
                'GET',
                '/api/health',
                use_cache=False
            )
            return response.get('status') == 'healthy'
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_all_data_parallel(self) -> Dict[str, pd.DataFrame]:
        """
        Get all data from API in parallel
        
        Returns:
            Dictionary mapping data type to DataFrame
        """
        tasks = {
            'yarn_inventory': self.get_yarn_active(),
            'greige_g00': self.get_greige_inventory('g00'),
            'greige_g02': self.get_greige_inventory('g02'),
            'finished_i01': self.get_finished_inventory('i01'),
            'finished_f01': self.get_finished_inventory('f01'),
            'yarn_po': self.get_yarn_po(),
            'knit_orders': self.get_knit_orders(),
            'styles': self.get_styles(),
            'yarn_expected': self.get_yarn_expected(),
            'sales_activity': self.get_sales_activity(),
            'yarn_demand': self.get_yarn_demand(),
            'sales_order_plan': self.get_sales_order_plan()
        }
        
        # Execute all tasks in parallel
        results = await asyncio.gather(
            *tasks.values(),
            return_exceptions=True
        )
        
        # Process results
        data = {}
        for (name, _), result in zip(tasks.items(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to load {name}: {result}")
                data[name] = pd.DataFrame()
            else:
                data[name] = result
                logger.info(f"Loaded {name}: {len(result)} rows")
        
        return data
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get client status
        
        Returns:
            Status dictionary
        """
        status = {
            'authenticated': self.auth_manager.is_session_valid(),
            'auth_status': self.auth_manager.auth_status.value,
            'circuit_breaker': self.circuit_breaker.get_status(),
            'base_url': self.base_url
        }
        
        if self.cache:
            status['cache'] = self.cache.get_stats()
        
        if self.auth_manager.session_info:
            status['session'] = self.auth_manager.get_session_info()
        
        return status


async def test_api_client():
    """Test API client functionality"""
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    from config.secure_api_config import get_api_config
    
    # Get configuration
    config_manager = get_api_config()
    config = config_manager.get_credentials()
    
    if not config.get('api_enabled'):
        print("API not configured. Set EFAB_USERNAME and EFAB_PASSWORD environment variables.")
        return
    
    # Test client
    async with EFabAPIClient(config) as client:
        print("Testing eFab API Client")
        print("=" * 50)
        
        # Health check
        print("\n1. Health Check:")
        healthy = await client.health_check()
        print(f"   API Health: {'✓ Healthy' if healthy else '✗ Unhealthy'}")
        
        # Get status
        print("\n2. Client Status:")
        status = client.get_status()
        print(f"   Authenticated: {status['authenticated']}")
        print(f"   Circuit Breaker: {status['circuit_breaker']['state']}")
        
        # Test individual endpoints
        print("\n3. Testing Endpoints:")
        
        try:
            # Yarn inventory
            yarn_df = await client.get_yarn_active()
            print(f"   Yarn Inventory: {len(yarn_df)} items")
            
            # Knit orders
            orders_df = await client.get_knit_orders()
            print(f"   Knit Orders: {len(orders_df)} orders")
            
            # Test parallel loading
            print("\n4. Testing Parallel Loading:")
            start_time = time.time()
            all_data = await client.get_all_data_parallel()
            elapsed = time.time() - start_time
            
            print(f"   Loaded {len(all_data)} datasets in {elapsed:.2f} seconds")
            for name, df in all_data.items():
                print(f"     {name}: {len(df)} rows")
                
        except Exception as e:
            print(f"   Error: {e}")
        
        # Final status
        print("\n5. Final Status:")
        final_status = client.get_status()
        if client.cache:
            print(f"   Cache entries: {final_status['cache']['total_entries']}")
        print(f"   Circuit breaker: {final_status['circuit_breaker']['state']}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_api_client())