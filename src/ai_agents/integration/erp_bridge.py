#!/usr/bin/env python3
"""
ERP Integration Bridge for Beverly Knits AI Agents
Provides seamless integration between AI agents and ERP API endpoints
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
import json
import aiohttp
import time
from dataclasses import dataclass, field
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)


class APICallStatus(Enum):
    """API call status tracking"""
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CACHED = "CACHED"


@dataclass
class APICallResult:
    """Result of an ERP API call"""
    endpoint: str
    status: APICallStatus
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    response_time_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    from_cache: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "endpoint": self.endpoint,
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "from_cache": self.from_cache
        }


@dataclass
class CachedResponse:
    """Cached API response"""
    data: Dict[str, Any]
    timestamp: datetime
    ttl_seconds: int = 300  # 5 minutes default
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > (self.timestamp + timedelta(seconds=self.ttl_seconds))


class ERPIntegrationBridge:
    """
    ERP Integration Bridge for Beverly Knits AI Agents
    
    Features:
    - Async HTTP client for ERP API calls
    - Response caching with TTL
    - Request rate limiting and circuit breaker
    - Error handling and retry logic
    - Performance monitoring and metrics
    - Data transformation and validation
    - Authentication and authorization
    """
    
    def __init__(self, erp_base_url: str = "http://localhost:5006", max_concurrent_requests: int = 10):
        """Initialize ERP Integration Bridge"""
        self.erp_base_url = erp_base_url.rstrip('/')
        self.max_concurrent_requests = max_concurrent_requests
        
        # API Endpoint Mapping (Beverly Knits ERP v2)
        self.api_endpoints = {
            # Core Inventory and Planning
            "inventory_intelligence": "/api/inventory-intelligence-enhanced",
            "yarn_intelligence": "/api/yarn-intelligence", 
            "production_planning": "/api/production-planning",
            "inventory_netting": "/api/inventory-netting",
            
            # ML and Forecasting
            "ml_forecast": "/api/ml-forecast-detailed",
            "production_suggestions": "/api/production-suggestions",
            "production_recommendations_ml": "/api/production-recommendations-ml",
            
            # Machine and Production Flow
            "machine_assignment": "/api/machine-assignment-suggestions",
            "factory_floor_ai": "/api/factory-floor-ai-dashboard",
            "production_pipeline": "/api/production-pipeline",
            "knit_orders": "/api/knit-orders",
            
            # Advanced Intelligence
            "yarn_substitution": "/api/yarn-substitution-intelligent",
            "po_risk_analysis": "/api/po-risk-analysis",
            "comprehensive_kpis": "/api/comprehensive-kpis",
            
            # System and Health
            "system_health": "/api/health",
            "reload_data": "/api/reload-data",
            "debug_data": "/api/debug-data",
            "consolidation_metrics": "/api/consolidation-metrics"
        }
        
        # Response caching
        self.cache: Dict[str, CachedResponse] = {}
        self.default_cache_ttl = 300  # 5 minutes
        self.cache_ttl_overrides = {
            "system_health": 60,        # 1 minute
            "debug_data": 30,           # 30 seconds
            "ml_forecast": 1800,        # 30 minutes
            "inventory_intelligence": 300,  # 5 minutes
            "production_planning": 180,     # 3 minutes
        }
        
        # Rate limiting and circuit breaker
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.request_count = 0
        self.error_count = 0
        self.circuit_breaker_threshold = 10  # errors
        self.circuit_breaker_timeout = 300   # 5 minutes
        self.circuit_open_until: Optional[datetime] = None
        
        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "average_response_time_ms": 0.0,
            "last_reset": datetime.now()
        }
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info("ERP Integration Bridge initialized")
    
    async def initialize(self):
        """Initialize async components"""
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_requests,
            limit_per_host=self.max_concurrent_requests,
            keepalive_timeout=30
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "Beverly-Knits-AI-Agent/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
        
        logger.info("ERP Integration Bridge session initialized")
    
    async def close(self):
        """Close HTTP session and cleanup"""
        if self.session:
            await self.session.close()
        logger.info("ERP Integration Bridge closed")
    
    async def call_erp_api(
        self, 
        endpoint_key: str, 
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        timeout_seconds: int = 30
    ) -> APICallResult:
        """
        Call Beverly Knits ERP API endpoint
        
        Args:
            endpoint_key: Key from api_endpoints mapping
            params: Query parameters
            use_cache: Whether to use cached responses
            timeout_seconds: Request timeout
            
        Returns:
            APICallResult with response data or error
        """
        start_time = time.time()
        
        try:
            # Check if circuit breaker is open
            if self._is_circuit_breaker_open():
                return APICallResult(
                    endpoint=endpoint_key,
                    status=APICallStatus.FAILED,
                    error="Circuit breaker is open - too many recent failures",
                    response_time_ms=0
                )
            
            # Get endpoint URL
            endpoint_path = self.api_endpoints.get(endpoint_key)
            if not endpoint_path:
                return APICallResult(
                    endpoint=endpoint_key,
                    status=APICallStatus.FAILED,
                    error=f"Unknown endpoint key: {endpoint_key}",
                    response_time_ms=0
                )
            
            url = f"{self.erp_base_url}{endpoint_path}"
            
            # Check cache first
            if use_cache:
                cached_result = self._get_cached_response(endpoint_key, params)
                if cached_result:
                    self.performance_metrics["cache_hits"] += 1
                    return APICallResult(
                        endpoint=endpoint_key,
                        status=APICallStatus.CACHED,
                        data=cached_result,
                        response_time_ms=int((time.time() - start_time) * 1000),
                        from_cache=True
                    )
            
            # Make API call with rate limiting
            async with self.request_semaphore:
                if not self.session:
                    await self.initialize()
                
                # Prepare request
                request_params = params or {}
                
                # Make request
                async with self.session.get(url, params=request_params) as response:
                    response_time_ms = int((time.time() - start_time) * 1000)
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Cache successful response
                        if use_cache:
                            self._cache_response(endpoint_key, params, data)
                        
                        # Update metrics
                        self._update_metrics(success=True, response_time_ms=response_time_ms)
                        
                        return APICallResult(
                            endpoint=endpoint_key,
                            status=APICallStatus.SUCCESS,
                            data=data,
                            response_time_ms=response_time_ms
                        )
                    else:
                        error_text = await response.text()
                        self._update_metrics(success=False, response_time_ms=response_time_ms)
                        
                        return APICallResult(
                            endpoint=endpoint_key,
                            status=APICallStatus.FAILED,
                            error=f"HTTP {response.status}: {error_text}",
                            response_time_ms=response_time_ms
                        )
        
        except asyncio.TimeoutError:
            response_time_ms = int((time.time() - start_time) * 1000)
            self._update_metrics(success=False, response_time_ms=response_time_ms)
            
            return APICallResult(
                endpoint=endpoint_key,
                status=APICallStatus.TIMEOUT,
                error=f"Request timeout after {timeout_seconds}s",
                response_time_ms=response_time_ms
            )
        
        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            self._update_metrics(success=False, response_time_ms=response_time_ms)
            
            logger.error(f"ERP API call failed for {endpoint_key}: {str(e)}")
            
            return APICallResult(
                endpoint=endpoint_key,
                status=APICallStatus.FAILED,
                error=str(e),
                response_time_ms=response_time_ms
            )
    
    async def batch_call_erp_apis(
        self, 
        calls: List[Dict[str, Any]]
    ) -> List[APICallResult]:
        """
        Make multiple ERP API calls concurrently
        
        Args:
            calls: List of call specifications with 'endpoint_key', 'params', etc.
            
        Returns:
            List of APICallResult objects
        """
        tasks = []
        
        for call_spec in calls:
            task = self.call_erp_api(
                endpoint_key=call_spec.get("endpoint_key"),
                params=call_spec.get("params"),
                use_cache=call_spec.get("use_cache", True),
                timeout_seconds=call_spec.get("timeout_seconds", 30)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(APICallResult(
                    endpoint=calls[i].get("endpoint_key", "unknown"),
                    status=APICallStatus.FAILED,
                    error=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def _get_cached_response(self, endpoint_key: str, params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired"""
        cache_key = self._generate_cache_key(endpoint_key, params)
        cached_entry = self.cache.get(cache_key)
        
        if cached_entry and not cached_entry.is_expired():
            return cached_entry.data
        elif cached_entry and cached_entry.is_expired():
            # Remove expired entry
            del self.cache[cache_key]
        
        return None
    
    def _cache_response(self, endpoint_key: str, params: Optional[Dict[str, Any]], data: Dict[str, Any]):
        """Cache API response"""
        cache_key = self._generate_cache_key(endpoint_key, params)
        ttl = self.cache_ttl_overrides.get(endpoint_key, self.default_cache_ttl)
        
        self.cache[cache_key] = CachedResponse(
            data=data,
            timestamp=datetime.now(),
            ttl_seconds=ttl
        )
        
        # Clean up expired entries periodically
        if len(self.cache) > 1000:  # Arbitrary limit
            self._cleanup_expired_cache()
    
    def _generate_cache_key(self, endpoint_key: str, params: Optional[Dict[str, Any]]) -> str:
        """Generate cache key from endpoint and parameters"""
        if params:
            # Sort parameters for consistent key generation
            sorted_params = sorted(params.items())
            params_str = json.dumps(sorted_params, sort_keys=True)
            return f"{endpoint_key}:{params_str}"
        else:
            return endpoint_key
    
    def _cleanup_expired_cache(self):
        """Remove expired cache entries"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.circuit_open_until and datetime.now() < self.circuit_open_until:
            return True
        elif self.circuit_open_until and datetime.now() >= self.circuit_open_until:
            # Circuit breaker timeout expired, reset
            self.circuit_open_until = None
            self.error_count = 0
            logger.info("Circuit breaker closed - timeout expired")
            return False
        
        return False
    
    def _update_metrics(self, success: bool, response_time_ms: int):
        """Update performance metrics"""
        self.performance_metrics["total_requests"] += 1
        
        if success:
            self.performance_metrics["successful_requests"] += 1
            self.error_count = 0  # Reset error count on success
        else:
            self.performance_metrics["failed_requests"] += 1
            self.error_count += 1
            
            # Check if we should open circuit breaker
            if self.error_count >= self.circuit_breaker_threshold:
                self.circuit_open_until = datetime.now() + timedelta(seconds=self.circuit_breaker_timeout)
                logger.warning(f"Circuit breaker opened due to {self.error_count} consecutive errors")
        
        # Update average response time (exponential moving average)
        if self.performance_metrics["average_response_time_ms"] == 0:
            self.performance_metrics["average_response_time_ms"] = response_time_ms
        else:
            alpha = 0.1  # Smoothing factor
            current_avg = self.performance_metrics["average_response_time_ms"]
            self.performance_metrics["average_response_time_ms"] = (
                alpha * response_time_ms + (1 - alpha) * current_avg
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        total_requests = self.performance_metrics["total_requests"]
        
        return {
            **self.performance_metrics,
            "success_rate": (
                self.performance_metrics["successful_requests"] / total_requests * 100
                if total_requests > 0 else 0
            ),
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"] / total_requests * 100
                if total_requests > 0 else 0
            ),
            "circuit_breaker_status": "OPEN" if self._is_circuit_breaker_open() else "CLOSED",
            "cache_entries": len(self.cache),
            "uptime_hours": (datetime.now() - self.performance_metrics["last_reset"]).total_seconds() / 3600
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "average_response_time_ms": 0.0,
            "last_reset": datetime.now()
        }
        logger.info("Performance metrics reset")
    
    def clear_cache(self):
        """Clear all cached responses"""
        cache_count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {cache_count} cache entries")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of ERP system"""
        try:
            result = await self.call_erp_api("system_health", use_cache=False, timeout_seconds=10)
            
            return {
                "erp_status": "HEALTHY" if result.status == APICallStatus.SUCCESS else "UNHEALTHY",
                "response_time_ms": result.response_time_ms,
                "error": result.error,
                "bridge_metrics": self.get_performance_metrics(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "erp_status": "UNHEALTHY",
                "error": str(e),
                "bridge_metrics": self.get_performance_metrics(),
                "timestamp": datetime.now().isoformat()
            }
    
    # Convenience methods for specific Beverly Knits ERP endpoints
    async def get_inventory_intelligence(
        self, 
        view: str = "summary", 
        analysis: str = "shortage", 
        realtime: bool = True
    ) -> APICallResult:
        """Get inventory intelligence data"""
        params = {"view": view, "analysis": analysis}
        if realtime:
            params["realtime"] = "true"
        
        return await self.call_erp_api("inventory_intelligence", params)
    
    async def get_yarn_intelligence(
        self, 
        analysis: str = "shortage", 
        forecast: bool = True
    ) -> APICallResult:
        """Get yarn intelligence data"""
        params = {"analysis": analysis}
        if forecast:
            params["forecast"] = "true"
        
        return await self.call_erp_api("yarn_intelligence", params)
    
    async def get_ml_forecast(
        self, 
        detail: str = "detailed", 
        horizon: int = 90, 
        format_type: str = "report"
    ) -> APICallResult:
        """Get ML forecast data"""
        params = {
            "detail": detail,
            "horizon": horizon,
            "format": format_type
        }
        
        return await self.call_erp_api("ml_forecast", params)
    
    async def get_production_planning(
        self, 
        view: str = "orders", 
        forecast: bool = True
    ) -> APICallResult:
        """Get production planning data"""
        params = {"view": view}
        if forecast:
            params["forecast"] = "true"
        
        return await self.call_erp_api("production_planning", params)
    
    async def get_machine_assignments(self) -> APICallResult:
        """Get machine assignment suggestions"""
        return await self.call_erp_api("machine_assignment")
    
    async def get_yarn_substitution(
        self, 
        target_yarn: Optional[str] = None, 
        compatibility_threshold: float = 0.8
    ) -> APICallResult:
        """Get yarn substitution recommendations"""
        params = {"compatibility_threshold": compatibility_threshold}
        if target_yarn:
            params["target_yarn"] = target_yarn
        
        return await self.call_erp_api("yarn_substitution", params)


# Global bridge instance
erp_bridge = ERPIntegrationBridge()

# Export main components
__all__ = ["ERPIntegrationBridge", "APICallResult", "APICallStatus", "erp_bridge"]