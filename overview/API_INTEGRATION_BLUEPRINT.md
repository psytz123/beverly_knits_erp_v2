# Beverly Knits ERP v3 - API Integration Blueprint (Production-Ready Architecture)
# Created: 2025-01-28
# Modified: 2025-01-28

## Executive Summary
This blueprint implements a resilient, scalable API-first architecture for Beverly Knits ERP v3, incorporating industry best practices for fault tolerance, performance optimization, and maintainability. All components follow microservices patterns with proper separation of concerns.

## Table of Contents
1. [Core Architecture Principles](#core-architecture-principles)
2. [Resilient API Client Implementation](#resilient-api-client-implementation)
3. [Service Integration Patterns](#service-integration-patterns)
4. [Data Mapping & Transformation](#data-mapping--transformation)
5. [Caching Strategy](#caching-strategy)
6. [Error Handling & Recovery](#error-handling--recovery)
7. [Monitoring & Observability](#monitoring--observability)
8. [Security Implementation](#security-implementation)
9. [Performance Optimization](#performance-optimization)
10. [Implementation Roadmap](#implementation-roadmap)

## Core Architecture Principles

### Design Goals
- **Zero downtime**: System remains operational even when external APIs fail
- **Sub-second response**: 95th percentile response time < 1 second
- **Horizontal scalability**: Support 10x growth without architecture changes
- **Clean separation**: Each service handles single responsibility
- **Type safety**: 100% type coverage for public APIs

### Technology Stack
```yaml
Language: Python 3.11+
Framework: FastAPI (async-first)
Cache: Redis with Redis Sentinel
Queue: RabbitMQ with clustering
Database: PostgreSQL 14+ with connection pooling
Monitoring: Prometheus + Grafana
Tracing: OpenTelemetry with Jaeger
API Gateway: Kong/Nginx with rate limiting
```

### API Integration Architecture Diagram
```mermaid
graph TB
    subgraph "External Systems"
        EFAB[eFab API<br/>Production Data]
        SP[SharePoint<br/>Documents & Reports]
        ACCT[Accounting System<br/>Financial Data]
        WMS[Warehouse System<br/>Inventory Data]
    end

    subgraph "API Gateway Layer"
        KONG[Kong API Gateway<br/>Rate Limiting & Auth]
    end

    subgraph "Integration Services"
        IS[Integration Service<br/>Circuit Breaker & Retry]
        CACHE[Redis Cache<br/>Response Caching]
        QUEUE[RabbitMQ<br/>Async Processing]
    end

    subgraph "Core Services"
        PROD[Production Service]
        INV[Inventory Service]
        ORDER[Order Service]
        FORE[Forecasting Service]
    end

    subgraph "Data Layer"
        PG[(PostgreSQL<br/>Transactional Data)]
        MONGO[(MongoDB<br/>Document Store)]
        REDIS[(Redis<br/>Session & Cache)]
    end

    EFAB -->|REST API| KONG
    SP -->|Graph API| KONG
    ACCT -->|SOAP/REST| KONG
    WMS -->|REST API| KONG

    KONG --> IS
    IS --> CACHE
    IS --> QUEUE
    IS --> PROD
    IS --> INV
    IS --> ORDER
    IS --> FORE

    PROD --> PG
    INV --> PG
    ORDER --> PG
    FORE --> MONGO

    CACHE --> REDIS
    QUEUE --> REDIS
```

## Resilient API Client Implementation

### Base Client Architecture
```python
# services/api_integration/resilient_client.py
"""Resilient API client with circuit breaker, retry, and caching.
Created: 2025-01-28
"""
from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import aiohttp
import redis.asyncio as redis
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from circuit_breaker import CircuitBreaker
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()

T = TypeVar('T', bound=BaseModel)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ClientConfig:
    """API client configuration."""
    base_url: str
    api_key: str
    timeout_seconds: float = 30.0
    max_retries: int = 3
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: int = 60
    connection_limit: int = 100
    connection_limit_per_host: int = 30
    cache_ttl_seconds: int = 300
    enable_cache: bool = True


class ResilientAPIClient(Generic[T]):
    """Base resilient API client with all protection patterns."""

    def __init__(
        self,
        config: ClientConfig,
        redis_client: Optional[redis.Redis] = None,
        response_model: Optional[type[T]] = None
    ) -> None:
        """Initialize resilient client with configuration."""
        self.config = config
        self.response_model = response_model
        self.redis_client = redis_client or redis.Redis(
            decode_responses=True,
            connection_pool=redis.ConnectionPool(
                max_connections=50,
                health_check_interval=30
            )
        )

        # Circuit breaker setup
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_failure_threshold,
            recovery_timeout=config.circuit_recovery_timeout,
            expected_exception=aiohttp.ClientError,
            name=f"api_client_{config.base_url}"
        )

        # Connection pool configuration
        self.connector = aiohttp.TCPConnector(
            limit=config.connection_limit,
            limit_per_host=config.connection_limit_per_host,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=True,
            keepalive_timeout=30
        )

        # Timeout configuration
        self.timeout = aiohttp.ClientTimeout(
            total=config.timeout_seconds,
            connect=5,
            sock_read=10,
            sock_connect=5
        )

        self.session: Optional[aiohttp.ClientSession] = None
        self._closed = False

    async def __aenter__(self) -> ResilientAPIClient:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def initialize(self) -> None:
        """Initialize client session."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "BeverlyKnitsERP/3.0"
                }
            )

    async def close(self) -> None:
        """Clean up resources."""
        if not self._closed and self.session:
            await self.session.close()
            await asyncio.sleep(0.25)  # Allow connection cleanup
            self._closed = True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            aiohttp.ClientError,
            asyncio.TimeoutError
        )),
        reraise=True
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        if self.circuit_breaker.is_open:
            logger.warning(
                "Circuit breaker open",
                endpoint=endpoint,
                method=method
            )
            raise RuntimeError("Circuit breaker is open")

        try:
            url = f"{self.config.base_url}{endpoint}"
            async with self.session.request(
                method=method,
                url=url,
                **kwargs
            ) as response:
                response.raise_for_status()
                data = await response.json()

                # Record successful call
                self.circuit_breaker.record_success()

                return data

        except aiohttp.ClientError as e:
            # Record failed call
            self.circuit_breaker.record_failure()
            logger.error(
                "API request failed",
                endpoint=endpoint,
                method=method,
                error=str(e)
            )
            raise

    async def get_with_cache(
        self,
        endpoint: str,
        cache_key: Optional[str] = None,
        **params
    ) -> T:
        """GET request with caching."""
        cache_key = cache_key or f"api:{endpoint}:{str(params)}"

        # Try cache first
        if self.config.enable_cache:
            cached = await self.redis_client.get(cache_key)
            if cached:
                logger.info("Cache hit", key=cache_key)
                return self.response_model.parse_raw(cached)

        # Make API call
        data = await self._make_request("GET", endpoint, params=params)

        # Cache response
        if self.config.enable_cache:
            await self.redis_client.setex(
                cache_key,
                self.config.cache_ttl_seconds,
                self.response_model(**data).json()
            )

        return self.response_model(**data)

    async def post(
        self,
        endpoint: str,
        data: BaseModel,
        **kwargs
    ) -> T:
        """POST request with validation."""
        response = await self._make_request(
            "POST",
            endpoint,
            json=data.dict(exclude_none=True),
            **kwargs
        )
        return self.response_model(**response)

    async def health_check(self) -> bool:
        """Check API health status."""
        try:
            await self._make_request("GET", "/health")
            return True
        except Exception:
            return False


# Service-specific implementations

class EFabAPIClient(ResilientAPIClient):
    """eFab API client with domain-specific methods."""

    async def get_orders(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> list[Dict[str, Any]]:
        """Fetch orders within date range."""
        return await self.get_with_cache(
            "/api/v1/orders",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )

    async def get_inventory(
        self,
        location: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """Fetch current inventory levels."""
        params = {"location": location} if location else {}
        return await self.get_with_cache(
            "/api/v1/inventory",
            **params
        )

    async def update_production_status(
        self,
        order_id: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update production order status."""
        data = {
            "order_id": order_id,
            "status": status,
            "metadata": metadata or {},
            "updated_at": datetime.now().isoformat()
        }
        return await self.post(
            f"/api/v1/orders/{order_id}/status",
            data=data
        )
```

## Service Integration Patterns

### Integration Flow Diagram
```mermaid
sequenceDiagram
    participant Client
    participant Gateway as API Gateway
    participant Circuit as Circuit Breaker
    participant Cache as Redis Cache
    participant Service as Core Service
    participant External as External API

    Client->>Gateway: Request
    Gateway->>Circuit: Forward Request

    alt Cache Hit
        Circuit->>Cache: Check Cache
        Cache-->>Circuit: Return Cached Data
        Circuit-->>Gateway: Response
        Gateway-->>Client: Response
    else Cache Miss
        Circuit->>External: API Call
        alt API Success
            External-->>Circuit: Response
            Circuit->>Cache: Store Response
            Circuit-->>Gateway: Response
            Gateway-->>Client: Response
        else API Failure
            Circuit->>Service: Fallback Logic
            Service-->>Circuit: Fallback Response
            Circuit-->>Gateway: Response
            Gateway-->>Client: Response
        end
    end
```

### API Gateway Configuration
```python
# gateway/main.py
"""API Gateway with load balancing and rate limiting.
Created: 2025-01-28
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
from typing import Dict, Any
import asyncio
from datetime import datetime

app = FastAPI(
    title="Beverly Knits API Gateway",
    version="3.0.0",
    docs_url="/api/docs"
)

# Service registry
SERVICES = {
    "production": ["http://production-service:5001", "http://production-service-2:5001"],
    "inventory": ["http://inventory-service:5002", "http://inventory-service-2:5002"],
    "forecasting": ["http://ml-service:5003"],
    "ai": ["http://ai-service:5004"],
    "analytics": ["http://analytics-service:5005"]
}

# Load balancer state
service_indices = {service: 0 for service in SERVICES}


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: int = 100, per: int = 60):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = datetime.now()

    async def check(self, key: str) -> bool:
        """Check if request is allowed."""
        current = datetime.now()
        time_passed = (current - self.last_check).total_seconds()
        self.last_check = current
        self.allowance += time_passed * (self.rate / self.per)

        if self.allowance > self.rate:
            self.allowance = self.rate

        if self.allowance < 1.0:
            return False

        self.allowance -= 1.0
        return True


rate_limiter = RateLimiter(rate=1000, per=60)


@app.middleware("http")
async def add_rate_limiting(request: Request, call_next):
    """Rate limiting middleware."""
    client_id = request.client.host

    if not await rate_limiter.check(client_id):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"}
        )

    response = await call_next(request)
    return response


async def get_next_service(service_name: str) -> str:
    """Round-robin load balancing."""
    if service_name not in SERVICES:
        raise HTTPException(404, f"Service {service_name} not found")

    services = SERVICES[service_name]
    index = service_indices[service_name]
    service_url = services[index]

    # Update index for round-robin
    service_indices[service_name] = (index + 1) % len(services)

    return service_url


@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(service: str, path: str, request: Request):
    """Proxy requests to appropriate service."""
    service_url = await get_next_service(service)
    url = f"{service_url}/{path}"

    async with httpx.AsyncClient() as client:
        # Forward request
        response = await client.request(
            method=request.method,
            url=url,
            headers=request.headers,
            params=request.query_params,
            content=await request.body()
        )

        return JSONResponse(
            content=response.json(),
            status_code=response.status_code,
            headers=dict(response.headers)
        )
```

## Data Mapping & Transformation

### Unified Data Mapper
```python
# services/data_mapping/mapper.py
"""Unified data mapping for consistent column standardization.
Created: 2025-01-28
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from pydantic import BaseModel, Field, validator


class DataSource(Enum):
    """Supported data sources."""
    EFAB = "efab"
    QUADS = "quads"
    SHAREPOINT = "sharepoint"
    MANUAL = "manual"


@dataclass
class ColumnMapping:
    """Column mapping configuration."""
    source_column: str
    target_column: str
    data_type: type
    transformer: Optional[callable] = None
    default_value: Any = None


class DataMapper:
    """Universal data mapper for all sources."""

    # Standard column mappings
    STANDARD_MAPPINGS = {
        DataSource.EFAB: [
            ColumnMapping("StyleCode", "style_code", str),
            ColumnMapping("FabricType", "fabric_type", str),
            ColumnMapping("YarnWeight", "yarn_weight", float),
            ColumnMapping("OrderQty", "order_quantity", int),
            ColumnMapping("DeliveryDate", "delivery_date", pd.Timestamp),
            ColumnMapping("Customer", "customer_name", str),
            ColumnMapping("PONumber", "po_number", str),
            ColumnMapping("ColorCode", "color_code", str),
            ColumnMapping("Size", "size", str),
            ColumnMapping("Status", "status", str, str.lower)
        ],
        DataSource.QUADS: [
            ColumnMapping("style", "style_code", str),
            ColumnMapping("fabric", "fabric_type", str),
            ColumnMapping("yarn_wt", "yarn_weight", float),
            ColumnMapping("qty", "order_quantity", int),
            ColumnMapping("ship_date", "delivery_date", pd.Timestamp),
            ColumnMapping("cust_name", "customer_name", str),
            ColumnMapping("po_num", "po_number", str),
            ColumnMapping("color", "color_code", str),
            ColumnMapping("sz", "size", str),
            ColumnMapping("stat", "status", str, str.lower)
        ]
    }

    @classmethod
    def map_dataframe(
        cls,
        df: pd.DataFrame,
        source: DataSource
    ) -> pd.DataFrame:
        """Map DataFrame columns to standard format."""
        mappings = cls.STANDARD_MAPPINGS.get(source, [])
        result = pd.DataFrame()

        for mapping in mappings:
            if mapping.source_column in df.columns:
                col_data = df[mapping.source_column]

                # Apply transformer if provided
                if mapping.transformer:
                    col_data = col_data.apply(mapping.transformer)

                # Handle missing values
                if mapping.default_value is not None:
                    col_data = col_data.fillna(mapping.default_value)

                # Type conversion
                if mapping.data_type != str:
                    col_data = col_data.astype(mapping.data_type)

                result[mapping.target_column] = col_data
            else:
                # Add column with default values if missing
                result[mapping.target_column] = mapping.default_value

        return result

    @classmethod
    def validate_mapping(
        cls,
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> tuple[bool, List[str]]:
        """Validate that all required columns are present."""
        missing = [col for col in required_columns if col not in df.columns]
        return len(missing) == 0, missing


class OrderDataModel(BaseModel):
    """Validated order data model."""
    style_code: str = Field(..., min_length=1, max_length=50)
    fabric_type: str = Field(..., min_length=1)
    yarn_weight: float = Field(..., gt=0, le=1000)
    order_quantity: int = Field(..., ge=1)
    delivery_date: datetime
    customer_name: str = Field(..., min_length=1)
    po_number: str = Field(..., min_length=1)
    color_code: str = Field(..., min_length=1, max_length=20)
    size: str = Field(..., min_length=1, max_length=10)
    status: str = Field(..., regex="^(pending|processing|completed|cancelled)$")

    @validator('delivery_date')
    def delivery_date_future(cls, v):
        """Ensure delivery date is not in the past."""
        if v < datetime.now():
            raise ValueError('Delivery date must be in the future')
        return v
```

## Caching Strategy

### Redis Cache Manager
```python
# services/cache/cache_manager.py
"""Distributed caching with Redis Sentinel for HA.
Created: 2025-01-28
"""
from typing import Optional, Any, Union
import json
import pickle
from datetime import timedelta
import redis.asyncio as redis
from redis.asyncio.sentinel import Sentinel
import hashlib
import structlog

logger = structlog.get_logger()


class CacheManager:
    """High-availability cache manager with Redis Sentinel."""

    def __init__(
        self,
        sentinels: list[tuple[str, int]],
        service_name: str = "mymaster",
        db: int = 0,
        decode_responses: bool = True,
        max_connections: int = 50
    ):
        """Initialize cache with Sentinel for HA."""
        self.sentinel = Sentinel(
            sentinels,
            socket_connect_timeout=0.5,
            decode_responses=decode_responses,
            db=db
        )
        self.service_name = service_name
        self.pool = redis.ConnectionPool(
            max_connections=max_connections,
            health_check_interval=30
        )

    async def get_client(self) -> redis.Redis:
        """Get Redis client from Sentinel."""
        return self.sentinel.master_for(
            self.service_name,
            connection_pool=self.pool
        )

    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate namespaced cache key."""
        return f"{namespace}:{key}"

    def _hash_key(self, data: Any) -> str:
        """Generate hash key for complex data."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def get(
        self,
        key: str,
        namespace: str = "default"
    ) -> Optional[Any]:
        """Get value from cache."""
        client = await self.get_client()
        full_key = self._generate_key(namespace, key)

        try:
            value = await client.get(full_key)
            if value:
                # Try JSON first, fallback to pickle
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return pickle.loads(value)
            return None
        except redis.RedisError as e:
            logger.error("Cache get error", key=full_key, error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None,
        namespace: str = "default"
    ) -> bool:
        """Set value in cache with optional TTL."""
        client = await self.get_client()
        full_key = self._generate_key(namespace, key)

        # Convert timedelta to seconds
        if isinstance(ttl, timedelta):
            ttl = int(ttl.total_seconds())

        try:
            # Try JSON first, fallback to pickle
            try:
                serialized = json.dumps(value, default=str)
            except (TypeError, ValueError):
                serialized = pickle.dumps(value)

            if ttl:
                await client.setex(full_key, ttl, serialized)
            else:
                await client.set(full_key, serialized)
            return True
        except redis.RedisError as e:
            logger.error("Cache set error", key=full_key, error=str(e))
            return False

    async def delete(
        self,
        key: str,
        namespace: str = "default"
    ) -> bool:
        """Delete key from cache."""
        client = await self.get_client()
        full_key = self._generate_key(namespace, key)

        try:
            await client.delete(full_key)
            return True
        except redis.RedisError as e:
            logger.error("Cache delete error", key=full_key, error=str(e))
            return False

    async def invalidate_pattern(
        self,
        pattern: str,
        namespace: str = "default"
    ) -> int:
        """Invalidate all keys matching pattern."""
        client = await self.get_client()
        full_pattern = self._generate_key(namespace, pattern)

        try:
            keys = await client.keys(full_pattern)
            if keys:
                return await client.delete(*keys)
            return 0
        except redis.RedisError as e:
            logger.error("Cache invalidate error", pattern=full_pattern, error=str(e))
            return 0


# Cache decorators for easy use

def cache_result(
    ttl: Union[int, timedelta] = 300,
    namespace: str = "function",
    key_prefix: Optional[str] = None
):
    """Decorator to cache function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{key_prefix or func.__name__}:{str(args)}:{str(kwargs)}"

            # Try to get from cache
            cache_manager = CacheManager(
                sentinels=[("localhost", 26379)],
                service_name="mymaster"
            )
            cached = await cache_manager.get(cache_key, namespace)

            if cached is not None:
                return cached

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl, namespace)

            return result
        return wrapper
    return decorator
```

## Error Handling & Recovery

### Global Error Handler
```python
# services/errors/error_handler.py
"""Comprehensive error handling with recovery strategies.
Created: 2025-01-28
"""
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import traceback
import asyncio
from datetime import datetime
import structlog

logger = structlog.get_logger()


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    COMPENSATE = "compensate"
    ESCALATE = "escalate"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: type[Exception]
    severity: ErrorSeverity
    strategy: RecoveryStrategy
    max_retries: int = 3
    fallback_value: Optional[Any] = None
    compensation_func: Optional[Callable] = None
    metadata: Dict[str, Any] = None


class ErrorHandler:
    """Global error handler with recovery strategies."""

    # Error mapping configuration
    ERROR_MAPPINGS = {
        ConnectionError: ErrorContext(
            error_type=ConnectionError,
            severity=ErrorSeverity.HIGH,
            strategy=RecoveryStrategy.RETRY,
            max_retries=5
        ),
        TimeoutError: ErrorContext(
            error_type=TimeoutError,
            severity=ErrorSeverity.MEDIUM,
            strategy=RecoveryStrategy.FALLBACK
        ),
        ValueError: ErrorContext(
            error_type=ValueError,
            severity=ErrorSeverity.LOW,
            strategy=RecoveryStrategy.ESCALATE
        ),
        RuntimeError: ErrorContext(
            error_type=RuntimeError,
            severity=ErrorSeverity.CRITICAL,
            strategy=RecoveryStrategy.CIRCUIT_BREAK
        )
    }

    @classmethod
    async def handle_error(
        cls,
        error: Exception,
        context: Optional[ErrorContext] = None,
        operation_name: str = "unknown"
    ) -> Optional[Any]:
        """Handle error with appropriate recovery strategy."""
        # Get context from mapping if not provided
        if not context:
            context = cls.ERROR_MAPPINGS.get(
                type(error),
                ErrorContext(
                    error_type=type(error),
                    severity=ErrorSeverity.MEDIUM,
                    strategy=RecoveryStrategy.ESCALATE
                )
            )

        # Log error with context
        logger.error(
            "Error occurred",
            operation=operation_name,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=context.severity.value,
            strategy=context.strategy.value,
            traceback=traceback.format_exc()
        )

        # Apply recovery strategy
        if context.strategy == RecoveryStrategy.RETRY:
            return await cls._retry_strategy(error, context, operation_name)
        elif context.strategy == RecoveryStrategy.FALLBACK:
            return await cls._fallback_strategy(context)
        elif context.strategy == RecoveryStrategy.CIRCUIT_BREAK:
            return await cls._circuit_break_strategy(error, operation_name)
        elif context.strategy == RecoveryStrategy.COMPENSATE:
            return await cls._compensate_strategy(context)
        else:  # ESCALATE
            return await cls._escalate_strategy(error, operation_name)

    @classmethod
    async def _retry_strategy(
        cls,
        error: Exception,
        context: ErrorContext,
        operation_name: str
    ) -> Optional[Any]:
        """Retry with exponential backoff."""
        for attempt in range(context.max_retries):
            wait_time = 2 ** attempt
            logger.info(
                "Retrying operation",
                operation=operation_name,
                attempt=attempt + 1,
                wait_time=wait_time
            )
            await asyncio.sleep(wait_time)
            # Retry logic would go here
        return None

    @classmethod
    async def _fallback_strategy(
        cls,
        context: ErrorContext
    ) -> Optional[Any]:
        """Use fallback value or function."""
        if context.fallback_value is not None:
            logger.info("Using fallback value")
            return context.fallback_value
        return None

    @classmethod
    async def _circuit_break_strategy(
        cls,
        error: Exception,
        operation_name: str
    ) -> None:
        """Open circuit breaker to prevent cascade failures."""
        logger.critical(
            "Circuit breaker opened",
            operation=operation_name,
            error=str(error)
        )
        # Circuit breaker logic would go here
        raise error

    @classmethod
    async def _compensate_strategy(
        cls,
        context: ErrorContext
    ) -> Optional[Any]:
        """Execute compensation function."""
        if context.compensation_func:
            logger.info("Executing compensation")
            return await context.compensation_func()
        return None

    @classmethod
    async def _escalate_strategy(
        cls,
        error: Exception,
        operation_name: str
    ) -> None:
        """Escalate to monitoring/alerting system."""
        logger.error(
            "Escalating error",
            operation=operation_name,
            error=str(error)
        )
        # Send to monitoring system
        raise error


# Error recovery middleware
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with recovery."""
    result = await ErrorHandler.handle_error(
        exc,
        operation_name=f"{request.method} {request.url.path}"
    )

    if result is not None:
        return JSONResponse(
            status_code=200,
            content={"data": result, "recovered": True}
        )

    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "path": str(request.url)
        }
    )
```

## Monitoring & Observability

### Metrics Collection
```python
# services/monitoring/metrics.py
"""Prometheus metrics collection and OpenTelemetry tracing.
Created: 2025-01-28
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import time
from functools import wraps
from typing import Callable

# Prometheus metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

# OpenTelemetry setup
resource = Resource.create({"service.name": "beverly-knits-erp"})
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831
)

span_processor = BatchSpanProcessor(jaeger_exporter)
provider.add_span_processor(span_processor)

tracer = trace.get_tracer(__name__)


def track_metrics(endpoint: str):
    """Decorator to track API metrics."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Start timer
            start_time = time.time()

            # Start trace span
            with tracer.start_as_current_span(f"api.{endpoint}") as span:
                span.set_attribute("endpoint", endpoint)

                try:
                    # Execute function
                    result = await func(*args, **kwargs)

                    # Record success metrics
                    api_requests_total.labels(
                        method="GET",
                        endpoint=endpoint,
                        status="success"
                    ).inc()

                    span.set_attribute("status", "success")
                    return result

                except Exception as e:
                    # Record error metrics
                    api_requests_total.labels(
                        method="GET",
                        endpoint=endpoint,
                        status="error"
                    ).inc()

                    span.set_attribute("status", "error")
                    span.set_attribute("error", str(e))
                    raise

                finally:
                    # Record duration
                    duration = time.time() - start_time
                    api_request_duration.labels(
                        method="GET",
                        endpoint=endpoint
                    ).observe(duration)

        return wrapper
    return decorator


# Health check endpoint
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI()


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()
```

## Security Implementation

### API Security Layer
```python
# services/security/auth.py
"""JWT authentication and API key management.
Created: 2025-01-28
"""
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import structlog

logger = structlog.get_logger()

# Security configuration
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token security
security = HTTPBearer()


class TokenManager:
    """JWT token management."""

    @staticmethod
    def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + (
            expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        to_encode.update({"exp": expire, "type": "access"})

        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def create_refresh_token(
        data: Dict[str, Any]
    ) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})

        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def verify_token(
        token: str,
        token_type: str = "access"
    ) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                SECRET_KEY,
                algorithms=[ALGORITHM]
            )

            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token type"
                )

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials"
            )


class APIKeyManager:
    """API key management for service-to-service auth."""

    @staticmethod
    def generate_api_key() -> str:
        """Generate secure API key."""
        return f"bk_{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage."""
        return pwd_context.hash(api_key)

    @staticmethod
    def verify_api_key(
        plain_key: str,
        hashed_key: str
    ) -> bool:
        """Verify API key against hash."""
        return pwd_context.verify(plain_key, hashed_key)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """Dependency to get current authenticated user."""
    token = credentials.credentials
    payload = TokenManager.verify_token(token)

    # Extract user information
    user = {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "roles": payload.get("roles", [])
    }

    logger.info(
        "User authenticated",
        user_id=user["user_id"],
        email=user["email"]
    )

    return user


# Role-based access control
def require_role(required_role: str):
    """Decorator to require specific role."""
    def decorator(func):
        @wraps(func)
        async def wrapper(
            *args,
            current_user: Dict[str, Any] = Depends(get_current_user),
            **kwargs
        ):
            if required_role not in current_user.get("roles", []):
                raise HTTPException(
                    status_code=403,
                    detail="Insufficient permissions"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator
```

## Performance Optimization

### Connection Pool Management
```python
# services/database/connection_pool.py
"""Database connection pooling with monitoring.
Created: 2025-01-28
"""
from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
from contextlib import contextmanager
import structlog
from typing import Generator

logger = structlog.get_logger()

# Database configuration
DATABASE_URL = "postgresql://user:pass@localhost/beverly_knits"


class DatabasePool:
    """Managed database connection pool."""

    def __init__(
        self,
        url: str = DATABASE_URL,
        pool_size: int = 20,
        max_overflow: int = 40,
        pool_timeout: float = 30.0,
        pool_recycle: int = 3600
    ):
        """Initialize connection pool with monitoring."""
        self.engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,  # Test connections before use
            echo_pool=True,  # Log pool checkouts/checkins
            future=True  # Use SQLAlchemy 2.0 style
        )

        # Session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False
        )

        # Add pool monitoring
        self._setup_pool_monitoring()

    def _setup_pool_monitoring(self):
        """Set up connection pool monitoring."""
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Log new connection creation."""
            connection_record.info['connect_time'] = datetime.now()
            logger.info(
                "Database connection created",
                pool_size=self.engine.pool.size(),
                overflow=self.engine.pool.overflow()
            )

        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Log connection checkout."""
            logger.debug(
                "Connection checked out",
                checked_out=self.engine.pool.checkedout()
            )

        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Log connection checkin."""
            logger.debug(
                "Connection checked in",
                checked_out=self.engine.pool.checkedout()
            )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def health_check(self) -> bool:
        """Check database health."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False

    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status."""
        return {
            "size": self.engine.pool.size(),
            "checked_out": self.engine.pool.checkedout(),
            "overflow": self.engine.pool.overflow(),
            "total": self.engine.pool.size() + self.engine.pool.overflow()
        }


# Singleton instance
db_pool = DatabasePool()


# Dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database session."""
    with db_pool.get_session() as session:
        yield session
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
1. **Set up infrastructure**
   - Deploy Redis Sentinel cluster
   - Configure RabbitMQ with clustering
   - Set up PostgreSQL with replication
   - Deploy monitoring stack (Prometheus + Grafana)

2. **Implement base services**
   - Create resilient API client
   - Set up connection pooling
   - Implement cache manager
   - Configure error handling

### Phase 2: Service Extraction (Week 3-4)
1. **Break down monolithic file**
   - Extract production service (< 500 LOC)
   - Extract inventory service (< 500 LOC)
   - Extract forecasting service (< 500 LOC)
   - Extract AI agent service (< 500 LOC)

2. **Implement API Gateway**
   - Set up Kong/Nginx
   - Configure rate limiting
   - Implement load balancing
   - Add authentication middleware

### Phase 3: Integration (Week 5-6)
1. **Connect services**
   - Implement service mesh
   - Configure message queues
   - Set up event bus
   - Test end-to-end flows

2. **Data migration**
   - Migrate to single PostgreSQL instance
   - Implement data mapping layer
   - Set up ETL pipelines
   - Validate data consistency

### Phase 4: Optimization (Week 7-8)
1. **Performance tuning**
   - Optimize connection pools
   - Tune cache TTLs
   - Implement query optimization
   - Add database indexes

2. **Monitoring & Testing**
   - Set up comprehensive monitoring
   - Implement distributed tracing
   - Create integration tests
   - Performance benchmarking

### Phase 5: Deployment (Week 9-10)
1. **Production readiness**
   - Security audit
   - Load testing
   - Disaster recovery testing
   - Documentation completion

2. **Go-live**
   - Blue-green deployment
   - Monitor metrics
   - Gradual traffic migration
   - Performance validation

## Success Metrics
- **Response Time**: P95 < 1 second
- **Availability**: 99.9% uptime
- **Error Rate**: < 0.1%
- **Code Quality**: Cyclomatic complexity < 10
- **Test Coverage**: > 85%
- **Connection Pool Efficiency**: > 90% reuse
- **Cache Hit Rate**: > 80%

## Conclusion
This blueprint provides a production-ready API integration architecture that addresses all critical issues identified in the audit. By implementing these patterns, Beverly Knits ERP will achieve enterprise-grade reliability, scalability, and maintainability.