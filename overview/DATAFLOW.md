# Beverly Knits ERP v3 - Data Flow Architecture (Production-Ready Patterns)
# Created: 2025-01-28
# Modified: 2025-01-28

## Executive Summary

This document implements a resilient, scalable data flow architecture for Beverly Knits ERP v3. All patterns follow industry best practices with proper connection pooling, async processing, transactional consistency, and fault tolerance. The architecture eliminates single points of failure and ensures data integrity across distributed services.

## Table of Contents
1. [Core Data Architecture](#core-data-architecture)
2. [Connection Pool Implementation](#connection-pool-implementation)
3. [Data Source Abstraction](#data-source-abstraction)
4. [ETL Pipeline Architecture](#etl-pipeline-architecture)
5. [Event Streaming](#event-streaming)
6. [Data Consistency Patterns](#data-consistency-patterns)
7. [Caching Strategy](#caching-strategy)
8. [Data Validation Framework](#data-validation-framework)
9. [Monitoring & Observability](#monitoring--observability)
10. [Implementation Roadmap](#implementation-roadmap)

## Core Data Architecture

### Data Flow Diagram
```mermaid
graph TB
    subgraph "Data Sources"
        EFAB[eFab API<br/>Production Data]
        SP[SharePoint<br/>Sales Reports]
        WMS[Warehouse<br/>Inventory Data]
        SALES[Sales Activity<br/>Customer Orders]
    end

    subgraph "Data Ingestion Layer"
        ETL[ETL Pipeline<br/>Async Processing]
        VALID[Data Validation<br/>Schema Checking]
        CLEAN[Data Cleansing<br/>Quality Assurance]
    end

    subgraph "Data Processing"
        TRANS[Data Transformation<br/>Business Rules]
        ENRICH[Data Enrichment<br/>Cross-Reference]
        AGGR[Data Aggregation<br/>Analytics]
    end

    subgraph "Data Storage"
        PG_PROD[(Production DB<br/>Transactional)]
        PG_INV[(Inventory DB<br/>Stock Levels)]
        PG_ORDER[(Order DB<br/>Customer Data)]
        MONGO[(MongoDB<br/>Documents)]
        REDIS[(Redis<br/>Cache)]
    end

    subgraph "Data Consumers"
        PROD_SVC[Production Service]
        INV_SVC[Inventory Service]
        ORDER_SVC[Order Service]
        FORE_SVC[Forecasting Service]
        DASH[Analytics Dashboard]
    end

    EFAB --> ETL
    SP --> ETL
    WMS --> ETL
    SALES --> ETL

    ETL --> VALID
    VALID --> CLEAN
    CLEAN --> TRANS
    TRANS --> ENRICH
    ENRICH --> AGGR

    AGGR --> PG_PROD
    AGGR --> PG_INV
    AGGR --> PG_ORDER
    AGGR --> MONGO
    AGGR --> REDIS

    PG_PROD --> PROD_SVC
    PG_INV --> INV_SVC
    PG_ORDER --> ORDER_SVC
    MONGO --> FORE_SVC
    REDIS --> DASH
```

### Production Flow Mapping
```mermaid
graph LR
    subgraph "Sales Layer"
        SO[Sales Orders<br/>Customer Demand]
        FORE[Forecast<br/>Predicted Demand]
    end

    subgraph "Production Planning"
        KO[Knit Orders<br/>Production Instructions]
        SCHED[Production Schedule<br/>Machine Allocation]
    end

    subgraph "Material Flow"
        YARN[Yarn Inventory<br/>Raw Materials]
        G00[G00: Greige Fabric<br/>Work in Progress]
        G02[G02: Processed<br/>Quality Check]
        I01[I01: Inspection<br/>Quality Control]
        F01[F01: Finished Goods<br/>Ready to Ship]
    end

    SO --> KO
    FORE --> KO
    KO --> SCHED
    SCHED --> YARN
    YARN --> G00
    G00 --> G02
    G02 --> I01
    I01 --> F01
```

### High-Level Data Flow
```
External Sources → Adapters → Message Queue → Processing Services →
Repository Layer → Connection Pool → Database → Cache → API → Clients
```

### Technology Stack
- **Primary Database**: PostgreSQL 14+ with streaming replication
- **Cache**: Redis Sentinel cluster for HA
- **Message Queue**: RabbitMQ with clustering
- **Document Store**: MongoDB for unstructured data
- **Search**: ElasticSearch for full-text search
- **Object Storage**: S3-compatible for files
- **Stream Processing**: Apache Kafka for real-time events

## Connection Pool Implementation

### Database Connection Pool Manager
```python
# infrastructure/database/connection_manager.py
"""Production-ready database connection pooling.
Created: 2025-01-28
"""
from __future__ import annotations

from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.exc import DBAPIError, DisconnectionError
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any, Callable
import structlog
import asyncio
from datetime import datetime, timedelta
import threading

logger = structlog.get_logger()


class ConnectionPoolManager:
    """Enterprise-grade connection pool manager with monitoring."""

    def __init__(
        self,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 40,
        pool_timeout: float = 30.0,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        echo_pool: bool = False
    ):
        """Initialize connection pool with optimal settings."""
        self.database_url = database_url
        self.pool_config = {
            "pool_size": pool_size,
            "max_overflow": max_overflow,
            "pool_timeout": pool_timeout,
            "pool_recycle": pool_recycle,
            "pool_pre_ping": pool_pre_ping,
            "echo_pool": echo_pool
        }

        # Performance metrics
        self.metrics = {
            "connections_created": 0,
            "connections_recycled": 0,
            "connections_failed": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "overflow_connections": 0,
            "wait_time_total": 0.0,
            "query_count": 0,
            "error_count": 0
        }

        self._lock = threading.Lock()
        self._initialize_engine()
        self._setup_event_listeners()

    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine with connection pool."""
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=self.pool_config["pool_size"],
            max_overflow=self.pool_config["max_overflow"],
            pool_timeout=self.pool_config["pool_timeout"],
            pool_recycle=self.pool_config["pool_recycle"],
            pool_pre_ping=self.pool_config["pool_pre_ping"],
            echo_pool=self.pool_config["echo_pool"],
            future=True,  # SQLAlchemy 2.0 style
            connect_args={
                "connect_timeout": 10,
                "application_name": "beverly_knits_erp",
                "options": "-c statement_timeout=30000"  # 30 second statement timeout
            }
        )

        # Session factory with scoped sessions for thread safety
        session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
            class_=Session
        )
        self.SessionLocal = scoped_session(session_factory)

    def _setup_event_listeners(self) -> None:
        """Set up event listeners for monitoring."""
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Track new connection creation."""
            connection_record.info['connect_time'] = datetime.now()
            connection_record.info['query_count'] = 0

            with self._lock:
                self.metrics["connections_created"] += 1

            # Set connection parameters
            with dbapi_conn.cursor() as cursor:
                cursor.execute("SET lock_timeout = '10s'")
                cursor.execute("SET idle_in_transaction_session_timeout = '60s'")

            logger.info(
                "Database connection created",
                pool_size=self.engine.pool.size(),
                overflow=self.engine.pool.overflow(),
                total_connections=self.metrics["connections_created"]
            )

        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Track connection checkout from pool."""
            checkout_time = datetime.now()
            connection_record.info['checkout_time'] = checkout_time

            with self._lock:
                self.metrics["active_connections"] += 1

            # Calculate wait time
            if hasattr(connection_record.info, 'checkin_time'):
                wait_time = (checkout_time - connection_record.info['checkin_time']).total_seconds()
                self.metrics["wait_time_total"] += wait_time

            logger.debug(
                "Connection checked out",
                active=self.metrics["active_connections"],
                pool_size=self.engine.pool.size()
            )

        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Track connection checkin to pool."""
            connection_record.info['checkin_time'] = datetime.now()

            with self._lock:
                self.metrics["active_connections"] -= 1
                self.metrics["idle_connections"] += 1

            # Reset connection state
            with dbapi_conn.cursor() as cursor:
                cursor.execute("DISCARD ALL")

            logger.debug(
                "Connection checked in",
                active=self.metrics["active_connections"],
                idle=self.metrics["idle_connections"]
            )

        @event.listens_for(self.engine, "invalidate")
        def receive_invalidate(dbapi_conn, connection_record, exception):
            """Handle connection invalidation."""
            with self._lock:
                self.metrics["connections_failed"] += 1

            logger.error(
                "Connection invalidated",
                exception=str(exception),
                failed_total=self.metrics["connections_failed"]
            )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
            self.metrics["query_count"] += 1
        except DBAPIError as e:
            session.rollback()
            self.metrics["error_count"] += 1

            if e.connection_invalidated:
                logger.error("Connection invalidated, removing from pool")
                session.bind.pool.invalidate()

            raise
        except Exception as e:
            session.rollback()
            self.metrics["error_count"] += 1
            logger.error("Database error", error=str(e))
            raise
        finally:
            session.close()
            self.SessionLocal.remove()

    async def execute_async(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute query asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._execute_sync,
            query,
            params
        )

    def _execute_sync(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute query synchronously."""
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            return result.fetchall()

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on connection pool."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                db_responsive = result.scalar() == 1

            pool_status = self.get_pool_status()

            return {
                "healthy": db_responsive,
                "responsive": db_responsive,
                "pool_status": pool_status,
                "metrics": self.metrics
            }
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "healthy": False,
                "responsive": False,
                "error": str(e),
                "metrics": self.metrics
            }

    def get_pool_status(self) -> Dict[str, int]:
        """Get current pool status."""
        pool = self.engine.pool
        return {
            "size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total": pool.size() + pool.overflow(),
            "available": pool.size() - pool.checkedout()
        }

    def close(self) -> None:
        """Close all connections and cleanup."""
        self.SessionLocal.remove()
        self.engine.dispose()
        logger.info("Connection pool closed", metrics=self.metrics)


# Global connection pool instance
db_pool = ConnectionPoolManager(
    database_url="postgresql://user:pass@localhost/beverly_knits",
    pool_size=20,
    max_overflow=40
)
```

## Data Source Abstraction

### Unified Data Adapter Pattern
```python
# services/data_adapters/base_adapter.py
"""Base adapter for all data sources.
Created: 2025-01-28
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator
from datetime import datetime
import pandas as pd
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


class DataSourceConfig(BaseModel):
    """Configuration for data source."""
    source_type: str
    connection_string: str
    retry_attempts: int = 3
    timeout_seconds: int = 30
    batch_size: int = 1000
    enable_cache: bool = True


class DataAdapter(ABC):
    """Abstract base adapter for data sources."""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.metrics = {
            "records_read": 0,
            "records_written": 0,
            "errors": 0,
            "last_sync": None
        }

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source."""
        pass

    @abstractmethod
    async def read_data(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Read data from source."""
        pass

    @abstractmethod
    async def write_data(
        self,
        data: pd.DataFrame,
        target: str
    ) -> int:
        """Write data to source."""
        pass

    @abstractmethod
    async def stream_data(
        self,
        query: str,
        batch_size: Optional[int] = None
    ) -> AsyncIterator[pd.DataFrame]:
        """Stream data in batches."""
        pass

    async def validate_connection(self) -> bool:
        """Validate connection is active."""
        try:
            await self.connect()
            await self.disconnect()
            return True
        except Exception as e:
            logger.error(
                "Connection validation failed",
                source=self.config.source_type,
                error=str(e)
            )
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        return self.metrics


class PostgreSQLAdapter(DataAdapter):
    """PostgreSQL data adapter implementation."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.connection = None

    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        import asyncpg
        self.connection = await asyncpg.connect(
            self.config.connection_string,
            timeout=self.config.timeout_seconds
        )
        logger.info("Connected to PostgreSQL")

    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self.connection:
            await self.connection.close()
            logger.info("Disconnected from PostgreSQL")

    async def read_data(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Read data from PostgreSQL."""
        try:
            records = await self.connection.fetch(query, *(params or {}).values())
            df = pd.DataFrame(records)
            self.metrics["records_read"] += len(df)
            return df
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error("PostgreSQL read error", error=str(e))
            raise

    async def write_data(
        self,
        data: pd.DataFrame,
        target: str
    ) -> int:
        """Write data to PostgreSQL using COPY."""
        try:
            # Convert DataFrame to tuples
            records = [tuple(row) for row in data.values]

            # Use COPY for bulk insert
            await self.connection.copy_records_to_table(
                target,
                records=records,
                columns=list(data.columns)
            )

            self.metrics["records_written"] += len(records)
            return len(records)

        except Exception as e:
            self.metrics["errors"] += 1
            logger.error("PostgreSQL write error", error=str(e))
            raise

    async def stream_data(
        self,
        query: str,
        batch_size: Optional[int] = None
    ) -> AsyncIterator[pd.DataFrame]:
        """Stream data in batches."""
        batch_size = batch_size or self.config.batch_size

        async with self.connection.transaction():
            cursor = await self.connection.cursor(query)

            while True:
                records = await cursor.fetch(batch_size)
                if not records:
                    break

                df = pd.DataFrame(records)
                self.metrics["records_read"] += len(df)
                yield df


class EFabAPIAdapter(DataAdapter):
    """eFab API data adapter."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.session = None

    async def connect(self) -> None:
        """Connect to eFab API."""
        import aiohttp
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.config.connection_string}",
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        )
        logger.info("Connected to eFab API")

    async def disconnect(self) -> None:
        """Disconnect from eFab API."""
        if self.session:
            await self.session.close()
            logger.info("Disconnected from eFab API")

    async def read_data(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Read data from eFab API."""
        try:
            async with self.session.get(query, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                df = pd.DataFrame(data["results"] if "results" in data else data)
                self.metrics["records_read"] += len(df)
                return df

        except Exception as e:
            self.metrics["errors"] += 1
            logger.error("eFab API read error", error=str(e))
            raise

    async def write_data(
        self,
        data: pd.DataFrame,
        target: str
    ) -> int:
        """Write data to eFab API."""
        try:
            records = data.to_dict("records")

            async with self.session.post(
                target,
                json={"records": records}
            ) as response:
                response.raise_for_status()

                self.metrics["records_written"] += len(records)
                return len(records)

        except Exception as e:
            self.metrics["errors"] += 1
            logger.error("eFab API write error", error=str(e))
            raise

    async def stream_data(
        self,
        query: str,
        batch_size: Optional[int] = None
    ) -> AsyncIterator[pd.DataFrame]:
        """Stream paginated data from API."""
        batch_size = batch_size or self.config.batch_size
        offset = 0

        while True:
            params = {
                "limit": batch_size,
                "offset": offset
            }

            df = await self.read_data(query, params)

            if df.empty:
                break

            yield df
            offset += batch_size


class SharePointAdapter(DataAdapter):
    """SharePoint file adapter."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.client = None

    async def connect(self) -> None:
        """Connect to SharePoint."""
        from office365.runtime.auth.authentication_context import AuthenticationContext
        from office365.sharepoint.client_context import ClientContext

        ctx_auth = AuthenticationContext(self.config.connection_string)
        ctx_auth.acquire_token_for_user(
            username="user@domain.com",
            password="password"
        )

        self.client = ClientContext(
            self.config.connection_string,
            ctx_auth
        )
        logger.info("Connected to SharePoint")

    async def read_data(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Read files from SharePoint."""
        try:
            # Download file
            file_content = self.client.web.get_file_by_server_relative_url(query)
            self.client.load(file_content)
            self.client.execute_query()

            # Parse based on file type
            if query.endswith('.xlsx'):
                df = pd.read_excel(file_content.content)
            elif query.endswith('.csv'):
                df = pd.read_csv(file_content.content)
            else:
                raise ValueError(f"Unsupported file type: {query}")

            self.metrics["records_read"] += len(df)
            return df

        except Exception as e:
            self.metrics["errors"] += 1
            logger.error("SharePoint read error", error=str(e))
            raise
```

## ETL Pipeline Architecture

### Async ETL Pipeline Manager
```python
# services/etl/pipeline_manager.py
"""Resilient ETL pipeline with error recovery.
Created: 2025-01-28
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any, Callable, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import structlog
from asyncio import Queue, QueueEmpty

logger = structlog.get_logger()


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    PAUSED = "paused"


@dataclass
class PipelineStage:
    """ETL pipeline stage definition."""
    name: str
    handler: Callable
    retry_count: int = 3
    timeout_seconds: int = 300
    error_handler: Optional[Callable] = None
    parallel: bool = False


@dataclass
class PipelineResult:
    """Pipeline execution result."""
    pipeline_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime]
    stages_completed: int
    total_stages: int
    records_processed: int
    errors: List[str]
    metrics: Dict[str, Any]


class ETLPipeline:
    """Resilient ETL pipeline with async execution."""

    def __init__(
        self,
        name: str,
        source_adapter: DataAdapter,
        target_adapter: DataAdapter,
        batch_size: int = 1000,
        max_workers: int = 4
    ):
        self.name = name
        self.source = source_adapter
        self.target = target_adapter
        self.batch_size = batch_size
        self.max_workers = max_workers

        self.stages: List[PipelineStage] = []
        self.status = PipelineStatus.PENDING
        self.metrics = {
            "records_extracted": 0,
            "records_transformed": 0,
            "records_loaded": 0,
            "batches_processed": 0,
            "errors": 0,
            "retries": 0,
            "duration_seconds": 0
        }

        # Processing queues
        self.extract_queue: Queue = Queue(maxsize=10)
        self.transform_queue: Queue = Queue(maxsize=10)
        self.load_queue: Queue = Queue(maxsize=10)

    def add_stage(self, stage: PipelineStage) -> None:
        """Add processing stage to pipeline."""
        self.stages.append(stage)
        logger.info(f"Added stage: {stage.name}")

    async def execute(
        self,
        query: str,
        target: str
    ) -> PipelineResult:
        """Execute ETL pipeline."""
        start_time = datetime.now()
        self.status = PipelineStatus.RUNNING
        errors = []

        try:
            # Connect to sources
            await self.source.connect()
            await self.target.connect()

            # Create worker tasks
            workers = []

            # Extraction workers
            for _ in range(max(1, self.max_workers // 3)):
                workers.append(
                    asyncio.create_task(
                        self._extract_worker(query)
                    )
                )

            # Transformation workers
            for _ in range(self.max_workers):
                workers.append(
                    asyncio.create_task(
                        self._transform_worker()
                    )
                )

            # Load workers
            for _ in range(max(1, self.max_workers // 2)):
                workers.append(
                    asyncio.create_task(
                        self._load_worker(target)
                    )
                )

            # Wait for completion
            await asyncio.gather(*workers, return_exceptions=True)

            self.status = PipelineStatus.COMPLETED

        except Exception as e:
            self.status = PipelineStatus.FAILED
            errors.append(str(e))
            logger.error("Pipeline failed", error=str(e))

        finally:
            # Cleanup
            await self.source.disconnect()
            await self.target.disconnect()

            end_time = datetime.now()
            self.metrics["duration_seconds"] = (
                end_time - start_time
            ).total_seconds()

        return PipelineResult(
            pipeline_id=f"{self.name}_{start_time.isoformat()}",
            status=self.status,
            start_time=start_time,
            end_time=end_time,
            stages_completed=len(self.stages),
            total_stages=len(self.stages),
            records_processed=self.metrics["records_loaded"],
            errors=errors,
            metrics=self.metrics
        )

    async def _extract_worker(self, query: str) -> None:
        """Extract data from source."""
        try:
            async for batch in self.source.stream_data(query, self.batch_size):
                await self.extract_queue.put(batch)
                self.metrics["records_extracted"] += len(batch)
                self.metrics["batches_processed"] += 1

                logger.debug(
                    "Extracted batch",
                    size=len(batch),
                    total=self.metrics["records_extracted"]
                )

        except Exception as e:
            self.metrics["errors"] += 1
            logger.error("Extraction error", error=str(e))
            raise

        finally:
            # Signal completion
            await self.extract_queue.put(None)

    async def _transform_worker(self) -> None:
        """Transform data batches."""
        while True:
            try:
                # Get batch from extract queue
                batch = await asyncio.wait_for(
                    self.extract_queue.get(),
                    timeout=60
                )

                if batch is None:
                    # End of data
                    await self.transform_queue.put(None)
                    break

                # Apply transformation stages
                transformed = batch
                for stage in self.stages:
                    if stage.name.startswith("transform_"):
                        try:
                            transformed = await self._execute_stage(
                                stage,
                                transformed
                            )
                        except Exception as e:
                            if stage.error_handler:
                                transformed = await stage.error_handler(
                                    transformed,
                                    e
                                )
                            else:
                                raise

                # Put in load queue
                await self.transform_queue.put(transformed)
                self.metrics["records_transformed"] += len(transformed)

            except QueueEmpty:
                continue
            except Exception as e:
                self.metrics["errors"] += 1
                logger.error("Transformation error", error=str(e))

    async def _load_worker(self, target: str) -> None:
        """Load data to target."""
        while True:
            try:
                # Get batch from transform queue
                batch = await asyncio.wait_for(
                    self.transform_queue.get(),
                    timeout=60
                )

                if batch is None:
                    # End of data
                    break

                # Load to target
                records_written = await self.target.write_data(
                    batch,
                    target
                )

                self.metrics["records_loaded"] += records_written

                logger.debug(
                    "Loaded batch",
                    size=records_written,
                    total=self.metrics["records_loaded"]
                )

            except QueueEmpty:
                continue
            except Exception as e:
                self.metrics["errors"] += 1
                logger.error("Load error", error=str(e))

                # Retry logic
                if self.metrics["retries"] < 3:
                    self.metrics["retries"] += 1
                    await self.transform_queue.put(batch)
                else:
                    raise

    async def _execute_stage(
        self,
        stage: PipelineStage,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Execute pipeline stage with retry."""
        for attempt in range(stage.retry_count):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    stage.handler(data),
                    timeout=stage.timeout_seconds
                )
                return result

            except asyncio.TimeoutError:
                logger.warning(
                    f"Stage {stage.name} timed out",
                    attempt=attempt + 1
                )
                if attempt == stage.retry_count - 1:
                    raise

            except Exception as e:
                logger.error(
                    f"Stage {stage.name} failed",
                    attempt=attempt + 1,
                    error=str(e)
                )
                if attempt == stage.retry_count - 1:
                    raise

                await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## Event Streaming

### Kafka Event Stream Processor
```python
# services/streaming/kafka_processor.py
"""Real-time event streaming with Kafka.
Created: 2025-01-28
"""
from typing import Optional, Dict, Any, Callable, List
import asyncio
import json
from datetime import datetime
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
import structlog

logger = structlog.get_logger()


class EventStreamProcessor:
    """Kafka-based event streaming processor."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "beverly-knits-erp"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.handlers: Dict[str, List[Callable]] = {}

    async def start(self) -> None:
        """Start event streaming."""
        # Initialize producer
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode(),
            compression_type="snappy",
            max_batch_size=16384,
            linger_ms=10
        )
        await self.producer.start()
        logger.info("Event producer started")

    async def stop(self) -> None:
        """Stop event streaming."""
        if self.producer:
            await self.producer.stop()

        for consumer in self.consumers.values():
            await consumer.stop()

        logger.info("Event streaming stopped")

    async def publish_event(
        self,
        topic: str,
        event: Dict[str, Any],
        key: Optional[str] = None
    ) -> None:
        """Publish event to stream."""
        try:
            # Add metadata
            event["timestamp"] = datetime.now().isoformat()
            event["source"] = "beverly-knits-erp"

            # Send to Kafka
            await self.producer.send_and_wait(
                topic,
                value=event,
                key=key.encode() if key else None
            )

            logger.info(
                "Event published",
                topic=topic,
                key=key
            )

        except KafkaError as e:
            logger.error(
                "Failed to publish event",
                topic=topic,
                error=str(e)
            )
            raise

    async def subscribe(
        self,
        topic: str,
        handler: Callable
    ) -> None:
        """Subscribe to event stream."""
        if topic not in self.consumers:
            # Create consumer
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda v: json.loads(v.decode()),
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                max_poll_records=100
            )

            await consumer.start()
            self.consumers[topic] = consumer

            # Start consumer task
            asyncio.create_task(
                self._consume_events(topic, consumer)
            )

        # Register handler
        if topic not in self.handlers:
            self.handlers[topic] = []

        self.handlers[topic].append(handler)
        logger.info(f"Subscribed to topic: {topic}")

    async def _consume_events(
        self,
        topic: str,
        consumer: AIOKafkaConsumer
    ) -> None:
        """Consume events from stream."""
        try:
            async for msg in consumer:
                event = msg.value

                # Call handlers
                for handler in self.handlers.get(topic, []):
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(
                            "Handler error",
                            topic=topic,
                            error=str(e)
                        )

        except Exception as e:
            logger.error(
                "Consumer error",
                topic=topic,
                error=str(e)
            )
```

## Data Consistency Patterns

### Saga Pattern Implementation
```python
# services/consistency/saga_manager.py
"""Distributed transaction management with Saga pattern.
Created: 2025-01-28
"""
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import uuid
from datetime import datetime
import structlog

logger = structlog.get_logger()


class SagaStatus(Enum):
    """Saga execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


@dataclass
class SagaStep:
    """Individual saga step."""
    name: str
    action: Callable
    compensation: Callable
    timeout_seconds: int = 30
    retry_count: int = 3


@dataclass
class SagaResult:
    """Saga execution result."""
    saga_id: str
    status: SagaStatus
    completed_steps: List[str]
    failed_step: Optional[str]
    error: Optional[str]


class SagaManager:
    """Manages distributed transactions using Saga pattern."""

    def __init__(self):
        self.active_sagas: Dict[str, SagaStatus] = {}

    async def execute_saga(
        self,
        steps: List[SagaStep]
    ) -> SagaResult:
        """Execute saga with automatic compensation."""
        saga_id = str(uuid.uuid4())
        self.active_sagas[saga_id] = SagaStatus.RUNNING

        completed_steps = []
        failed_step = None
        error_message = None

        try:
            # Execute forward steps
            for step in steps:
                try:
                    await self._execute_step(step)
                    completed_steps.append(step.name)
                    logger.info(f"Completed step: {step.name}")

                except Exception as e:
                    failed_step = step.name
                    error_message = str(e)
                    logger.error(
                        f"Step {step.name} failed",
                        error=str(e)
                    )

                    # Start compensation
                    self.active_sagas[saga_id] = SagaStatus.COMPENSATING
                    await self._compensate(steps, completed_steps)

                    self.active_sagas[saga_id] = SagaStatus.COMPENSATED
                    break

            else:
                # All steps completed successfully
                self.active_sagas[saga_id] = SagaStatus.COMPLETED

        except Exception as e:
            self.active_sagas[saga_id] = SagaStatus.FAILED
            error_message = str(e)

        return SagaResult(
            saga_id=saga_id,
            status=self.active_sagas[saga_id],
            completed_steps=completed_steps,
            failed_step=failed_step,
            error=error_message
        )

    async def _execute_step(self, step: SagaStep) -> Any:
        """Execute single saga step with retry."""
        for attempt in range(step.retry_count):
            try:
                result = await asyncio.wait_for(
                    step.action(),
                    timeout=step.timeout_seconds
                )
                return result

            except asyncio.TimeoutError:
                if attempt == step.retry_count - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

            except Exception as e:
                if attempt == step.retry_count - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def _compensate(
        self,
        steps: List[SagaStep],
        completed_steps: List[str]
    ) -> None:
        """Compensate completed steps in reverse order."""
        for step in reversed(steps):
            if step.name in completed_steps:
                try:
                    await step.compensation()
                    logger.info(f"Compensated step: {step.name}")
                except Exception as e:
                    logger.error(
                        f"Compensation failed for {step.name}",
                        error=str(e)
                    )
```

## Caching Strategy

### Multi-Layer Cache Implementation
```python
# services/cache/multi_layer_cache.py
"""Multi-layer caching with L1/L2/L3 strategy.
Created: 2025-01-28
"""
from typing import Optional, Any, Union, Dict
from datetime import timedelta
import pickle
import asyncio
from functools import wraps
import redis.asyncio as redis
import aiomcache
import structlog

logger = structlog.get_logger()


class MultiLayerCache:
    """Three-layer cache: Memory (L1), Memcached (L2), Redis (L3)."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        memcached_host: str = "localhost",
        memcached_port: int = 11211,
        l1_max_size: int = 1000
    ):
        # L1: In-memory cache (fastest, smallest)
        self.l1_cache: Dict[str, Any] = {}
        self.l1_max_size = l1_max_size

        # L2: Memcached (fast, medium)
        self.l2_cache = aiomcache.Client(
            memcached_host,
            memcached_port
        )

        # L3: Redis (slower, largest, persistent)
        self.l3_cache = redis.from_url(
            redis_url,
            decode_responses=False
        )

        self.metrics = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "sets": 0
        }

    async def get(
        self,
        key: str,
        namespace: str = "default"
    ) -> Optional[Any]:
        """Get value from cache (waterfall through layers)."""
        full_key = f"{namespace}:{key}"

        # Check L1 (memory)
        if full_key in self.l1_cache:
            self.metrics["l1_hits"] += 1
            return self.l1_cache[full_key]

        # Check L2 (Memcached)
        try:
            value = await self.l2_cache.get(full_key.encode())
            if value:
                self.metrics["l2_hits"] += 1
                # Promote to L1
                self._set_l1(full_key, pickle.loads(value))
                return pickle.loads(value)
        except Exception as e:
            logger.warning(f"L2 cache error: {e}")

        # Check L3 (Redis)
        try:
            value = await self.l3_cache.get(full_key)
            if value:
                self.metrics["l3_hits"] += 1
                deserialized = pickle.loads(value)
                # Promote to L1 and L2
                self._set_l1(full_key, deserialized)
                await self._set_l2(full_key, deserialized, 3600)
                return deserialized
        except Exception as e:
            logger.warning(f"L3 cache error: {e}")

        self.metrics["misses"] += 1
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None,
        namespace: str = "default"
    ) -> bool:
        """Set value in all cache layers."""
        full_key = f"{namespace}:{key}"

        if isinstance(ttl, timedelta):
            ttl = int(ttl.total_seconds())

        # Set in all layers
        self._set_l1(full_key, value)
        await self._set_l2(full_key, value, ttl or 3600)
        await self._set_l3(full_key, value, ttl)

        self.metrics["sets"] += 1
        return True

    def _set_l1(self, key: str, value: Any) -> None:
        """Set in L1 cache with LRU eviction."""
        if len(self.l1_cache) >= self.l1_max_size:
            # Simple LRU: remove first item
            self.l1_cache.pop(next(iter(self.l1_cache)))

        self.l1_cache[key] = value

    async def _set_l2(
        self,
        key: str,
        value: Any,
        ttl: int
    ) -> None:
        """Set in L2 cache."""
        try:
            await self.l2_cache.set(
                key.encode(),
                pickle.dumps(value),
                exptime=ttl
            )
        except Exception as e:
            logger.warning(f"L2 cache set error: {e}")

    async def _set_l3(
        self,
        key: str,
        value: Any,
        ttl: Optional[int]
    ) -> None:
        """Set in L3 cache."""
        try:
            serialized = pickle.dumps(value)
            if ttl:
                await self.l3_cache.setex(key, ttl, serialized)
            else:
                await self.l3_cache.set(key, serialized)
        except Exception as e:
            logger.warning(f"L3 cache set error: {e}")

    async def invalidate(
        self,
        key: str,
        namespace: str = "default"
    ) -> None:
        """Invalidate key from all cache layers."""
        full_key = f"{namespace}:{key}"

        # Remove from all layers
        self.l1_cache.pop(full_key, None)

        try:
            await self.l2_cache.delete(full_key.encode())
        except Exception:
            pass

        try:
            await self.l3_cache.delete(full_key)
        except Exception:
            pass

    def get_metrics(self) -> Dict[str, int]:
        """Get cache metrics."""
        total_hits = (
            self.metrics["l1_hits"] +
            self.metrics["l2_hits"] +
            self.metrics["l3_hits"]
        )
        total_requests = total_hits + self.metrics["misses"]

        if total_requests > 0:
            hit_rate = (total_hits / total_requests) * 100
        else:
            hit_rate = 0

        return {
            **self.metrics,
            "hit_rate": hit_rate
        }
```

## Data Validation Framework

### Schema Validation Pipeline
```python
# services/validation/schema_validator.py
"""Data validation framework with Pydantic.
Created: 2025-01-28
"""
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, validator, Field
from datetime import datetime
import pandas as pd
import structlog

logger = structlog.get_logger()


class ValidationResult(BaseModel):
    """Validation result container."""
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    records_validated: int
    records_failed: int


class OrderDataSchema(BaseModel):
    """Order data validation schema."""
    order_id: str = Field(..., min_length=1, max_length=50)
    customer_id: str = Field(..., min_length=1)
    product_id: str = Field(..., regex="^PROD-[0-9]+$")
    quantity: int = Field(..., gt=0, le=10000)
    unit_price: float = Field(..., gt=0, le=1000000)
    order_date: datetime
    delivery_date: datetime
    status: str = Field(..., regex="^(pending|processing|shipped|delivered|cancelled)$")

    @validator('delivery_date')
    def delivery_after_order(cls, v, values):
        """Ensure delivery date is after order date."""
        if 'order_date' in values and v < values['order_date']:
            raise ValueError('Delivery date must be after order date')
        return v

    @validator('quantity')
    def quantity_reasonable(cls, v):
        """Check quantity is reasonable."""
        if v > 5000:
            logger.warning(f"Large quantity detected: {v}")
        return v


class DataValidator:
    """Comprehensive data validation framework."""

    def __init__(self):
        self.schemas: Dict[str, Type[BaseModel]] = {
            "orders": OrderDataSchema
        }
        self.validation_rules: List[Callable] = []

    def add_schema(
        self,
        name: str,
        schema: Type[BaseModel]
    ) -> None:
        """Register validation schema."""
        self.schemas[name] = schema

    def add_rule(self, rule: Callable) -> None:
        """Add custom validation rule."""
        self.validation_rules.append(rule)

    async def validate_dataframe(
        self,
        df: pd.DataFrame,
        schema_name: str
    ) -> ValidationResult:
        """Validate DataFrame against schema."""
        if schema_name not in self.schemas:
            return ValidationResult(
                valid=False,
                errors=[f"Unknown schema: {schema_name}"],
                records_validated=0,
                records_failed=0
            )

        schema = self.schemas[schema_name]
        errors = []
        warnings = []
        failed_records = 0

        # Validate each row
        for idx, row in df.iterrows():
            try:
                # Convert row to dict and validate
                record = row.to_dict()
                schema(**record)

                # Apply custom rules
                for rule in self.validation_rules:
                    rule_result = await rule(record)
                    if rule_result:
                        warnings.append(
                            f"Row {idx}: {rule_result}"
                        )

            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
                failed_records += 1

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors[:100],  # Limit error messages
            warnings=warnings[:50],
            records_validated=len(df),
            records_failed=failed_records
        )

    async def validate_stream(
        self,
        stream: AsyncIterator[pd.DataFrame],
        schema_name: str
    ) -> AsyncIterator[ValidationResult]:
        """Validate streaming data."""
        async for batch in stream:
            result = await self.validate_dataframe(
                batch,
                schema_name
            )
            yield result
```

## Monitoring & Observability

### Data Pipeline Monitoring
```python
# services/monitoring/data_monitor.py
"""Data pipeline monitoring and alerting.
Created: 2025-01-28
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog

logger = structlog.get_logger()

# Metrics
data_processed = Counter(
    'data_records_processed_total',
    'Total records processed',
    ['pipeline', 'stage', 'status']
)

pipeline_duration = Histogram(
    'pipeline_duration_seconds',
    'Pipeline execution duration',
    ['pipeline'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]
)

active_pipelines = Gauge(
    'active_pipelines',
    'Number of active pipelines'
)

data_quality_score = Gauge(
    'data_quality_score',
    'Data quality score',
    ['dataset']
)

error_rate = Summary(
    'data_error_rate',
    'Data processing error rate',
    ['pipeline', 'error_type']
)


class DataMonitor:
    """Monitor data pipelines and quality."""

    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = {
            "error_rate": 0.05,  # 5% error rate
            "latency_p99": 300,  # 5 minutes
            "quality_score": 0.90  # 90% quality
        }

    def record_pipeline_execution(
        self,
        pipeline: str,
        duration: float,
        records: int,
        errors: int
    ) -> None:
        """Record pipeline execution metrics."""
        pipeline_duration.labels(pipeline=pipeline).observe(duration)

        data_processed.labels(
            pipeline=pipeline,
            stage="complete",
            status="success"
        ).inc(records - errors)

        if errors > 0:
            data_processed.labels(
                pipeline=pipeline,
                stage="complete",
                status="error"
            ).inc(errors)

            error_pct = errors / records if records > 0 else 0
            error_rate.labels(
                pipeline=pipeline,
                error_type="processing"
            ).observe(error_pct)

            # Check threshold
            if error_pct > self.thresholds["error_rate"]:
                self._create_alert(
                    "High Error Rate",
                    f"Pipeline {pipeline} error rate: {error_pct:.2%}",
                    "critical"
                )

    def record_data_quality(
        self,
        dataset: str,
        score: float
    ) -> None:
        """Record data quality metrics."""
        data_quality_score.labels(dataset=dataset).set(score)

        if score < self.thresholds["quality_score"]:
            self._create_alert(
                "Low Data Quality",
                f"Dataset {dataset} quality: {score:.2%}",
                "warning"
            )

    def _create_alert(
        self,
        title: str,
        message: str,
        severity: str
    ) -> None:
        """Create monitoring alert."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "message": message,
            "severity": severity
        }
        self.alerts.append(alert)
        logger.warning(f"Alert: {title}", **alert)

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        recent_alerts = [
            a for a in self.alerts
            if datetime.fromisoformat(a["timestamp"]) >
            datetime.now() - timedelta(hours=1)
        ]

        critical_alerts = [
            a for a in recent_alerts
            if a["severity"] == "critical"
        ]

        return {
            "healthy": len(critical_alerts) == 0,
            "active_pipelines": active_pipelines._value.get(),
            "recent_alerts": len(recent_alerts),
            "critical_alerts": len(critical_alerts),
            "alerts": recent_alerts[-10:]  # Last 10 alerts
        }
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
1. **Set up connection pooling**
   - Deploy PostgreSQL with replication
   - Configure connection pool (20/40)
   - Implement health checks
   - Add monitoring

2. **Implement cache layer**
   - Deploy Redis Sentinel
   - Set up multi-layer cache
   - Configure TTLs
   - Add cache warming

### Phase 2: Data Adapters (Week 2)
1. **Create data adapters**
   - PostgreSQL adapter
   - eFab API adapter
   - SharePoint adapter
   - File system adapter

2. **Implement validation**
   - Schema definitions
   - Validation rules
   - Error handling
   - Quality scoring

### Phase 3: ETL Pipelines (Week 3)
1. **Build ETL framework**
   - Async pipeline manager
   - Error recovery
   - Retry logic
   - Progress tracking

2. **Create pipelines**
   - Order sync pipeline
   - Inventory update pipeline
   - Production data pipeline
   - Analytics pipeline

### Phase 4: Event Streaming (Week 4)
1. **Deploy Kafka**
   - Set up cluster
   - Create topics
   - Configure retention
   - Add monitoring

2. **Implement streaming**
   - Event publishers
   - Event consumers
   - Stream processing
   - Dead letter queues

### Phase 5: Production (Week 5)
1. **Performance tuning**
   - Optimize queries
   - Tune cache
   - Adjust pool sizes
   - Index optimization

2. **Monitoring setup**
   - Deploy Grafana
   - Create dashboards
   - Set up alerts
   - Document runbooks

## Success Metrics
- **Connection Pool Efficiency**: > 90% reuse rate
- **Cache Hit Rate**: > 80%
- **ETL Throughput**: > 10,000 records/second
- **Data Quality Score**: > 95%
- **Pipeline Success Rate**: > 99.5%
- **Latency**: P95 < 100ms for reads
- **Availability**: 99.9% uptime

## Conclusion
This data flow architecture provides enterprise-grade reliability, scalability, and performance. All components follow best practices with proper abstractions, error handling, and monitoring to ensure data integrity and system resilience.