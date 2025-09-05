# Beverly Knits ERP v2 - Comprehensive Unified Implementation Plan

**Document Version:** 2.0 (Enhanced)  
**Date:** January 2025  
**Timeline:** 7 weeks (excluding security)  
**Focus:** Architecture, Performance, Quality, Feature Completion  

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current State Deep Analysis](#current-state-deep-analysis)
3. [Implementation Strategy](#implementation-strategy)
4. [Detailed Phase Breakdown](#detailed-phase-breakdown)
5. [Code Patterns & Examples](#code-patterns--examples)
6. [Rollback & Contingency Plans](#rollback--contingency-plans)
7. [Success Metrics & Monitoring](#success-metrics--monitoring)
8. [Risk Management](#risk-management)
9. [Post-Implementation Support](#post-implementation-support)

---

## Executive Summary

### Project Scope
Transform Beverly Knits ERP v2 from an 18,076-line monolith into a modern, microservices-based architecture with:
- **117 Python files** to be reorganized
- **127 API endpoints** to be consolidated to 50
- **7 existing services** to be properly integrated
- **67 bare except clauses** to be fixed
- **42 pass statements** to be implemented
- **Dashboard UI** preserved (locked from changes)

### Key Objectives
1. **Decompose monolith** into manageable services
2. **Improve performance** by 10-100x
3. **Consolidate data access** from 4+ loaders to 1
4. **Achieve 90% test coverage**
5. **Reduce technical debt** to maintainable levels

### Critical Success Factors
- Zero downtime during migration
- Maintain backward compatibility
- Preserve all business functionality
- Improve developer productivity by 50%

---

## Current State Deep Analysis

### System Architecture
```
Current Structure (117 Python files):
├── src/core/beverly_comprehensive_erp.py (18,076 lines)
├── src/services/ (7 partially integrated services)
├── src/data_loaders/ (4 competing implementations)
├── src/api/ (scattered endpoint definitions)
├── web/ (5 dashboard HTML files - DO NOT MODIFY)
└── tests/ (39 test files, ~30% coverage)
```

### Technical Debt Inventory

#### Code Quality Issues
| Issue Type | Count | Priority | Impact |
|------------|-------|----------|---------|
| Bare except clauses | 67 | HIGH | Silent failures, debugging nightmare |
| Pass statements | 42 | HIGH | Incomplete functionality |
| TODO/FIXME comments | 7 | MEDIUM | Missing features |
| Blocking operations | 17 | HIGH | Performance bottlenecks |
| DataFrame.iterrows() | 157 | HIGH | 10-100x slower than vectorized |
| SELECT * queries | Multiple | MEDIUM | Memory waste, slow queries |

#### Data Management Issues
- **Column name variations**: 15+ different naming patterns
- **Data loader confusion**: 4 competing implementations
- **Cache fragmentation**: No unified strategy
- **File path hardcoding**: Throughout codebase

### Business Context
- **1,199 yarn items** tracked
- **28,653 BOM entries**
- **194 production orders** (154 assigned, 40 pending)
- **91 work centers** with 285 machines
- **557,671 lbs** production workload

---

## Implementation Strategy

### Phased Approach
```mermaid
gantt
    title Beverly Knits ERP Refactoring Timeline
    dateFormat YYYY-MM-DD
    section Phase 1
    Service Extraction    :2025-01-08, 10d
    section Phase 2
    Data Consolidation   :2025-01-22, 5d
    section Phase 3
    Performance Opt      :2025-01-29, 5d
    section Phase 4
    API Consolidation    :2025-02-05, 5d
    section Phase 5
    Feature Completion   :2025-02-12, 5d
    section Phase 6
    Testing & Quality    :2025-02-19, 5d
```

### Migration Principles
1. **Strangler Fig Pattern**: Gradually replace monolith
2. **Feature Flags**: Control rollout per feature
3. **Parallel Running**: Old and new side-by-side
4. **Incremental Testing**: Test each extraction
5. **Continuous Monitoring**: Track all metrics

---

## Detailed Phase Breakdown

## PHASE 1: Service Extraction & Architecture (Days 1-10)

### Day 1: Environment Setup & Planning
```bash
# Morning (2 hours)
1. Create new project structure
   mkdir -p src/{domain,application,infrastructure}
   mkdir -p src/domain/{entities,services,interfaces}
   mkdir -p src/application/{commands,queries,handlers}
   mkdir -p src/infrastructure/{repositories,cache,external}

2. Set up dependency injection framework
   pip install dependency-injector
   pip install python-dotenv

3. Create configuration management
   - config/settings.py
   - config/database.py
   - config/cache.py

# Afternoon (4 hours)
4. Analyze monolith dependencies
   - Map class relationships
   - Identify extraction order
   - Document API endpoints
```

### Day 2-3: Extract Inventory Service
```python
# Step 1: Create domain entity
# src/domain/entities/yarn.py
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class Yarn:
    """Core yarn domain entity"""
    yarn_id: str
    description: str
    theoretical_balance: float
    allocated: float  # Already stored as negative
    on_order: float
    min_stock_level: float = 0.0
    lead_time_days: int = 14
    last_updated: Optional[datetime] = None
    
    @property
    def planning_balance(self) -> float:
        """Calculate planning balance"""
        return self.theoretical_balance + self.allocated + self.on_order
    
    def has_shortage(self) -> bool:
        """Check if yarn has shortage"""
        return self.planning_balance < self.min_stock_level
    
    def days_of_stock(self, daily_usage: float) -> float:
        """Calculate days of stock remaining"""
        if daily_usage <= 0:
            return float('inf')
        return self.planning_balance / daily_usage

# Step 2: Create repository interface
# src/domain/interfaces/yarn_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.entities import Yarn

class IYarnRepository(ABC):
    @abstractmethod
    async def get_by_id(self, yarn_id: str) -> Optional[Yarn]:
        pass
    
    @abstractmethod
    async def get_all(self) -> List[Yarn]:
        pass
    
    @abstractmethod
    async def get_shortages(self, threshold: float = 0) -> List[Yarn]:
        pass
    
    @abstractmethod
    async def update(self, yarn: Yarn) -> bool:
        pass

# Step 3: Create service
# src/services/inventory/inventory_service.py
class InventoryService:
    def __init__(self, repository: IYarnRepository, cache: ICacheService):
        self.repository = repository
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    async def calculate_planning_balance(self, yarn_id: str) -> PlanningBalance:
        """Calculate planning balance with caching"""
        cache_key = f"planning_balance:{yarn_id}"
        
        # Try cache first
        if cached := await self.cache.get(cache_key):
            return PlanningBalance.from_dict(cached)
        
        # Get from repository
        yarn = await self.repository.get_by_id(yarn_id)
        if not yarn:
            raise YarnNotFoundError(f"Yarn {yarn_id} not found")
        
        balance = PlanningBalance(
            yarn_id=yarn_id,
            theoretical=yarn.theoretical_balance,
            allocated=yarn.allocated,
            on_order=yarn.on_order,
            total=yarn.planning_balance
        )
        
        # Cache result
        await self.cache.set(cache_key, balance.to_dict(), ttl=300)
        return balance
```

### Day 4-5: Extract Forecasting Service
```python
# src/services/forecasting/forecasting_service.py
class ForecastingService:
    def __init__(self, ml_config: MLConfig):
        self.models = self._initialize_models(ml_config)
        self.accuracy_monitor = AccuracyMonitor()
        self.cache = MultiTierCache()
    
    def _initialize_models(self, config: MLConfig) -> Dict[str, Model]:
        """Initialize all ML models"""
        return {
            'arima': ARIMAModel(config.arima),
            'prophet': ProphetModel(config.prophet),
            'lstm': LSTMModel(config.lstm),
            'xgboost': XGBoostModel(config.xgboost)
        }
    
    async def predict_demand(
        self, 
        style_id: str, 
        horizon_days: int = 30,
        confidence: float = 0.95
    ) -> DemandForecast:
        """Generate ensemble forecast"""
        # Check cache
        cache_key = f"forecast:{style_id}:{horizon_days}"
        if cached := await self.cache.get(cache_key):
            return DemandForecast.from_dict(cached)
        
        # Get predictions from each model
        predictions = await asyncio.gather(*[
            self._get_model_prediction(model, style_id, horizon_days)
            for model in self.models.values()
        ])
        
        # Ensemble combination
        ensemble_pred = self._combine_predictions(predictions, confidence)
        
        # Monitor accuracy
        self.accuracy_monitor.record(style_id, ensemble_pred)
        
        # Cache and return
        await self.cache.set(cache_key, ensemble_pred.to_dict(), ttl=3600)
        return ensemble_pred
```

### Day 6-7: Implement Dependency Injection
```python
# src/infrastructure/container.py
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    """Application dependency injection container"""
    
    # Configuration
    config = providers.Configuration()
    
    # Infrastructure - Singletons
    database = providers.Singleton(
        DatabaseConnection,
        connection_string=config.database.url,
        pool_size=config.database.pool_size
    )
    
    redis_cache = providers.Singleton(
        RedisCache,
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db
    )
    
    # Data Loader - Singleton
    data_loader = providers.Singleton(
        UnifiedDataLoader,
        file_path=config.data.base_path,
        cache=redis_cache
    )
    
    # Repositories - Factory (new instance each time)
    yarn_repository = providers.Factory(
        YarnRepository,
        db=database,
        cache=redis_cache,
        data_loader=data_loader
    )
    
    bom_repository = providers.Factory(
        BOMRepository,
        db=database,
        data_loader=data_loader
    )
    
    order_repository = providers.Factory(
        OrderRepository,
        db=database,
        cache=redis_cache
    )
    
    # Services - Factory with dependencies
    inventory_service = providers.Factory(
        InventoryService,
        yarn_repository=yarn_repository,
        bom_repository=bom_repository,
        cache=redis_cache
    )
    
    forecasting_service = providers.Factory(
        ForecastingService,
        ml_config=config.ml,
        data_repository=yarn_repository
    )
    
    production_service = providers.Factory(
        ProductionService,
        order_repository=order_repository,
        inventory_service=inventory_service,
        forecasting_service=forecasting_service
    )

# Usage in application
container = Container()
container.config.from_yaml('config/settings.yaml')

# Wire up Flask app
app = Flask(__name__)
container.wire(modules=[__name__])

@app.route('/api/v2/inventory')
@inject
def get_inventory(
    inventory_service: InventoryService = Provide[Container.inventory_service]
):
    return inventory_service.get_inventory_status()
```

### Day 8-10: Complete Service Integration
```python
# src/application/orchestrator.py
class ProductionOrchestrator:
    """Orchestrates complex workflows across services"""
    
    def __init__(self, container: Container):
        self.inventory = container.inventory_service()
        self.forecasting = container.forecasting_service()
        self.production = container.production_service()
        self.yarn = container.yarn_service()
    
    async def complete_production_workflow(
        self, 
        order_request: ProductionOrderRequest
    ) -> ProductionWorkflowResult:
        """
        Complete production workflow:
        1. Check yarn availability
        2. Generate forecast
        3. Calculate requirements
        4. Create production plan
        5. Assign to machines
        """
        
        # Step 1: Check current inventory
        inventory_status = await self.inventory.get_current_status()
        
        # Step 2: Generate demand forecast
        forecast = await self.forecasting.predict_demand(
            style_id=order_request.style_id,
            horizon_days=order_request.lead_time
        )
        
        # Step 3: Calculate material requirements
        requirements = await self.yarn.calculate_requirements(
            style_id=order_request.style_id,
            quantity=order_request.quantity,
            include_forecast=True
        )
        
        # Step 4: Check availability
        shortages = self._identify_shortages(
            requirements, 
            inventory_status
        )
        
        if shortages:
            # Generate purchase orders
            purchase_orders = await self._generate_purchase_orders(shortages)
            
            # Adjust production plan
            adjusted_plan = await self.production.create_partial_plan(
                order_request,
                available_inventory=inventory_status
            )
        else:
            # Create full production plan
            adjusted_plan = await self.production.create_production_plan(
                order_request
            )
        
        # Step 5: Machine assignment
        machine_assignment = await self.production.assign_to_machines(
            adjusted_plan
        )
        
        return ProductionWorkflowResult(
            plan=adjusted_plan,
            machine_assignment=machine_assignment,
            material_requirements=requirements,
            shortages=shortages,
            forecast=forecast
        )
```

## PHASE 2: Data Layer Consolidation (Days 11-15)

### Day 11-12: Unify Data Loaders
```python
# src/infrastructure/data/unified_data_loader.py
class UnifiedDataLoader:
    """Single source of truth for all data loading"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.cache = CacheManager("data_loader", ttl=900)
        self.column_mapper = ColumnMapper()
        self.validator = DataValidator()
        
        # Initialize data sources in priority order
        self.sources = [
            FileDataSource(config.file_path),
            APIDataSource(config.api_url) if config.use_api else None,
            DatabaseDataSource(config.db_connection)
        ]
        self.sources = [s for s in self.sources if s]
    
    @cached(ttl=900)
    async def load_yarn_inventory(self) -> pd.DataFrame:
        """Load yarn inventory with fallback strategy"""
        for source in self.sources:
            try:
                self.logger.info(f"Loading yarn inventory from {source.__class__.__name__}")
                data = await source.load("yarn_inventory")
                
                if data is not None and not data.empty:
                    # Standardize columns
                    data = self.column_mapper.standardize(data)
                    
                    # Validate data
                    data = self.validator.validate_yarn_data(data)
                    
                    # Handle data type conversions
                    data = self._convert_data_types(data)
                    
                    return data
                    
            except DataSourceException as e:
                self.logger.warning(f"Failed to load from {source}: {e}")
                continue
        
        raise DataLoadException("All data sources failed for yarn_inventory")
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types"""
        numeric_columns = [
            'theoretical_balance', 'allocated', 'on_order', 
            'planning_balance', 'quantity'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                # Remove commas and dollar signs
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace(',', '').str.replace('$', '')
                # Convert to float
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
```

### Day 13: Column Standardization
```python
# src/infrastructure/data/column_mapper.py
class ColumnMapper:
    """Handle all column name variations"""
    
    # Master mapping configuration
    MAPPINGS = {
        'yarn_id': ['Desc#', 'desc_num', 'YarnID', 'yarn_id', 'Yarn ID'],
        'planning_balance': ['Planning Balance', 'Planning_Balance', 'planning balance'],
        'style_id': ['fStyle#', 'Style#', 'style_num', 'Style'],
        'quantity': ['Qty', 'Quantity', 'quantity', 'Amount'],
        'balance': ['Balance (lbs)', 'Balance', 'balance_lbs'],
        'theoretical_balance': ['Theoretical Balance', 'theoretical_balance', 'Theo Balance'],
        'allocated': ['Allocated', 'allocated', 'Alloc'],
        'on_order': ['On Order', 'on_order', 'OnOrder']
    }
    
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize all column names"""
        standardized = df.copy()
        
        # Create reverse mapping
        reverse_map = {}
        for standard, variations in self.MAPPINGS.items():
            for var in variations:
                reverse_map[var] = standard
        
        # Rename columns
        standardized = standardized.rename(columns=reverse_map)
        
        # Log unmapped columns
        unmapped = set(standardized.columns) - set(self.MAPPINGS.keys())
        if unmapped:
            self.logger.warning(f"Unmapped columns: {unmapped}")
        
        return standardized
    
    def validate_required_columns(
        self, 
        df: pd.DataFrame, 
        required: List[str]
    ) -> ValidationResult:
        """Validate DataFrame has required columns"""
        result = ValidationResult()
        
        for req_col in required:
            if req_col not in df.columns:
                # Try to find variation
                found = False
                for col in df.columns:
                    if col in self.MAPPINGS.get(req_col, []):
                        found = True
                        break
                
                if not found:
                    result.add_error(f"Missing required column: {req_col}")
        
        return result
```

### Day 14-15: Implement Caching Strategy
```python
# src/infrastructure/cache/multi_tier_cache.py
class MultiTierCache:
    """Multi-level caching for optimal performance"""
    
    def __init__(self):
        # L1: In-memory cache (microseconds)
        self.l1_memory = LRUCache(maxsize=100)
        self.l1_hits = 0
        self.l1_misses = 0
        
        # L2: Redis cache (milliseconds)
        self.l2_redis = redis.Redis(
            host='localhost', 
            port=6379,
            decode_responses=True
        )
        self.l2_hits = 0
        self.l2_misses = 0
        
        # Cache strategies by data type
        self.strategies = {
            'yarn_inventory': CacheStrategy(ttl=900, warm_on_start=True),
            'bom_data': CacheStrategy(ttl=3600, warm_on_start=True),
            'production_orders': CacheStrategy(ttl=60, warm_on_start=False),
            'ml_predictions': CacheStrategy(ttl=1800, warm_on_start=False)
        }
    
    async def get(self, key: str, data_type: str = 'default') -> Optional[Any]:
        """Get from cache with fallback chain"""
        
        # L1: Check memory cache
        if value := self.l1_memory.get(key):
            self.l1_hits += 1
            return value
        self.l1_misses += 1
        
        # L2: Check Redis cache
        try:
            if value := await self.l2_redis.get(key):
                self.l2_hits += 1
                # Promote to L1
                self.l1_memory.put(key, value)
                return json.loads(value)
        except redis.RedisError:
            pass
        self.l2_misses += 1
        
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        data_type: str = 'default'
    ):
        """Set in all cache levels"""
        strategy = self.strategies.get(data_type, CacheStrategy())
        
        # L1: Memory cache
        self.l1_memory.put(key, value)
        
        # L2: Redis cache with TTL
        try:
            await self.l2_redis.setex(
                key, 
                strategy.ttl, 
                json.dumps(value, cls=CustomJSONEncoder)
            )
        except redis.RedisError as e:
            self.logger.error(f"Redis set failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.l1_hits + self.l1_misses
        return {
            'l1_hit_rate': self.l1_hits / total_requests if total_requests > 0 else 0,
            'l2_hit_rate': self.l2_hits / self.l2_misses if self.l2_misses > 0 else 0,
            'l1_size': len(self.l1_memory),
            'total_requests': total_requests
        }

# Cache warming implementation
class CacheWarmer:
    """Proactive cache warming"""
    
    def __init__(self, cache: MultiTierCache, data_loader: UnifiedDataLoader):
        self.cache = cache
        self.loader = data_loader
        self.tasks = []
    
    async def warm_on_startup(self):
        """Warm critical data on application startup"""
        critical_data = [
            ('yarn_inventory', self.loader.load_yarn_inventory),
            ('bom_data', self.loader.load_bom_data),
            ('work_centers', self.loader.load_work_centers)
        ]
        
        tasks = []
        for data_type, loader_func in critical_data:
            tasks.append(self._warm_data(data_type, loader_func))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to warm {critical_data[i][0]}: {result}")
```

## PHASE 3: Performance Optimization (Days 16-20)

### Day 16: Eliminate DataFrame.iterrows()
```python
# src/optimization/dataframe_optimizer.py
class DataFrameOptimizer:
    """Replace iterrows with vectorized operations"""
    
    @staticmethod
    def optimize_planning_balance_calculation(df: pd.DataFrame) -> pd.DataFrame:
        """
        BEFORE (157 instances like this):
        for index, row in df.iterrows():
            df.at[index, 'planning_balance'] = (
                row['theoretical_balance'] + 
                row['allocated'] + 
                row['on_order']
            )
        
        AFTER (100x faster):
        """
        df['planning_balance'] = (
            df['theoretical_balance'] + 
            df['allocated'] + 
            df['on_order']
        )
        return df
    
    @staticmethod
    def optimize_shortage_detection(df: pd.DataFrame) -> pd.DataFrame:
        """
        BEFORE:
        shortages = []
        for index, row in df.iterrows():
            if row['planning_balance'] < row['min_stock']:
                shortages.append(row)
        
        AFTER (50x faster):
        """
        return df[df['planning_balance'] < df['min_stock']].copy()
    
    @staticmethod
    def optimize_bom_explosion(df: pd.DataFrame, quantity: float) -> pd.DataFrame:
        """
        BEFORE:
        for index, row in df.iterrows():
            df.at[index, 'required'] = row['quantity_per'] * quantity
        
        AFTER (Vectorized):
        """
        df['required'] = df['quantity_per'] * quantity
        return df
```

### Day 17: Database Query Optimization
```python
# src/infrastructure/database/query_optimizer.py
class QueryOptimizer:
    """Optimize database queries"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.query_cache = {}
    
    async def optimize_yarn_query(self, conditions: Dict = None) -> pd.DataFrame:
        """
        BEFORE:
        SELECT * FROM yarn_inventory WHERE status = 'active'
        
        AFTER:
        """
        columns = [
            'yarn_id', 'description', 'theoretical_balance',
            'allocated', 'on_order', 'min_stock', 'lead_time'
        ]
        
        query = f"""
        SELECT {', '.join(columns)}
        FROM yarn_inventory
        WHERE status = 'active'
        """
        
        if conditions:
            where_clauses = []
            params = []
            for key, value in conditions.items():
                where_clauses.append(f"{key} = %s")
                params.append(value)
            query += " AND " + " AND ".join(where_clauses)
        
        query += " ORDER BY yarn_id LIMIT 1000"
        
        return await self.db.fetch_df(query, params)
    
    async def batch_fetch(
        self, 
        table: str, 
        ids: List[str], 
        batch_size: int = 1000
    ) -> pd.DataFrame:
        """Fetch records in batches to avoid memory issues"""
        results = []
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            placeholders = ','.join(['%s'] * len(batch_ids))
            
            query = f"""
            SELECT * FROM {table} 
            WHERE id IN ({placeholders})
            """
            
            batch_result = await self.db.fetch_df(query, batch_ids)
            results.append(batch_result)
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def create_indexes(self):
        """Create missing database indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_yarn_planning ON yarn_inventory(planning_balance)",
            "CREATE INDEX IF NOT EXISTS idx_yarn_status ON yarn_inventory(status)",
            "CREATE INDEX IF NOT EXISTS idx_order_status ON production_orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_order_date ON production_orders(scheduled_date)",
            "CREATE INDEX IF NOT EXISTS idx_bom_style ON bom(style_id, yarn_id)",
            "CREATE INDEX IF NOT EXISTS idx_machine_center ON machines(work_center_id)"
        ]
        
        for index_sql in indexes:
            try:
                self.db.execute(index_sql)
                self.logger.info(f"Created index: {index_sql[:50]}...")
            except Exception as e:
                self.logger.warning(f"Index may already exist: {e}")
```

### Day 18: Remove Blocking Operations
```python
# src/optimization/async_processor.py
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Callable, Any

class AsyncProcessor:
    """Convert blocking operations to async"""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
    
    async def replace_blocking_sleep(self, seconds: int):
        """
        BEFORE:
        time.sleep(60)  # Blocks entire thread
        
        AFTER:
        """
        await asyncio.sleep(seconds)  # Non-blocking
    
    async def process_heavy_calculation(self, func: Callable, *args, **kwargs):
        """Run CPU-intensive tasks in process pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.process_pool, 
            func, 
            *args, 
            **kwargs
        )
    
    async def batch_process_async(
        self, 
        items: List[Any], 
        processor: Callable,
        max_concurrent: int = 10
    ) -> List[Any]:
        """Process items concurrently with semaphore"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await processor(item)
        
        tasks = [process_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)

# Background task scheduler
class BackgroundScheduler:
    """Replace blocking scheduled tasks"""
    
    def __init__(self):
        self.tasks = {}
        self.running = False
    
    def add_periodic_task(
        self, 
        name: str, 
        func: Callable, 
        interval: int
    ):
        """Add task to run periodically"""
        self.tasks[name] = {
            'func': func,
            'interval': interval,
            'last_run': None
        }
    
    async def start(self):
        """Start background scheduler"""
        self.running = True
        
        while self.running:
            current_time = time.time()
            
            for name, task in self.tasks.items():
                if task['last_run'] is None or \
                   current_time - task['last_run'] >= task['interval']:
                    # Run task in background
                    asyncio.create_task(self._run_task(name, task))
                    task['last_run'] = current_time
            
            await asyncio.sleep(1)  # Check every second
    
    async def _run_task(self, name: str, task: Dict):
        """Run individual task with error handling"""
        try:
            await task['func']()
            self.logger.info(f"Task {name} completed successfully")
        except Exception as e:
            self.logger.error(f"Task {name} failed: {e}")
```

### Day 19-20: Memory Optimization
```python
# src/optimization/memory_optimizer.py
class MemoryOptimizer:
    """Reduce DataFrame memory usage by 50-90%"""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame, deep: bool = True) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        start_mem = df.memory_usage(deep=deep).sum() / 1024**2
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = MemoryOptimizer._downcast_numeric(df[col], 'integer')
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = MemoryOptimizer._downcast_numeric(df[col], 'float')
        
        # Convert strings to categories
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            
            if num_unique / num_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
        
        # Optimize datetime columns
        for col in df.select_dtypes(include=['datetime']).columns:
            # Use datetime64[ns] instead of object
            df[col] = pd.to_datetime(df[col])
        
        end_mem = df.memory_usage(deep=deep).sum() / 1024**2
        reduction_pct = 100 * (1 - end_mem / start_mem)
        
        print(f"Memory usage reduced by {reduction_pct:.1f}%")
        print(f"({start_mem:.1f} MB → {end_mem:.1f} MB)")
        
        return df
    
    @staticmethod
    def _downcast_numeric(series: pd.Series, dtype: str) -> pd.Series:
        """Downcast numeric types to smallest possible"""
        if dtype == 'integer':
            return pd.to_numeric(series, downcast='integer')
        elif dtype == 'float':
            return pd.to_numeric(series, downcast='float')
        return series

# Connection pooling
class ConnectionPool:
    """Database connection pooling"""
    
    def __init__(self, connection_string: str, pool_size: int = 20):
        self.engine = create_engine(
            connection_string,
            pool_size=pool_size,
            max_overflow=40,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True  # Verify connections before use
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool"""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    @contextmanager
    def get_session(self):
        """Get ORM session from pool"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
```

## PHASE 4: API Consolidation (Days 21-25)

### Day 21-22: Consolidate Endpoints
```python
# src/api/v2/consolidated_routes.py
from flask import Blueprint, request, jsonify
from typing import Dict, Any

api_v2 = Blueprint('api_v2', __name__, url_prefix='/api/v2')

# Consolidated inventory endpoint (replaces 5 endpoints)
@api_v2.route('/inventory', methods=['GET'])
@inject
async def inventory_endpoint(
    inventory_service: InventoryService = Provide[Container.inventory_service]
):
    """
    Consolidated inventory endpoint
    Replaces:
    - /api/yarn-inventory
    - /api/yarn-data  
    - /api/inventory-intelligence-enhanced
    - /api/real-time-inventory-dashboard
    - /api/emergency-shortage-dashboard
    
    Query Parameters:
    - view: summary|detailed|yarn|shortage
    - analysis: none|shortage|forecast
    - realtime: true|false
    - format: json|csv|excel
    """
    params = InventoryQueryParams(
        view=request.args.get('view', 'summary'),
        analysis=request.args.get('analysis', 'none'),
        realtime=request.args.get('realtime', 'false') == 'true',
        format=request.args.get('format', 'json')
    )
    
    result = await inventory_service.get_inventory(params)
    
    if params.format == 'csv':
        return result.to_csv(), 200, {'Content-Type': 'text/csv'}
    elif params.format == 'excel':
        return result.to_excel(), 200, {'Content-Type': 'application/vnd.ms-excel'}
    else:
        return jsonify(result.to_dict())

# Consolidated production endpoint (replaces 6 endpoints)
@api_v2.route('/production', methods=['GET', 'POST'])
async def production_endpoint():
    """
    Replaces:
    - /api/production-planning
    - /api/production-status
    - /api/production-pipeline
    - /api/production-recommendations-ml
    - /api/machine-assignment-suggestions
    - /api/production-flow
    """
    if request.method == 'GET':
        view = request.args.get('view', 'status')
        
        if view == 'status':
            return await production_service.get_status()
        elif view == 'planning':
            return await production_service.get_planning()
        elif view == 'recommendations':
            return await production_service.get_ml_recommendations()
        elif view == 'machines':
            return await production_service.get_machine_assignments()
    
    else:  # POST
        command = CreateProductionOrderCommand(request.json)
        return await production_service.create_order(command)

# Backward compatibility middleware
class DeprecatedEndpointMiddleware:
    """Handle deprecated endpoints with redirects"""
    
    MAPPINGS = {
        '/api/yarn-inventory': '/api/v2/inventory?view=yarn',
        '/api/production-status': '/api/v2/production?view=status',
        '/api/ml-forecasting': '/api/v2/forecast',
        '/api/inventory-intelligence-enhanced': '/api/v2/inventory?analysis=intelligence',
        # ... 40+ more mappings
    }
    
    def __init__(self, app):
        self.app = app
    
    def __call__(self, environ, start_response):
        path = environ['PATH_INFO']
        
        if path in self.MAPPINGS:
            # Log deprecated usage
            logger.warning(f"Deprecated endpoint used: {path}")
            
            # Get new path
            new_path = self.MAPPINGS[path]
            
            # Preserve query parameters
            if query_string := environ.get('QUERY_STRING'):
                new_path += '&' + query_string
            
            # Send redirect
            start_response('301 Moved Permanently', [('Location', new_path)])
            return [b'']
        
        return self.app(environ, start_response)
```

### Day 23-25: Complete Feature Implementation
```python
# Fix pass statements (42 occurrences)
# Example: src/api/v2/base.py cache decorator
def cache_response(ttl: int = 300):
    """Decorator to cache endpoint responses"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{request.full_path}"
            
            # Try cache
            if cached := await cache.get(cache_key):
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator

# Implement fabric production API
@api_v2.route('/fabric-production', methods=['GET'])
async def fabric_production_endpoint():
    """Complete fabric production implementation"""
    analyzer = FabricProductionAnalyzer(data_loader)
    
    params = FabricQueryParams(
        start_date=request.args.get('start_date'),
        end_date=request.args.get('end_date'),
        style_filter=request.args.get('style'),
        include_forecast=request.args.get('forecast', 'false') == 'true'
    )
    
    production_data = await analyzer.analyze_production(params)
    demand_data = await analyzer.analyze_demand(params)
    capacity_data = await analyzer.get_capacity_utilization()
    
    return jsonify({
        'status': 'success',
        'production': production_data,
        'demand': demand_data,
        'capacity': capacity_data,
        'generated_at': datetime.utcnow().isoformat()
    })
```

## PHASE 5: Testing & Quality (Days 26-30)

### Day 26-27: Unit Testing
```python
# tests/unit/services/test_inventory_service.py
import pytest
from unittest.mock import Mock, AsyncMock
from src.services.inventory import InventoryService

class TestInventoryService:
    @pytest.fixture
    def mock_repository(self):
        repo = Mock()
        repo.get_by_id = AsyncMock()
        repo.get_all = AsyncMock()
        repo.get_shortages = AsyncMock()
        return repo
    
    @pytest.fixture
    def mock_cache(self):
        cache = Mock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        return cache
    
    @pytest.fixture
    def service(self, mock_repository, mock_cache):
        return InventoryService(mock_repository, mock_cache)
    
    @pytest.mark.asyncio
    async def test_calculate_planning_balance(self, service, mock_repository):
        """Test planning balance calculation"""
        # Arrange
        mock_yarn = Mock(
            yarn_id="Y001",
            theoretical_balance=100.0,
            allocated=-20.0,  # Already negative
            on_order=50.0
        )
        mock_yarn.planning_balance = 130.0
        mock_repository.get_by_id.return_value = mock_yarn
        
        # Act
        result = await service.calculate_planning_balance("Y001")
        
        # Assert
        assert result.total == 130.0
        assert result.theoretical == 100.0
        assert result.allocated == -20.0
        assert result.on_order == 50.0
        mock_repository.get_by_id.assert_called_once_with("Y001")
    
    @pytest.mark.asyncio
    async def test_detect_shortages(self, service, mock_repository):
        """Test shortage detection"""
        # Arrange
        mock_yarns = [
            Mock(yarn_id="Y001", planning_balance=-10, min_stock=20),  # Shortage
            Mock(yarn_id="Y002", planning_balance=30, min_stock=20),   # OK
            Mock(yarn_id="Y003", planning_balance=5, min_stock=10),    # Shortage
        ]
        mock_repository.get_all.return_value = mock_yarns
        
        # Act
        shortages = await service.detect_shortages()
        
        # Assert
        assert len(shortages) == 2
        assert shortages[0].yarn_id == "Y001"
        assert shortages[1].yarn_id == "Y003"
    
    @pytest.mark.asyncio
    async def test_caching(self, service, mock_repository, mock_cache):
        """Test that caching works correctly"""
        # Arrange
        cached_value = {'yarn_id': 'Y001', 'planning_balance': 130}
        mock_cache.get.return_value = cached_value
        
        # Act
        result = await service.calculate_planning_balance("Y001")
        
        # Assert
        mock_cache.get.assert_called_once()
        mock_repository.get_by_id.assert_not_called()  # Should not hit DB
```

### Day 28: Integration Testing
```python
# tests/integration/test_production_workflow.py
import pytest
from src.container import Container

class TestProductionWorkflow:
    @pytest.fixture
    async def container(self):
        container = Container()
        container.config.from_dict({
            'database': {'url': 'postgresql://test:test@localhost/test'},
            'redis': {'host': 'localhost', 'port': 6379}
        })
        await container.init_resources()
        yield container
        await container.shutdown_resources()
    
    @pytest.mark.asyncio
    async def test_complete_production_planning(self, container):
        """Test end-to-end production planning workflow"""
        # Get orchestrator
        orchestrator = container.production_orchestrator()
        
        # Create test order
        order_request = ProductionOrderRequest(
            order_id="TEST001",
            style_id="S001",
            quantity=100,
            due_date=datetime.now() + timedelta(days=14)
        )
        
        # Execute workflow
        result = await orchestrator.complete_production_workflow(order_request)
        
        # Assertions
        assert result.plan is not None
        assert result.machine_assignment is not None
        assert len(result.material_requirements) > 0
        
        # Check for shortages
        if result.shortages:
            assert len(result.shortages) > 0
            assert result.plan.status == 'partial'
        else:
            assert result.plan.status == 'complete'
```

### Day 29-30: Performance Testing
```python
# tests/performance/test_api_performance.py
import asyncio
import aiohttp
import time
from statistics import mean, stdev

class TestAPIPerformance:
    BASE_URL = "http://localhost:5006"
    
    async def test_response_times(self):
        """Test that all endpoints respond within 200ms"""
        endpoints = [
            '/api/v2/inventory',
            '/api/v2/production',
            '/api/v2/forecast',
            '/api/v2/yarn'
        ]
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                times = []
                
                # Make 100 requests
                for _ in range(100):
                    start = time.perf_counter()
                    async with session.get(f"{self.BASE_URL}{endpoint}") as response:
                        await response.text()
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)
                
                # Calculate statistics
                avg_time = mean(times)
                std_time = stdev(times)
                p95_time = sorted(times)[95]
                
                print(f"{endpoint}:")
                print(f"  Average: {avg_time:.2f}ms")
                print(f"  Std Dev: {std_time:.2f}ms")
                print(f"  P95: {p95_time:.2f}ms")
                
                # Assert P95 < 200ms
                assert p95_time < 200, f"{endpoint} P95 time {p95_time}ms exceeds 200ms"
    
    async def test_concurrent_load(self):
        """Test system under concurrent load"""
        async def make_request(session, endpoint):
            async with session.get(f"{self.BASE_URL}{endpoint}") as response:
                return response.status
        
        async with aiohttp.ClientSession() as session:
            # Create 200 concurrent requests
            tasks = []
            for _ in range(200):
                endpoint = '/api/v2/inventory'
                tasks.append(make_request(session, endpoint))
            
            start = time.perf_counter()
            results = await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - start
            
            # Check success rate
            success_count = sum(1 for r in results if r == 200)
            success_rate = success_count / len(results)
            
            print(f"Concurrent Load Test:")
            print(f"  Requests: {len(results)}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Total Time: {elapsed:.2f}s")
            print(f"  Requests/sec: {len(results)/elapsed:.2f}")
            
            assert success_rate >= 0.99, f"Success rate {success_rate} below 99%"
```

---

## Code Patterns & Examples

### Repository Pattern
```python
# src/infrastructure/repositories/base_repository.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional

T = TypeVar('T')

class BaseRepository(ABC, Generic[T]):
    """Base repository for data access"""
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[T]:
        pass
    
    @abstractmethod
    async def get_all(self, limit: int = 1000) -> List[T]:
        pass
    
    @abstractmethod
    async def save(self, entity: T) -> T:
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        pass

# Implementation
class YarnRepository(BaseRepository[Yarn]):
    def __init__(self, db, cache):
        self.db = db
        self.cache = cache
    
    async def get_by_id(self, yarn_id: str) -> Optional[Yarn]:
        # Check cache first
        cache_key = f"yarn:{yarn_id}"
        if cached := await self.cache.get(cache_key):
            return Yarn.from_dict(cached)
        
        # Query database
        query = "SELECT * FROM yarn_inventory WHERE yarn_id = %s"
        result = await self.db.fetch_one(query, [yarn_id])
        
        if result:
            yarn = Yarn.from_db_row(result)
            await self.cache.set(cache_key, yarn.to_dict())
            return yarn
        
        return None
```

### CQRS Pattern
```python
# Commands (Write)
class UpdatePlanningBalanceCommand:
    def __init__(self, yarn_id: str, new_balance: float):
        self.yarn_id = yarn_id
        self.new_balance = new_balance

class UpdatePlanningBalanceHandler:
    async def handle(self, command: UpdatePlanningBalanceCommand):
        yarn = await self.repository.get_by_id(command.yarn_id)
        yarn.update_planning_balance(command.new_balance)
        await self.repository.save(yarn)
        await self.event_bus.publish(
            PlanningBalanceUpdatedEvent(yarn.yarn_id, command.new_balance)
        )

# Queries (Read)
class GetInventoryStatusQuery:
    def __init__(self, include_forecast: bool = False):
        self.include_forecast = include_forecast

class GetInventoryStatusHandler:
    async def handle(self, query: GetInventoryStatusQuery):
        inventory = await self.repository.get_current_status()
        if query.include_forecast:
            forecast = await self.forecasting_service.get_forecast()
            inventory.add_forecast(forecast)
        return inventory
```

### Error Handling Pattern
```python
# Custom exceptions
class DomainException(Exception):
    """Base domain exception"""
    pass

class YarnNotFoundError(DomainException):
    """Yarn not found in system"""
    pass

class InsufficientInventoryError(DomainException):
    """Not enough inventory available"""
    pass

# Global error handler
@app.errorhandler(DomainException)
def handle_domain_error(e):
    logger.warning(f"Domain error: {e}")
    return jsonify({'error': str(e)}), 400

@app.errorhandler(Exception)
def handle_unexpected_error(e):
    logger.exception("Unexpected error occurred")
    return jsonify({'error': 'Internal server error'}), 500
```

---

## Rollback & Contingency Plans

### Feature Flag System
```python
# src/utils/feature_flags.py
class FeatureFlags:
    FLAGS = {
        'use_new_inventory_service': False,
        'use_unified_data_loader': False,
        'use_consolidated_api': False,
        'enable_async_processing': False,
        'enable_ml_forecasting': True
    }
    
    @classmethod
    def is_enabled(cls, flag: str, user_id: str = None) -> bool:
        """Check if feature is enabled"""
        if flag not in cls.FLAGS:
            return False
        
        # Check for gradual rollout
        if user_id:
            # Use hash for consistent assignment
            rollout_percentage = cls.get_rollout_percentage(flag)
            user_hash = hash(user_id) % 100
            return user_hash < rollout_percentage
        
        return cls.FLAGS[flag]
    
    @classmethod
    def enable(cls, flag: str, percentage: int = 100):
        """Enable feature for percentage of users"""
        cls.FLAGS[flag] = True
        cls.set_rollout_percentage(flag, percentage)
```

### Rollback Procedures
```python
# scripts/rollback.py
#!/usr/bin/env python3
import sys
import subprocess
from datetime import datetime

class RollbackManager:
    def __init__(self):
        self.backup_path = "/backups"
        self.log_file = "/var/log/rollback.log"
    
    def create_backup(self):
        """Create backup before deployment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"pre_deployment_{timestamp}"
        
        # Backup database
        subprocess.run([
            "pg_dump", 
            "-h", "localhost",
            "-U", "beverly_knits",
            "-d", "erp_db",
            "-f", f"{self.backup_path}/{backup_name}.sql"
        ])
        
        # Backup code
        subprocess.run([
            "tar", "-czf",
            f"{self.backup_path}/{backup_name}.tar.gz",
            "/app/src"
        ])
        
        return backup_name
    
    def rollback_to(self, backup_name: str):
        """Rollback to specific backup"""
        # Disable all feature flags
        FeatureFlags.disable_all()
        
        # Restore database
        subprocess.run([
            "psql",
            "-h", "localhost",
            "-U", "beverly_knits",
            "-d", "erp_db",
            "-f", f"{self.backup_path}/{backup_name}.sql"
        ])
        
        # Restore code
        subprocess.run([
            "tar", "-xzf",
            f"{self.backup_path}/{backup_name}.tar.gz",
            "-C", "/"
        ])
        
        # Restart services
        subprocess.run(["systemctl", "restart", "beverly-erp"])
        
        self.log(f"Rolled back to {backup_name}")
```

### Monitoring & Alerts
```python
# src/monitoring/health_checks.py
class HealthMonitor:
    def __init__(self):
        self.checks = []
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5%
            'response_time_p95': 500,  # 500ms
            'memory_usage': 0.8,  # 80%
            'cpu_usage': 0.7  # 70%
        }
    
    async def run_health_checks(self) -> HealthStatus:
        """Run all health checks"""
        results = {}
        
        # Check error rate
        error_rate = await self.get_error_rate()
        results['error_rate'] = {
            'value': error_rate,
            'healthy': error_rate < self.alert_thresholds['error_rate']
        }
        
        # Check response times
        p95_time = await self.get_response_time_p95()
        results['response_time'] = {
            'value': p95_time,
            'healthy': p95_time < self.alert_thresholds['response_time_p95']
        }
        
        # Check resources
        memory = await self.get_memory_usage()
        results['memory'] = {
            'value': memory,
            'healthy': memory < self.alert_thresholds['memory_usage']
        }
        
        # Determine overall health
        all_healthy = all(r['healthy'] for r in results.values())
        
        if not all_healthy:
            await self.send_alert(results)
        
        return HealthStatus(
            healthy=all_healthy,
            checks=results,
            timestamp=datetime.utcnow()
        )
```

---

## Success Metrics & Monitoring

### Key Performance Indicators
```python
# src/monitoring/kpi_tracker.py
class KPITracker:
    METRICS = {
        'api_response_time_p95': Gauge('api_response_time_p95', 'API response time 95th percentile'),
        'error_rate': Gauge('error_rate', 'Error rate percentage'),
        'cache_hit_rate': Gauge('cache_hit_rate', 'Cache hit rate percentage'),
        'active_users': Gauge('active_users', 'Number of active users'),
        'yarn_shortage_count': Gauge('yarn_shortage_count', 'Number of yarn shortages'),
        'production_efficiency': Gauge('production_efficiency', 'Production efficiency percentage')
    }
    
    async def update_metrics(self):
        """Update all KPI metrics"""
        # API metrics
        self.METRICS['api_response_time_p95'].set(
            await self.calculate_response_time_p95()
        )
        
        # Error metrics
        self.METRICS['error_rate'].set(
            await self.calculate_error_rate()
        )
        
        # Business metrics
        shortages = await self.inventory_service.get_shortages()
        self.METRICS['yarn_shortage_count'].set(len(shortages))
```

### Dashboard Metrics
| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| API Response Time (P95) | <200ms | >500ms |
| Error Rate | <0.1% | >1% |
| Cache Hit Rate | >90% | <70% |
| Test Coverage | 90% | <80% |
| Memory Usage | <1GB | >1.5GB |
| CPU Usage | <60% | >80% |

---

## Risk Management

### Technical Risks & Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|-------------------|
| Service extraction breaks functionality | Medium | High | • Comprehensive testing<br>• Feature flags<br>• Parallel running |
| Performance regression | Low | High | • Continuous benchmarking<br>• Load testing<br>• Rollback plan |
| Data inconsistency | Low | Critical | • Data validation<br>• Transaction management<br>• Audit logs |
| API breaking changes | Medium | Medium | • Versioning<br>• Deprecation warnings<br>• Backward compatibility |
| Memory leaks | Low | High | • Memory profiling<br>• Resource cleanup<br>• Monitoring |

### Contingency Procedures

1. **If extraction fails**: Revert to monolith, analyze failure, retry with smaller scope
2. **If performance degrades**: Enable caching, optimize queries, scale horizontally
3. **If data corruption**: Restore from backup, replay transaction log
4. **If API breaks**: Roll back API changes, maintain old endpoints temporarily

---

## Post-Implementation Support

### Week 1 Post-Launch
- Daily health checks
- Monitor all metrics
- Address critical bugs
- Gather user feedback

### Week 2-4
- Performance tuning
- Minor bug fixes
- Documentation updates
- Knowledge transfer sessions

### Ongoing Maintenance
- Weekly performance reviews
- Monthly security audits
- Quarterly architecture reviews
- Continuous improvement sprints

---

## Appendices

### A. File Mapping
```
Old Structure → New Structure
src/core/beverly_comprehensive_erp.py → 
  ├── src/services/inventory/
  ├── src/services/forecasting/
  ├── src/services/production/
  └── src/api/v2/routes.py
```

### B. Database Migrations
```sql
-- Add missing indexes
CREATE INDEX idx_yarn_planning ON yarn_inventory(planning_balance);
CREATE INDEX idx_order_status ON production_orders(status);
CREATE INDEX idx_bom_style ON bom(style_id, yarn_id);

-- Add audit columns
ALTER TABLE yarn_inventory ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE production_orders ADD COLUMN created_by VARCHAR(100);
```

### C. Configuration Files
```yaml
# config/settings.yaml
database:
  url: postgresql://user:pass@localhost/beverly_knits
  pool_size: 20
  
redis:
  host: localhost
  port: 6379
  
api:
  rate_limit: 100
  timeout: 30
  
ml:
  models: [arima, prophet, lstm, xgboost]
  retrain_interval: 604800  # Weekly
```

---

**Document Status**: Complete  
**Version**: 2.0 (Comprehensive)  
**Last Updated**: January 2025  
**Ready for Implementation**: ✅

This comprehensive plan provides all details needed for successful implementation, including specific code patterns, rollback procedures, and contingency plans.