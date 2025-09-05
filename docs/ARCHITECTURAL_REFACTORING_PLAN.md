# Beverly Knits ERP v2 - Architectural Refactoring & Optimization Plan

**Document Version:** 1.0  
**Date:** September 5, 2025  
**Focus:** Architecture, Performance, Code Quality  
**Security:** Deferred (separate implementation track)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Architectural Issues](#core-architectural-issues)
3. [Monolith Decomposition Strategy](#monolith-decomposition-strategy)
4. [Service Architecture Design](#service-architecture-design)
5. [Data Layer Consolidation](#data-layer-consolidation)
6. [API Consolidation Plan](#api-consolidation-plan)
7. [Code Quality Improvements](#code-quality-improvements)
8. [Performance Optimization](#performance-optimization)
9. [Implementation Roadmap](#implementation-roadmap)
10. [Testing Strategy](#testing-strategy)
11. [Success Metrics](#success-metrics)

---

## Executive Summary

### Current State
- **18,000-line monolithic file** (`beverly_comprehensive_erp.py`)
- **127 API endpoints** in single file
- **60+ bare except clauses** hiding errors
- **42 pass statements** indicating incomplete functionality
- **Multiple competing data loaders** causing confusion
- **No clear separation of concerns**
- **Performance bottlenecks** from synchronous operations

### Target State
- **Microservices architecture** with clear boundaries
- **Clean architecture patterns** (Domain, Application, Infrastructure layers)
- **Single unified data loader** with proper caching
- **50 consolidated API endpoints** (from 127)
- **90% test coverage** with automated testing
- **50% performance improvement** in response times

### Timeline
**6 weeks** for complete architectural refactoring (excluding security implementation)

---

## Core Architectural Issues

### 1. The Monolith Problem

**File:** `src/core/beverly_comprehensive_erp.py` (18,000 lines)

**Current Structure:**
```python
# Everything mixed together
class ManufacturingSupplyChainAI:
    def __init__(self):
        # 12+ major components initialized
        self.inventory_analyzer = InventoryAnalyzer()
        self.sales_engine = SalesForecastingEngine()
        self.capacity_planner = CapacityPlanningEngine()
        self.yarn_calculator = YarnRequirementCalculator()
        self.inventory_tracker = MultiStageInventoryTracker()
        self.dashboard_manager = ProductionDashboardManager()
        self.scheduler = ProductionScheduler()
        self.mrp = TimePhasedMRP()
        # ... plus 127 API endpoints below
```

**Issues:**
- Impossible to test individual components
- Changes affect entire system
- No clear module boundaries
- Memory inefficient (loads everything)
- Deployment requires entire system restart

### 2. Data Layer Chaos

**Current State:**
- 4+ data loader implementations competing
- Hardcoded paths throughout codebase
- Column name inconsistencies
- No unified caching strategy

**Files:**
```
src/data_loaders/
├── unified_data_loader.py (1000+ lines)
├── parallel_data_loader.py (competing implementation)
├── optimized_data_loader.py (another implementation)
├── efab_api_loader.py (API integration, disabled)
└── po_delivery_loader.py (specialized loader)
```

### 3. Code Quality Issues

**Statistics:**
- **60+ bare except clauses** - Silent failures
- **42 pass statements** - Stub implementations
- **50+ placeholder returns** - Functions returning None
- **17 blocking operations** - `time.sleep()` calls
- **7 TODO comments** - Critical unfinished features

---

## Monolith Decomposition Strategy

### Phase 1: Extract Core Services

#### 1.1 Inventory Service

**Extract from monolith:**
```python
# NEW: src/services/inventory/inventory_service.py
class InventoryService:
    """
    Handles all inventory-related operations
    - Planning balance calculations
    - Shortage detection
    - Stock level management
    """
    
    def __init__(self, repository: InventoryRepository):
        self.repository = repository
        self.cache = CacheManager("inventory", ttl=900)
    
    def calculate_planning_balance(self, yarn_id: str) -> PlanningBalance:
        """
        Planning Balance = Theoretical Balance + Allocated + On Order
        Note: Allocated is already negative in source data
        """
        yarn = self.repository.get_yarn(yarn_id)
        return PlanningBalance(
            theoretical=yarn.theoretical_balance,
            allocated=yarn.allocated,  # Already negative
            on_order=yarn.on_order,
            total=yarn.theoretical_balance + yarn.allocated + yarn.on_order
        )
```

#### 1.2 Forecasting Service

**Extract from monolith:**
```python
# NEW: src/services/forecasting/forecasting_service.py
class ForecastingService:
    """
    ML-powered demand forecasting
    - ARIMA, Prophet, LSTM, XGBoost models
    - Ensemble predictions
    - Accuracy monitoring
    """
    
    def __init__(self, ml_config: MLConfig):
        self.models = self._initialize_models(ml_config)
        self.accuracy_monitor = AccuracyMonitor()
    
    async def predict_demand(
        self, 
        style_id: str, 
        horizon_days: int = 30
    ) -> DemandForecast:
        # Ensemble prediction logic
        predictions = await self._gather_predictions(style_id, horizon_days)
        return self._ensemble_predict(predictions)
```

#### 1.3 Production Planning Service

**Extract from monolith:**
```python
# NEW: src/services/production/production_service.py
class ProductionPlanningService:
    """
    Production planning and scheduling
    - 6-phase planning engine
    - Machine assignment
    - Capacity planning
    """
    
    def __init__(self, planning_engine: SixPhasePlanningEngine):
        self.engine = planning_engine
        self.machine_mapper = MachineMapper()
    
    def generate_production_plan(
        self, 
        orders: List[ProductionOrder]
    ) -> ProductionPlan:
        # Six-phase planning implementation
        return self.engine.plan(orders)
```

### Phase 2: Create Domain Layer

#### Domain Entities

```python
# src/domain/entities/yarn.py
@dataclass
class Yarn:
    """Domain entity for yarn"""
    yarn_id: str
    description: str
    theoretical_balance: float
    allocated: float  # Stored as negative
    on_order: float
    min_stock_level: float
    lead_time_days: int
    
    @property
    def planning_balance(self) -> float:
        """Calculate planning balance"""
        return self.theoretical_balance + self.allocated + self.on_order
    
    def has_shortage(self) -> bool:
        """Check if yarn has shortage"""
        return self.planning_balance < self.min_stock_level

# src/domain/entities/production_order.py
@dataclass
class ProductionOrder:
    """Domain entity for production order"""
    order_id: str
    style_id: str
    quantity: float
    machine_id: Optional[int]
    work_center: str
    status: OrderStatus
    scheduled_date: datetime
    
    def can_assign_to_machine(self, machine: Machine) -> bool:
        """Business rule: Check if order can be assigned to machine"""
        return (
            self.machine_id is None and
            machine.work_center == self.work_center and
            machine.is_available()
        )
```

#### Domain Services

```python
# src/domain/services/bom_explosion_service.py
class BOMExplosionService:
    """
    Domain service for BOM explosion logic
    Multi-level BOM calculation
    """
    
    def explode_bom(
        self, 
        style_id: str, 
        quantity: float
    ) -> List[MaterialRequirement]:
        """
        Calculate material requirements for style
        Handles multi-level BOMs
        """
        bom_items = self.bom_repository.get_bom_for_style(style_id)
        requirements = []
        
        for item in bom_items:
            requirement = MaterialRequirement(
                yarn_id=item.yarn_id,
                quantity_required=item.quantity_per_unit * quantity,
                level=item.bom_level
            )
            requirements.append(requirement)
            
            # Recursive for multi-level
            if item.has_sub_components:
                sub_requirements = self.explode_bom(
                    item.component_id, 
                    requirement.quantity_required
                )
                requirements.extend(sub_requirements)
        
        return requirements
```

---

## Service Architecture Design

### Clean Architecture Layers

```
┌─────────────────────────────────────┐
│         Presentation Layer          │
│     (API Controllers, Web UI)       │
├─────────────────────────────────────┤
│        Application Layer            │
│   (Use Cases, Command/Query)        │
├─────────────────────────────────────┤
│         Domain Layer                │
│   (Entities, Domain Services)       │
├─────────────────────────────────────┤
│      Infrastructure Layer           │
│ (Database, Cache, External APIs)    │
└─────────────────────────────────────┘
```

### Dependency Flow

```python
# src/infrastructure/dependency_injection.py
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    """Dependency injection container"""
    
    # Configuration
    config = providers.Configuration()
    
    # Infrastructure
    database = providers.Singleton(
        Database,
        connection_string=config.database.url
    )
    
    cache = providers.Singleton(
        RedisCache,
        host=config.redis.host,
        port=config.redis.port
    )
    
    # Repositories
    yarn_repository = providers.Factory(
        YarnRepository,
        db=database,
        cache=cache
    )
    
    bom_repository = providers.Factory(
        BOMRepository,
        db=database
    )
    
    # Domain Services
    bom_explosion_service = providers.Factory(
        BOMExplosionService,
        bom_repository=bom_repository
    )
    
    # Application Services
    inventory_service = providers.Factory(
        InventoryService,
        yarn_repository=yarn_repository,
        bom_service=bom_explosion_service
    )
    
    forecasting_service = providers.Factory(
        ForecastingService,
        config=config.ml
    )
```

### Service Communication

```python
# src/application/orchestrator.py
class ProductionOrchestrator:
    """
    Orchestrates complex workflows across services
    """
    
    def __init__(
        self,
        inventory_service: InventoryService,
        forecasting_service: ForecastingService,
        production_service: ProductionPlanningService
    ):
        self.inventory = inventory_service
        self.forecasting = forecasting_service
        self.production = production_service
    
    async def plan_production_with_forecast(
        self, 
        orders: List[ProductionOrder]
    ) -> ComprehensivePlan:
        """
        Complex workflow orchestration
        """
        # 1. Get current inventory
        inventory_status = await self.inventory.get_current_status()
        
        # 2. Get demand forecast
        forecasts = await asyncio.gather(*[
            self.forecasting.predict_demand(order.style_id)
            for order in orders
        ])
        
        # 3. Calculate material requirements
        requirements = self.inventory.calculate_requirements(
            orders, 
            forecasts
        )
        
        # 4. Generate production plan
        plan = await self.production.generate_production_plan(
            orders,
            requirements,
            inventory_status
        )
        
        return ComprehensivePlan(
            production_plan=plan,
            material_requirements=requirements,
            forecasts=forecasts,
            inventory_status=inventory_status
        )
```

---

## Data Layer Consolidation

### Single Data Loader Implementation

```python
# src/infrastructure/data/data_loader.py
class UnifiedDataLoader:
    """
    Single source of truth for all data loading
    Replaces 4+ competing implementations
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.cache = CacheManager()
        self.validator = DataValidator()
        self.column_standardizer = ColumnStandardizer()
        
        # Data source priority
        self.sources = [
            FileDataSource(config.file_path),
            APIDataSource(config.api_url, enabled=config.use_api),
            DatabaseDataSource(config.db_connection)
        ]
    
    @cached(ttl=900)
    def load_yarn_inventory(self) -> pd.DataFrame:
        """Load yarn inventory with fallback strategy"""
        for source in self.sources:
            try:
                data = source.load("yarn_inventory")
                if data is not None:
                    # Standardize column names
                    data = self.column_standardizer.standardize(data)
                    # Validate data
                    data = self.validator.validate_yarn_data(data)
                    return data
            except DataSourceException:
                continue
        
        raise DataLoadException("All data sources failed")
    
    def _handle_column_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle column name variations"""
        column_mappings = {
            'Planning Balance': 'planning_balance',
            'Planning_Balance': 'planning_balance',
            'Desc#': 'description',
            'desc_num': 'description',
            'YarnID': 'yarn_id',
            'fStyle#': 'style_id',
            'Style#': 'style_id'
        }
        
        for old_name, new_name in column_mappings.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        return df
```

### Repository Pattern Implementation

```python
# src/infrastructure/repositories/yarn_repository.py
class YarnRepository:
    """
    Data access for yarn entities
    Single point of data access
    """
    
    def __init__(self, data_loader: UnifiedDataLoader):
        self.loader = data_loader
        self.cache = CacheManager("yarn_repo", ttl=300)
    
    @cached
    def get_yarn(self, yarn_id: str) -> Yarn:
        """Get single yarn by ID"""
        df = self.loader.load_yarn_inventory()
        yarn_data = df[df['yarn_id'] == yarn_id].iloc[0]
        
        return Yarn(
            yarn_id=yarn_data['yarn_id'],
            description=yarn_data['description'],
            theoretical_balance=float(yarn_data['theoretical_balance']),
            allocated=float(yarn_data['allocated']),  # Already negative
            on_order=float(yarn_data['on_order']),
            min_stock_level=float(yarn_data.get('min_stock', 0)),
            lead_time_days=int(yarn_data.get('lead_time', 14))
        )
    
    def get_yarns_with_shortage(self) -> List[Yarn]:
        """Get all yarns with shortage"""
        df = self.loader.load_yarn_inventory()
        
        # Calculate planning balance
        df['planning_balance'] = (
            df['theoretical_balance'] + 
            df['allocated'] +  # Already negative
            df['on_order']
        )
        
        # Filter shortages
        shortage_df = df[df['planning_balance'] < df['min_stock']]
        
        return [
            self._df_row_to_yarn(row) 
            for _, row in shortage_df.iterrows()
        ]
```

---

## API Consolidation Plan

### Current State: 127 Endpoints

**Problem:** Too many endpoints with overlapping functionality

### Target State: 50 Consolidated Endpoints

#### Consolidation Strategy

```python
# src/api/v2/consolidated_routes.py
from flask import Blueprint, request, jsonify
from src.application.queries import *
from src.application.commands import *

api_v2 = Blueprint('api_v2', __name__, url_prefix='/api/v2')

@api_v2.route('/inventory', methods=['GET'])
def inventory_endpoint():
    """
    Consolidated inventory endpoint
    Replaces:
    - /api/yarn-inventory
    - /api/yarn-data
    - /api/inventory-intelligence-enhanced
    - /api/real-time-inventory-dashboard
    - /api/emergency-shortage-dashboard
    """
    view = request.args.get('view', 'summary')
    analysis = request.args.get('analysis', 'none')
    realtime = request.args.get('realtime', 'false') == 'true'
    
    query = InventoryQuery(
        view=view,
        analysis=analysis,
        realtime=realtime
    )
    
    result = query_handler.handle(query)
    return jsonify(result)

@api_v2.route('/production', methods=['GET', 'POST'])
def production_endpoint():
    """
    Consolidated production endpoint
    Replaces:
    - /api/production-planning
    - /api/production-status
    - /api/production-pipeline
    - /api/production-recommendations-ml
    - /api/machine-assignment-suggestions
    """
    if request.method == 'GET':
        view = request.args.get('view', 'status')
        return production_query_handler.handle(
            ProductionQuery(view=view)
        )
    else:
        command = CreateProductionPlanCommand(
            request.json
        )
        return command_handler.handle(command)

@api_v2.route('/forecast', methods=['GET'])
def forecast_endpoint():
    """
    Consolidated forecasting endpoint
    Replaces:
    - /api/ml-forecast-detailed
    - /api/ml-forecasting
    - /api/sales-forecast-analysis
    - /api/forecast-demand
    """
    type = request.args.get('type', 'demand')
    horizon = int(request.args.get('horizon', 30))
    model = request.args.get('model', 'ensemble')
    
    query = ForecastQuery(
        forecast_type=type,
        horizon_days=horizon,
        model=model
    )
    
    return jsonify(forecast_handler.handle(query))
```

#### Migration Strategy with Redirects

```python
# src/api/middleware/deprecation_handler.py
class DeprecationMiddleware:
    """
    Handles deprecated endpoint redirects
    """
    
    ENDPOINT_MAPPINGS = {
        '/api/yarn-inventory': '/api/v2/inventory?view=yarn',
        '/api/production-status': '/api/v2/production?view=status',
        '/api/ml-forecasting': '/api/v2/forecast?type=demand',
        # ... 40+ more mappings
    }
    
    def __call__(self, environ, start_response):
        path = environ['PATH_INFO']
        
        if path in self.ENDPOINT_MAPPINGS:
            # Log deprecated usage
            logger.warning(f"Deprecated endpoint accessed: {path}")
            
            # Redirect to new endpoint
            new_path = self.ENDPOINT_MAPPINGS[path]
            
            # Preserve query parameters
            if environ.get('QUERY_STRING'):
                new_path += '&' + environ['QUERY_STRING']
            
            # Return redirect response
            start_response(
                '301 Moved Permanently',
                [('Location', new_path)]
            )
            return [b'Endpoint moved']
        
        return self.app(environ, start_response)
```

---

## Code Quality Improvements

### 1. Fix Bare Except Clauses

**Current Problem:** 60+ instances of silent failures

```python
# BEFORE: Silent failure
try:
    result = process_data()
except:
    pass  # Hides all errors!

# AFTER: Proper error handling
try:
    result = process_data()
except DataProcessingError as e:
    logger.error(f"Data processing failed: {e}")
    raise ServiceException("Failed to process data", original_error=e)
except Exception as e:
    logger.exception("Unexpected error in data processing")
    raise SystemException("System error occurred", original_error=e)
```

### 2. Replace Pass Statements

**Current Problem:** 42 stub implementations

```python
# BEFORE: Stub implementation
class YarnAnalyzer:
    def analyze(self):
        pass  # TODO: Implement

# AFTER: Proper implementation
class YarnAnalyzer:
    def analyze(self) -> AnalysisResult:
        """
        Analyze yarn inventory for patterns and issues
        """
        inventory = self.repository.get_all_yarns()
        
        shortages = self._detect_shortages(inventory)
        trends = self._analyze_trends(inventory)
        recommendations = self._generate_recommendations(
            shortages, 
            trends
        )
        
        return AnalysisResult(
            shortages=shortages,
            trends=trends,
            recommendations=recommendations
        )
```

### 3. Fix Placeholder Returns

**Current Problem:** 50+ functions returning None inappropriately

```python
# BEFORE: Placeholder return
def calculate_requirement():
    # TODO: Implement
    return None  # Causes downstream errors

# AFTER: Meaningful implementation
def calculate_requirement() -> MaterialRequirement:
    """
    Calculate material requirements
    Raises exception if cannot calculate
    """
    if not self.has_required_data():
        raise InsufficientDataError(
            "Cannot calculate requirements: missing data"
        )
    
    requirement = MaterialRequirement(
        quantity=self._calculate_quantity(),
        timing=self._calculate_timing(),
        priority=self._determine_priority()
    )
    
    return requirement
```

### 4. Remove Blocking Operations

**Current Problem:** 17 instances of `time.sleep()`

```python
# BEFORE: Blocking operation
def wait_for_data():
    time.sleep(60)  # Blocks entire thread!
    return fetch_data()

# AFTER: Non-blocking async
async def wait_for_data():
    await asyncio.sleep(60)  # Non-blocking
    return await fetch_data_async()

# OR: Use scheduling
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()
scheduler.add_job(
    fetch_data, 
    'interval', 
    seconds=60,
    id='data_fetch'
)
```

---

## Performance Optimization

### 1. Database Query Optimization

```python
# BEFORE: Inefficient N+1 queries
def get_yarn_details():
    yarns = db.query("SELECT * FROM yarns")
    for yarn in yarns:
        # N+1 problem - query for each yarn
        yarn['orders'] = db.query(
            f"SELECT * FROM orders WHERE yarn_id = {yarn['id']}"
        )
    return yarns

# AFTER: Optimized single query
def get_yarn_details():
    query = """
    SELECT 
        y.*,
        json_agg(o.*) as orders
    FROM yarns y
    LEFT JOIN orders o ON o.yarn_id = y.id
    GROUP BY y.id
    """
    return db.query(query)
```

### 2. Implement Caching Strategy

```python
# src/infrastructure/cache/cache_strategy.py
class CacheStrategy:
    """
    Intelligent caching with TTL and invalidation
    """
    
    CACHE_CONFIGS = {
        'yarn_inventory': {
            'ttl': 900,  # 15 minutes
            'invalidate_on': ['yarn_update', 'inventory_sync'],
            'compression': True
        },
        'production_plan': {
            'ttl': 300,  # 5 minutes
            'invalidate_on': ['order_change', 'machine_update'],
            'compression': False
        },
        'ml_forecast': {
            'ttl': 3600,  # 1 hour
            'invalidate_on': ['model_retrain'],
            'compression': True
        }
    }
    
    def cache_with_strategy(self, key: str, data_type: str):
        config = self.CACHE_CONFIGS.get(data_type)
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                cache_key = f"{data_type}:{key}"
                
                # Try cache first
                cached = await self.redis.get(cache_key)
                if cached:
                    return self._decompress(cached) if config['compression'] else cached
                
                # Compute if not cached
                result = await func(*args, **kwargs)
                
                # Store in cache
                to_cache = self._compress(result) if config['compression'] else result
                await self.redis.setex(
                    cache_key, 
                    config['ttl'], 
                    to_cache
                )
                
                return result
            return wrapper
        return decorator
```

### 3. Async Processing

```python
# src/application/async_processor.py
class AsyncProcessor:
    """
    Handle heavy operations asynchronously
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.loop = asyncio.get_event_loop()
    
    async def process_batch_async(
        self, 
        items: List[Any], 
        processor_func: Callable
    ) -> List[Any]:
        """
        Process items in parallel
        """
        tasks = []
        
        for item in items:
            task = self.loop.run_in_executor(
                self.executor,
                processor_func,
                item
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

# Usage example
async def calculate_all_planning_balances():
    processor = AsyncProcessor()
    yarn_ids = repository.get_all_yarn_ids()
    
    balances = await processor.process_batch_async(
        yarn_ids,
        calculate_single_balance
    )
    
    return balances
```

### 4. Connection Pooling

```python
# src/infrastructure/database/connection_pool.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

class DatabasePool:
    """
    Efficient database connection pooling
    """
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=20,  # Number of connections
            max_overflow=40,  # Maximum overflow connections
            pool_timeout=30,  # Timeout for getting connection
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo_pool=True  # Log pool checkouts/checkins
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def get_session(self):
        """Get database session from pool"""
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self):
        """Provide transactional scope"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
```

---

## Implementation Roadmap

### Week 1: Foundation & Preparation

#### Day 1-2: Setup & Analysis
- [ ] Create new project structure
- [ ] Set up dependency injection framework
- [ ] Analyze monolith dependencies
- [ ] Create extraction plan

#### Day 3-5: Core Extractions
- [ ] Extract InventoryAnalyzer class
- [ ] Extract SalesForecastingEngine class
- [ ] Extract CapacityPlanningEngine class
- [ ] Create service interfaces

### Week 2: Service Implementation

#### Day 6-8: Domain Layer
- [ ] Create domain entities
- [ ] Implement domain services
- [ ] Define business rules
- [ ] Add value objects

#### Day 9-10: Repository Pattern
- [ ] Implement YarnRepository
- [ ] Implement BOMRepository
- [ ] Implement OrderRepository
- [ ] Add caching layer

### Week 3: Data Layer Consolidation

#### Day 11-13: Unified Data Loader
- [ ] Consolidate 4+ data loaders into one
- [ ] Implement fallback strategy
- [ ] Add column standardization
- [ ] Create validation layer

#### Day 14-15: Caching Strategy
- [ ] Implement Redis caching
- [ ] Add cache invalidation
- [ ] Configure TTL strategies
- [ ] Add compression for large datasets

### Week 4: API Consolidation

#### Day 16-18: New API Structure
- [ ] Create v2 API blueprint
- [ ] Implement consolidated endpoints
- [ ] Add request validation
- [ ] Create OpenAPI documentation

#### Day 19-20: Migration Support
- [ ] Implement redirect middleware
- [ ] Add deprecation warnings
- [ ] Create compatibility layer
- [ ] Update client documentation

### Week 5: Code Quality & Performance

#### Day 21-22: Error Handling
- [ ] Replace 60+ bare except clauses
- [ ] Fix 42 pass statements
- [ ] Replace placeholder returns
- [ ] Add proper logging

#### Day 23-24: Performance Optimization
- [ ] Optimize database queries
- [ ] Remove blocking operations
- [ ] Implement connection pooling
- [ ] Add async processing

#### Day 25: Load Testing
- [ ] Create load test scenarios
- [ ] Run performance benchmarks
- [ ] Identify bottlenecks
- [ ] Apply optimizations

### Week 6: Testing & Deployment

#### Day 26-27: Test Coverage
- [ ] Unit tests (target 90%)
- [ ] Integration tests
- [ ] Performance tests
- [ ] End-to-end tests

#### Day 28-29: Documentation
- [ ] API documentation
- [ ] Architecture diagrams
- [ ] Deployment guide
- [ ] Migration guide

#### Day 30: Deployment Preparation
- [ ] Create deployment scripts
- [ ] Set up monitoring
- [ ] Prepare rollback plan
- [ ] Final testing

---

## Testing Strategy

### Unit Testing (Target: 90% Coverage)

```python
# tests/unit/services/test_inventory_service.py
import pytest
from unittest.mock import Mock, MagicMock
from src.services.inventory import InventoryService

class TestInventoryService:
    @pytest.fixture
    def service(self):
        mock_repo = Mock()
        mock_cache = Mock()
        return InventoryService(mock_repo, mock_cache)
    
    def test_calculate_planning_balance(self, service):
        """Test planning balance calculation"""
        # Arrange
        mock_yarn = Mock(
            theoretical_balance=100.0,
            allocated=-20.0,  # Already negative
            on_order=50.0
        )
        service.repository.get_yarn.return_value = mock_yarn
        
        # Act
        result = service.calculate_planning_balance("Y001")
        
        # Assert
        assert result.total == 130.0  # 100 + (-20) + 50
        service.repository.get_yarn.assert_called_once_with("Y001")
    
    def test_detect_shortages(self, service):
        """Test shortage detection"""
        # Arrange
        mock_yarns = [
            Mock(yarn_id="Y001", planning_balance=10, min_stock=20),  # Shortage
            Mock(yarn_id="Y002", planning_balance=30, min_stock=20),  # OK
        ]
        service.repository.get_all_yarns.return_value = mock_yarns
        
        # Act
        shortages = service.detect_shortages()
        
        # Assert
        assert len(shortages) == 1
        assert shortages[0].yarn_id == "Y001"
```

### Integration Testing

```python
# tests/integration/test_production_workflow.py
import pytest
from src.container import Container

class TestProductionWorkflow:
    @pytest.fixture
    def container(self):
        container = Container()
        container.config.from_dict({
            'database': {'url': 'sqlite:///:memory:'},
            'redis': {'host': 'localhost', 'port': 6379}
        })
        return container
    
    async def test_complete_production_planning(self, container):
        """Test end-to-end production planning"""
        # Get services
        inventory = container.inventory_service()
        forecasting = container.forecasting_service()
        production = container.production_service()
        orchestrator = container.production_orchestrator()
        
        # Create test data
        orders = [
            ProductionOrder(
                order_id="PO001",
                style_id="S001",
                quantity=100
            )
        ]
        
        # Execute workflow
        plan = await orchestrator.plan_production_with_forecast(orders)
        
        # Assertions
        assert plan.production_plan is not None
        assert len(plan.material_requirements) > 0
        assert plan.forecasts[0].confidence >= 0.7
```

### Performance Testing

```python
# tests/performance/test_api_performance.py
import locust
from locust import HttpUser, task, between

class ERPUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def get_inventory(self):
        """Test inventory endpoint performance"""
        with self.client.get(
            "/api/v2/inventory?view=summary",
            catch_response=True
        ) as response:
            if response.elapsed.total_seconds() > 0.2:
                response.failure(f"Too slow: {response.elapsed.total_seconds()}s")
    
    @task(2)
    def get_production_plan(self):
        """Test production planning endpoint"""
        self.client.get("/api/v2/production?view=status")
    
    @task(1)
    def get_forecast(self):
        """Test forecasting endpoint"""
        self.client.get("/api/v2/forecast?horizon=30")

# Run with: locust -f test_api_performance.py --host=http://localhost:5006
```

---

## Success Metrics

### Technical Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Monolith Size | 18,000 lines | <1,000 lines | Line count |
| API Endpoints | 127 | 50 | Endpoint count |
| Code Coverage | <40% | 90% | pytest-cov |
| Response Time (p95) | 200ms | 100ms | APM monitoring |
| Bare Except Clauses | 60+ | 0 | Static analysis |
| Pass Statements | 42 | 0 | Code analysis |
| Memory Usage | 2GB | 1GB | System monitoring |
| Concurrent Users | 50 | 200 | Load testing |

### Code Quality Metrics

| Metric | Current | Target | Tool |
|--------|---------|--------|------|
| Cyclomatic Complexity | >20 | <10 | radon |
| Maintainability Index | C | A | radon |
| Technical Debt | High | Low | SonarQube |
| Duplication | 15% | <3% | SonarQube |
| Code Smells | 200+ | <20 | SonarQube |

### Business Impact

| Metric | Current | Target | Impact |
|--------|---------|--------|--------|
| Deployment Frequency | Weekly | Daily | Faster features |
| Mean Time to Recovery | 2 hours | 15 min | Better reliability |
| Bug Rate | 15/week | 3/week | Quality improvement |
| Development Velocity | 20 pts | 35 pts | Productivity gain |

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Service extraction breaks functionality | Medium | High | Comprehensive testing, gradual extraction |
| Performance regression | Low | High | Continuous benchmarking, profiling |
| Data inconsistency | Low | Critical | Validation layers, data contracts |
| Integration failures | Medium | Medium | Contract testing, mocking |

### Mitigation Strategies

1. **Feature Flags**: Gradual rollout of new services
2. **Parallel Running**: Run old and new systems in parallel
3. **Comprehensive Testing**: 90% coverage before switching
4. **Monitoring**: Real-time performance and error tracking
5. **Rollback Plan**: Quick revert capability

---

## Conclusion

This architectural refactoring plan focuses on transforming the Beverly Knits ERP v2 from a monolithic application into a well-structured, maintainable system using clean architecture principles. 

### Key Deliverables

1. **Decomposed Services**: Extract 8+ services from monolith
2. **Clean Architecture**: Implement proper layer separation
3. **Unified Data Layer**: Single data loader replacing 4+ implementations
4. **Consolidated APIs**: Reduce from 127 to 50 endpoints
5. **90% Test Coverage**: Comprehensive testing suite
6. **50% Performance Improvement**: Through optimization and caching

### Timeline
**6 weeks** for complete implementation (excluding security work)

### Next Steps
1. Review and approve plan
2. Allocate development resources
3. Set up new project structure
4. Begin service extraction
5. Establish monitoring and metrics

---

**Document Status:** Complete  
**Version:** 1.0  
**Date:** September 5, 2025  
**Contact:** architecture-team@beverly-knits.com