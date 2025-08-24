# Service Extraction Plan for Beverly Knits ERP Monolith

## Executive Summary

The main `beverly_comprehensive_erp.py` file contains 13,500+ lines of code. This document outlines a systematic plan to extract this monolith into microservices while maintaining functionality and minimizing disruption.

## Current State Analysis

### Monolith Structure (beverly_comprehensive_erp.py)

The monolith contains:
- **4 Main Classes** (~3,000 lines)
  - `InventoryAnalyzer` (Lines 267-326)
  - `InventoryManagementPipeline` (Lines 327-494)
  - `SalesForecastingEngine` (Lines 495-1652)
  - `CapacityPlanningEngine` (Lines 1653-2820)
  
- **80+ API Endpoints** (~5,000 lines)
- **Helper Functions** (~2,000 lines)
- **Data Processing Logic** (~3,000 lines)

### Already Extracted Services

✅ **Completed Extractions:**
- `services/capacity_planning_service.py`
- `services/inventory_analyzer_service.py`
- `services/sales_forecasting_service.py`
- `services/yarn_requirement_service.py`
- `services/service_manager.py`

## Proposed Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway Service                      │
│                  (Authentication & Routing)                  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
│   Inventory    │  │   Production    │  │   Forecasting   │
│    Service     │  │     Service     │  │     Service     │
└────────────────┘  └─────────────────┘  └─────────────────┘
        │                     │                     │
┌───────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
│     Yarn       │  │    Planning     │  │       ML        │
│  Intelligence  │  │     Service     │  │     Service     │
└────────────────┘  └─────────────────┘  └─────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Data Service    │
                    │   (Shared Data)   │
                    └───────────────────┘
```

## Phase 1: Core Service Extraction (Week 1)

### 1.1 Inventory Service
**Target Lines: ~2,500**

```python
# src/services/inventory_service.py
class InventoryService:
    - analyze_inventory()
    - get_yarn_status()
    - calculate_shortages()
    - get_procurement_plan()
    - manage_allocations()
```

**Endpoints to Move:**
- `/api/inventory-analysis`
- `/api/inventory-intelligence`
- `/api/yarn-intelligence`
- `/api/real-time-inventory`
- `/api/emergency-yarns`

### 1.2 Production Service
**Target Lines: ~2,000**

```python
# src/services/production_service.py
class ProductionService:
    - get_production_schedule()
    - manage_knit_orders()
    - track_wip()
    - calculate_capacity()
    - optimize_production()
```

**Endpoints to Move:**
- `/api/production-planning`
- `/api/production-pipeline`
- `/api/production-suggestions`
- `/api/knit-orders/*`

### 1.3 Forecasting Service
**Target Lines: ~1,800**

```python
# src/services/forecasting_service.py
class ForecastingService:
    - generate_sales_forecast()
    - forecast_demand()
    - calculate_seasonality()
    - ensemble_predictions()
```

**Endpoints to Move:**
- `/api/sales-forecast`
- `/api/ml-forecast`
- `/api/forecast/*`
- `/api/retrain-ml`

## Phase 2: Supporting Service Extraction (Week 2)

### 2.1 Planning Service
**Target Lines: ~1,500**

```python
# src/services/planning_service.py
class PlanningService:
    - execute_planning_phases()
    - material_requirements_planning()
    - capacity_planning()
    - supply_chain_optimization()
```

**Endpoints to Move:**
- `/api/planning-phases`
- `/api/six-phase-planning`
- `/api/execute-planning`

### 2.2 Analytics Service
**Target Lines: ~1,200**

```python
# src/services/analytics_service.py
class AnalyticsService:
    - generate_reports()
    - calculate_kpis()
    - trend_analysis()
    - performance_metrics()
```

**Endpoints to Move:**
- `/api/supply-chain-analysis`
- `/api/capacity-analysis`
- `/api/metrics/*`

### 2.3 Data Management Service
**Target Lines: ~1,000**

```python
# src/services/data_management_service.py
class DataManagementService:
    - load_data()
    - validate_data()
    - sync_data()
    - export_data()
```

**Endpoints to Move:**
- `/api/reload-data`
- `/api/data-sync`
- `/api/export/*`

## Phase 3: API Gateway Implementation (Week 3)

### 3.1 Create API Gateway
```python
# src/api_gateway.py
class APIGateway:
    def __init__(self):
        self.services = {
            'inventory': InventoryService(),
            'production': ProductionService(),
            'forecasting': ForecastingService(),
            'planning': PlanningService(),
            'analytics': AnalyticsService(),
            'data': DataManagementService()
        }
    
    def route_request(self, endpoint, method, data):
        # Route to appropriate service
        pass
```

### 3.2 Implement Service Registry
```python
# src/services/service_registry.py
class ServiceRegistry:
    - register_service()
    - discover_service()
    - health_check()
    - load_balancing()
```

## Phase 4: Communication Layer (Week 4)

### 4.1 Inter-Service Communication
```python
# src/services/service_communication.py
class ServiceCommunication:
    - sync_call()  # Synchronous HTTP/gRPC
    - async_call()  # Async with message queue
    - event_publish()  # Event-driven
    - event_subscribe()
```

### 4.2 Message Queue Integration
- Implement RabbitMQ/Redis Pub-Sub for async communication
- Event-driven architecture for real-time updates

## Implementation Strategy

### Step-by-Step Process for Each Service

1. **Extract Business Logic**
   ```python
   # Before (in monolith)
   @app.route('/api/inventory-analysis')
   def inventory_analysis():
       # 200 lines of logic
       pass
   
   # After (in service)
   class InventoryService:
       def analyze_inventory(self, params):
           # Same logic, now reusable
           pass
   ```

2. **Create Service Interface**
   ```python
   class IInventoryService(ABC):
       @abstractmethod
       def analyze_inventory(self, params): pass
   ```

3. **Implement Service Adapter**
   ```python
   class InventoryServiceAdapter:
       def __init__(self, service: IInventoryService):
           self.service = service
       
       def handle_request(self, request):
           # Validate, transform, call service
           pass
   ```

4. **Update Monolith to Use Service**
   ```python
   # Gradual migration
   @app.route('/api/inventory-analysis')
   def inventory_analysis():
       return inventory_service.handle_request(request)
   ```

## Database Separation Strategy

### Current: Shared Database
```
Monolith → Single Database
```

### Target: Service-Specific Databases
```
Inventory Service → Inventory DB
Production Service → Production DB
Forecasting Service → Forecast DB
```

### Migration Steps:
1. Identify data boundaries
2. Create service-specific schemas
3. Implement data synchronization
4. Gradual data migration

## Testing Strategy

### 1. Unit Tests for Each Service
```python
# tests/services/test_inventory_service.py
def test_inventory_analysis():
    service = InventoryService()
    result = service.analyze_inventory(test_data)
    assert result['status'] == 'success'
```

### 2. Integration Tests
```python
# tests/integration/test_service_communication.py
def test_inventory_production_integration():
    inventory = InventoryService()
    production = ProductionService()
    # Test inter-service communication
```

### 3. Contract Tests
```python
# tests/contracts/test_api_contracts.py
def test_inventory_api_contract():
    # Ensure API compatibility
    pass
```

## Monitoring & Observability

### Service-Level Monitoring
```python
# src/monitoring/service_monitor.py
class ServiceMonitor:
    - track_latency()
    - monitor_errors()
    - measure_throughput()
    - health_checks()
```

### Distributed Tracing
- Implement OpenTelemetry for request tracing
- Correlation IDs for request tracking
- Service dependency mapping

## Risk Mitigation

### 1. Feature Flags
```python
FEATURE_FLAGS = {
    'use_inventory_service': False,  # Gradual rollout
    'use_production_service': False,
    'use_forecasting_service': False
}
```

### 2. Fallback Mechanisms
```python
def get_inventory_data():
    try:
        return inventory_service.get_data()
    except ServiceUnavailableError:
        return monolith_fallback.get_inventory_data()
```

### 3. Gradual Migration
- Start with read-only endpoints
- Move write operations gradually
- Keep monolith as fallback

## Success Metrics

### Technical Metrics
- **Code Reduction**: 13,500 → <1,000 lines per service
- **Response Time**: <200ms per service call
- **Deployment Time**: 30min → 5min per service
- **Test Coverage**: >80% per service

### Business Metrics
- **System Availability**: 99.9% uptime
- **Feature Velocity**: 2x faster delivery
- **Maintenance Cost**: 40% reduction
- **Scalability**: Independent service scaling

## Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Core Services | Inventory, Production, Forecasting services |
| 2 | Supporting Services | Planning, Analytics, Data services |
| 3 | API Gateway | Gateway implementation, service registry |
| 4 | Communication | Inter-service communication, message queue |
| 5 | Testing | Full test coverage, integration tests |
| 6 | Deployment | Docker containers, orchestration |
| 7 | Migration | Data migration, gradual rollout |
| 8 | Optimization | Performance tuning, monitoring |

## Next Steps

1. **Immediate Actions**
   - Set up service directory structure
   - Create service interfaces
   - Begin extracting inventory service

2. **Week 1 Goals**
   - Extract 3 core services
   - Implement basic service communication
   - Create integration tests

3. **Success Criteria**
   - All services running independently
   - No functionality loss
   - Improved performance metrics

## Conclusion

This extraction plan provides a systematic approach to breaking down the Beverly Knits ERP monolith into manageable microservices. The gradual migration strategy ensures business continuity while improving system architecture, maintainability, and scalability.