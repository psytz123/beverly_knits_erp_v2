# API Consolidation Plan - Beverly Knits ERP v2

## Executive Summary

The Beverly Knits ERP currently has **107 API endpoints** with significant redundancy and inconsistent naming. This consolidation plan will reduce the API surface to **~25 well-structured endpoints** (76% reduction) while maintaining backward compatibility.

## Current State Analysis

### API Endpoint Statistics
- **Total Endpoints**: 107
- **Already Consolidated**: ~30 endpoints (with redirects)
- **Remaining Unconsolidated**: 77 endpoints
- **Endpoint Groups**: 40+ different prefixes
- **Duplicate Functionality**: ~60% overlap

### Current Issues
1. **Security Risk**: 95+ endpoints need individual authentication
2. **Maintenance Burden**: 107 route handlers in a 17,697-line file
3. **Inconsistent Naming**: Multiple patterns for similar operations
4. **Documentation Complexity**: Difficult to document 107 endpoints
5. **Performance**: Redundant code paths and caching logic

## Consolidation Strategy

### Design Principles
1. **RESTful Resource-Based**: `/api/v2/{resource}`
2. **Parameter-Driven Views**: `?view=dashboard&format=json`
3. **Consistent Naming**: Predictable URL patterns
4. **Version Namespacing**: `/api/v2/` for new APIs
5. **Backward Compatibility**: 301 redirects from old endpoints

## Detailed Consolidation Mapping

### 1. Inventory Management APIs

#### Current Endpoints (17 total)
```
/api/inventory-analysis
/api/inventory-analysis/complete
/api/inventory-analysis/yarn-shortages
/api/inventory-analysis/stock-risks
/api/inventory-analysis/forecast-vs-stock
/api/inventory-analysis/yarn-requirements
/api/inventory-analysis/action-items
/api/inventory-analysis/dashboard-data
/api/inventory-overview
/api/inventory-netting
/api/inventory-intelligence-enhanced ✅
/api/real-time-inventory
/api/real-time-inventory-dashboard
/api/multi-stage-inventory
/api/safety-stock
/api/dynamic-eoq
/api/ai/inventory-intelligence
```

#### Consolidated Endpoint
```http
GET /api/v2/inventory

Parameters:
  operation: analysis|netting|multi-stage|safety-stock|eoq
  view: overview|dashboard|real-time|complete|action-items
  analysis_type: yarn-shortages|stock-risks|forecast-comparison
  format: summary|detailed|report
  ai_enhanced: true|false
  real_time: true|false

Examples:
  /api/v2/inventory?operation=analysis&view=dashboard
  /api/v2/inventory?operation=netting&format=detailed
  /api/v2/inventory?operation=safety-stock&ai_enhanced=true
```

### 2. Yarn Management APIs

#### Current Endpoints (15 total)
```
/api/yarn
/api/yarn-data
/api/yarn-intelligence ✅
/api/yarn-intelligence-enhanced
/api/yarn-shortage-analysis
/api/yarn-shortage-timeline
/api/yarn-forecast-shortages
/api/yarn-aggregation
/api/yarn-requirements-calculation
/api/yarn-substitution-intelligent ✅
/api/yarn-substitution-opportunities
/api/yarn-alternatives
/api/validate-substitution
/api/ai/yarn-forecast/<yarn_id>
/api/inventory-analysis/yarn-requirements
```

#### Consolidated Endpoint
```http
GET /api/v2/yarn
POST /api/v2/yarn (for calculations)

Parameters:
  operation: analysis|forecast|substitution|requirements|validation
  analysis_type: shortage|timeline|aggregation|intelligence
  view: data|opportunities|alternatives
  yarn_id: {specific_yarn_id} (optional)
  include_forecast: true|false
  include_substitutes: true|false
  ai_enhanced: true|false

Examples:
  /api/v2/yarn?operation=analysis&analysis_type=shortage
  /api/v2/yarn?operation=substitution&view=opportunities
  /api/v2/yarn?operation=forecast&yarn_id=YARN001
```

### 3. Production Management APIs

#### Current Endpoints (14 total)
```
/api/production-planning ✅
/api/production-data
/api/production-orders
/api/production-orders/create
/api/production-pipeline
/api/production-schedule
/api/production-suggestions
/api/production-recommendations-ml
/api/production-metrics-enhanced
/api/production-capacity
/api/production-machine-mapping
/api/production-plan-forecast
/api/ai-production-insights
/api/ai-production-forecast
```

#### Consolidated Endpoint
```http
GET /api/v2/production
POST /api/v2/production (for create operations)

Parameters:
  resource: orders|schedule|pipeline|metrics|capacity|mapping
  operation: list|create|analyze|forecast
  view: data|suggestions|insights
  ai_enhanced: true|false (for ML recommendations)
  include_forecast: true|false
  format: summary|detailed|enhanced

Examples:
  /api/v2/production?resource=orders&operation=list
  /api/v2/production?resource=schedule&include_forecast=true
  /api/v2/production?resource=metrics&format=enhanced
  POST /api/v2/production?resource=orders&operation=create
```

### 4. ML/AI Forecasting APIs

#### Current Endpoints (12 total)
```
/api/ml-forecasting
/api/ml-forecast-detailed ✅
/api/ml-forecast-report
/api/ml-validation-summary
/api/consistency-forecast
/api/sales-forecast-analysis
/api/ai/yarn-forecast/<yarn_id>
/api/ai/ensemble-forecast
/api/ai/optimize-safety-stock
/api/ai/reorder-recommendation
/api/pipeline/forecast
/api/retrain-ml
```

#### Consolidated Endpoint
```http
GET /api/v2/forecast
POST /api/v2/forecast (for operations)

Parameters:
  model: ml|ensemble|consistency|arima|prophet|lstm
  target: yarn|inventory|production|sales
  operation: predict|validate|retrain|optimize
  output: summary|detailed|validation|report
  horizon: {days} (forecast horizon)
  yarn_id: {specific_id} (optional)

Examples:
  /api/v2/forecast?model=ensemble&target=yarn&output=detailed
  /api/v2/forecast?model=ml&operation=validate&output=report
  POST /api/v2/forecast?operation=retrain&model=ensemble
```

### 5. Backtest APIs

#### Current Endpoints (6 total)
```
/api/backtest/run
/api/backtest/models
/api/backtest/accuracy
/api/backtest/fabric-comprehensive
/api/backtest/yarn-comprehensive
/api/backtest/full-report
```

#### Consolidated Endpoint
```http
GET /api/v2/backtest
POST /api/v2/backtest (for run operations)

Parameters:
  operation: run|list-models|get-accuracy|generate-report
  scope: fabric|yarn|full|specific-model
  model: {model_name} (optional)
  date_range: {start_date},{end_date}
  format: summary|comprehensive|detailed

Examples:
  /api/v2/backtest?operation=get-accuracy&scope=yarn
  POST /api/v2/backtest?operation=run&scope=full
  /api/v2/backtest?operation=generate-report&format=comprehensive
```

### 6. Factory Floor & Machine APIs

#### Current Endpoints (7 total)
```
/api/machines-status
/api/machine-utilization
/api/machine-assignment-suggestions
/api/work-center-capacity
/api/factory-floor-status
/api/factory-floor-ai-dashboard
/api/ai-bottleneck-detection
```

#### Consolidated Endpoint
```http
GET /api/v2/factory

Parameters:
  resource: machines|work-centers|floor
  metrics: status|utilization|capacity|assignments
  analysis: bottlenecks|suggestions|optimization
  view: dashboard|detailed|summary
  ai_insights: true|false
  work_center_id: {id} (optional)
  machine_id: {id} (optional)

Examples:
  /api/v2/factory?resource=machines&metrics=utilization
  /api/v2/factory?resource=floor&view=dashboard&ai_insights=true
  /api/v2/factory?analysis=bottlenecks&resource=work-centers
```

### 7. Emergency & Risk Management APIs

#### Current Endpoints (4 total)
```
/api/emergency-shortage
/api/emergency-shortage-dashboard
/api/emergency-procurement
/api/po-risk-analysis
```

#### Consolidated Endpoint
```http
GET /api/v2/emergency

Parameters:
  type: shortage|procurement|risk-analysis
  resource: yarn|production|purchase-orders
  view: dashboard|analysis|recommendations
  severity: critical|high|medium|low
  include_mitigation: true|false

Examples:
  /api/v2/emergency?type=shortage&resource=yarn&view=dashboard
  /api/v2/emergency?type=risk-analysis&resource=purchase-orders
  /api/v2/emergency?type=procurement&severity=critical
```

### 8. Supply Chain APIs

#### Current Endpoints (5 total)
```
/api/supply-chain-analysis
/api/supply-chain-analysis-cached
/api/supplier-intelligence
/api/supplier-risk-scoring
/api/six-phase-planning
```

#### Consolidated Endpoint
```http
GET /api/v2/supply-chain

Parameters:
  operation: analysis|planning|risk-scoring
  resource: suppliers|materials|planning-phases
  phase: 1-6 (for planning)
  use_cache: true|false
  include_intelligence: true|false
  format: summary|detailed|report

Examples:
  /api/v2/supply-chain?operation=analysis&use_cache=true
  /api/v2/supply-chain?operation=planning&phase=1
  /api/v2/supply-chain?resource=suppliers&operation=risk-scoring
```

### 9. Planning & Execution APIs

#### Current Endpoints (5 total)
```
/api/planning-phases
/api/planning-status
/api/planning/execute
/api/execute-planning (GET & POST)
/api/time-phased-planning
```

#### Consolidated Endpoint
```http
GET /api/v2/planning
POST /api/v2/planning (for execute operations)

Parameters:
  operation: status|phases|execute|time-phased
  phase: 1-6 (specific phase)
  horizon: {days} (planning horizon)
  mode: simulation|execute
  format: summary|detailed|status

Examples:
  /api/v2/planning?operation=status
  /api/v2/planning?operation=phases&format=detailed
  POST /api/v2/planning?operation=execute&phase=1&mode=simulation
```

### 10. Analytics & KPIs APIs

#### Current Endpoints (4 total)
```
/api/comprehensive-kpis
/api/executive-insights
/api/advanced-optimization
/api/ai-optimization-recommendations
```

#### Consolidated Endpoint
```http
GET /api/v2/analytics

Parameters:
  type: kpis|insights|optimization
  level: executive|operational|tactical
  include_recommendations: true|false
  ai_enhanced: true|false
  time_period: daily|weekly|monthly|custom
  format: dashboard|report|detailed

Examples:
  /api/v2/analytics?type=kpis&level=executive
  /api/v2/analytics?type=optimization&ai_enhanced=true
  /api/v2/analytics?type=insights&format=dashboard
```

## Implementation Roadmap

### Phase 1: Infrastructure Setup (Week 1)
**Objective**: Create v2 API foundation

Tasks:
1. **Day 1-2**: Create `/src/api/v2/` directory structure
   ```
   src/api/v2/
   ├── __init__.py
   ├── base.py (shared utilities)
   ├── inventory.py
   ├── yarn.py
   ├── production.py
   ├── forecast.py
   ├── factory.py
   ├── emergency.py
   ├── supply_chain.py
   ├── planning.py
   └── analytics.py
   ```

2. **Day 3**: Implement base parameter parsing and validation
   ```python
   class APIv2Base:
       def parse_parameters(self, request):
           # Common parameter parsing logic
           pass
       
       def validate_parameters(self, params, schema):
           # Parameter validation using marshmallow
           pass
   ```

3. **Day 4-5**: Create response standardization
   ```python
   def standard_response(data, status="success", metadata=None):
       return {
           "status": status,
           "timestamp": datetime.utcnow().isoformat(),
           "data": data,
           "metadata": metadata or {}
       }
   ```

### Phase 2: Endpoint Implementation (Week 2)
**Objective**: Build consolidated endpoints

Priority Order:
1. **Day 1**: Inventory endpoint (17 → 1)
2. **Day 2**: Yarn endpoint (15 → 1)
3. **Day 3**: Production endpoint (14 → 1)
4. **Day 4**: Forecast endpoint (12 → 1)
5. **Day 5**: Factory, Emergency, Supply Chain endpoints

### Phase 3: Migration Support (Week 3)
**Objective**: Ensure backward compatibility

Tasks:
1. **Day 1-2**: Implement 301 redirects
   ```python
   @app.route("/api/inventory-analysis")
   def redirect_inventory_analysis():
       return redirect("/api/v2/inventory?operation=analysis", 301)
   ```

2. **Day 3**: Create parameter mapping logic
   ```python
   PARAMETER_MAPPINGS = {
       "/api/inventory-analysis": {
           "params": {"operation": "analysis", "view": "default"}
       }
   }
   ```

3. **Day 4-5**: Update dashboard compatibility layer
   ```javascript
   // Dashboard compatibility wrapper
   function apiCall(oldEndpoint) {
       const newEndpoint = mapToV2(oldEndpoint);
       return fetch(newEndpoint);
   }
   ```

### Phase 4: Testing & Documentation (Week 4)
**Objective**: Validate and document

Tasks:
1. **Day 1-2**: Comprehensive testing
   - Unit tests for each v2 endpoint
   - Integration tests for parameter combinations
   - Performance benchmarks

2. **Day 3**: Load testing
   - Test concurrent requests
   - Validate caching behavior
   - Check redirect performance

3. **Day 4-5**: Documentation
   - OpenAPI 3.0 specification
   - Migration guide
   - Parameter reference

## Benefits Analysis

### Quantitative Benefits
| Metric | Current | After Consolidation | Improvement |
|--------|---------|-------------------|-------------|
| Total Endpoints | 107 | 25 | -76% |
| Code Lines | 17,697 | ~8,000 | -55% |
| Auth Points | 107 | 25 | -76% |
| Documentation Pages | 107 | 25 | -76% |
| Test Cases | 500+ | 200 | -60% |
| Cache Keys | 107 | 25 | -76% |

### Qualitative Benefits
1. **Security**: Easier to audit and secure 25 endpoints vs 107
2. **Maintainability**: Clear resource-based structure
3. **Performance**: Better caching with parameter-based keys
4. **Developer Experience**: Predictable, consistent API patterns
5. **Documentation**: Cleaner, more comprehensive docs

## Migration Guide

### For Frontend Developers
```javascript
// Old way
fetch('/api/inventory-analysis')
fetch('/api/inventory-overview')
fetch('/api/real-time-inventory')

// New way
fetch('/api/v2/inventory?operation=analysis')
fetch('/api/v2/inventory?view=overview')
fetch('/api/v2/inventory?real_time=true')
```

### For Backend Developers
```python
# Old way - multiple route handlers
@app.route("/api/inventory-analysis")
def inventory_analysis(): pass

@app.route("/api/inventory-overview")
def inventory_overview(): pass

# New way - single parameterized handler
@app.route("/api/v2/inventory")
def inventory_v2():
    operation = request.args.get('operation', 'analysis')
    view = request.args.get('view', 'default')
    return handle_inventory_request(operation, view)
```

## Testing Strategy

### Unit Tests
```python
def test_inventory_consolidation():
    # Test all parameter combinations
    responses = []
    for operation in ['analysis', 'netting', 'safety-stock']:
        for view in ['overview', 'dashboard', 'detailed']:
            response = client.get(f'/api/v2/inventory?operation={operation}&view={view}')
            assert response.status_code == 200
```

### Integration Tests
```python
def test_backward_compatibility():
    # Verify redirects work correctly
    old_response = client.get('/api/inventory-analysis')
    assert old_response.status_code == 301
    assert '/api/v2/inventory' in old_response.location
```

### Performance Tests
```python
def test_consolidated_performance():
    # Ensure consolidation doesn't degrade performance
    start = time.time()
    for _ in range(100):
        client.get('/api/v2/inventory?operation=analysis')
    elapsed = time.time() - start
    assert elapsed < 20  # Less than 200ms per request
```

## Security Considerations

### Authentication Strategy
```python
@auth_required
@app.route("/api/v2/<resource>")
def unified_handler(resource):
    # Single authentication point for each resource
    if not user_has_permission(resource, request.args):
        return forbidden()
    return process_request(resource, request.args)
```

### Input Validation
```python
from marshmallow import Schema, fields, validate

class InventoryQuerySchema(Schema):
    operation = fields.Str(validate=validate.OneOf([
        'analysis', 'netting', 'safety-stock', 'eoq'
    ]))
    view = fields.Str(validate=validate.OneOf([
        'overview', 'dashboard', 'detailed'
    ]))
    format = fields.Str(default='summary')
```

### Rate Limiting
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=get_remote_address)

@limiter.limit("100/minute")
@app.route("/api/v2/<resource>")
def rate_limited_handler(resource):
    pass
```

## Monitoring & Analytics

### Metrics to Track
1. **Usage Patterns**
   - Most used parameter combinations
   - Peak usage times
   - Resource consumption by endpoint

2. **Performance Metrics**
   - Response times per resource
   - Cache hit rates
   - Database query counts

3. **Migration Progress**
   - Old vs new endpoint usage
   - Redirect counts
   - Deprecation warnings triggered

### Dashboard Updates
```javascript
// Update dashboard to show consolidation metrics
const ConsolidationMetrics = {
    oldEndpointCalls: 0,
    newEndpointCalls: 0,
    redirectCount: 0,
    deprecationWarnings: [],
    
    track(endpoint) {
        if (endpoint.startsWith('/api/v2/')) {
            this.newEndpointCalls++;
        } else {
            this.oldEndpointCalls++;
            this.deprecationWarnings.push({
                endpoint,
                timestamp: Date.now()
            });
        }
    }
};
```

## Rollback Plan

### Feature Flags
```python
FEATURE_FLAGS = {
    'use_v2_apis': False,
    'enable_redirects': True,
    'show_deprecation_warnings': True
}

if FEATURE_FLAGS['use_v2_apis']:
    register_v2_endpoints()
else:
    register_legacy_endpoints()
```

### Gradual Rollout
1. **Week 1**: Enable v2 APIs in parallel with v1
2. **Week 2**: Start redirecting 10% of traffic
3. **Week 3**: Increase to 50% redirect
4. **Week 4**: 100% redirect with fallback
5. **Week 5**: Deprecate old endpoints

## Success Criteria

### Technical Metrics
- [ ] All 107 endpoints mapped to 25 v2 endpoints
- [ ] 100% backward compatibility via redirects
- [ ] No performance degradation (< 200ms p95)
- [ ] 100% test coverage for v2 endpoints
- [ ] Zero breaking changes for dashboard

### Business Metrics
- [ ] 50% reduction in API-related support tickets
- [ ] 75% faster new feature implementation
- [ ] 90% developer satisfaction with new API
- [ ] 60% reduction in documentation maintenance time

## Conclusion

This API consolidation will transform the Beverly Knits ERP from a sprawling 107-endpoint system to a clean, maintainable 25-endpoint architecture. The implementation maintains full backward compatibility while providing significant benefits in security, performance, and developer experience.

**Total Investment**: 4 weeks of development
**Expected ROI**: 60% reduction in maintenance effort, 75% faster feature development
**Risk Level**: Low (with proper testing and gradual rollout)

---

*Document Version*: 1.0  
*Created*: 2025-09-05  
*Status*: Ready for Review  
*Next Steps*: Approve and begin Phase 1 implementation