# Beverly Knits ERP v2 - Complete API-Dashboard Architecture Reference

**Document Version**: 1.0
**Last Updated**: September 13, 2025
**System Version**: Beverly Knits ERP v2 (Production Ready)
**Primary Maintainer**: Development Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Complete API Inventory](#complete-api-inventory)
4. [Dashboard Tab-to-API Mapping](#dashboard-tab-to-api-mapping)
5. [API Consolidation System](#api-consolidation-system)
6. [Data Flow Architecture](#data-flow-architecture)
7. [Recovery Procedures](#recovery-procedures)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Historical Context](#historical-context)
10. [References & Related Documents](#references--related-documents)

---

## Executive Summary

The Beverly Knits ERP v2 system is a production-ready textile manufacturing ERP with real-time inventory intelligence, ML-powered forecasting, and 6-phase supply chain optimization. This document provides the definitive mapping between the 95+ API endpoints and their corresponding dashboard data points to ensure rapid system recovery and maintenance.

### Key System Statistics
- **Total APIs**: 95+ endpoints (consolidated to ~50 active)
- **Dashboard Tabs**: 7 primary tabs with JavaScript-driven interfaces
- **Data Records**: 41,596+ total (1,199 yarns, 28,653 BOM entries, 195 production orders)
- **Inventory Value**: $4,939,491 tracked
- **Main Ports**: 5006 (primary ERP), 8000 (wrapper service)

---

## System Architecture Overview

### Core Components

#### 1. Main Application Server
**File**: `/src/core/beverly_comprehensive_erp.py` (7,000+ lines)
**Port**: 5006
**Framework**: Flask with CORS support
**Status**: Production ready, handles 41,596+ data records

**Key Classes**:
- `InventoryAnalyzer`: Core inventory analysis with Planning Balance calculations
- `InventoryManagementPipeline`: Orchestrates inventory operations
- `SalesForecastingEngine`: ML-powered demand forecasting
- `CapacityPlanningEngine`: Production capacity planning

#### 2. Dashboard Interface
**File**: `/web/consolidated_dashboard.html` (16,000+ lines)
**Framework**: Vanilla JavaScript with TailwindCSS
**Features**: 7 tabs, real-time updates, responsive design
**Update Mechanism**: `fetchAPI()` wrapper with error handling

#### 3. ERP Wrapper Service
**Location**: `/erp-wrapper/`
**Port**: 8000
**Framework**: FastAPI/Uvicorn
**Purpose**: External system integration and API wrapping

### Data Architecture

#### Primary Data Sources
```
/data/production/5/ERP Data/
├── yarn_inventory.csv (1,199 items, $4.9M value)
├── BOM_updated.csv (28,653 style-to-yarn mappings)
├── eFab_Knit_Orders.csv (195 production orders)
├── eFab_SO_List.csv (133 sales orders)
├── eFab_Styles_20250902.xlsx (11,416 style records)
└── 8-28-2025/ (latest eFab data snapshots)
```

#### Data Loading Architecture
1. **OptimizedDataLoader**: 100x+ speed improvement with caching
2. **ParallelDataLoader**: 4x speed with concurrent loading
3. **UnifiedCacheManager**: Memory + Redis with TTL=300s

---

## Complete API Inventory

### Core System APIs (Always Active)

#### 1. Comprehensive KPIs
```http
GET /api/comprehensive-kpis
```
**Purpose**: Central dashboard metrics
**Returns**: Inventory value, active orders, alerts, forecast accuracy
**Used By**: Overview tab, all dashboard status cards
**Data Source**: Real-time calculation from all data sources

#### 2. System Health & Control
```http
GET /api/reload-data                    # Force data refresh
GET /api/debug-data                     # Data loading diagnostics
GET /api/consolidation-metrics          # API consolidation status
GET /api/cache-stats                    # Cache performance metrics
POST /api/cache-clear                   # Clear all caches
```

### Inventory Management APIs

#### Primary Inventory Intelligence
```http
GET /api/inventory-intelligence-enhanced
```
**Parameters**:
- `view`: `summary|dashboard|complete|full`
- `analysis`: `standard|shortage|optimization`
- `realtime`: `true|false`
- `ai`: `true|false`

**Dashboard Usage**:
- Overview tab: Main inventory metrics
- Inventory tab: Complete inventory analysis
- Real-time status updates across all tabs

**Consolidates**: 7 legacy inventory endpoints

#### Yarn Intelligence System
```http
GET /api/yarn-intelligence
```
**Parameters**:
- `view`: `full|data|summary`
- `analysis`: `standard|shortage|requirements`
- `forecast`: `true|false`
- `yarn_id`: `specific_yarn_identifier`
- `include_timing`: `true|false`

**Dashboard Usage**:
- Overview tab: Critical yarn shortages
- Inventory tab: Yarn analysis and alternatives
- Production tab: Yarn requirements for orders

**Consolidates**: 6 legacy yarn endpoints

#### Inventory Netting & Requirements
```http
GET /api/inventory-netting              # Multi-level BOM netting
GET /api/bom-explosion-net-requirements # BOM explosion calculations
GET /api/yarn-requirements-calculation  # Yarn requirement analysis
```

### Production Management APIs

#### Production Planning (Consolidated)
```http
GET /api/production-planning
```
**Parameters**:
- `view`: `planning|orders|data|metrics`
- `forecast`: `true|false`
- `include_capacity`: `true|false`

**Dashboard Usage**:
- Production tab: Production planning data
- Overview tab: Production status metrics

**Consolidates**: 3 legacy production endpoints

#### Production Pipeline & Workflow
```http
GET /api/production-pipeline            # Real-time production flow
GET /api/production-suggestions         # AI-powered recommendations
GET /api/po-risk-analysis              # Purchase order risk assessment
GET /api/production-recommendations-ml  # ML-based production optimization
```

#### Knit Orders Management
```http
GET /api/knit-orders                   # Primary knit orders data
GET /api/knit-orders-analysis          # Order analysis and metrics
GET /api/knit-orders-styles            # Style mapping for orders
POST /api/knit-orders/generate         # Create new knit orders
```

**Dashboard Usage**:
- Knit Orders tab: Complete order management
- Production tab: Order status and assignments
- Machine Planning tab: Order-to-machine assignments

#### Machine & Capacity Planning
```http
GET /api/factory-floor-ai-dashboard     # Machine planning dashboard data
GET /api/machine-assignment-suggestions # ML-powered machine assignments
GET /api/machine-utilization           # Machine capacity utilization
GET /api/work-center-capacity          # Work center capacity analysis
GET /api/capacity-bottlenecks          # Bottleneck identification
```

**Dashboard Usage**:
- Machine Planning tab: Complete machine scheduling interface
- Production tab: Capacity and utilization metrics

### ML & Forecasting APIs

#### ML Forecasting (Consolidated)
```http
GET /api/ml-forecast-detailed
```
**Parameters**:
- `detail`: `full|summary|metrics`
- `format`: `json|report|chart`
- `compare`: `stock|orders|capacity`
- `horizon`: `30|60|90|180` (days)
- `source`: `ml|pipeline|hybrid`

**Dashboard Usage**:
- Forecasting tab: ML model performance and predictions
- Overview tab: Forecast accuracy metrics
- Analytics tab: Forecast vs actual analysis

**Consolidates**: 4 legacy ML endpoints

#### Forecasting Support APIs
```http
GET /api/ml-validation-summary         # Model validation metrics
POST /api/retrain-ml                   # Trigger model retraining
GET /api/fabric-forecast-integrated    # Fabric demand forecasting
GET /api/consistency-forecast          # Data consistency forecasting
```

### Emergency & Risk Management APIs

#### Emergency Shortage Management
```http
GET /api/emergency-shortage-dashboard
```
**Parameters**:
- `view`: `dashboard|procurement|analysis`
- `type`: `all|yarn|fabric|materials`
- `urgency`: `critical|high|medium|all`

**Dashboard Usage**:
- Inventory tab: Critical shortage alerts
- Production tab: Production-blocking shortages

#### Risk Analysis APIs
```http
GET /api/supplier-risk-scoring         # Supplier risk assessment
GET /api/supply-chain-analysis         # Supply chain risk analysis
GET /api/yarn-shortage-timeline        # Timeline-based shortage analysis
```

### Analytics & Optimization APIs

#### Advanced Analytics
```http
GET /api/advanced-optimization         # AI inventory optimization
GET /api/executive-insights           # Executive-level analytics
GET /api/ai/inventory-intelligence    # AI-enhanced inventory insights
GET /api/procurement-recommendations  # Procurement optimization
```

**Dashboard Usage**:
- Analytics tab: Advanced optimization recommendations
- Overview tab: Executive summary metrics

#### Substitution & Alternatives
```http
GET /api/yarn-substitution-intelligent # Intelligent yarn substitutions
GET /api/yarn-alternatives            # Alternative yarn options
GET /api/validate-substitution        # Substitution validation
```

### Fabric & Textile APIs

#### Fabric Management
```http
POST /api/fabric/convert              # Fabric unit conversions
GET /api/fabric/specs                 # Fabric specifications
POST /api/fabric/yarn-requirements    # Fabric yarn requirement calculation
GET /api/fabric-production           # Fabric production metrics
```

#### Textile BOM
```http
POST /api/textile-bom                # Textile bill of materials processing
GET /api/style-mapping              # Style number mappings
```

### Testing & Development APIs

#### Backtesting & Validation
```http
GET /api/backtest/fabric-comprehensive # Fabric forecast backtesting
GET /api/backtest/yarn-comprehensive   # Yarn forecast backtesting
GET /api/backtest/full-report         # Complete backtest analysis
GET /api/backtest/models              # Available backtest models
GET /api/backtest/accuracy            # Backtest accuracy metrics
```

---

## Dashboard Tab-to-API Mapping

### Overview Tab (`overview-tab`)
**Primary Function**: `loadDashboard()` → `loadOverviewData()`

**API Calls**:
1. `/api/comprehensive-kpis` - Core metrics (inventory value, active orders, alerts)
2. `/api/yarn-intelligence` - Yarn shortage analysis and criticality
3. `/api/inventory-intelligence-enhanced?realtime=true` - Real-time inventory status

**Data Points Updated**:
- `#inventoryValue` - Total inventory value ($4,939,491)
- `#activeOrders` - Active knit orders count (133)
- `#alertCount` - Critical alerts (457)
- `#costSavings` - Procurement savings
- Status cards: Inventory, Production, Demand, Planning status

**Loading Function Location**: Line 4003 in consolidated_dashboard.html

### Production Tab (`production-tab`)
**Primary Function**: `loadProductionData()`

**API Calls**:
1. `/api/production-planning` - Production planning data
2. `/api/production-pipeline` - Real-time production flow
3. `/api/knit-orders` - Active knit orders
4. `/api/po-risk-analysis` - Purchase order risk assessment
5. `/api/production-suggestions` - AI recommendations

**Data Points Updated**:
- Production order tables with style, quantity, machine assignments
- PO risk analysis with shortage predictions
- Production suggestions and optimization recommendations
- Machine utilization and capacity metrics

**Loading Function Location**: Line 4033 in consolidated_dashboard.html

### Inventory Tab (`inventory-tab`)
**Primary Function**: `loadInventoryTab()` → `loadInventoryData()`

**API Calls**:
1. `/api/inventory-intelligence-enhanced` - Complete inventory analysis
2. `/api/yarn-intelligence` - Yarn-specific intelligence and shortages
3. `/api/yarn-substitution-intelligent` - Alternative yarn options

**Data Points Updated**:
- Inventory status tables with Planning Balance calculations
- Critical yarn shortages list (457 items requiring attention)
- Yarn alternatives and substitution recommendations
- Inventory risk assessments and optimization suggestions

**Loading Function Location**: Line 6649 in consolidated_dashboard.html

### Machine Planning Tab (`machine-planning-tab`)
**Primary Function**: `loadMachinePlanningData()`

**API Calls**:
1. `/api/factory-floor-ai-dashboard` - Machine planning dashboard data
2. `/api/machine-assignment-suggestions` - ML-powered machine assignments
3. `/api/knit-orders` - Orders requiring machine assignment (40 unassigned)

**Data Points Updated**:
- Machine schedule board with work center groupings
- Machine utilization by work center (285 total machines, 91 work centers)
- Unassigned order recommendations
- Production workload distribution (557,671 lbs total)

**Loading Function Location**: Line 13961 in consolidated_dashboard.html

### Forecasting Tab (`forecasting-tab`)
**Primary Function**: `loadForecastingTab()` → `loadMLForecastingData()`

**API Calls**:
1. `/api/ml-forecast-detailed?detail=summary` - ML model performance
2. `/api/ml-validation-summary` - Model validation metrics
3. `/api/yarn-intelligence` - Yarn demand forecasting

**Data Points Updated**:
- ML model accuracy metrics (92.5% current accuracy)
- Forecast vs actual comparisons
- Model training status and performance
- Risk assessment based on forecast reliability

**Loading Function Location**: Line 8282 in consolidated_dashboard.html

### Analytics Tab (`analytics-tab`)
**Primary Function**: `loadAnalyticsTab()`

**API Calls**:
1. `/api/advanced-optimization` - AI optimization recommendations
2. `/api/executive-insights` - Executive-level analytics
3. `/api/comprehensive-kpis` - Performance metrics
4. `/api/ai/inventory-intelligence` - AI-enhanced insights

**Data Points Updated**:
- Executive dashboard with key performance indicators
- Optimization recommendations and potential savings
- Efficiency metrics and trend analysis
- Advanced analytics and business intelligence

### Knit Orders Tab (`knit-orders-tab`)
**Primary Function**: `loadKnitOrders()`

**API Calls**:
1. `/api/knit-orders` - Complete knit orders data (195 orders)
2. `/api/knit-orders-analysis` - Order analysis metrics
3. `/api/knit-orders-styles` - Style mapping data (optional)

**Data Points Updated**:
- Knit orders table with status, style, quantity, machine assignment
- Order summary statistics (154 assigned, 40 pending assignment)
- Style mapping and BOM information
- Order creation and management interfaces

**Loading Function Location**: Line 4174 in consolidated_dashboard.html

---

## API Consolidation System

### Overview
The system implements a comprehensive API consolidation strategy that reduced 95+ endpoints to approximately 50 active endpoints through parameter-based views and intelligent redirects.

### Feature Flags Configuration
**File**: `/src/config/feature_flags.py`

```python
FEATURE_FLAGS = {
    "api_consolidation_enabled": True,        # Master consolidation switch
    "redirect_deprecated_apis": True,         # Auto-redirect old endpoints
    "log_deprecated_usage": True,            # Monitor deprecated usage
    "enforce_new_apis": False,               # Block deprecated entirely
    "dashboard_compatibility_layer": True    # Dashboard compatibility
}
```

### Consolidation Mapping
**Reference File**: `/docs/api_mapping.json`

#### Major Consolidations:

**Inventory APIs → `/api/inventory-intelligence-enhanced`**:
- `/api/inventory-analysis` → `?view=full`
- `/api/inventory-overview` → `?view=summary`
- `/api/real-time-inventory` → `?realtime=true`
- `/api/ai/inventory-intelligence` → `?ai=true`

**Yarn APIs → `/api/yarn-intelligence`**:
- `/api/yarn-data` → `?view=data`
- `/api/yarn-shortage-analysis` → `?analysis=shortage`
- `/api/yarn-forecast-shortages` → `?forecast=true&analysis=shortage`

**Production APIs → `/api/production-planning`**:
- `/api/production-data` → `?view=data`
- `/api/production-orders` → `?view=orders`
- `/api/production-plan-forecast` → `?forecast=true`

**ML APIs → `/api/ml-forecast-detailed`**:
- `/api/ml-forecasting` → `?detail=summary`
- `/api/ml-forecast-report` → `?format=report`

### Redirect Middleware
**File**: `/src/api/consolidation_middleware.py`

**Functions**:
- `deprecated_api()`: Handles automatic redirects
- `redirect_to_new_api()`: Parameter mapping and forwarding
- `intercept_deprecated_endpoints()`: Middleware integration

**Monitoring**: `/api/consolidation-metrics` provides real-time consolidation statistics

### Emergency Rollback
```python
# Emergency rollback procedure
from src.config.feature_flags import emergency_rollback
emergency_rollback()  # Disables all consolidation features
```

---

## Data Flow Architecture

### Data Loading Pipeline

#### 1. Unified Data Loader
**File**: `/src/data_loaders/unified_data_loader.py`
**Performance**: Loads 41,596 records in ~1.56 seconds
**Features**: Parallel loading, caching, error recovery

**Loading Sequence**:
1. Knit Orders: 195 records (0.54s)
2. Yarn Inventory: 1,199 records (0.57s)
3. Sales Orders: 133 records (0.62s)
4. Styles: 11,416 records (1.07s)
5. BOM: 28,653 records (1.56s)

#### 2. Cache Management
**File**: `/src/utils/cache_manager.py`
**Configuration**: TTL=300s, max_items=1000
**Backends**: Memory + Redis with intelligent eviction

**Cache Performance Monitoring**:
```http
GET /api/cache-stats    # Hit/miss ratios, performance metrics
POST /api/cache-clear   # Emergency cache clearing
```

#### 3. Data Consistency Manager
**File**: `/src/data_consistency/consistency_manager.py`
**Purpose**: Handles column name variations and data standardization

**Column Standardization**:
- 'Planning Balance' vs 'Planning_Balance'
- 'Desc#' vs 'desc_num' vs 'YarnID'
- 'fStyle#' vs 'Style#' for style mapping

### Real-Time Update Mechanism

#### Dashboard fetchAPI() Wrapper
**Location**: Line 3514 in consolidated_dashboard.html

```javascript
async function fetchAPI(endpoint, options = {}) {
    // Handles URL construction, error recovery, and response parsing
    // Supports parameter-based API calls
    // Implements automatic retry with fallback chains
}
```

**Features**:
- Automatic `/api/` prefix handling
- Parameter preservation for consolidated endpoints
- Error handling with user-friendly messages
- Response caching and deduplication

#### Auto-Refresh System
- Overview tab: 30-second KPI updates
- Production tab: 60-second pipeline updates
- Inventory tab: Real-time shortage monitoring
- Machine Planning: 5-minute schedule updates

---

## Recovery Procedures

### System Startup Sequence

#### 1. Port Management
```bash
# Check for port conflicts
lsof -i :5006
lsof -i :8000

# Kill existing processes if needed
pkill -f "python3.*beverly"
lsof -i :5006 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

#### 2. Cache Clearing
```bash
# Clear all caches
rm -rf /tmp/bki_cache/*
rm -rf cache/*

# Via API (if server is running)
curl -X POST http://localhost:5006/api/cache-clear
```

#### 3. Service Startup
```bash
# Primary ERP application
python3 src/core/beverly_comprehensive_erp.py

# ERP wrapper service (if needed)
cd erp-wrapper && uvicorn app.main:app --port 8000 --host 0.0.0.0
```

#### 4. Data Validation
```bash
# Force data reload
curl http://localhost:5006/api/reload-data

# Verify system health
curl http://localhost:5006/api/debug-data | python3 -m json.tool

# Check consolidation status
curl http://localhost:5006/api/consolidation-metrics | python3 -m json.tool
```

### Emergency Recovery Scenarios

#### Scenario 1: API Consolidation Issues
**Symptoms**: Dashboard shows errors, deprecated endpoint warnings

**Solution**:
```python
# Emergency rollback
from src.config.feature_flags import emergency_rollback
emergency_rollback()

# Or manual flag adjustment
FEATURE_FLAGS["api_consolidation_enabled"] = False
FEATURE_FLAGS["redirect_deprecated_apis"] = False
```

#### Scenario 2: Data Loading Failures
**Symptoms**: Dashboard shows loading errors, empty data

**Solution**:
```bash
# 1. Check data file existence
ls -la "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/"

# 2. Clear all caches
rm -rf /tmp/bki_cache/*

# 3. Restart with data reload
pkill -f "python3.*beverly"
python3 src/core/beverly_comprehensive_erp.py
curl http://localhost:5006/api/reload-data
```

#### Scenario 3: Dashboard JavaScript Errors
**Symptoms**: Tabs not loading, API calls failing

**Solution**:
1. Check browser console for specific errors
2. Verify API endpoints with direct curl calls
3. Clear browser cache and hard refresh (Ctrl+Shift+R)
4. Check `fetchAPI()` function for endpoint construction errors

#### Scenario 4: Port Conflicts
**Symptoms**: "Address already in use" errors

**Solution**:
```bash
# Find and kill competing processes
sudo netstat -tulpn | grep :5006
sudo kill -9 [PID]

# Or comprehensive cleanup
sudo pkill -f beverly
sudo pkill -f uvicorn
```

### Data Recovery Procedures

#### Missing Data Files
**Primary Data Location**: `/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/`

**Critical Files**:
- `yarn_inventory.csv` (1,199 items)
- `BOM_updated.csv` (28,653 entries)
- `eFab_Knit_Orders.csv` (195 orders)
- `eFab_SO_List.csv` (133 sales orders)

**Recovery Steps**:
1. Check backup locations in `8-28-2025/` subdirectory
2. Verify file permissions and accessibility
3. Use SharePoint integration if available
4. Restore from backup if necessary

#### Column Name Issues
**Common Problems**:
- 'Planning Balance' vs 'Planning_Balance'
- Missing or renamed columns in data files

**Solution**: DataConsistencyManager handles most variations automatically, but manual intervention may be required for new column name patterns.

---

## Troubleshooting Guide

### Common Issues & Solutions

#### 1. "Day 0 Emergency fixes not available"
**Error**: `[DAY0] Emergency fixes not available: No module named 'scripts'`
**Impact**: None - this is expected, core functionality works
**Solution**: Ignore - Day 0 fixes are standalone utilities

#### 2. ML Training Data Format Issues
**Error**: Price format errors during ML training
**Cause**: Sales data contains "$" prefixes in price columns
**Solution**:
```python
df['price'] = df['price'].str.replace('$', '').astype(float)
```

#### 3. JSON Serialization Errors (NaN values)
**Error**: "NaN is not JSON serializable"
**Solution**:
```python
# Convert NaN values before JSON serialization
value = float(value) if pd.notna(value) else 0
date_value = str(value) if pd.notna(value) else ''
```

#### 4. Machine Planning Tab Errors
**Error**: "fetchAPI is not a function"
**Solution**: Ensure `fetchAPI` function is loaded before tab initialization (Line 3514)

#### 5. API Parameter Issues
**Error**: Consolidated endpoints not returning expected data
**Cause**: Incorrect parameter formatting
**Solution**: Use documented parameter combinations from API mapping

### Performance Issues

#### Slow Data Loading
**Symptoms**: Dashboard takes >5 seconds to load
**Solutions**:
1. Clear cache: `rm -rf /tmp/bki_cache/*`
2. Check data file sizes and accessibility
3. Monitor cache hit ratios: `GET /api/cache-stats`
4. Restart server to refresh data loaders

#### High Memory Usage
**Symptoms**: Server becomes unresponsive, memory errors
**Solutions**:
1. Restart server to clear memory leaks
2. Reduce cache TTL settings
3. Limit concurrent API calls from dashboard
4. Monitor with `htop` or system tools

#### API Response Timeouts
**Symptoms**: Dashboard shows timeout errors
**Solutions**:
1. Check server logs for specific endpoint issues
2. Verify data file accessibility
3. Clear caches and restart
4. Temporarily disable real-time updates

### Dashboard-Specific Issues

#### Tab Loading Failures
**Symptoms**: Specific tabs show "Loading..." indefinitely
**Debug Steps**:
1. Check browser console for JavaScript errors
2. Test API endpoints directly with curl
3. Verify `switchTab()` function is properly defined
4. Check tab-specific loading functions exist

#### Real-Time Updates Not Working
**Symptoms**: Data doesn't refresh automatically
**Solutions**:
1. Check `fetchAPI()` wrapper for errors
2. Verify WebSocket connections (if applicable)
3. Clear browser cache
4. Check network connectivity

---

## Historical Context

### System Evolution
The Beverly Knits ERP v2 represents a complete evolution from a collection of separate scripts to a unified, production-ready manufacturing ERP system.

#### Version History:
- **v1.0** (2024): Initial prototype with separate data loaders
- **v1.5** (Early 2025): Consolidated core functionality
- **v2.0** (August 2025): API consolidation and production readiness
- **v2.1** (September 2025): Current production version

#### Major Milestones:
1. **API Consolidation Project** (August 2025): Reduced 95+ endpoints to ~50
2. **Dashboard Unification** (August 2025): Single consolidated dashboard
3. **ML Integration** (September 2025): Production-ready forecasting
4. **Performance Optimization** (September 2025): 100x+ speed improvements

### Design Decisions

#### Why API Consolidation?
- **Problem**: 95+ redundant endpoints causing maintenance overhead
- **Solution**: Parameter-based consolidated endpoints with intelligent redirects
- **Result**: 47% reduction in endpoint count, improved maintainability

#### Why Monolithic Architecture?
- **Rationale**: Faster development, simpler deployment for manufacturing environment
- **Trade-offs**: Acknowledged, but acceptable for current scale
- **Future**: Microservices architecture planned for enterprise scale

#### Why Flask over FastAPI?
- **Historical**: System evolved from Flask-based prototypes
- **Integration**: Extensive existing Flask ecosystem integration
- **Performance**: Adequate for current scale with optimization opportunities

### Lessons Learned

#### API Design:
- Parameter-based views more flexible than endpoint proliferation
- Automatic redirects enable smooth migrations
- Feature flags essential for safe rollouts

#### Data Architecture:
- Parallel loading critical for large datasets
- Cache management must handle TTL and invalidation intelligently
- Column name standardization required for real-world data

#### Dashboard Development:
- JavaScript error handling must be comprehensive
- Real-time updates require careful state management
- Mobile responsiveness important for manufacturing environments

---

## References & Related Documents

### Primary Documentation
- **CLAUDE.md**: System commands and configuration reference
- **README.md**: Quick start and installation guide
- **API_MAPPING_DOCUMENTATION.md**: Detailed API consolidation documentation
- **DEPLOYMENT_READY.md**: Production deployment procedures

### Technical Documentation
- **`/docs/api_mapping.json`**: Complete API consolidation mapping
- **`/docs/technical/`**: Detailed technical analyses and reports
- **`/src/config/feature_flags.py`**: Feature flag configuration
- **`/src/api/consolidated_endpoints.py`**: Consolidated endpoint implementations

### Configuration Files
- **`requirements.txt`**: Python dependencies
- **`/src/config/ml_config.py`**: ML model configurations
- **`/src/database/database_config.json`**: Database configuration
- **`/config/efab_config.json`**: eFab integration settings

### Test Documentation
- **`/tests/`**: Comprehensive test suite
- **`/tests/test_api_consolidation.py`**: API consolidation validation
- **`/tests/performance/`**: Performance benchmarking
- **`/tests/e2e/`**: End-to-end workflow testing

### Data Documentation
- **`/data/production/5/`**: Primary production data location
- **`/docs/technical/DATA_MAPPING_REFERENCE.md`**: Data field mappings
- **`/docs/technical/MAPPING/`**: Production flow and data mappings

### Deployment Documentation
- **`/deployment/`**: Docker and Kubernetes configurations
- **`/docs/deployment/PRODUCTION_DEPLOYMENT_GUIDE.md`**: Production deployment
- **`/docs/deployment/RAILWAY_DEPLOYMENT_GUIDE.md`**: Railway platform deployment

### External References
- **Flask Documentation**: https://flask.palletsprojects.com/
- **TailwindCSS Documentation**: https://tailwindcss.com/docs
- **Chart.js Documentation**: https://www.chartjs.org/docs/
- **Pandas Documentation**: https://pandas.pydata.org/docs/

### API Testing Tools
```bash
# Quick API validation
curl http://localhost:5006/api/comprehensive-kpis
curl http://localhost:5006/api/consolidation-metrics
curl http://localhost:5006/api/debug-data

# Dashboard access
http://localhost:5006/consolidated
```

### Support Contacts
- **Primary Dashboard**: http://localhost:5006/consolidated
- **API Documentation**: Available via `/api/` endpoints
- **System Health**: `/api/consolidation-metrics`, `/api/cache-stats`
- **Emergency Procedures**: See Recovery Procedures section above

---

**End of Document**

*This documentation represents the complete API-Dashboard architecture as of September 13, 2025. For updates or corrections, modify this file and update the version header.*