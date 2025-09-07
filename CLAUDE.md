# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

Beverly Knits ERP v2 - Production-ready textile manufacturing ERP with real-time inventory intelligence, ML-powered forecasting, and 6-phase supply chain optimization.

**Current System Stats**:
- 1,200+ yarn items actively tracked (5x increase via API integration)
- 28,653+ BOM entries (style to yarn mappings) 
- 195 production orders with real-time tracking
- 91 work centers with 285 total machines
- 557,671+ lbs total production workload
- Machine utilization tracking via eFab Knit Orders integration
- Real-time yarn shortage detection with API-first data loading
- Complete API-first architecture with file fallbacks

## Primary Commands

### Server Operations
```bash
# Start main ERP server (Port 5006)
python3 src/core/beverly_comprehensive_erp.py

# Start ERP wrapper service (Port 8000) - Required for API integration
cd erp-wrapper/
uvicorn app.main:app --port 8000
# OR with Docker: docker-compose up -d

# Kill existing servers if port conflict
pkill -f "python3.*beverly"
lsof -i :5006 | grep LISTEN | awk '{print $2}' | xargs kill -9
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Start with Makefile (alternative)
make run          # Production mode
make run-dev      # Development mode with debug
```

### Testing
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests
pytest -m e2e           # End-to-end tests
pytest -n auto          # Parallel execution

# Run single test
pytest tests/unit/test_inventory.py::test_yarn_shortage_calculation -v

# Test API consolidation
pytest tests/test_api_consolidation.py -v

# Using Makefile
make test         # All tests
make test-unit    # Unit tests only
make test-cov     # With coverage report
```

### Code Quality
```bash
# From project root
make lint          # Run linters (ruff, flake8, pylint)
make format        # Format code (black, isort)
make type-check    # Run mypy type checking

# Manual commands
black src/ tests/
isort src/ tests/
ruff check src/
pylint src/
```

### Data Management
```bash
# Clear cache when data issues occur
rm -rf /tmp/bki_cache/*

# Force data reload via API
curl -s http://localhost:5006/api/reload-data

# Debug data loading and API integration
curl -s http://localhost:5006/api/debug-data | python3 -m json.tool
curl -s http://localhost:8000/health  # Check wrapper service health
curl -s http://localhost:8000/api/yarn/active | head -c 1000  # Test API data

# Check consolidation metrics
curl -s http://localhost:5006/api/consolidation-metrics | python3 -m json.tool

# Sync data from SharePoint
make sync-data
make validate
```

### ML Operations
```bash
# Test ML configuration
python3 src/config/ml_config.py

# Run ML backtest
python3 scripts/ml_backtest.py --save-results

# Train specific model
python3 scripts/ml_training_pipeline.py --model xgboost --force

# Deploy model to production
python3 scripts/ml_training_pipeline.py --deploy xgboost
```

### Emergency Fixes & Utilities
```bash
# Run Day 0 health check
python3 scripts/day0_emergency_fixes.py --health-check

# Apply Day 0 fixes to main ERP
python3 scripts/apply_day0_fixes.py

# Validate all fixes
python3 scripts/day0_emergency_fixes.py --validate
```

## High-Level Architecture

### Core Monolithic Application
`src/core/beverly_comprehensive_erp.py` (7000+ lines) - Flask application with:
- **InventoryAnalyzer**: Core inventory analysis engine with Planning Balance calculations
- **InventoryManagementPipeline**: Orchestrates inventory operations and workflow
- **SalesForecastingEngine**: ML-powered demand forecasting with ensemble methods
- **CapacityPlanningEngine**: Production capacity planning and scheduling

### Data Loading Architecture (API-First)
The system uses a comprehensive API-first data loading strategy:
1. **ERP Wrapper Service** (`erp-wrapper/`): FastAPI proxy handling eFab authentication and session management
2. **APIDataLoader** (`src/core/beverly_comprehensive_erp.py`): Primary data loading via wrapper service APIs
3. **ParallelDataLoader** (`src/data_loaders/parallel_data_loader.py`): 4x speed fallback with concurrent file loading
4. **UnifiedCacheManager** (`src/utils/cache_manager.py`): Memory + Redis caching with TTL

**Data Flow Priority**: API → Parallel Loader → File System → Error

### ERP Wrapper Service (NEW as of Sep 2025)
**Location**: `erp-wrapper/` - FastAPI microservice for eFab ERP integration
- **Port**: 8000 (required for main ERP to function)
- **Purpose**: Handles authentication, session management, and API proxying for eFab ERP
- **Features**: Auto-login, 5-minute caching, retry logic, health monitoring

**Critical Endpoints**:
- `/api/yarn/active` - Real-time yarn inventory (1200+ items)
- `/api/knit-orders` - Production orders 
- `/api/sales-orders` - Sales data
- `/api/greige/g00`, `/api/greige/g02` - Production stages
- `/api/finished/i01`, `/api/finished/f01` - QC and finished goods

**Architecture**: Browser ↔ Main ERP (5006) ↔ Wrapper (8000) ↔ eFab APIs

### API Consolidation Architecture (NEW as of Aug 2025)
- **45+ deprecated endpoints** automatically redirect to consolidated endpoints
- **Redirect middleware** in `intercept_deprecated_endpoints()` 
- **JavaScript compatibility layer** in dashboard for client-side handling
- **Feature flags** in `/src/config/feature_flags.py` for rollback control
- Monitor with `/api/consolidation-metrics`

### Service Modules
```
src/
├── core/                  # Main application & API data loading
│   └── beverly_comprehensive_erp.py  # APIDataLoader + Flask app
├── services/              # Modular business services
│   ├── inventory_analyzer_service.py
│   ├── inventory_pipeline_service.py
│   ├── sales_forecasting_service.py
│   └── capacity_planning_service.py
├── yarn_intelligence/     # Yarn management & substitution
│   ├── yarn_intelligence_enhanced.py
│   ├── yarn_substitution_intelligent.py
│   └── yarn_interchangeability_analyzer.py
├── production/           # Production planning
│   ├── six_phase_planning_engine.py
│   ├── enhanced_production_pipeline.py
│   └── enhanced_production_suggestions_v2.py
├── forecasting/         # ML forecasting
│   ├── enhanced_forecasting_engine.py
│   ├── forecast_accuracy_monitor.py
│   └── forecast_auto_retrain.py
├── config/              # Configuration management
│   └── ml_config.py     # ML model configurations
└── scripts/             # Utility scripts
    ├── day0_emergency_fixes.py    # Critical data fixes
    ├── ml_backtest.py              # ML backtesting
    └── ml_training_pipeline.py    # Automated training

erp-wrapper/             # FastAPI microservice (Port 8000)
├── app/
│   ├── main.py         # FastAPI application
│   ├── efab_client.py  # eFab API client
│   └── auth.py         # Session management
└── docker-compose.yml  # Container orchestration
```

## Data Flow & Field Mappings

### Critical Data Paths
```
Primary: /mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/
8-28-2025 subfolder: Contains latest eFab_Knit_Orders.csv and other current data
```

### Key Data Files & Their Purpose
1. **yarn_inventory.csv** - CORRECTED: Now uses Yarn_ID_Master.csv with proper yarn IDs (18000-19000 range)
2. **BOM_updated.csv** - Bill of Materials (28,653 entries mapping styles to yarns)
3. **eFab_Knit_Orders.csv** - 194 production orders (154 assigned, 40 unassigned)
4. **QuadS_greigeFabricList_(1).xlsx** - Style to Work Center mappings (columns C=style, D=work_center)
5. **Machine Report fin1.csv** - Machine to Work Center mappings (WC column=machine patterns, MACH=machine IDs)
6. **Sales Activity Report.csv** - Historical sales data for forecasting
7. **Yarn_ID_Master.csv** - MASTER yarn inventory file with correct yarn IDs and planning balances

### Work Center & Machine Structure
- **Work Center Pattern**: `x.xx.xx.X` where:
  - First digit = knit construction
  - Second pair = machine diameter  
  - Third pair = needle cut
  - Letter = type (F/M/C/V etc.)
  - Example: `9.38.20.F` = construction 9, diameter 38, needle 20, type F
- **Machine IDs**: Simple numeric values (e.g., 161, 224, 110)
- **Machine Pattern Mapping**: Each work center can have multiple machines

### Column Name Handling
The system handles multiple column name variations:
- 'Planning Balance' vs 'Planning_Balance' 
- 'Desc#' vs 'desc_num' vs 'YarnID'
- 'fStyle#' vs 'Style#' for style mapping
- 'Balance (lbs)' may contain commas that need cleaning

### Production Flow Stages
```
G00 (Greige) → G02 (Greige Stage 2) → I01 (QC) → F01 (Finished Goods)
```

## API Endpoints (Post-Consolidation)

### Critical Dashboard APIs
All working at `/api/`:
- `production-planning` - Production schedule with parameter support
- `inventory-intelligence-enhanced` - Inventory analytics with views
- `ml-forecast-detailed` - ML predictions with format options
- `inventory-netting` - Multi-level netting calculations
- `comprehensive-kpis` - Complete KPI metrics
- `yarn-intelligence` - Yarn analysis with shortage detection
- `production-suggestions` - AI-powered recommendations
- `po-risk-analysis` - Risk assessment
- `production-pipeline` - Real-time production flow
- `yarn-substitution-intelligent` - ML-based substitutions
- `production-recommendations-ml` - ML recommendations
- `knit-orders` - Order management
- `machine-assignment-suggestions` - Suggests machines for unassigned orders using QuadS mappings
- `factory-floor-ai-dashboard` - Machine planning data with work center groupings

### Consolidated Endpoint Parameters
```
GET /api/inventory-intelligence-enhanced?view=summary&analysis=shortage&realtime=true
GET /api/ml-forecast-detailed?detail=full&format=report&horizon=90
GET /api/yarn-intelligence?analysis=shortage&forecast=true
GET /api/production-planning?view=orders&forecast=true
```

## Dashboard Access

Primary dashboard: http://localhost:5006/consolidated

The dashboard includes an API compatibility layer that automatically handles deprecated endpoints.

## Common Issues & Solutions

### Port Issues
```bash
# Main ERP server runs on port 5006, wrapper service on port 8000
lsof -i :5006  # Main ERP
lsof -i :8000  # ERP wrapper service

# Kill if needed
lsof -i :5006 | grep LISTEN | awk '{print $2}' | xargs kill -9
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

### ERP Wrapper Service Issues
If yarn data shows 0 items or API integration fails:
1. **Check wrapper service health**: `curl http://localhost:8000/health`
2. **Verify authentication**: Check eFab credentials in `erp-wrapper/.env`
3. **Test direct API**: `curl http://localhost:8000/api/yarn/active`
4. **Check logs**: `docker-compose logs -f` (if using Docker)
5. **Restart wrapper**: `cd erp-wrapper/ && uvicorn app.main:app --port 8000`
6. **Force main ERP to use API**: Look for `[API] Loaded yarn inventory via API` in startup logs

### Data Loading Issues
1. **API-FIRST ARCHITECTURE (Sep 2025)**: System now primarily loads data via ERP wrapper APIs
2. Check wrapper service: `curl -s http://localhost:8000/health`
3. Check API data: `curl -s http://localhost:8000/api/yarn/active | head -c 1000`
4. Clear cache: `rm -rf /tmp/bki_cache/*`
5. Reload data: `curl http://localhost:5006/api/reload-data`
6. Restart services:
   ```bash
   # Kill both services
   pkill -f "python3.*beverly" && pkill -f "uvicorn.*app.main"
   # Start wrapper first, then main ERP
   cd erp-wrapper/ && uvicorn app.main:app --port 8000 &
   cd .. && python3 src/core/beverly_comprehensive_erp.py
   ```

### eFab API Integration 
**Corrected Domains & Endpoints (as of Sep 2025):**
- eFab ERP: `https://efab.bkiapps.com` (NOT bklapps.com)
- QuadS: `https://quads.bkiapps.com`

**Active API Endpoints:**
```
# eFab APIs
https://efab.bkiapps.com/api/report/yarn_demand_ko
https://efab.bkiapps.com/api/report/yarn_demand
https://efab.bkiapps.com/api/yarn-po
https://efab.bkiapps.com/api/report/yarn_expected
https://efab.bkiapps.com/api/yarn/active

# QuadS APIs
https://quads.bkiapps.com/api/styles/greige/active
https://quads.bkiapps.com/api/styles/finished/active
```

**Authentication Required:** All endpoints require session cookies for access.

### Column Name Errors (RESOLVED)
**FIXED (Sep 2025)**: Column naming issues resolved with yarn data correction:
- System now handles both 'Planning Balance' and 'Planning_Balance' formats
- UTF-8 BOM characters removed from headers
- Currency formatting standardized ($ signs and commas cleaned)
- Use `python3 fix_yarn_data.py` for any new data formatting issues

### Day 0 Fixes Not Loading
If you see `[DAY0] Emergency fixes not available: No module named 'scripts'`:
- This is expected - Day 0 fixes work standalone but aren't integrated
- Run fixes manually: `python3 scripts/day0_emergency_fixes.py --health-check`
- Core functionality still works through existing code

### ML Training Data Format Issues
If ML training fails with price format errors:
- Sales data contains "$" prefixes in price columns
- Preprocess data before training: `df['price'] = df['price'].str.replace('$', '').astype(float)`
- Use `Unit Price` or `Line Price` columns from sales data

### API Consolidation Rollback
If issues arise with consolidated APIs:
```python
# In /src/config/feature_flags.py
FEATURE_FLAGS = {
    "api_consolidation_enabled": False,  # Disable consolidation
    "redirect_deprecated_apis": False,   # Stop redirects
}
```
Then restart the server.

### Machine Planning Dashboard Issues
If Machine Planning tab shows errors:
- Check `fetchAPI` function is used (not `fetchWithErrorHandling`)
- Ensure NaN values are handled in API responses (convert to null/0)
- Work centers display full pattern (e.g., `9.38.20.F`), not just first digit
- Machine workloads are loaded from `eFab_Knit_Orders.csv` Machine column

### JSON Serialization Errors
If APIs return NaN or invalid JSON:
- Wrap date fields: `str(value) if pd.notna(value) else ''`
- Convert numeric fields: `float(value) if pd.notna(value) else 0`
- Use `int()` for counts to avoid float serialization

## Testing Requirements

### Coverage Targets
- Overall: 80% minimum
- Critical paths: 90% minimum
- Focus areas:
  - Planning Balance calculations (negative Allocated values)
  - Style mapping (fStyle# ↔ Style#)
  - Yarn shortage detection
  - API redirects and parameter handling

### Test Organization
```
tests/
├── unit/              # Business logic
├── integration/       # API endpoints  
├── e2e/              # Workflows
├── performance/      # Load testing
└── test_api_consolidation.py  # API consolidation tests
```

## Performance Metrics

### Current Benchmarks
- Data Load: 1-2 seconds with parallel loading
- API Response: <200ms for most endpoints
- Dashboard Load: <3 seconds full render
- Cache Hit Rate: 70-90% typical

### System Capacity
- Yarn Items: 1,200+ tracked (5x increase via API integration)
- BOM Entries: 28,653+ 
- Sales Records: 6,946+ (live API data)
- Production Orders: 195+ active (real-time tracking)

## ML Models

### Available Models
- ARIMA, Prophet (time series)
- LSTM (deep learning)
- XGBoost (gradient boosting)
- Ensemble (combines all)

### Accuracy Targets
- 9-week horizon: 90% accuracy
- 30-day forecast: 95% accuracy
- Fallback chain: Ensemble → Single model → Statistical → Last known

## Dashboard Lock Policy

⚠️ **DASHBOARD UI IS LOCKED** - No visual/style changes allowed to:
- `web/consolidated_dashboard.html`
- Any styling in `beverly_comprehensive_erp.py`

Allowed changes:
- ✅ Fix API calls and data processing
- ✅ Improve error handling
- ✅ Optimize performance
- ❌ NO color, layout, or style changes

## Dependencies

Core packages (see requirements.txt for full list):
- flask>=3.0.0
- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- prophet>=1.1.0
- xgboost>=2.0.0
- openpyxl>=3.1.0
- redis>=4.5.0

Install: `pip install -r requirements.txt` or `make install`