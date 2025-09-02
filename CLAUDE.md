# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

Beverly Knits ERP v2 - Production-ready textile manufacturing ERP with real-time inventory intelligence, ML-powered forecasting, and 6-phase supply chain optimization. The system manages 1,199 yarn items, 28,653 BOM entries, and 194 active production orders.

**Current System Health**: 75% operational (Day 0 fixes implemented, Phase 3 testing complete, Phase 4 ML configured)

## Primary Commands

### Server Operations
```bash
# Start server (Port 5006, NOT 5005 or 5003)
python3 src/core/beverly_comprehensive_erp.py

# Kill existing server if port conflict
pkill -f "python3.*beverly"
lsof -i :5006 | grep LISTEN | awk '{print $2}' | xargs kill -9

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

# Debug data loading
curl -s http://localhost:5006/api/debug-data | python3 -m json.tool

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

### Data Loading Architecture
The system uses a multi-tier data loading strategy:
1. **OptimizedDataLoader** (`src/data_loaders/optimized_data_loader.py`): 100x+ speed with caching
2. **ParallelDataLoader** (`src/data_loaders/parallel_data_loader.py`): 4x speed with concurrent loading
3. **UnifiedCacheManager** (`src/utils/cache_manager.py`): Memory + Redis caching with TTL

### API Consolidation Architecture (NEW as of Aug 2025)
- **45+ deprecated endpoints** automatically redirect to consolidated endpoints
- **Redirect middleware** in `intercept_deprecated_endpoints()` 
- **JavaScript compatibility layer** in dashboard for client-side handling
- **Feature flags** in `/src/config/feature_flags.py` for rollback control
- Monitor with `/api/consolidation-metrics`

### Service Modules
```
src/
├── services/               # Modular business services
│   ├── inventory_analyzer_service.py
│   ├── inventory_pipeline_service.py
│   ├── sales_forecasting_service.py
│   └── capacity_planning_service.py
├── yarn_intelligence/      # Yarn management & substitution
│   ├── yarn_intelligence_enhanced.py
│   ├── yarn_substitution_intelligent.py
│   └── yarn_interchangeability_analyzer.py
├── production/            # Production planning
│   ├── six_phase_planning_engine.py
│   ├── enhanced_production_pipeline.py
│   └── enhanced_production_suggestions_v2.py
├── forecasting/          # ML forecasting
│   ├── enhanced_forecasting_engine.py
│   ├── forecast_accuracy_monitor.py
│   └── forecast_auto_retrain.py
├── config/               # Configuration management
│   └── ml_config.py      # ML model configurations
└── scripts/              # Utility scripts
    ├── day0_emergency_fixes.py    # Critical data fixes
    ├── ml_backtest.py              # ML backtesting
    └── ml_training_pipeline.py    # Automated training
```

## Data Flow & Field Mappings

### Critical Data Paths
```
Primary: /mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/
Fallback: /mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/
```

### Key Data Files
1. **yarn_inventory.xlsx** - Contains 'Planning Balance' column (with space)
2. **BOM_updated.csv** - Bill of Materials (preferred over Style_BOM.csv)
3. **eFab_Knit_Orders.xlsx** - Active production orders
4. **Sales Activity Report.csv** - Historical sales data

### Column Name Handling
The system handles multiple column name variations:
- 'Planning Balance' vs 'Planning_Balance' 
- 'Desc#' vs 'desc_num' vs 'YarnID'
- 'fStyle#' vs 'Style#' for style mapping

### Production Flow Stages
```
G00 (Greige) → G02 (Greige Stage 2) → I01 (QC) → F01 (Finished Goods)
```

## API Endpoints (Post-Consolidation)

### Critical Dashboard APIs (12 endpoints)
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
# Server runs on port 5006 (documentation may show 5005 or 5003)
lsof -i :5006
# Kill if needed
lsof -i :5006 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

### Data Loading Issues
1. Check file exists: `ls -la "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/yarn_inventory.xlsx"`
2. Clear cache: `rm -rf /tmp/bki_cache/*`
3. Reload data: `curl http://localhost:5006/api/reload-data`
4. Restart server: `pkill -f "python3.*beverly" && python3 src/core/beverly_comprehensive_erp.py`

### Column Name Errors
The system handles multiple column name formats. If you see "Planning Balance" errors:
- Check both 'Planning Balance' and 'Planning_Balance' 
- Use hasattr() checks before accessing DataFrame columns
- Implement fallback logic for column variations

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
- Yarn Items: 1,198+ tracked
- BOM Entries: 28,653+ 
- Sales Records: 10,338+
- Production Orders: 221+ active

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