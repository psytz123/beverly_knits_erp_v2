# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

Beverly Knits ERP System - A comprehensive textile manufacturing ERP with real-time inventory intelligence, 6-phase supply chain planning, and ML-powered forecasting. The system manages the complete production pipeline from yarn inventory to finished goods, with sophisticated demand planning and optimization capabilities.

## ⚠️ DASHBOARD LOCK STATUS: ACTIVE ⚠️
**LOCKED AS OF: 2025-08-10 19:00**

### Dashboard Files - LOCKED FROM UI CHANGES:
- **consolidated_dashboard.html** - Primary dashboard (NO visual/layout/style changes)
- **unified_dashboard.html**, **production_dashboard.html**, **dashboard_live.html**, **integrated_dashboard.html** - Legacy dashboards
- **beverly_comprehensive_erp.py** - UI/styling endpoints are LOCKED

### ONLY ALLOWED CHANGES:
- ✅ Fix button functionality and API calls
- ✅ Improve data calculations and processing
- ✅ Enhance error handling and recovery
- ✅ Optimize backend performance
- ❌ NO changes to colors, layouts, styles, or visual elements

## Primary Commands

### Quick Start
```bash
# Start server (Port 5005 - NOT 5003)
python3 beverly_comprehensive_erp.py

# Kill existing server if port conflict
pkill -f "python3.*beverly"
lsof -i :5005 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

### Testing
```bash
# Run all tests with coverage
pytest tests/ -v --cov=beverly_comprehensive_erp --cov-report=html

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests
pytest -m e2e           # End-to-end tests
pytest -n auto          # Parallel execution

# Run single test
pytest tests/unit/test_inventory.py::test_yarn_shortage_calculation -v
```

### Code Quality (from bkai/ directory)
```bash
make lint          # Run linters (ruff, flake8, pylint)
make format        # Format code (black, isort)
make type-check    # Run mypy type checking
make test-cov      # Run tests with coverage report
```

### Data Management & Debugging
```bash
# Clear cache when data issues occur
rm -rf /tmp/bki_cache/*

# Force data reload via API
curl -s http://localhost:5005/api/reload-data

# Debug data loading
curl -s http://localhost:5005/api/debug-data

# Check yarn inventory file exists
ls -la "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5/yarn_inventory*.xlsx"
```

## High-Level Architecture

### Core Monolithic Application (`beverly_comprehensive_erp.py`)
Single Flask application (7000+ lines) containing four main classes:
- `InventoryAnalyzer` (Line 267): Core inventory analysis engine
- `InventoryManagementPipeline` (Line 327): Pipeline orchestration
- `SalesForecastingEngine` (Line 495): ML-powered forecasting
- `CapacityPlanningEngine` (Line 1653): Production capacity planning

### Key Supporting Modules
- `optimized_data_loader.py`: High-performance data loading with caching (100x+ speed improvement)
- `parallel_data_loader.py`: Concurrent data loading (2.31s for 52k records)
- `six_phase_planning_engine.py`: Supply chain planning orchestration
- `yarn_intelligence_enhanced.py`: Advanced yarn shortage analysis
- `yarn_interchangeability_analyzer.py`: ML-based yarn substitution recommendations
- `enhanced_production_pipeline.py`: Real-time production flow tracking
- `cache_manager.py`: Unified caching with memory/Redis support

### Performance Optimization Components
- 4-phase optimization roadmap implemented achieving 75% faster data loading
- Parallel processing for data ingestion
- Client-side request batching and lazy loading
- Database connection pooling
- Real-time ML pipeline with online learning capabilities

## Data Flow & Critical Field Mappings

### Primary Data Directory
```
/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/
```
Note: Data files are distributed across dated subdirectories (08-04, 08-06, 08-09) and the "5" subdirectory

### Key Data Files & Their Roles
1. **yarn_inventory*.xlsx** ⚠️ CRITICAL
   - Located in dated subdirectories (e.g., 08-06/yarn_inventory (1).xlsx)
   - Contains `Planning_Balance` column (may appear as `Planning_Ballance` with typo)
   - Fields: Desc#, Planning_Balance, Allocated, On_Order, Theoretical_Balance, Consumed

2. **Sales Activity Report (6).csv** - Sales transactions in "5" directory (Style#)
3. **eFab_Styles_*.xlsx** - MASTER MAPPING (fStyle# ↔ Style#) - various dates
4. **eFab_Knit_Orders_*.xlsx** - Production orders in dated subdirectories
5. **Style_BOM.csv** or **BOM_updated.csv** - Bill of Materials in "5" directory
6. **eFab_Inventory_[F01|G00|G02|I01]*.xlsx** - Stage inventories in dated subdirectories

### Production Flow Stages
```
G00 (Greige Stage 1) → G02 (Greige Stage 2) → I01 (QC Inspection) → F01 (Finished Goods)
```

### Critical Field Mappings
- **Style References**: `fStyle#` in inventory files ↔ `Style#` in production/BOM
- **Yarn IDs**: Standardized to `Desc#` (was: Yarn_ID, YarnID, etc.)
- **Planning Balance Formula**: `Planning_Balance = Theoretical_Balance + Allocated + On_Order`
  - NOTE: Allocated values are ALREADY NEGATIVE in data files

## Critical API Endpoints

### Yarn & Inventory Intelligence
- `GET /api/yarn-intelligence` - Comprehensive yarn analysis with substitutions
- `GET /api/inventory-intelligence-enhanced` - Enhanced inventory metrics
- `GET /api/yarn-aggregation` - Interchangeable yarn groups
- `GET /api/yarn-substitution-intelligent` - ML-based substitution recommendations
- `GET /api/yarn-forecast-shortages` - BOM-based forecasted shortages

### Planning & Production
- `GET /api/six-phase-planning` - Execute 6-phase planning engine
- `GET /api/production-pipeline` - Real-time production flow
- `POST /api/fabric/yarn-requirements` - Calculate yarn from fabric specs
- `GET /api/production-planning` - Production schedule and capacity

### ML & Forecasting
- `GET /api/ml-forecast-report` - ML forecast summary
- `GET /api/ml-forecast-detailed` - Detailed predictions
- `POST /api/retrain-ml` - Trigger model retraining

### Performance & Health
- `GET /api/health` - Health check
- `GET /api/debug-data` - Debug data loading
- `GET /api/reload-data` - Force cache refresh
- `GET /api/cache-stats` - Cache performance metrics

## Dashboard Access

After starting server:
1. **Consolidated Dashboard**: http://localhost:5005/consolidated (⭐ PRIMARY)
2. **Test Production Functions**: http://localhost:5005/test_production_functions.html

### Production Tab Manual Loading (if needed)
```javascript
// In browser console (F12)
loadForecastedOrders()
loadAIProductionSuggestions()
loadCurrentYarnShortages()
loadForecastedShortages()
loadYarnAlternatives()
```

## Weekly Demand Calculation Logic

The system calculates weekly demand using a tiered approach:
1. **From Consumed Data**: Monthly consumption / 4.3 weeks (if available)
2. **From Allocated Amount**: Allocated quantity / 8-week production cycle (if no consumption)
3. **Minimal Default**: 10 lbs/week (only when no data available)

Note: `Consumed` values in data files are negative (amount used), converted to positive for calculations.

## Common Issues & Solutions

### Data Not Loading
1. Check file exists: `ls -la "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5/yarn_inventory*.xlsx"`
2. Clear cache: `rm -rf /tmp/bki_cache/*`
3. Restart server: `pkill -f "python3.*beverly" && python3 beverly_comprehensive_erp.py`

### Port Conflicts
- Server runs on port 5005 (NOT 5003 as some docs state)
- Check port: `lsof -i :5005`
- Force kill: `lsof -i :5005 | grep LISTEN | awk '{print $2}' | xargs kill -9`

### Planning Balance Issues
- Ensure using correct formula (Allocated is already negative in files)
- Check column name variations: `Planning_Balance` or `Planning_Ballance`
- Planning Balance < 0 means need to order, NOT out of stock
- Theoretical Balance is actual physical inventory on hand

## Performance Targets & Metrics

### Current Performance (After Optimization)
- Data Load Time: ~1-2 seconds with parallel loading
- Cache Hit Rate: Variable depending on cache state
- API Response Time: Depends on data availability
- Dashboard Refresh: Varies based on data size
- Concurrent Users: Flask single-threaded by default

### System Capacity
- Yarn Inventory: 1,198 items tracked
- Style BOM: 28,653 entries
- Sales Transactions: 10,338+ records
- Production Orders: 221+ active orders

## ML Models Available

- **Forecasting**: ARIMA, Prophet, LSTM, XGBoost, Ensemble
- **Accuracy**: 90-95% potential with ensemble methods
- **Fallback Strategy**: Ensemble → Single model → Statistical → Last known value
- **Online Learning**: River ML for real-time adaptation

## Testing Strategy

### Test Coverage Requirements
- Target: 80% for critical paths
- Focus areas:
  - Planning Balance formula with negative Allocated values
  - fStyle# ↔ Style# mapping completeness
  - Unit conversions (yards ↔ pounds using QuadS data)
  - Weekly demand calculations from consumed/allocated data

### Test Files Organization
```
tests/
├── unit/           # Business logic tests
├── integration/    # API endpoint tests
├── e2e/           # Workflow tests
└── performance/   # Load and speed tests
```

## Dependencies

Core requirements (from requirements.txt):
- flask>=3.0.0
- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- prophet>=1.1.0 (forecasting)
- xgboost>=2.0.0 (optional ML)
- openpyxl>=3.1.0 (Excel files)
- redis>=4.5.0 (caching)
- celery>=5.3.0 (async tasks)

Install: `pip install -r requirements.txt`