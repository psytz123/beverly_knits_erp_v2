# Beverly Knits ERP System Overhaul - Project Handoff Document

## 🚨 CRITICAL INFORMATION - READ FIRST

### System Access
- **Main Application**: `/mnt/c/finalee/Agent-MCP-1-ddd/Agent-MCP-1-dd/BKI_comp/beverly_comprehensive_erp.py`
- **Port**: 5005 (NOT 5003)
- **Start Command**: `python3 beverly_comprehensive_erp.py`
- **Kill Command**: `pkill -f "python3.*beverly"`

### Current Critical Issues
1. **System crashes frequently** - memory leaks, data loading failures
2. **Performance degradation** - API timeouts, slow page loads
3. **Data inconsistencies** - Planning Balance calculations incorrect
4. **SharePoint sync broken** - requires manual intervention
5. **Monolithic architecture** - 7000+ line single file

### Business Constraints
- **MUST preserve ALL custom business calculations** (extensive throughout code)
- **System must remain operational during refactoring** (parallel run required)
- **Forecast accuracy is CRITICAL** - target 90%+ at 9-week horizon
- **Timeline**: 2 months to production
- **Users**: Currently 1 executive, adding purchasing/planning users soon

### DO NOT TOUCH
- **Dashboard UI/Styling** - Files are LOCKED from visual changes
- **Custom Business Logic** - Extract but NEVER modify calculations
- **API Endpoints** - Must maintain backward compatibility

---

## 📋 PROJECT OVERVIEW

### Goal
Transform an unstable, monolithic ERP system into a stable, modular, production-ready application while maintaining 100% business logic compatibility and improving forecast accuracy to 90%+ at 9-week horizon.

### Current State
- **Architecture**: 7000+ line monolithic Flask application
- **Database**: SQLite (production) - causing concurrency issues
- **Data Pipeline**: 3 competing implementations, unclear which works
- **Testing**: Minimal coverage, many test files but incomplete
- **Performance**: Degrading under load, memory leaks
- **Codebase**: 30+ duplicate dashboard files, redundant implementations

### Target State
- **Architecture**: Modular service-based architecture
- **Database**: PostgreSQL with connection pooling
- **Data Pipeline**: Single unified, performant implementation
- **Testing**: 80%+ coverage with automated CI/CD
- **Performance**: <2s page load, <200ms API response
- **Codebase**: Clean, organized, documented

---

## 🎯 EXECUTION PLAN

### PHASE 1: STABILIZATION (Days 1-10)
**Goal: Stop the bleeding - fix critical stability issues**

#### Day 1-2: Performance Analysis
```python
# TASK: Create performance_analysis.py
"""
Requirements:
1. Profile all API endpoints in beverly_comprehensive_erp.py
2. Identify memory leaks using tracemalloc
3. Test all 3 data loaders with actual data from /ERP Data/5/
4. Output: performance_report.json with bottlenecks ranked by impact
"""

# Key files to analyze:
- beverly_comprehensive_erp.py (main app)
- optimized_data_loader.py
- parallel_data_loader.py
- cache_manager.py

# Run profiling on these critical endpoints:
- /api/yarn-intelligence
- /api/execute-planning
- /api/ml-forecast-detailed
```

#### Day 3-4: Database Migration
```python
# TASK: Create migrate_to_postgresql.py
"""
CRITICAL: Must preserve ALL business logic
1. Export all SQLite data
2. Create PostgreSQL schema with indexes on:
   - Desc# (yarn ID)
   - Style# (style references)
   - Planning_Balance
3. Implement connection pooling (min=5, max=20)
4. Add rollback capability
"""

# Database connection pattern to implement:
from contextlib import contextmanager
import psycopg2.pool

class DatabaseManager:
    def __init__(self):
        self.pool = psycopg2.pool.ThreadedConnectionPool(5, 20, **db_config)
    
    @contextmanager
    def get_connection(self):
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)
```

#### Day 5-6: Unified Data Loader
```python
# TASK: Create unified_data_loader.py
"""
Combine best features from all 3 loaders:
- Parallel processing from parallel_data_loader.py
- Caching from optimized_data_loader.py
- Error handling from standard loader

MUST HANDLE:
- Planning_Balance vs Planning_Ballance (typo in data)
- Negative Allocated values (already negative in files)
- Missing columns gracefully
"""

# Critical data files to test with:
/ERP Data/5/yarn_inventory (4).xlsx  # Has Planning_Balance column
/ERP Data/5/Style_BOM.csv           # Bill of materials
Sales Activity Report.csv           # Sales data
```

#### Day 7-8: Fix Critical Bugs
```python
# TASK: Create bug_fixes.py to patch beverly_comprehensive_erp.py
"""
Fix these specific issues:
1. Planning Balance Formula:
   CORRECT: Planning_Balance = Theoretical_Balance + Allocated + On_Order
   (Allocated is ALREADY negative in files)

2. Memory leaks in data loading:
   - Add explicit garbage collection
   - Limit DataFrame sizes in memory
   - Clear cache periodically

3. Add timeout handling to all API endpoints
"""
```

#### Day 9-10: Validation
```bash
# TASK: Create validation suite
# Test that all critical endpoints still work:
curl http://localhost:5005/api/yarn-intelligence
curl http://localhost:5005/api/execute-planning
curl http://localhost:5005/api/ml-forecast-detailed

# Verify business calculations unchanged
python validate_business_logic.py
```

### PHASE 2: MODULARIZATION (Days 11-20)
**Goal: Break apart monolith WITHOUT changing functionality**

#### Day 11-13: Extract Core Services
```python
# TASK: Extract classes from beverly_comprehensive_erp.py

# Create services/inventory_analyzer_service.py
"""
Line 359-418: class InventoryAnalyzer
PRESERVE EXACTLY - all methods, all calculations
"""

# Create services/sales_forecasting_service.py
"""
Line 587-1792: class SalesForecastingEngine
CRITICAL: Keep ALL ML models and fallback logic
"""

# Create services/capacity_planning_service.py
"""
Line 2052-2144: class CapacityPlanningEngine
"""

# Create services/business_rules.py
"""
Extract ALL custom calculations:
- Weekly demand calculations
- Planning balance formulas
- Yarn substitution logic
- Unit conversions (yards to pounds)
"""
```

#### Day 14-16: Create Service Layer
```python
# TASK: Refactor beverly_comprehensive_erp.py
"""
1. Replace embedded classes with imports
2. Create service initialization pattern:

from services.inventory_analyzer_service import InventoryAnalyzer
from services.sales_forecasting_service import SalesForecastingEngine

class ERPServiceManager:
    def __init__(self):
        self.inventory = InventoryAnalyzer()
        self.forecasting = SalesForecastingEngine()
        # ... etc

3. Update all routes to use service manager
4. TEST EVERY ENDPOINT after changes
"""
```

#### Day 17-20: Integration Testing
```python
# TASK: Create integration test suite
"""
Test critical workflows end-to-end:
1. Data upload -> Processing -> Storage
2. Forecast request -> ML models -> Results
3. Planning execution -> 6-phase engine -> Output
4. Yarn shortage -> Substitution -> Recommendations
"""
```

### PHASE 3: FORECASTING ENHANCEMENT (Days 21-30)
**Goal: Achieve 90%+ accuracy at 9-week horizon**

#### Day 21-23: ML Model Optimization
```python
# TASK: Create enhanced_forecasting_engine.py
"""
Current models to optimize:
- Prophet (best for seasonality)
- XGBoost (complex patterns)
- ARIMA (baseline)

Implementation requirements:
1. Ensemble approach with weighted voting
2. 9-week forecast horizon focus
3. Weekly automatic retraining
4. Both historical AND forward-looking data

Key data sources:
- Historical: 'Consumed' column (negative values = usage)
- Forward: Sales orders and committed demand
"""

# Model configuration for 9-week accuracy:
FORECAST_CONFIG = {
    'horizon_weeks': 9,
    'retrain_frequency': 'weekly',
    'min_accuracy_threshold': 0.90,
    'ensemble_weights': {
        'prophet': 0.4,
        'xgboost': 0.35,
        'arima': 0.25
    }
}
```

#### Day 24-26: Dual Forecast System
```python
# TASK: Implement hybrid forecasting
"""
Create two parallel forecasts:

1. Historical-based (from consumed data):
   - Use 'Consumed' column from yarn_inventory
   - Monthly consumption / 4.3 = weekly demand
   
2. Order-based (from forward orders):
   - Sales Activity Report data
   - Committed orders from eFab_Knit_Orders
   
3. Combine with confidence weighting:
   final_forecast = (historical * 0.6) + (order_based * 0.4)
"""
```

#### Day 27-30: Accuracy Validation
```python
# TASK: Create forecast_accuracy_monitor.py
"""
1. Track predictions vs actuals
2. Calculate MAPE, RMSE, MAE
3. Auto-adjust ensemble weights based on performance
4. Alert if accuracy drops below 90%
"""
```

### PHASE 4: CLEANUP & ORGANIZATION (Days 31-40)
**Goal: Remove technical debt and organize codebase**

#### Day 31-33: Remove Duplicates
```python
# TASK: Create cleanup_script.py
"""
Files to consolidate/remove:
- Keep ONLY consolidated_dashboard.html
- Remove 30+ other dashboard*.html files
- Remove backup and test files
- Consolidate requirements*.txt files

CRITICAL: Before deleting:
1. Check for unique functionality
2. Update all imports/references
3. Create backup in /cleanup_backup_[timestamp]/
"""
```

#### Day 34-36: Restructure Project
```bash
# TASK: Reorganize directory structure
# Target structure:
beverly_erp/
├── api/
│   ├── routes/          # Organized by domain
│   └── middleware/      # Auth, validation, etc
├── services/
│   ├── core/           # Business logic services
│   └── integrations/   # External system connectors
├── models/
│   ├── database/       # SQLAlchemy models
│   └── domain/         # Business domain models
├── utils/
│   └── helpers/        # Utility functions
├── config/
│   └── settings.py     # Configuration management
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── main.py            # Entry point (renamed from beverly_comprehensive_erp.py)
```

#### Day 37-40: Documentation
```python
# TASK: Generate documentation
"""
1. API documentation using OpenAPI/Swagger
2. Business logic documentation
3. Deployment guide
4. Troubleshooting guide
"""
```

### PHASE 5: TESTING & VALIDATION (Days 41-50)
**Goal: Comprehensive test coverage and validation**

#### Day 41-45: Test Suite Development
```python
# TASK: Create comprehensive test suite
"""
Required test coverage:
1. Unit tests for ALL business calculations
2. Integration tests for critical workflows
3. Performance benchmarks
4. Load testing (50 concurrent users)

Focus areas:
- Planning Balance calculations
- Forecast accuracy
- Yarn substitution logic
- Data pipeline reliability
"""
```

#### Day 46-50: Final Validation
```python
# TASK: Complete system validation
"""
1. Run full regression test suite
2. Validate all business calculations
3. Performance benchmarking
4. Security audit
5. Create rollback plan
"""
```

### PHASE 6: DEPLOYMENT PREPARATION (Days 51-60)
**Goal: Production-ready deployment**

#### Day 51-55: Infrastructure Setup
```yaml
# TASK: Create production deployment configuration
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5005:5005"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/beverly_erp
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
      
  db:
    image: postgres:14
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  redis:
    image: redis:7
    
  nginx:
    image: nginx
    ports:
      - "80:80"
      - "443:443"
```

#### Day 56-60: Production Cutover
```bash
# TASK: Production deployment plan
"""
1. Database migration in maintenance window
2. Deploy new application alongside old
3. Gradual traffic migration
4. Monitor for issues
5. Full cutover after validation
"""
```

---

## 🔧 TECHNICAL DETAILS

### Key Business Calculations to Preserve

```python
# 1. Planning Balance (CRITICAL - often wrong)
# CORRECT FORMULA:
planning_balance = theoretical_balance + allocated + on_order
# Note: Allocated is ALREADY NEGATIVE in data files

# 2. Weekly Demand Calculation
if consumed_data_exists:
    weekly_demand = abs(monthly_consumed) / 4.3
elif allocated_exists:
    weekly_demand = allocated_qty / 8  # 8-week production cycle
else:
    weekly_demand = 10  # minimal default

# 3. Yarn Substitution Score
similarity_score = (
    color_match_weight * 0.3 +
    composition_match_weight * 0.4 +
    weight_match_weight * 0.3
)
```

### Critical Data Mappings

```python
# Style Reference Mapping
# fStyle# (in inventory files) ↔ Style# (in production/BOM)

# Yarn ID Standardization
# All variations map to 'Desc#':
yarn_id_mappings = {
    'Yarn_ID': 'Desc#',
    'YarnID': 'Desc#',
    'Yarn ID': 'Desc#',
    'Desc': 'Desc#'
}

# Column Name Variations (handle typos)
column_mappings = {
    'Planning_Ballance': 'Planning_Balance',  # Common typo
    'Theoratical_Balance': 'Theoretical_Balance'
}
```

### Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| Page Load | 5-10s | <2s |
| API Response | 500ms-2s | <200ms |
| Data Load (52k records) | 10-15s | <3s |
| Memory Usage | 4-6GB | <2GB |
| Concurrent Users | 5-10 | 50+ |
| Forecast Accuracy (9wk) | 70-80% | >90% |

### Testing Checklist

- [ ] All API endpoints return correct data
- [ ] Planning Balance calculations are accurate
- [ ] Forecast accuracy meets 90% threshold
- [ ] Data pipeline handles all file formats
- [ ] Memory usage stays under 2GB
- [ ] Page loads under 2 seconds
- [ ] System handles 50 concurrent users
- [ ] All business calculations preserved
- [ ] Backward compatibility maintained
- [ ] No data loss during migration

---

## 📚 REFERENCE MATERIALS

### Critical Files to Understand

1. **beverly_comprehensive_erp.py** - Main application (7000+ lines)
2. **six_phase_planning_engine.py** - Supply chain planning logic
3. **yarn_intelligence_enhanced.py** - Yarn shortage and substitution
4. **ml_forecast_integration.py** - ML forecasting implementation
5. **optimized_data_loader.py** - Current best data loader

### Key Documentation

- `/CLAUDE.md` - System overview and constraints
- `/BKI_comp/CLAUDE.md` - Detailed implementation notes
- `/ML_FUNCTIONALITY_STATUS.md` - ML capabilities and limitations
- `/COMPLETE_MAPPING_REFERENCE.md` - Data field mappings

### Data File Locations

```
/ERP Data/5/
├── yarn_inventory (4).xlsx    # Master inventory with Planning_Balance
├── Style_BOM.csv              # Bill of Materials
├── Sales Activity Report.csv  # Sales transactions
└── eFab_Knit_Orders_*.xlsx   # Production orders
```

---

## ⚠️ CRITICAL WARNINGS

1. **NEVER modify business calculations without validation**
2. **Dashboard HTML files are LOCKED - no UI changes allowed**
3. **System must remain operational during refactoring**
4. **Test EVERY change against production data**
5. **Maintain backward compatibility for all APIs**
6. **Document every custom calculation you encounter**
7. **Create backups before any destructive operation**

---

## 🚀 GETTING STARTED

### Immediate First Steps

1. **Set up development environment**:
```bash
cd /mnt/c/finalee/Agent-MCP-1-ddd/Agent-MCP-1-dd/BKI_comp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Start the application**:
```bash
python3 beverly_comprehensive_erp.py
# Browse to http://localhost:5005/consolidated
```

3. **Run performance analysis** (Day 1 task):
```bash
# Create and run performance_analysis.py as specified above
```

4. **Review critical business logic**:
```bash
# Search for custom calculations
grep -n "Planning.Balance\|weekly.demand\|yarn.substitution" beverly_comprehensive_erp.py
```

5. **Check data pipeline**:
```bash
# Test data loading with actual files
python3 -c "from optimized_data_loader import OptimizedDataLoader; loader = OptimizedDataLoader(); print(loader.load_yarn_inventory())"
```

---

## 📞 ESCALATION & QUESTIONS

### Known Complex Areas

1. **Planning Balance Calculation** - Line 3000-3100 in main file
2. **ML Model Ensemble** - Line 600-900 in SalesForecastingEngine
3. **Yarn Substitution Logic** - yarn_intelligence_enhanced.py
4. **6-Phase Planning** - six_phase_planning_engine.py
5. **Data Column Mappings** - Multiple variations throughout

### When You Get Stuck

1. Check existing documentation in CLAUDE.md files
2. Review test files for expected behavior
3. Use git history to understand evolution
4. Test with actual production data from /ERP Data/5/
5. Preserve and document any unclear business logic

---

## ✅ SUCCESS CRITERIA

The project is complete when:

1. **Stability**: Zero crashes in 48-hour continuous operation
2. **Performance**: All targets met (see Performance Targets table)
3. **Accuracy**: 90%+ forecast accuracy at 9-week horizon
4. **Testing**: 80%+ code coverage with passing tests
5. **Architecture**: Modular, maintainable codebase
6. **Documentation**: Complete API and business logic documentation
7. **Deployment**: Successfully running in production environment
8. **Users**: Executive, Purchasing, and Planning users can operate system

---

## 💡 FINAL NOTES

This is a critical production system for a textile manufacturing company. The complexity comes from:
- Extensive custom business logic throughout
- Poor initial architecture (evolved, not designed)
- Multiple failed refactoring attempts
- Real-time operational dependency

The key to success is:
1. **Incremental changes** - Don't break what works
2. **Extensive testing** - Validate every change
3. **Business logic preservation** - Extract but don't modify
4. **Performance focus** - Current issues are blocking operations
5. **Clear documentation** - Future maintainers will thank you

Good luck! This system is challenging but the business impact of stabilizing it is significant.