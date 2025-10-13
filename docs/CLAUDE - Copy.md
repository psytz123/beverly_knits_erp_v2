# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

Beverly Knits ERP v2 - Production-ready textile manufacturing ERP with real-time inventory intelligence, ML-powered forecasting, and 6-phase supply chain optimization.

**Current System Stats**:
- 1,199 yarn items tracked
- 28,653 BOM entries (style to yarn mappings)
- 194 production orders (154 assigned to machines, 40 pending assignment)
- 91 work centers with 285 total machines
- 557,671 lbs total production workload
- Machine utilization tracking via eFab Knit Orders integration

## Primary Commands

### Server Operations
```bash
# Start server with Yarn Demand scheduler enabled (Port 5006)
export EFAB_SESSION="aMdcwNLa0ov0pcbWcQ_zb5wyPLSkYF_B"  # Update session cookie as needed
export ENABLE_YARN_SCHEDULER=true
export FILTER_NONPRODUCTION_YARNS=true  # Show all yarns with negative balances
python3 src/core/beverly_comprehensive_erp.py

# Kill existing server if port conflict
pkill -f "python3.*beverly"
lsof -i :5006 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Start with Makefile (alternative)
make run          # Production mode
make run-dev      # Development mode with debug

# Manual Yarn Demand refresh (when scheduler is running)
curl -X POST http://localhost:5006/api/manual-yarn-refresh
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

### Yarn Demand Scheduler & Time-Phased Reporting (NEW as of Sep 2025)
- **Automated downloads** from eFab API at 10 AM and 12 PM daily
- **EFabReportDownloader** (`src/data_loaders/efab_report_downloader.py`): Downloads Yarn Demand reports
- **PODeliveryLoader** (`src/data_loaders/po_delivery_loader.py`): Processes time-phased PO deliveries
- **TimePhasedPlanning** (`src/production/time_phased_planning.py`): Core MRP calculations
- **Expected_Yarn_Report.xlsx**: Downloaded report saved to `/data/production/5/ERP Data/`
- **Session cookie** required in `EFAB_SESSION` environment variable (expires ~24 hours)
- Shows ALL yarns with negative balances at ANY point in time-phased view

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
8-28-2025 subfolder: Contains latest eFab_Knit_Orders.csv and other current data
```

### Key Data Files & Their Purpose
1. **yarn_inventory.xlsx** - Contains 'Planning Balance' column (with space)
2. **BOM_updated.csv** - Bill of Materials (28,653 entries mapping styles to yarns)
3. **eFab_Knit_Orders.csv** - 194 production orders (154 assigned, 40 unassigned)
4. **QuadS_greigeFabricList_(1).xlsx** - Style to Work Center mappings (columns C=style, D=work_center)
5. **Machine Report fin1.csv** - Machine to Work Center mappings (WC column=machine patterns, MACH=machine IDs)
6. **Sales Activity Report.csv** - Historical sales data for forecasting

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

### Yarn Demand Scheduler Issues
If the scheduler isn't downloading reports:
1. **Check session cookie**: The `EFAB_SESSION` cookie expires after ~24 hours
   ```bash
   # Update session cookie and restart
   export EFAB_SESSION="new_cookie_value_from_browser"
   ```
2. **Manual refresh**: `curl -X POST http://localhost:5006/api/manual-yarn-refresh`
3. **Verify file downloaded**: `ls -la "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/Expected_Yarn_Report.xlsx"`
4. **Check scheduler status**: Look for `[SCHEDULER]` messages in server logs

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

## Python Excellence Standards

### Core Principles
1. **ALWAYS use type hints** for all function parameters and returns
2. **ALWAYS write tests first** (TDD) - tests before implementation
3. **NEVER use bare except:** statements - catch specific exceptions
4. **NEVER exceed 500 LOC** per file or 50 LOC per function
5. **ALWAYS handle errors** with specific exceptions and logging

### Code Quality Rules
- **Cyclomatic complexity** ≤ 10 per function
- **Code duplication** < 3% in new code
- **Test coverage** ≥ 85% for new code, ≥ 70% for branches
- **Type coverage** 100% of public functions
- **Import order**: stdlib → third-party → local (with blank lines between)

### Before Writing Code
```bash
# Check for existing similar code
rg "function_name" src/
grep -r "similar_pattern" .

# Verify no breaking changes
pytest tests/ -v

# Check current code quality
mypy src/ --strict
ruff check src/
```

### Python Code Template
```python
"""Module purpose and usage description."""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Result:
    """Result container with validation."""
    value: float
    status: str

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("Value must be non-negative")


def function_name(
    param1: str,
    param2: Optional[int] = None,
    *,  # Force keyword-only args after this
    validate: bool = True
) -> Result:
    """Brief description of function purpose.

    Args:
        param1: Description of first parameter.
        param2: Optional parameter description.
        validate: Whether to validate input.

    Returns:
        Result object containing processed value and status.

    Raises:
        ValueError: If validation fails.
        TypeError: If types are incorrect.

    Examples:
        >>> function_name("test", param2=42)
        Result(value=42.0, status='success')
    """
    if validate and not param1:
        raise ValueError("param1 cannot be empty")

    try:
        # Implementation with specific error handling
        result = process_data(param1, param2)
        logger.info(f"Processed {param1}: {result}")
        return Result(value=result, status='success')

    except SpecificError as e:
        logger.exception(f"Failed to process {param1}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise RuntimeError(f"Processing failed: {e}") from e
```

### Testing Template
```python
import pytest
from typing import Any
from unittest.mock import patch, MagicMock


class TestFunctionName:
    """Test suite for function_name."""

    @pytest.fixture
    def valid_input(self) -> Dict[str, Any]:
        """Provide valid test input."""
        return {"param1": "test", "param2": 42}

    def test_normal_operation(self, valid_input: Dict[str, Any]) -> None:
        """Test standard flow with valid input."""
        result = function_name(**valid_input)
        assert result.value == 42.0
        assert result.status == 'success'

    def test_raises_on_invalid_input(self) -> None:
        """Test that invalid input raises appropriate error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            function_name("", validate=True)

    @pytest.mark.parametrize("input_val,expected", [
        ("test1", 1.0),
        ("test2", 2.0),
    ])
    def test_various_inputs(self, input_val: str, expected: float) -> None:
        """Test various input values."""
        result = function_name(input_val)
        assert result.value == expected
```

### Required Quality Checks
After EVERY Python file change, run:
```bash
# 1. Type checking (MUST PASS)
mypy src/ --strict

# 2. Linting (MUST PASS)
ruff check src/
black --check src/

# 3. Tests (MUST PASS with >85% coverage)
pytest tests/ -v --cov=src --cov-report=term-missing

# 4. Complexity check
radon cc src/ -nc

# Or use Makefile shortcuts:
make quality  # Runs all checks
make fix      # Auto-fixes formatting issues
```

### Error Handling Standards
```python
# GOOD: Specific exceptions with context
try:
    data = load_data(file_path)
except FileNotFoundError as e:
    logger.error(f"Data file not found: {file_path}")
    raise
except pd.errors.ParserError as e:
    logger.exception(f"Failed to parse data file: {e}")
    return default_data()

# BAD: Bare except or too broad
try:
    data = load_data(file_path)
except:  # NEVER DO THIS
    pass
```

### Logging Standards
```python
# Use appropriate log levels
logger.debug("Detailed info for debugging: %s", variable)
logger.info("Normal operation: processed %d items", count)
logger.warning("Unexpected but handled: %s", warning_msg)
logger.error("Error occurred but continuing: %s", error)
logger.exception("Failed with traceback:")  # Includes stack trace
logger.critical("System failure, shutting down")

# NEVER log sensitive data
logger.info(f"User {user_id} logged in")  # Good
logger.info(f"Password: {password}")  # NEVER DO THIS
```

### Performance Considerations
- Use `async/await` for I/O operations
- Profile before optimizing: `python -m cProfile -s cumtime script.py`
- Cache expensive operations with `@functools.lru_cache`
- Use generators for large datasets
- Prefer list comprehensions over loops for simple transformations

### Documentation Requirements
- Every module needs a docstring explaining purpose
- Every public function needs a complete docstring
- Complex logic needs inline comments
- Update README.md when adding new features
- Document breaking changes in CHANGELOG.md

### SOLID Principles Application
1. **Single Responsibility**: Each function/class does ONE thing
2. **Open/Closed**: Extend via inheritance, don't modify existing
3. **Liskov Substitution**: Subtypes must be substitutable
4. **Interface Segregation**: No forced unused dependencies
5. **Dependency Inversion**: Depend on abstractions, not concretions

### Modern Python Features to Use
- Python 3.10+: Pattern matching with `match/case`
- Type hints: Use `Optional`, `Union`, `Literal`, `TypeAlias`
- Dataclasses for data containers
- Pathlib for file operations (not `os.path`)
- Context managers for resource management
- f-strings for formatting (not `.format()` or `%`)

### Features to Avoid
- Global variables (use dependency injection)
- Mutable default arguments (use `None` and create in function)
- Dynamic attribute creation with `setattr`
- `eval()` and `exec()` (security risks)
- Deep inheritance (prefer composition)
- Monkey patching (breaks type checking)