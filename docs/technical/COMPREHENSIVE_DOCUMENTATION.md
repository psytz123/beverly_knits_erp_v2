---
Generated: 2025-09-01 22:26:15
Version: 2.0.0
Status: Production Ready
---
## Table of Contents

- [Beverly Knits ERP v2 - Comprehensive Documentation](#beverly-knits-erp-v2---comprehensive-documentation)
  - [Executive Summary](#executive-summary)
    - [Key Capabilities](#key-capabilities)
    - [System Metrics](#system-metrics)
  - [System Architecture](#system-architecture)
    - [High-Level Architecture](#high-level-architecture)
    - [Component Architecture](#component-architecture)
  - [API Documentation](#api-documentation)
    - [Base URL](#base-url)
    - [Authentication](#authentication)
    - [Critical Endpoints](#critical-endpoints)
  - [Deployment &amp; Operations Guide](#deployment--operations-guide)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Edit .env with your settings](#edit-.env-with-your-settings)
- [Server runs on port 5006](#server-runs-on-port-5006)
  - [Testing Strategy](#testing-strategy)
    - [Test Coverage](#test-coverage)
    - [Running Tests](#running-tests)
    - [Test Organization](#test-organization)
    - [Key Test Cases](#key-test-cases)
  - [Data Dictionary](#data-dictionary)
    - [Yarn Inventory Fields](#yarn-inventory-fields)
    - [Production Stages](#production-stages)
    - [BOM Structure](#bom-structure)
    - [Sales Data](#sales-data)
  - [Troubleshooting Guide](#troubleshooting-guide)
    - [Common Issues &amp; Solutions](#common-issues--solutions)
  - [Development Roadmap](#development-roadmap)
    - [Completed Features ✅](#completed-features-✅)
    - [In Progress 🚧](#in-progress-🚧)
    - [Planned Features 📋](#planned-features-📋)
    - [Technical Debt](#technical-debt)

# Beverly Knits ERP v2 - Comprehensive Documentation

## Executive Summary

Beverly Knits ERP v2 is a production-ready textile manufacturing Enterprise Resource Planning system designed to optimize the complete supply chain from yarn procurement to finished goods delivery. The system provides real-time inventory intelligence, ML-powered demand forecasting, and sophisticated 6-phase supply chain planning capabilities.

### Key Capabilities

- **Real-time Inventory Management**: Track 1,198+ yarn items with Planning Balance calculations
- **ML-Powered Forecasting**: 90% accuracy at 9-week horizon using ensemble methods
- **6-Phase Supply Chain Planning**: Optimize from yarn procurement through finished goods
- **Production Flow Tracking**: Monitor products through 5 production stages (G00→G02→I01→F01→P01)
- **Intelligent Yarn Substitution**: ML-based recommendations for interchangeable yarns
- **Performance Optimization**: 100x+ speed improvement with parallel data loading and caching

### System Metrics

- **Data Volume**: 28,653+ BOM entries, 10,338+ sales records, 221+ active orders
- **Response Time**: <200ms for most API endpoints
- **Dashboard Load**: <3 seconds full render
- **Cache Hit Rate**: 70-90% typical performance

## System Architecture

### High-Level Architecture

The system follows a monolithic architecture with modular service components:

```
┌─────────────────────────────────────────────────────────┐
│                   Web Dashboard                          │
│              (consolidated_dashboard.html)               │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                Flask Application                         │
│          (beverly_comprehensive_erp.py)                  │
├──────────────────────────────────────────────────────────┤
│  Core Classes:                                           │
│  • InventoryAnalyzer - Inventory analysis engine         │
│  • InventoryManagementPipeline - Pipeline orchestration  │
│  • SalesForecastingEngine - ML forecasting               │
│  • CapacityPlanningEngine - Production planning          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                Service Layer                             │
├──────────────────────────────────────────────────────────┤
│  • Yarn Intelligence Services                            │
│  • Production Planning Services                          │
│  • Forecasting Services                                  │
│  • Data Loading Services                                 │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Data Layer                                  │
├──────────────────────────────────────────────────────────┤
│  • Excel Files (yarn_inventory, BOM, orders)             │
│  • CSV Files (sales data, reports)                       │
│  • Cache Layer (Memory + Redis)                          │
│  • SQLite Database (production tracking)                 │
└──────────────────────────────────────────────────────────┘
```

### Component Architecture

#### Core Application (beverly_comprehensive_erp.py)

- **Lines**: 7000+
- **Responsibilities**: Request handling, business logic orchestration
- **Key Methods**: Data loading, API endpoints, dashboard serving

#### Data Loading Architecture

```
OptimizedDataLoader (100x speed)
    ├── Parallel processing
    ├── Chunk-based loading
    └── Multi-level caching
  
ParallelDataLoader (4x speed)
    ├── Concurrent file reading
    ├── Thread pool execution
    └── Batch processing
  
UnifiedCacheManager
    ├── Memory cache (L1)
    ├── Redis cache (L2)
    └── TTL management
```

#### ML Architecture

```
Forecasting Models:
├── ARIMA (Time series)
├── Prophet (Seasonal patterns)
├── LSTM (Deep learning)
├── XGBoost (Gradient boosting)
└── Ensemble (Combined predictions)

Accuracy: 90-95% with ensemble methods
Fallback: Ensemble → Single → Statistical → Last known
```

## API Documentation

### Base URL

```
http://localhost:5006
```

### Authentication

Currently no authentication required (development mode)

### Critical Endpoints

#### Yarn & Inventory Intelligence

**GET /api/yarn-intelligence**

- **Description**: Comprehensive yarn analysis with shortage detection
- **Parameters**:
  - `analysis`: Type of analysis (shortage, forecast, all)
  - `forecast`: Include forecast data (true/false)
- **Response**: JSON with yarn shortages, recommendations, and forecasts

**GET /api/inventory-intelligence-enhanced**

- **Description**: Enhanced inventory metrics and analytics
- **Parameters**:
  - `view`: Data view (summary, detailed)
  - `realtime`: Force real-time calculation (true/false)
- **Response**: JSON with inventory KPIs, trends, and insights

#### Production Planning

**GET /api/production-planning**

- **Description**: Production schedule and capacity planning
- **Parameters**:
  - `view`: Planning view (orders, capacity, schedule)
  - `forecast`: Include forecast data (true/false)
- **Response**: JSON with production plans and schedules

**GET /api/production-pipeline**

- **Description**: Real-time production flow tracking
- **Response**: JSON with stage-wise production status

**GET /api/six-phase-planning**

- **Description**: Execute 6-phase supply chain planning
- **Response**: JSON with complete planning results

#### ML & Forecasting

**GET /api/ml-forecast-detailed**

- **Description**: Detailed ML predictions
- **Parameters**:
  - `detail`: Level of detail (summary, full)
  - `format`: Response format (json, report)
  - `horizon`: Forecast horizon in days
- **Response**: JSON with predictions and confidence intervals

**POST /api/retrain-ml**

- **Description**: Trigger ML model retraining
- **Body**: Optional training parameters
- **Response**: JSON with training results

#### System & Health

**GET /api/health**

- **Description**: System health check
- **Response**: JSON with system status

**GET /api/debug-data**

- **Description**: Debug data loading issues
- **Response**: JSON with detailed data status

**GET /api/cache-stats**

- **Description**: Cache performance metrics
- **Response**: JSON with cache statistics

## Deployment & Operations Guide

### Prerequisites

- Python 3.8+
- Redis (optional, for caching)
- 4GB+ RAM recommended
- Excel file access to data directory

### Installation

1. **Clone Repository**

```bash
git clone <repository-url>
cd beverly_knits_erp_v2
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure Environment**

```bash
cp config/.env.example config/.env
# Edit .env with your settings
```

4. **Start Server**

```bash
python3 src/core/beverly_comprehensive_erp.py
# Server runs on port 5006
```

### Docker Deployment

```bash
# Build image
docker build -f deployment/docker/Dockerfile -t beverly-erp .

# Run container
docker run -p 5006:5006 -v /path/to/data:/data beverly-erp
```

### Production Deployment

#### Using Docker Compose

```bash
docker-compose -f deployment/docker/docker-compose.prod.yml up -d
```

#### Railway Deployment

- Use Dockerfile.railway for Railway platform
- Set environment variables in Railway dashboard
- Mount data volume for persistent storage

### Monitoring & Maintenance

#### Health Checks

```bash
curl http://localhost:5006/api/health
```

#### Clear Cache

```bash
rm -rf /tmp/bki_cache/*
curl http://localhost:5006/api/reload-data
```

#### View Logs

```bash
tail -f logs/application.log
```

### Troubleshooting

#### Port Conflicts

```bash
lsof -i :5006
kill -9 <PID>
```

#### Data Loading Issues

1. Check file paths in CLAUDE.md
2. Clear cache: `rm -rf /tmp/bki_cache/*`
3. Restart server
4. Check debug endpoint: `/api/debug-data`

#### Performance Issues

1. Monitor cache hit rate: `/api/cache-stats`
2. Check memory usage
3. Enable Redis for better caching
4. Use production deployment configuration

## Testing Strategy

### Test Coverage

- **Current Coverage**: Target 80% for critical paths
- **Critical Areas**:
  - Planning Balance calculations
  - Style mapping (fStyle# ↔ Style#)
  - Yarn shortage detection
  - API endpoints

### Running Tests

#### All Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

#### By Category

```bash
pytest -m unit           # Unit tests
pytest -m integration    # Integration tests
pytest -m e2e           # End-to-end tests
pytest -n auto          # Parallel execution
```

#### Specific Tests

```bash
pytest tests/unit/test_inventory.py::test_yarn_shortage_calculation -v
```

### Test Organization

```
tests/
├── unit/              # Business logic tests
├── integration/       # API endpoint tests
├── e2e/              # Workflow tests
├── performance/      # Load and speed tests
└── conftest.py       # Test fixtures
```

### Key Test Cases

#### Inventory Tests

- Planning Balance formula validation
- Negative allocated values handling
- Shortage detection accuracy
- Weekly demand calculations

#### Production Tests

- Stage flow validation (G00→G02→I01→F01→P01)
- BOM explosion accuracy
- Capacity constraint checking
- Lead time calculations

#### Forecasting Tests

- Model accuracy validation
- Fallback chain testing
- Ensemble performance
- Online learning updates

### Performance Testing

```bash
# Load testing
locust -f tests/performance/locustfile.py --host=http://localhost:5006

# Benchmark specific endpoints
python tests/performance/benchmark_api.py
```

## Data Dictionary

### Yarn Inventory Fields

| Field               | Type   | Description                       | Example             |
| ------------------- | ------ | --------------------------------- | ------------------- |
| Desc#               | String | Unique yarn identifier            | "19003"             |
| Description         | String | Yarn description                  | "100% Cotton White" |
| Planning_Balance    | Float  | Available + On Order - Allocated  | -500.0              |
| Theoretical_Balance | Float  | Physical inventory on hand        | 1000.0              |
| Allocated           | Float  | Amount allocated (negative value) | -1500.0             |
| On_Order            | Float  | Amount on order from suppliers    | 0.0                 |
| Consumed            | Float  | Monthly consumption (negative)    | -250.0              |
| Cost/Pound          | Float  | Cost per pound of yarn            | 2.50                |

### Production Stages

| Stage | Name            | Description           | Status    |
| ----- | --------------- | --------------------- | --------- |
| G00   | Greige/Knitting | Raw knitted fabric    | WIP       |
| G02   | Finishing       | Dyeing and finishing  | WIP       |
| I01   | Inspection      | Quality control       | WIP       |
| F01   | Finished Goods  | Available for sale    | READY     |
| P01   | Allocated       | Reserved for shipment | COMMITTED |

### BOM Structure

| Field          | Type   | Description              | Example  |
| -------------- | ------ | ------------------------ | -------- |
| Style#         | String | Style identifier         | "ABC123" |
| fStyle#        | String | Fabric style reference   | "FAB456" |
| Yarn_ID        | String | Yarn identifier (Desc#)  | "19003"  |
| Percentage     | Float  | Yarn percentage in style | 0.45     |
| Usage_Per_Unit | Float  | Pounds per unit          | 1.2      |

### Sales Data

| Field        | Type    | Description         | Example         |
| ------------ | ------- | ------------------- | --------------- |
| Order_Number | String  | Unique order ID     | "SO-2024-001"   |
| Style#       | String  | Style ordered       | "ABC123"        |
| Quantity     | Integer | Units ordered       | 500             |
| Ship_Date    | Date    | Scheduled ship date | "2024-08-15"    |
| Customer     | String  | Customer name       | "Customer A"    |
| Status       | String  | Order status        | "In Production" |

### Key Formulas

**Planning Balance**

```
Planning_Balance = Theoretical_Balance + Allocated + On_Order
```

Note: Allocated is already negative in data files

**Weekly Demand**

```
If Consumed exists: abs(Consumed) / 4.3
Else if Allocated exists: abs(Allocated) / 8
Else: 10 (default minimum)
```

**Yarn Requirement**

```
Yarn_Required = Fabric_Yards x Conversion_Factor × BOM_Percentage
```

## Troubleshooting Guide

### Common Issues & Solutions

#### Data Not Loading

**Symptoms**: Empty dashboard, no data displayed
**Solutions**:

1. Check file paths exist:

```bash
ls -la "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/"
```

2. Clear cache:

```bash
rm -rf /tmp/bki_cache/*
```

3. Restart server:

```bash
pkill -f "python3.*beverly"
python3 src/core/beverly_comprehensive_erp.py
```

#### Planning Balance Incorrect

**Symptoms**: Negative values showing as positive
**Solutions**:

- Verify formula: Planning_Balance = Theoretical + Allocated + On_Order
- Check that Allocated values are negative in source data
- Review column name variations (Planning_Balance vs Planning_Ballance)

#### Slow Performance

**Symptoms**: Dashboard takes >10 seconds to load
**Solutions**:

1. Enable Redis caching
2. Check cache hit rate: `/api/cache-stats`
3. Use parallel data loader
4. Reduce dashboard refresh frequency

#### ML Forecasting Errors

**Symptoms**: Forecast returns errors or defaults
**Solutions**:

1. Check sales data availability
2. Verify minimum data points (30+ for ARIMA)
3. Trigger retraining: `POST /api/retrain-ml`
4. Check fallback chain is working

#### Database Errors

**Symptoms**: "disk I/O error" messages
**Solutions**:

1. Check database file permissions
2. Disable SQLite temporarily (set SQLITE_AVAILABLE = False)
3. Use mock data for testing
4. Recreate database if corrupted

## Development Roadmap

### Completed Features ✅

- Core inventory management system
- ML-powered forecasting (5 models)
- 6-phase supply chain planning
- Real-time production tracking
- Yarn substitution intelligence
- Performance optimization (100x+ improvement)
- Comprehensive caching system
- API consolidation

### In Progress 🚧

- Production flow tracking implementation
- Enhanced reporting capabilities
- Mobile-responsive dashboard
- Advanced analytics dashboard

### Planned Features 📋

#### Q1 2025

- [ ] Multi-tenant support
- [ ] Advanced user authentication
- [ ] Role-based access control
- [ ] Audit logging system

#### Q2 2025

- [ ] GraphQL API layer
- [ ] Real-time WebSocket updates
- [ ] Advanced ML model management
- [ ] Automated report generation

#### Q3 2025

- [ ] Mobile application
- [ ] Integration with external systems
- [ ] Advanced predictive analytics
- [ ] Automated procurement workflows

#### Q4 2025

- [ ] AI-powered decision support
- [ ] Complete automation framework
- [ ] Advanced optimization algorithms
- [ ] Enterprise integration suite

### Technical Debt

- Refactor monolithic application into microservices
- Implement comprehensive error handling
- Add request validation middleware
- Improve test coverage to 90%+
- Optimize database queries
- Implement connection pooling

### Performance Targets

- API response time: <100ms (currently ~200ms)
- Dashboard load: <2s (currently ~3s)
- Data processing: <500ms (currently ~1-2s)
- Concurrent users: 100+ (currently ~10-20)
