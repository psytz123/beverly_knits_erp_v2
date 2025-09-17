# Beverly Knits ERP v2 - Data Management Guide

## 🔄 API-Based Data Architecture

The ERP system is configured to use **API-based data fetching** from eFab and QuadS systems, replacing the previous SharePoint file upload model.

## 📊 Real-Time Data Integration

### How It Works:
1. **Direct API Connections**: System connects directly to eFab and QuadS APIs
2. **Automated Data Fetching**: No manual downloads or uploads required
3. **Real-time Updates**: Data is fetched on-demand with configurable caching
4. **Automatic Validation**: All data is validated and standardized automatically
5. **Session Management**: eFab sessions are managed automatically

### Automatic Process on ERP Startup:
```
1. Initialize API connections to eFab and QuadS
2. Validate session cookies and authentication
3. Fetch initial data sets via API endpoints
4. Process and cache data for optimal performance
5. Start yarn demand scheduler for periodic updates
```

## 🌐 API Integration Architecture

### Primary Data Sources

#### eFab ERP System
- **Base URL**: https://efab.bkiapps.com
- **Authentication**: Session-based using dancer.session cookie
- **Data Types**: Sales orders, knit orders, yarn inventory, styles
- **Update Frequency**: Real-time API calls with 30-minute caching

#### QuadS Manufacturing System
- **Base URL**: https://quads.bkiapps.com
- **Authentication**: Shared credentials with eFab
- **Data Types**: Greige styles, finished styles, work center mappings
- **Update Frequency**: Real-time API calls with 60-minute caching

### API Endpoint Integration

#### Sales & Production Data
```bash
# Fetch sales orders
GET /api/sales-order/plan/list

# Fetch knit orders with machine assignments
GET /api/knitorder/list

# Get active yarn inventory with planning balance
GET /api/yarn/active

# Retrieve style information
GET /api/styles
```

#### Inventory Staging Data
```bash
# Greige inventory stages
GET /api/greige/g00  # Stage 1
GET /api/greige/g02  # Stage 2

# Finished inventory stages
GET /api/finished/i01  # QC/Inspection
GET /api/finished/f01  # Finished Goods
```

#### Reporting & Analytics
```bash
# Yarn demand reports
GET /api/report/yarn_demand_ko
GET /api/report/yarn_demand

# Purchase order data
GET /api/yarn-po

# Expected deliveries (time-phased planning)
GET /api/report/yarn_expected
```

## 🔧 Data Processing & Validation

### Automatic Data Standardization

#### 1. **Column Name Mapping**
The system automatically handles various column name formats:
- `Desc#`, `Yarn_ID`, `YarnID` → **`Desc#`** (standardized)
- `Planning Balance` vs `Planning_Balance` → **`Planning_Balance`**
- `Style Number`, `Style #`, `fStyle#` → **`Style#`**
- `On Order`, `On-Order` → **`On_Order`**

#### 2. **Data Type Conversion**
- **Numeric fields**: Remove commas, handle NaN values, convert to proper numeric types
- **Date fields**: Parse multiple date formats, standardize to ISO format
- **Text fields**: Trim whitespace, handle null values
- **Unit standardization**: LB→LBS, YD→YDS for consistency

#### 3. **Business Logic Validation**
Each data type includes business rule validation:
- **Yarn Inventory**: Planning Balance calculations, shortage detection
- **Style BOM**: Style to yarn mappings, quantity validations
- **Production Orders**: Machine assignments, work center validations
- **Sales Data**: Date range validations, customer mappings

### Data Quality Monitoring

#### Real-time Quality Metrics
```bash
# Check data quality status
curl -s http://localhost:5006/api/comprehensive-kpis | jq '.data_quality'

# View consolidation metrics
curl -s http://localhost:5006/api/consolidation-metrics | jq '.data_freshness'
```

#### Quality Indicators Tracked:
- API response times and success rates
- Data freshness timestamps
- Record counts and completeness
- Validation error rates
- Cache hit/miss ratios

## ⏱️ Automated Yarn Demand Scheduler

### Time-Phased Planning Integration

The system includes automated yarn demand report processing:

#### Configuration
```bash
# Environment variables
export ENABLE_YARN_SCHEDULER=true
export FILTER_NONPRODUCTION_YARNS=true
export EFAB_SESSION="session_cookie_value"
```

#### Automatic Schedule
- **Downloads**: Yarn demand reports at 10 AM and 12 PM daily
- **Processing**: Automatic time-phased MRP calculations
- **Storage**: Reports saved to `/data/production/5/ERP Data/`
- **Integration**: Data integrated into planning balance calculations

#### Manual Operations
```bash
# Manual refresh
curl -X POST http://localhost:5006/api/manual-yarn-refresh

# Check scheduler status
curl -s http://localhost:5006/api/comprehensive-kpis | jq '.scheduler_status'

# View downloaded reports
ls -la "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/Expected_Yarn_Report.xlsx"
```

## 📁 Data Architecture Structure

### Current API-Based Structure
```
Beverly Knits ERP v2/
├── API Connections/
│   ├── eFab (efab.bkiapps.com)
│   │   ├── Sales Orders
│   │   ├── Knit Orders
│   │   ├── Yarn Inventory
│   │   └── Styles
│   └── QuadS (quads.bkiapps.com)
│       ├── Greige Styles
│       ├── Finished Styles
│       └── Work Centers
├── Cache Layer/
│   ├── Redis (30-60 min TTL)
│   └── Memory Cache
├── Processing Engine/
│   ├── Data Validation
│   ├── Planning Balance Calculations
│   └── Time-Phased MRP
└── Data Output/
    ├── Dashboard APIs
    ├── Reports
    └── Real-time Updates
```

### Legacy File-Based Structure (Obsolete)
```
❌ REPLACED: SharePoint → ZIP Download → Manual Upload → Processing
✅ CURRENT: eFab/QuadS APIs → Cache → Validation → Dashboard
```

## 🔄 Session Management

### eFab Session Handling

#### Automatic Session Management
The system automatically manages eFab session cookies:
- **Session Validation**: Checked before each API call
- **Automatic Renewal**: Attempts to refresh expired sessions
- **Fallback Handling**: Graceful degradation when sessions expire
- **Monitoring**: Session status visible in comprehensive KPIs

#### Manual Session Update
When sessions require manual renewal:
```bash
# 1. Login to eFab in browser
# 2. Copy session cookie from browser dev tools
# 3. Update environment variable
export EFAB_SESSION="new_session_cookie_value"

# 4. Restart ERP system to apply new session
systemctl restart beverly-erp
# OR for Docker:
docker-compose restart beverly-erp
```

#### Session Monitoring
```bash
# Check session status
curl -s http://localhost:5006/api/comprehensive-kpis | jq '.efab_session_status'

# Monitor for session expiration warnings in logs
grep -i "session" /var/log/beverly_erp.log
```

## 🚨 Troubleshooting API Integration

### Common Issues & Solutions

#### 1. API Connection Failures
```bash
# Check network connectivity
curl -I https://efab.bkiapps.com
curl -I https://quads.bkiapps.com

# Verify session cookie
curl -H "Cookie: dancer.session=$EFAB_SESSION" https://efab.bkiapps.com/api/test

# Check ERP system logs
tail -f /var/log/beverly_erp.log | grep -i "connection\|api\|error"
```

#### 2. Data Not Loading
```bash
# Force data reload
curl -X POST http://localhost:5006/api/reload-data

# Check specific endpoints
curl -s http://localhost:5006/api/yarn/active | jq '.status'
curl -s http://localhost:5006/api/knitorder/list | jq '.status'

# Clear cache and retry
redis-cli FLUSHDB  # If using Redis
# OR restart system for memory cache clear
```

#### 3. Session Expiration Issues
```bash
# Check session status
curl -s http://localhost:5006/api/comprehensive-kpis | jq '.auth_status'

# Update session (requires browser login to eFab)
# Copy new session cookie and update environment:
export EFAB_SESSION="new_cookie_value"
systemctl restart beverly-erp
```

#### 4. Yarn Scheduler Not Working
```bash
# Check scheduler configuration
curl -s http://localhost:5006/api/comprehensive-kpis | jq '.scheduler_enabled'

# Manual trigger
curl -X POST http://localhost:5006/api/manual-yarn-refresh

# Check for scheduler errors in logs
grep -i "scheduler\|yarn.*download" /var/log/beverly_erp.log
```

## 📈 Performance Optimization

### API Performance Tuning

#### Caching Strategy
- **eFab Data**: 30-minute TTL for yarn inventory, orders
- **QuadS Data**: 60-minute TTL for styles, work centers
- **ML Forecasts**: 4-hour TTL for predictions
- **Reports**: 2-hour TTL for yarn demand reports

#### Connection Optimization
```bash
# Connection pooling settings
export API_CONNECTION_POOL_SIZE=20
export API_REQUEST_TIMEOUT=30
export API_RETRY_ATTEMPTS=3

# Monitor API performance
curl -w "%{time_total}" -o /dev/null -s http://localhost:5006/api/yarn/active
```

### Monitoring Performance
```bash
# API response time monitoring
curl -s http://localhost:5006/api/consolidation-metrics | jq '.performance_metrics'

# Cache hit rate monitoring
curl -s http://localhost:5006/api/consolidation-metrics | jq '.cache_hit_rate'

# System resource monitoring
top -p $(pgrep -f beverly_comprehensive_erp)
```

## ✅ Benefits of API-Based Architecture

### Advantages Over File-Based System

1. **Real-time Data**: Immediate access to current information
2. **No Manual Intervention**: Fully automated data pipeline
3. **Data Consistency**: Single source of truth from eFab/QuadS
4. **Error Reduction**: Eliminates manual upload errors
5. **Audit Trail**: Complete API call logging and monitoring
6. **Scalability**: Can handle increased data volume automatically
7. **Security**: Secure API authentication vs file sharing
8. **Performance**: Cached data with configurable refresh rates

### System Reliability
- **Fault Tolerance**: Graceful handling of API failures
- **Automatic Retry**: Built-in retry logic for failed requests
- **Fallback Data**: Cached data available during outages
- **Health Monitoring**: Continuous system health tracking
- **Alert System**: Notifications for critical failures

## 🔧 Configuration Management

### Environment Variables
```bash
# Core API Configuration
ERP_BASE_URL=https://efab.bkiapps.com
QUADS_BASE_URL=https://quads.bkiapps.com
EFAB_SESSION=session_cookie_value

# Performance Settings
CACHE_TTL=3600
API_TIMEOUT=30
MAX_CONCURRENT_REQUESTS=100

# Scheduler Settings
ENABLE_YARN_SCHEDULER=true
SCHEDULER_INTERVAL_HOURS=2
FILTER_NONPRODUCTION_YARNS=true
```

### System Health Endpoints
```bash
# Overall system health
GET /api/comprehensive-kpis

# API connection status
GET /api/consolidation-metrics

# Manual data refresh
POST /api/reload-data

# Yarn scheduler control
POST /api/manual-yarn-refresh
```

---

The API-based data management system ensures reliable, real-time access to Beverly Knits manufacturing data while eliminating manual file handling processes!

*Data Management Guide v2.0.0 - Beverly Knits ERP System - Updated September 2025*