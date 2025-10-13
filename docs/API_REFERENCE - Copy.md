# Beverly Knits ERP v2 - API Reference

Complete API documentation for Beverly Knits ERP v2, covering all endpoints for eFab and QuadS integration, inventory management, and production planning.

## Base Configuration

```
Base URL: http://localhost:5006/api
Authentication: Session-based (eFab session cookie)
Content-Type: application/json
Rate Limit: No current limits (internal system)
```

## Environment Variables

```bash
# eFab ERP Configuration
ERP_BASE_URL=https://efab.bkiapps.com
ERP_LOGIN_URL=https://efab.bkiapps.com/login
ERP_API_PREFIX=/api
ERP_USERNAME=psytz
ERP_PASSWORD=big$cat
EFAB_SESSION=aMdcwNLa0ov0pcbWcQ_zb5wyPLSkYF_B  # Update as needed

# QuadS Configuration
QUADS_BASE_URL=https://quads.bkiapps.com
QUADS_LOGIN_URL=https://quads.bkiapps.com/LOGIN

# Session Management
SESSION_COOKIE_NAME=dancer.session
SESSION_STATE_PATH=/tmp/erp_session.json
```

---

## Primary Wrapper Endpoints

These endpoints fetch data directly from eFab and QuadS systems, replacing manual file uploads.

### Sales & Orders

#### Fetch Sales Orders
```http
GET /api/sales-order/plan/list
```

**Description**: Retrieves sales orders from eFab system for production planning.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "order_id": "SO123456",
      "customer": "Customer Name",
      "style": "Style123",
      "quantity": 1000,
      "due_date": "2025-10-15",
      "priority": "HIGH"
    }
  ],
  "count": 45,
  "last_updated": "2025-09-15T10:30:00Z"
}
```

#### Fetch Knit Orders
```http
GET /api/knitorder/list
```

**Description**: Retrieves knit orders from eFab system with machine assignments.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "order_id": "KO789",
      "style": "Style456",
      "quantity": 500,
      "machine": "M161",
      "work_center": "9.38.20.F",
      "status": "ASSIGNED"
    }
  ],
  "assigned_orders": 154,
  "unassigned_orders": 40,
  "total_workload_lbs": 557671
}
```

### Inventory Management

#### Active Yarn Inventory
```http
GET /api/yarn/active
```

**Description**: Retrieves current yarn inventory with planning balance calculations.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "yarn_id": "Y12345",
      "description": "Cotton 20/1",
      "planning_balance": -150.5,
      "on_hand": 1000.0,
      "allocated": 1150.5,
      "on_order": 500.0,
      "unit": "LBS"
    }
  ],
  "total_items": 1199,
  "shortage_items": 234
}
```

#### Fetch Styles
```http
GET /api/styles
```

**Description**: Retrieves style information from eFab system.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "style_id": "ST001",
      "style_name": "Basic Tee",
      "category": "Apparel",
      "work_center": "9.38.20.F"
    }
  ]
}
```

---

## QuadS Integration Endpoints

### Greige Styles

#### Active Greige Styles
```http
GET /api/styles/greige/active
```

**Description**: Retrieves active greige (unfinished) styles from QuadS system.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "style_id": "GR001",
      "fabric_type": "Jersey",
      "weight": "180GSM",
      "width": "60inch",
      "construction": "Single Jersey"
    }
  ]
}
```

#### Finished Styles
```http
GET /api/styles/finished/active
```

**Description**: Retrieves active finished styles from QuadS system.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "style_id": "FN001",
      "finished_type": "Dyed",
      "color": "Navy Blue",
      "process_stage": "F01"
    }
  ]
}
```

---

## Inventory Stage Endpoints

### Greige Inventory

#### Stage G00 (Greige Stage 1)
```http
GET /api/greige/g00
```

**Description**: Retrieves inventory at greige stage 1.

#### Stage G02 (Greige Stage 2)
```http
GET /api/greige/g02
```

**Description**: Retrieves inventory at greige stage 2.

### Finished Inventory

#### Stage I01 (QC/Inspection)
```http
GET /api/finished/i01
```

**Description**: Retrieves inventory at QC/inspection stage.

#### Stage F01 (Finished Goods)
```http
GET /api/finished/f01
```

**Description**: Retrieves finished goods inventory.

---

## Reporting Endpoints

### Yarn Demand Reports

#### Yarn Demand (KO Format)
```http
GET /api/report/yarn_demand_ko
```

**Description**: Yarn demand report in knit order format.

**Response:**
```json
{
  "status": "success",
  "report_date": "2025-09-15",
  "data": [
    {
      "yarn_id": "Y123",
      "demand_lbs": 500.0,
      "knit_orders": ["KO001", "KO002"],
      "due_date": "2025-10-01"
    }
  ]
}
```

#### Standard Yarn Demand
```http
GET /api/report/yarn_demand
```

**Description**: Standard yarn demand report.

#### Yarn Purchase Orders
```http
GET /api/yarn-po
```

**Description**: Retrieves yarn purchase orders.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "po_number": "PO123",
      "yarn_id": "Y456",
      "quantity": 1000.0,
      "delivery_date": "2025-10-15",
      "supplier": "Yarn Supplier Co"
    }
  ]
}
```

#### Expected Yarn Deliveries
```http
GET /api/report/yarn_expected
```

**Description**: Expected yarn delivery schedule from time-phased planning.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "yarn_id": "Y789",
      "expected_date": "2025-09-20",
      "quantity": 750.0,
      "source": "PO123"
    }
  ]
}
```

---

## Beverly Knits ERP Internal APIs

### Dashboard APIs

#### Production Planning
```http
GET /api/production-planning
```

**Query Parameters:**
- `view`: orders|summary|forecast
- `forecast`: true|false

**Description**: Production schedule with parameter support.

#### Inventory Intelligence Enhanced
```http
GET /api/inventory-intelligence-enhanced
```

**Query Parameters:**
- `view`: summary|detailed
- `analysis`: shortage|all
- `realtime`: true|false

**Description**: Inventory analytics with multiple view options.

#### ML Forecast Detailed
```http
GET /api/ml-forecast-detailed
```

**Query Parameters:**
- `detail`: full|summary
- `format`: json|report
- `horizon`: 30|60|90 (days)

**Description**: ML predictions with configurable detail levels.

#### Comprehensive KPIs
```http
GET /api/comprehensive-kpis
```

**Description**: Complete KPI metrics for system health monitoring.

#### Yarn Intelligence
```http
GET /api/yarn-intelligence
```

**Query Parameters:**
- `analysis`: shortage|substitution|all
- `forecast`: true|false

**Description**: Yarn analysis with shortage detection and substitution recommendations.

### Machine Planning

#### Machine Assignment Suggestions
```http
GET /api/machine-assignment-suggestions
```

**Description**: Suggests machines for unassigned orders using QuadS style mappings.

**Response:**
```json
{
  "status": "success",
  "unassigned_orders": [
    {
      "order_id": "KO001",
      "style": "Style123",
      "suggested_machines": [
        {
          "machine_id": "M161",
          "work_center": "9.38.20.F",
          "confidence": 0.95,
          "availability": "AVAILABLE"
        }
      ]
    }
  ]
}
```

#### Factory Floor AI Dashboard
```http
GET /api/factory-floor-ai-dashboard
```

**Description**: Machine planning data with work center groupings.

### System Management

#### Reload Data
```http
POST /api/reload-data
```

**Description**: Forces reload of all data from eFab and QuadS systems.

#### Manual Yarn Refresh
```http
POST /api/manual-yarn-refresh
```

**Description**: Manually triggers yarn demand report download from eFab.

#### Consolidation Metrics
```http
GET /api/consolidation-metrics
```

**Description**: API consolidation usage metrics and redirect statistics.

---

## Authentication & Session Management

### eFab Session Cookie
The system uses eFab session cookies for authentication. Update the `EFAB_SESSION` environment variable when the session expires (typically every 24 hours).

```bash
# Check session status
curl -H "Cookie: dancer.session=$EFAB_SESSION" https://efab.bkiapps.com/api/test

# Update session in environment
export EFAB_SESSION="new_session_cookie_value"
```

### Session Renewal
Sessions are automatically managed by the ERP system. Monitor server logs for `[SCHEDULER]` messages indicating session issues.

---

## Error Handling

### Standard Error Response
```json
{
  "status": "error",
  "message": "Description of the error",
  "code": "ERROR_CODE",
  "timestamp": "2025-09-15T14:30:00Z"
}
```

### Common Error Codes
- `SESSION_EXPIRED`: eFab session cookie has expired
- `CONNECTION_FAILED`: Unable to connect to eFab/QuadS
- `DATA_NOT_FOUND`: Requested data not available
- `INVALID_PARAMETERS`: Request parameters are invalid
- `SYSTEM_UNAVAILABLE`: External system temporarily unavailable

---

## Data Flow Architecture

### API-Based Data Flow (Current)
```
eFab API → Beverly ERP → Dashboard
QuadS API → Beverly ERP → Dashboard
                ↓
           Cache/Processing
```

### Replaced File-Based Flow (Obsolete)
```
SharePoint → Manual Download → File Upload → Processing
```

---

## Performance & Monitoring

### Response Times
- Primary wrapper endpoints: <500ms
- Internal ERP APIs: <200ms
- ML forecast APIs: <1000ms

### Caching
- eFab data: 30-minute TTL
- QuadS data: 60-minute TTL
- ML forecasts: 4-hour TTL

### Health Monitoring
```http
GET /api/comprehensive-kpis
```

Monitor this endpoint for system health, including:
- API response times
- Data freshness
- Cache hit rates
- Error rates

---

*API Reference v2.0.0 - Beverly Knits ERP System - Updated September 2025*