# Beverly Knits ERP - API Endpoint Catalog

## API Overview
- **Base URL**: `http://localhost:5006`
- **Content Type**: `application/json`
- **Authentication**: Session-based (some endpoints require authentication)

## Endpoint Categories

### 🏠 Core Application

#### GET `/`
**Description**: Main application dashboard
**Response**: HTML dashboard interface
**Authentication**: Not required

#### GET `/consolidated`
**Description**: Consolidated view dashboard
**Response**: HTML consolidated interface
**Authentication**: Not required

#### GET `/machine-schedule`
**Description**: Machine scheduling interface
**Response**: HTML machine schedule view
**Authentication**: Not required

#### GET `/ai-factory-floor`
**Description**: AI-powered factory floor monitoring
**Response**: HTML factory floor interface
**Authentication**: Not required

---

### 📊 KPIs and Metrics

#### GET `/api/comprehensive-kpis`
**Description**: Comprehensive key performance indicators
**Response**:
```json
{
  "inventory_metrics": {
    "total_value": 0,
    "turnover_rate": 0,
    "stockout_risk": []
  },
  "production_metrics": {
    "efficiency": 0,
    "on_time_delivery": 0,
    "capacity_utilization": 0
  },
  "financial_metrics": {
    "revenue": 0,
    "profit_margin": 0,
    "working_capital": 0
  }
}
```

#### GET `/api/production-metrics-enhanced`
**Description**: Enhanced production metrics with detailed analysis
**Response**: Production efficiency, machine utilization, quality metrics

---

### 📈 Planning & Optimization

#### GET `/api/planning-phases`
**Description**: Retrieve multi-phase planning configuration
**Response**:
```json
{
  "phases": [
    {
      "phase_id": 1,
      "name": "Demand Planning",
      "status": "pending",
      "progress": 0
    }
  ]
}
```

#### GET `/api/planning-status`
**Description**: Current planning execution status
**Response**: Planning phase status and progress

#### POST `/api/planning/execute`
**Description**: Execute planning phase
**Request Body**:
```json
{
  "phase": "demand_planning",
  "parameters": {}
}
```

#### GET `/api/six-phase-planning`
**Description**: Advanced six-phase planning system
**Response**: Comprehensive planning phases with optimization

---

### 📦 Inventory Management

#### GET `/api/real-time-inventory-dashboard`
**Description**: Real-time inventory status and alerts
**Response**:
```json
{
  "inventory_levels": [],
  "critical_items": [],
  "reorder_suggestions": []
}
```

#### GET `/api/yarn-data`
**Description**: Detailed yarn inventory information
**Response**: Yarn codes, quantities, locations, status

#### GET `/api/emergency-shortage-dashboard`
**Description**: Critical shortage alerts and recommendations
**Response**: Shortage items, impact analysis, alternatives

#### GET `/api/yarn-shortage-analysis`
**Description**: Predictive yarn shortage analysis
**Response**: Shortage predictions, risk assessment, mitigation strategies

---

### 🏭 Production Management

#### GET `/api/fabric-production`
**Description**: Fabric production tracking and status
**Response**: Production orders, completion status, quality metrics

#### GET `/api/knit-orders`
**Description**: Knitting order management
**Response**: Order list, status, priorities

#### GET `/api/knit-orders-styles`
**Description**: Style-specific knitting orders
**Response**: Styles in production, quantities, schedules

#### GET `/api/knit-orders-analysis`
**Description**: Knitting order analytics
**Response**: Performance metrics, bottlenecks, recommendations

#### POST `/api/knit-orders/generate`
**Description**: Generate new knitting orders
**Request Body**:
```json
{
  "style_id": "string",
  "quantity": 0,
  "due_date": "2025-01-18"
}
```

---

### 🤖 Machine Learning & Forecasting

#### GET `/api/ml-forecasting`
**Description**: ML-based demand forecasting
**Response**:
```json
{
  "forecast": [],
  "confidence_interval": [],
  "model_accuracy": 0
}
```

#### GET `/api/advanced-optimization`
**Description**: Advanced optimization algorithms
**Response**: Optimization results, recommendations, savings

#### GET `/api/sales-forecast-analysis`
**Description**: Sales forecasting with trend analysis
**Response**: Sales predictions, seasonal patterns, growth projections

#### GET `/api/fabric-forecast`
**Description**: Fabric demand forecasting
**Response**: Fabric requirements, lead time considerations

---

### 🛒 Procurement & Sourcing

#### GET `/api/procurement-recommendations`
**Description**: AI-driven procurement suggestions
**Response**:
```json
{
  "recommendations": [
    {
      "item": "string",
      "quantity": 0,
      "supplier": "string",
      "urgency": "high"
    }
  ]
}
```

#### GET/POST `/api/purchase-orders`
**Description**: Purchase order management
**GET Response**: List of purchase orders
**POST Request**:
```json
{
  "supplier_id": "string",
  "items": [],
  "delivery_date": "2025-01-18"
}
```

#### GET `/api/supplier-intelligence`
**Description**: Supplier performance analytics
**Response**: Supplier ratings, delivery performance, quality scores

#### GET `/api/yarn-alternatives`
**Description**: Alternative yarn suggestions
**Response**: Substitute options, compatibility scores

---

### 📋 BOM & Requirements

#### GET `/api/bom-explosion-net-requirements`
**Description**: Multi-level BOM explosion with netting
**Response**:
```json
{
  "bom_tree": {},
  "net_requirements": [],
  "total_cost": 0
}
```

#### POST `/api/textile-bom`
**Description**: Textile-specific BOM processing
**Request Body**:
```json
{
  "style_id": "string",
  "quantity": 0
}
```

#### GET `/api/yarn-requirements-calculation`
**Description**: Calculate yarn requirements
**Response**: Detailed yarn requirements by style and quantity

#### POST `/api/fabric/yarn-requirements`
**Description**: Convert fabric to yarn requirements
**Request Body**:
```json
{
  "fabric_type": "string",
  "quantity": 0,
  "width": 0
}
```

---

### 🔄 Data Conversion & Utilities

#### POST `/api/fabric/convert`
**Description**: Fabric unit conversion
**Request Body**:
```json
{
  "from_unit": "meters",
  "to_unit": "yards",
  "value": 0
}
```

#### GET `/api/fabric/specs`
**Description**: Fabric specifications database
**Response**: Fabric types, properties, specifications

#### GET `/api/validate-substitution`
**Description**: Validate material substitutions
**Response**: Validation results, compatibility scores

---

### 🛠️ System & Monitoring

#### GET `/api/debug-data`
**Description**: Debug information for troubleshooting
**Response**: System state, configurations, logs

#### GET `/api/cache-stats`
**Description**: Cache performance statistics
**Response**:
```json
{
  "hit_rate": 0,
  "miss_rate": 0,
  "total_requests": 0,
  "cache_size": 0
}
```

#### POST `/api/cache-clear`
**Description**: Clear system cache
**Response**: Cache cleared confirmation

#### GET `/api/consolidation-metrics`
**Description**: API consolidation metrics
**Response**: Consolidation status, deprecated endpoints, migration progress

#### GET `/api/reload-data`
**Description**: Reload system data from sources
**Response**: Reload status and statistics

---

## Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 500 | Internal Server Error |

## Common Response Format

### Success Response
```json
{
  "status": "success",
  "data": {},
  "message": "Operation completed successfully"
}
```

### Error Response
```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description",
    "details": {}
  }
}
```

## Rate Limiting
- Default: 100 requests per minute per IP
- Authenticated: 500 requests per minute per user
- Bulk operations: 10 requests per minute

## Authentication
Some endpoints require authentication. Include session cookie or API key in request headers:
```
Authorization: Bearer <api_key>
```
or
```
Cookie: session=<session_id>
```

## Pagination
List endpoints support pagination:
- `?page=1` - Page number (default: 1)
- `?per_page=20` - Items per page (default: 20, max: 100)
- `?sort_by=created_at` - Sort field
- `?order=desc` - Sort order (asc/desc)

## Filtering
Many endpoints support filtering:
- `?status=active` - Filter by status
- `?date_from=2025-01-01` - Date range start
- `?date_to=2025-12-31` - Date range end
- `?search=keyword` - Text search

## Webhooks
The system supports webhooks for real-time notifications:
- Inventory level changes
- Order status updates
- Production completions
- Forecast updates

## API Versioning
Currently using unversioned API. Future versions will use:
- `/api/v1/` - Version 1 (current)
- `/api/v2/` - Version 2 (planned)

## Testing Endpoints
Test endpoints available in development:
- `/test-early` - Early test page
- `/test-tabs` - Tab interface test
- `/final-test` - Final integration test
- `/test_dashboard.html` - Dashboard test

---

*Last Updated: 2025-01-18*
*API Version: 1.0*