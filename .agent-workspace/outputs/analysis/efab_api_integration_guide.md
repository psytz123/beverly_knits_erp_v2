# eFab API Integration Guide

**Server URL:** http://localhost:5006
**Base Path:** /api
**Data Source:** eFab ERP System (https://efab.bkiapps.com)
**API Type:** REST JSON
**Generated:** 2025-10-19

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [Caching Behavior](#caching-behavior)
5. [Endpoint Documentation](#endpoint-documentation)
   - [/api/knit-orders](#api-knit-orders)
   - [/api/yarn-intelligence](#api-yarn-intelligence)
   - [/api/inventory/pipeline-summary](#api-inventory-pipeline-summary)
6. [Error Handling](#error-handling)
7. [Example Implementations](#example-implementations)

---

## Overview

The eFab API Server is a Flask-based proxy that connects to the eFab ERP system and provides real-time manufacturing data. It acts as an intermediary layer with caching, rate limiting, and data transformation capabilities.

**Key Features:**
- Real-time data from eFab ERP
- 5-minute response caching
- Rate limiting (60 requests/minute by default)
- CORS enabled for web clients
- Data transformation for dashboard consumption

---

## Authentication

### Authentication Method: Cookie-Based Session

The API server uses a session cookie to authenticate with the eFab backend. Client applications connecting to localhost:5006 do NOT need to provide authentication credentials.

**Backend Authentication (Internal):**
```python
# Header structure used internally by the server
headers = {
    "Accept": "application/json",
    "Cookie": f"dancer.session={EFAB_SESSION}",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "X-Requested-With": "XMLHttpRequest"
}
```

**Session Configuration:**
- Session cookie stored in environment variable: `EFAB_SESSION`
- Loaded from `.env` file or secrets file
- Session obtained via: `python scripts/efab_login.py`

**Client Requirements:**
- No authentication headers required when calling localhost:5006
- All endpoints are publicly accessible on the local server

---

## Rate Limiting

**Default Configuration:**
- Rate: 60 requests per minute
- Storage: In-memory
- Response: HTTP 429 (Too Many Requests)

**Configuration Options:**
```bash
# Environment variables
ENABLE_RATE_LIMITING=true          # Enable/disable rate limiting
API_RATE_LIMIT="60 per minute"     # Rate limit value
RATE_LIMIT_STORAGE_URI="memory://" # Storage backend
```

**Rate Limit Response:**
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests, please slow down.",
  "retry_after": "60 seconds"
}
```

---

## Caching Behavior

**Cache Duration:** 5 minutes (300 seconds)

**Cache Strategy:**
- In-memory dictionary cache
- Cache key: `{endpoint}_{json_params}`
- Automatic cache invalidation after 5 minutes
- Cache hit logged as DEBUG level

**Cache Implementation:**
```python
CACHE_DURATION = timedelta(minutes=5)
data_cache: Dict[str, tuple[datetime, Any]] = {}
```

**Benefits:**
- Reduces load on eFab backend
- Faster response times for repeated requests
- Automatic staleness prevention

---

## Endpoint Documentation

### /api/knit-orders

**Description:** Retrieves all active knit orders from eFab with completion tracking and urgency analysis.

**HTTP Method:** GET
**URL:** `http://localhost:5006/api/knit-orders`
**Authentication:** None required
**Rate Limit:** 60/minute (shared)
**Cache TTL:** 5 minutes

#### Request

**Query Parameters:** None

**Example Request:**
```bash
curl http://localhost:5006/api/knit-orders
```

#### Response

**Success Response (200 OK):**

```json
{
  "orders": [
    {
      "ko_id": 12345,
      "order_id": 12345,
      "id": 12345,
      "serial_number": "KO-2024-001",
      "style": "Style-ABC-123",
      "customer": "Customer Name Inc.",
      "machine": "Machine-05",
      "qty_ordered": 1500.0,
      "qty_ordered_lbs": 1500.0,
      "qty_received": 750.0,
      "balance": 750.0,
      "balance_lbs": 750.0,
      "completion_percentage": 50.0,
      "days_until_due": 5,
      "status": "In Progress",
      "start_date": "2024-10-01T08:00:00",
      "requested_date": "2024-10-24T17:00:00",
      "is_active": true,
      "schedule_status": "On Schedule",
      "knitter": "John Doe",
      "purchase_order": "PO-2024-456",
      "uom": "lbs"
    }
  ],
  "total": 42,
  "source": "efab",
  "status": "ok"
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| ko_id | integer | Knit order ID (primary key) |
| order_id | integer | Alias for ko_id |
| id | integer | Alias for ko_id |
| serial_number | string | Human-readable order number |
| style | string | Style/product name from knit_style_base |
| customer | string | Customer name |
| machine | string | Assigned machine identifier |
| qty_ordered | float | Total quantity ordered (lbs) |
| qty_ordered_lbs | float | Alias for qty_ordered |
| qty_received | float | Quantity completed/received (lbs) |
| balance | float | Remaining quantity (qty_ordered - qty_received) |
| balance_lbs | float | Alias for balance |
| completion_percentage | float | Percentage complete (0-100) |
| days_until_due | integer/null | Days until requested_date (negative = overdue) |
| status | string | Order status (e.g., "In Progress", "Complete") |
| start_date | string (ISO) | Knit start date/time |
| requested_date | string (ISO) | Due date/time |
| is_active | boolean | Whether order is active |
| schedule_status | string | Schedule status description |
| knitter | string | Assigned knitter name |
| purchase_order | string | Related purchase order number |
| uom | string | Unit of measure (typically "lbs") |

**Sorting Logic:**
Orders are sorted by urgency:
1. Overdue orders (days_until_due < 0) - most overdue first
2. Due soon orders - sorted by days_until_due ascending
3. Within same due date - least complete first
4. Orders with no due date - sorted to end

**Error Response (500 Internal Server Error):**
```json
{
  "error": "Failed to fetch from eFab"
}
```

---

### /api/yarn-intelligence

**Description:** Provides comprehensive yarn inventory analysis with shortage detection and risk assessment.

**HTTP Method:** GET
**URL:** `http://localhost:5006/api/yarn-intelligence`
**Authentication:** None required
**Rate Limit:** 60/minute (shared)
**Cache TTL:** 5 minutes

#### Request

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| forecast | string | No | false | Enable forecast mode ("true"/"false") |

**Example Requests:**
```bash
# Normal mode - current inventory analysis
curl http://localhost:5006/api/yarn-intelligence

# Forecast mode - future shortage predictions
curl "http://localhost:5006/api/yarn-intelligence?forecast=true"
```

#### Response (Normal Mode)

**Success Response (200 OK):**

```json
{
  "criticality_analysis": {
    "yarns": [
      {
        "yarn_id": 18865,
        "description": "Cotton Blend 30/1",
        "supplier": "ABC Yarn Co.",
        "color": "Navy Blue",
        "theoretical_balance": 15353.22,
        "balance": 15353.22,
        "allocated": -2500.0,
        "planning_balance": 12853.22,
        "on_order": 0.0,
        "cost_per_pound": 3.45,
        "total_cost": 52968.61,
        "risk_level": "LOW",
        "priority_score": 87146.78
      }
    ],
    "summary": {
      "critical_count": 5,
      "high_count": 12,
      "medium_count": 23,
      "low_count": 145,
      "yarns_with_shortage": 8,
      "total_yarns": 185,
      "yarns_analyzed": 185
    }
  },
  "source": "efab",
  "timestamp": "2025-10-19T14:30:45.123456"
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| yarn_id | integer | Yarn description number (primary key) |
| description | string | Yarn description/name |
| supplier | string | Yarn supplier name |
| color | string | Yarn color name |
| theoretical_balance | float | Calculated: reconciled_qty + added + consumed + adjustments |
| balance | float | Alias for theoretical_balance |
| allocated | float | Allocated to orders (negative = committed) |
| planning_balance | float | theoretical_balance + on_order + allocated |
| on_order | float | Quantity on order from supplier |
| cost_per_pound | float | Average cost per pound |
| total_cost | float | Total inventory value |
| risk_level | string | "CRITICAL", "HIGH", "MEDIUM", or "LOW" |
| priority_score | float | Urgency score (higher = more urgent) |

**Risk Level Determination:**
```python
if planning_balance < 0:
    risk_level = 'CRITICAL'
elif planning_balance < 100:
    risk_level = 'HIGH'
elif planning_balance < 500:
    risk_level = 'MEDIUM'
else:
    risk_level = 'LOW'
```

**Priority Score Calculation:**
```python
risk_weight = {'CRITICAL': 100, 'HIGH': 50, 'MEDIUM': 25, 'LOW': 10}
shortage_magnitude = abs(min(planning_balance, 0))
priority_score = (
    risk_weight[risk_level] * 10000 +
    shortage_magnitude * 100 -
    planning_balance
)
```

#### Response (Forecast Mode)

**Success Response (200 OK):**

```json
{
  "forecast": {
    "predicted_shortages": [
      {
        "yarn_id": 11759,
        "description": "Polyester 150D",
        "forecasted_requirement": 2500.0,
        "current_inventory": -850.5,
        "net_shortage": -3350.5,
        "days_until_shortage": 0,
        "affected_orders": 0,
        "affected_styles": [],
        "urgency": "CRITICAL",
        "priority_score": 1333550.0
      }
    ],
    "total_shortage_count": 15,
    "critical_count": 5,
    "total_shortage_lbs": 12450.75
  },
  "source": "efab",
  "timestamp": "2025-10-19T14:30:45.123456"
}
```

**Forecast Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| yarn_id | integer | Yarn description number |
| description | string | Yarn description/name |
| forecasted_requirement | float | Projected 30-day demand |
| current_inventory | float | Current planning_balance |
| net_shortage | float | current_inventory - forecasted_requirement |
| days_until_shortage | integer | Estimated days until shortage (0 = already short) |
| affected_orders | integer | Number of affected orders (0 if not calculated) |
| affected_styles | array | List of affected styles (empty if not calculated) |
| urgency | string | "CRITICAL", "HIGH", "MEDIUM", or "LOW" |
| priority_score | float | Urgency score for sorting |

**Error Response (500 Internal Server Error):**
```json
{
  "error": "Failed to fetch from eFab"
}
```

---

### /api/inventory/pipeline-summary

**Description:** Provides a consolidated view of fabric inventory across all production stages (G00 → G02 → I01 → F01).

**HTTP Method:** GET
**URL:** `http://localhost:5006/api/inventory/pipeline-summary`
**Authentication:** None required
**Rate Limit:** 60/minute (shared)
**Cache TTL:** 5 minutes

#### Request

**Query Parameters:** None

**Example Request:**
```bash
curl http://localhost:5006/api/inventory/pipeline-summary
```

#### Response

**Success Response (200 OK):**

```json
{
  "status": "success",
  "pipeline": {
    "g00": {
      "stage": "G00 - Raw Greige",
      "items_count": 156,
      "total_on_hand": 45000.0,
      "total_available": 42500.0
    },
    "g02": {
      "stage": "G02 - Greige Processing",
      "items_count": 89,
      "total_on_hand": 28000.0,
      "total_available": 26500.0
    },
    "i01": {
      "stage": "I01 - QC/Inspection",
      "items_count": 67,
      "total_on_hand": 18500.0,
      "total_available": 17800.0
    },
    "f01": {
      "stage": "F01 - Finished Goods",
      "items_count": 234,
      "total_on_hand": 52000.0,
      "total_available": 48900.0
    }
  },
  "total_inventory_yards": 143500.0,
  "production_flow": "G00 → G02 → I01 → F01",
  "timestamp": "2025-10-19T14:30:45.123456"
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| status | string | "success" or "error" |
| pipeline.{stage}.stage | string | Stage name and description |
| pipeline.{stage}.items_count | integer | Number of inventory items in this stage |
| pipeline.{stage}.total_on_hand | float | Total quantity on hand (yards) |
| pipeline.{stage}.total_available | float | Total available quantity (yards) |
| total_inventory_yards | float | Sum of all on_hand quantities |
| production_flow | string | Visual representation of pipeline flow |
| timestamp | string (ISO) | Response generation timestamp |

**Production Pipeline Stages:**

1. **G00 - Raw Greige:** Unprocessed fabric fresh from knitting
2. **G02 - Greige Processing:** Fabric undergoing processing/treatment
3. **I01 - QC/Inspection:** Fabric in quality control inspection
4. **F01 - Finished Goods:** Completed fabric ready for shipment

**Error Response (500 Internal Server Error):**
```json
{
  "error": "Error message details",
  "status": "error"
}
```

---

## Error Handling

### Standard Error Response Format

All endpoints return consistent error responses:

```json
{
  "error": "Error message describing the issue",
  "status": "error"
}
```

### HTTP Status Codes

| Code | Meaning | When Used |
|------|---------|-----------|
| 200 | OK | Successful request |
| 400 | Bad Request | Invalid parameters or missing required fields |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error or eFab connection failure |

### Common Error Scenarios

**1. Rate Limit Exceeded (429):**
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests, please slow down.",
  "retry_after": "60 seconds"
}
```

**2. eFab Connection Failure (500):**
```json
{
  "error": "Failed to fetch from eFab"
}
```

**3. Invalid Session (500):**
```json
{
  "error": "eFab API error: 401 for api/yarn/active"
}
```

**4. JSON Parsing Error (500):**
```json
{
  "error": "JSON parse error for api/knitorder/list: ..."
}
```

### Error Logging

All errors are logged with:
- Timestamp
- Error message
- Stack trace (for exceptions)
- Request details

Example log entry:
```
2025-10-19 14:30:45 - efab_api_server - ERROR - Error in yarn_intelligence: Connection timeout
```

---

## Example Implementations

### Python with requests

```python
import requests
from typing import Dict, List, Optional

class EfabAPIClient:
    """Client for eFab API Server."""

    def __init__(self, base_url: str = "http://localhost:5006"):
        self.base_url = base_url
        self.session = requests.Session()

    def get_knit_orders(self) -> Dict:
        """Fetch all knit orders."""
        response = self.session.get(f"{self.base_url}/api/knit-orders")
        response.raise_for_status()
        return response.json()

    def get_yarn_intelligence(self, forecast: bool = False) -> Dict:
        """Fetch yarn intelligence data.

        Args:
            forecast: If True, returns forecast mode with shortage predictions
        """
        params = {"forecast": "true" if forecast else "false"}
        response = self.session.get(
            f"{self.base_url}/api/yarn-intelligence",
            params=params
        )
        response.raise_for_status()
        return response.json()

    def get_pipeline_summary(self) -> Dict:
        """Fetch inventory pipeline summary."""
        response = self.session.get(
            f"{self.base_url}/api/inventory/pipeline-summary"
        )
        response.raise_for_status()
        return response.json()

    def get_critical_yarns(self) -> List[Dict]:
        """Get list of yarns with CRITICAL risk level."""
        data = self.get_yarn_intelligence()
        yarns = data.get("criticality_analysis", {}).get("yarns", [])
        return [y for y in yarns if y["risk_level"] == "CRITICAL"]

    def get_overdue_orders(self) -> List[Dict]:
        """Get list of overdue knit orders."""
        data = self.get_knit_orders()
        orders = data.get("orders", [])
        return [
            o for o in orders
            if o.get("days_until_due") is not None
            and o["days_until_due"] < 0
        ]

# Usage example
if __name__ == "__main__":
    client = EfabAPIClient()

    # Get critical yarns
    critical = client.get_critical_yarns()
    print(f"Found {len(critical)} critical yarns")
    for yarn in critical[:5]:
        print(f"  - {yarn['yarn_id']}: {yarn['description']}")
        print(f"    Planning Balance: {yarn['planning_balance']:.2f} lbs")

    # Get overdue orders
    overdue = client.get_overdue_orders()
    print(f"\nFound {len(overdue)} overdue orders")
    for order in overdue[:5]:
        print(f"  - {order['serial_number']}: {order['style']}")
        print(f"    Overdue by: {abs(order['days_until_due'])} days")

    # Get pipeline summary
    pipeline = client.get_pipeline_summary()
    print(f"\nTotal inventory: {pipeline['total_inventory_yards']:,.0f} yards")
    for stage_key, stage_data in pipeline['pipeline'].items():
        print(f"  {stage_data['stage']}: {stage_data['total_on_hand']:,.0f} yards")
```

### JavaScript with fetch

```javascript
/**
 * eFab API Client
 */
class EfabAPIClient {
  constructor(baseUrl = 'http://localhost:5006') {
    this.baseUrl = baseUrl;
  }

  async getKnitOrders() {
    const response = await fetch(`${this.baseUrl}/api/knit-orders`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return response.json();
  }

  async getYarnIntelligence(forecast = false) {
    const params = new URLSearchParams({ forecast: forecast.toString() });
    const response = await fetch(
      `${this.baseUrl}/api/yarn-intelligence?${params}`
    );
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return response.json();
  }

  async getPipelineSummary() {
    const response = await fetch(
      `${this.baseUrl}/api/inventory/pipeline-summary`
    );
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return response.json();
  }

  async getCriticalYarns() {
    const data = await this.getYarnIntelligence();
    const yarns = data.criticality_analysis?.yarns || [];
    return yarns.filter(y => y.risk_level === 'CRITICAL');
  }

  async getOverdueOrders() {
    const data = await this.getKnitOrders();
    const orders = data.orders || [];
    return orders.filter(
      o => o.days_until_due !== null && o.days_until_due < 0
    );
  }
}

// Usage example
async function main() {
  const client = new EfabAPIClient();

  try {
    // Get critical yarns
    const critical = await client.getCriticalYarns();
    console.log(`Found ${critical.length} critical yarns`);
    critical.slice(0, 5).forEach(yarn => {
      console.log(`  - ${yarn.yarn_id}: ${yarn.description}`);
      console.log(`    Planning Balance: ${yarn.planning_balance.toFixed(2)} lbs`);
    });

    // Get overdue orders
    const overdue = await client.getOverdueOrders();
    console.log(`\nFound ${overdue.length} overdue orders`);
    overdue.slice(0, 5).forEach(order => {
      console.log(`  - ${order.serial_number}: ${order.style}`);
      console.log(`    Overdue by: ${Math.abs(order.days_until_due)} days`);
    });

    // Get pipeline summary
    const pipeline = await client.getPipelineSummary();
    console.log(`\nTotal inventory: ${pipeline.total_inventory_yards.toLocaleString()} yards`);
    for (const [key, stage] of Object.entries(pipeline.pipeline)) {
      console.log(`  ${stage.stage}: ${stage.total_on_hand.toLocaleString()} yards`);
    }
  } catch (error) {
    console.error('API Error:', error);
  }
}

main();
```

### curl Examples

```bash
#!/bin/bash

# Health check
echo "=== Health Check ==="
curl http://localhost:5006/api/health | jq

# Get knit orders
echo -e "\n=== Knit Orders ==="
curl http://localhost:5006/api/knit-orders | jq '.total'

# Get yarn intelligence (normal mode)
echo -e "\n=== Yarn Intelligence (Normal) ==="
curl http://localhost:5006/api/yarn-intelligence | jq '.criticality_analysis.summary'

# Get yarn intelligence (forecast mode)
echo -e "\n=== Yarn Intelligence (Forecast) ==="
curl "http://localhost:5006/api/yarn-intelligence?forecast=true" | jq '.forecast.total_shortage_count'

# Get pipeline summary
echo -e "\n=== Pipeline Summary ==="
curl http://localhost:5006/api/inventory/pipeline-summary | jq '.total_inventory_yards'

# Get critical yarns only
echo -e "\n=== Critical Yarns ==="
curl -s http://localhost:5006/api/yarn-intelligence | \
  jq '.criticality_analysis.yarns[] | select(.risk_level == "CRITICAL") | {yarn_id, description, planning_balance}'

# Get overdue orders
echo -e "\n=== Overdue Orders ==="
curl -s http://localhost:5006/api/knit-orders | \
  jq '.orders[] | select(.days_until_due != null and .days_until_due < 0) | {serial_number, style, days_until_due}'
```

---

## Additional Information

### Server Configuration

**Location:** `C:\finalee\beverly_knits_erp_v2\src\api\efab_api_server.py`

**Dependencies:**
- Flask (web framework)
- flask-cors (CORS support)
- flask-limiter (rate limiting)
- requests (HTTP client)
- python-dotenv (environment variables)

**Startup Command:**
```bash
python src/api/efab_api_server.py
```

### Environment Variables

```bash
# Required
EFAB_SESSION="your_session_cookie_here"

# Optional
EFAB_BASE_URL="https://efab.bkiapps.com"  # Default
ENABLE_RATE_LIMITING="true"                # Default
API_RATE_LIMIT="60 per minute"             # Default
RATE_LIMIT_STORAGE_URI="memory://"         # Default
SECRETS_FILE="path/to/secrets.json"        # Optional
```

### Health Check

**Endpoint:** `/api/health`
**Method:** GET
**Response:**
```json
{
  "status": "healthy",
  "data_source": "efab_direct",
  "efab_connected": true,
  "timestamp": "2025-10-19T14:30:45.123456"
}
```

### Logging

All API activity is logged to console with:
- Request URLs
- Response status
- Error details
- Cache hits/misses
- Data transformation info

**Log Format:**
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

**Log Levels:**
- INFO: Normal operations, successful requests
- WARNING: Non-fatal issues, data transformation warnings
- ERROR: Request failures, connection errors
- DEBUG: Cache hits, detailed diagnostics

---

## Integration Recommendations

### Best Practices

1. **Respect Rate Limits:** Implement exponential backoff for 429 responses
2. **Use Caching:** Don't poll more frequently than 5-minute cache TTL
3. **Handle Errors:** Always implement proper error handling
4. **Monitor Health:** Periodically check `/api/health` endpoint
5. **Log Requests:** Log all API calls for debugging

### Performance Optimization

1. **Batch Operations:** Combine multiple data needs into single requests
2. **Cache Client-Side:** Implement your own caching layer for frequently accessed data
3. **Use Forecast Mode Wisely:** Only request forecast data when needed
4. **Filter on Client:** Use API data and filter locally instead of polling repeatedly

### Security Considerations

1. **Local Only:** This API is designed for localhost access only
2. **No Public Exposure:** Do not expose port 5006 to the internet
3. **Session Security:** Keep EFAB_SESSION secret and rotate regularly
4. **CORS:** CORS is enabled for local development; restrict in production

---

## Appendix: Complete Endpoint List

Beyond the three primary endpoints documented above, the server provides many additional endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/health | GET | Health check |
| /api/knit-orders | GET | Knit orders list |
| /api/yarn-intelligence | GET | Yarn inventory analysis |
| /api/inventory/pipeline-summary | GET | Inventory pipeline summary |
| /api/time-phased-yarn-po | GET | Time-phased yarn purchase orders |
| /api/production-pipeline | GET | Production pipeline status |
| /api/dashboard-summary | GET | Dashboard summary data |
| /api/comprehensive-kpis | GET | Comprehensive KPIs |
| /api/ml-forecast-detailed | GET | Detailed ML forecasts |
| /api/forecasted-yarn-demand | GET | Forecasted yarn demand |
| /api/advanced-optimization | GET | Advanced optimization data |
| /api/inventory-intelligence-enhanced | GET | Enhanced inventory intelligence |
| /api/forecasted-sales | GET | Sales forecasts |
| /api/forecasted-production | GET | Production forecasts |
| /api/forecast-backtest | GET | Forecast backtesting results |
| /api/inventory/greige/g00 | GET | G00 greige inventory |
| /api/inventory/greige/g02 | GET | G02 greige inventory |
| /api/inventory/inspection/i01 | GET | I01 inspection inventory |
| /api/inventory/finished/f01 | GET | F01 finished goods inventory |
| /api/forecast/comprehensive | GET | Comprehensive forecast data |
| /api/forecast/upload-external | POST | Upload external forecast data |
| /api/forecast/accuracy-report | GET | Forecast accuracy metrics |
| /api/forecast/weight-recommendations | GET | Forecast weight recommendations |
| /api/forecast/proactive-production | GET | Proactive production suggestions |
| /api/production-planning | GET | Production planning data |
| /api/production-suggestions | GET | AI production suggestions |
| /api/material-shortages-real | GET | Real material shortages |
| /api/fabric-forecast-integrated | GET | Fabric forecast integrated view |
| /api/inventory-netting | GET | Inventory netting analysis |
| /api/factory-floor-ai-dashboard | GET | Factory floor AI dashboard |
| /api/machine-assignment-suggestions | GET | Machine assignment AI |
| /api/fabric-inquiry/search | POST | Fabric search |
| /api/retrain-ml | POST | Retrain ML models |
| /api/po-risk-analysis | GET | Purchase order risk analysis |

**Note:** Full documentation for additional endpoints available upon request.

---

**Document Version:** 1.0
**Last Updated:** 2025-10-19
**Maintainer:** Beverly Knits ERP Team
**Server Version:** eFab API Server v2.0
