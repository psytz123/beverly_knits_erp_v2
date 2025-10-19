# eFab API Quick Reference

**Server:** http://localhost:5006
**Last Updated:** 2025-10-19

---

## Quick Start

```bash
# Start the server
python src/api/efab_api_server.py

# Health check
curl http://localhost:5006/api/health
```

---

## Three Primary Endpoints

### 1. Knit Orders
```bash
# Get all knit orders
curl http://localhost:5006/api/knit-orders

# Python
import requests
response = requests.get('http://localhost:5006/api/knit-orders')
data = response.json()
orders = data['orders']
```

**Returns:**
- List of knit orders sorted by urgency
- Completion percentages
- Days until due
- Customer and style information

**Key Fields:**
- `serial_number` - Order number
- `style` - Product style
- `customer` - Customer name
- `completion_percentage` - 0-100%
- `days_until_due` - Negative = overdue
- `balance_lbs` - Remaining quantity

---

### 2. Yarn Intelligence

**Normal Mode (Current Inventory):**
```bash
curl http://localhost:5006/api/yarn-intelligence

# Python
data = requests.get('http://localhost:5006/api/yarn-intelligence').json()
yarns = data['criticality_analysis']['yarns']
summary = data['criticality_analysis']['summary']
```

**Forecast Mode (Future Shortages):**
```bash
curl "http://localhost:5006/api/yarn-intelligence?forecast=true"

# Python
params = {'forecast': 'true'}
data = requests.get('http://localhost:5006/api/yarn-intelligence', params=params).json()
shortages = data['forecast']['predicted_shortages']
```

**Key Fields:**
- `yarn_id` - Yarn identifier
- `description` - Yarn name
- `theoretical_balance` - Current on-hand
- `planning_balance` - Including on-order and allocated
- `risk_level` - CRITICAL, HIGH, MEDIUM, LOW
- `priority_score` - Higher = more urgent

**Risk Levels:**
- `CRITICAL`: planning_balance < 0
- `HIGH`: planning_balance < 100
- `MEDIUM`: planning_balance < 500
- `LOW`: planning_balance >= 500

---

### 3. Inventory Pipeline Summary

```bash
curl http://localhost:5006/api/inventory/pipeline-summary

# Python
data = requests.get('http://localhost:5006/api/inventory/pipeline-summary').json()
pipeline = data['pipeline']
total_yards = data['total_inventory_yards']
```

**Returns:**
- Inventory at each production stage
- G00 (Raw Greige) → G02 (Processing) → I01 (QC) → F01 (Finished)

**Key Fields:**
- `pipeline.g00.total_on_hand` - Raw greige yards
- `pipeline.g02.total_on_hand` - Processing yards
- `pipeline.i01.total_on_hand` - Inspection yards
- `pipeline.f01.total_on_hand` - Finished goods yards
- `total_inventory_yards` - Sum of all stages

---

## Authentication

No authentication required for client calls to localhost:5006.

Server authenticates to eFab using session cookie stored in environment.

---

## Rate Limiting

**Default:** 60 requests per minute per IP

**Rate limit response (429):**
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests, please slow down.",
  "retry_after": "60 seconds"
}
```

**Bypass:** Set `ENABLE_RATE_LIMITING=false` in environment

---

## Caching

**TTL:** 5 minutes

Responses are cached for 5 minutes. Repeated requests within this window return cached data.

**Cache key:** Endpoint + query parameters

---

## Error Handling

**Standard error response:**
```json
{
  "error": "Error message",
  "status": "error"
}
```

**HTTP Status Codes:**
- `200` - Success
- `400` - Bad request
- `429` - Rate limit exceeded
- `500` - Server error or eFab connection failure

---

## Common Use Cases

### Get Critical Yarns
```python
import requests

def get_critical_yarns():
    """Get yarns with CRITICAL risk level."""
    response = requests.get('http://localhost:5006/api/yarn-intelligence')
    data = response.json()
    yarns = data['criticality_analysis']['yarns']
    return [y for y in yarns if y['risk_level'] == 'CRITICAL']

critical = get_critical_yarns()
for yarn in critical:
    print(f"{yarn['yarn_id']}: {yarn['description']}")
    print(f"  Shortage: {yarn['planning_balance']:.2f} lbs")
```

### Get Overdue Orders
```python
def get_overdue_orders():
    """Get orders past their due date."""
    response = requests.get('http://localhost:5006/api/knit-orders')
    data = response.json()
    orders = data['orders']
    return [
        o for o in orders
        if o.get('days_until_due') is not None and o['days_until_due'] < 0
    ]

overdue = get_overdue_orders()
for order in overdue:
    print(f"{order['serial_number']}: {order['style']}")
    print(f"  Overdue by: {abs(order['days_until_due'])} days")
    print(f"  Completion: {order['completion_percentage']:.1f}%")
```

### Get Total Inventory Value
```python
def get_total_inventory_value():
    """Calculate total yarn inventory value."""
    response = requests.get('http://localhost:5006/api/yarn-intelligence')
    data = response.json()
    yarns = data['criticality_analysis']['yarns']
    return sum(y['total_cost'] for y in yarns)

total_value = get_total_inventory_value()
print(f"Total yarn inventory value: ${total_value:,.2f}")
```

### Monitor Pipeline Flow
```python
def get_pipeline_summary():
    """Get inventory at each production stage."""
    response = requests.get('http://localhost:5006/api/inventory/pipeline-summary')
    data = response.json()
    pipeline = data['pipeline']

    for stage_key in ['g00', 'g02', 'i01', 'f01']:
        stage = pipeline[stage_key]
        print(f"{stage['stage']}")
        print(f"  Items: {stage['items_count']}")
        print(f"  On Hand: {stage['total_on_hand']:,.0f} yards")
        print(f"  Available: {stage['total_available']:,.0f} yards")

get_pipeline_summary()
```

### Forecast Future Shortages
```python
def get_predicted_shortages(urgency_filter='CRITICAL'):
    """Get predicted yarn shortages."""
    params = {'forecast': 'true'}
    response = requests.get(
        'http://localhost:5006/api/yarn-intelligence',
        params=params
    )
    data = response.json()
    shortages = data['forecast']['predicted_shortages']

    if urgency_filter:
        shortages = [s for s in shortages if s['urgency'] == urgency_filter]

    return shortages

critical_shortages = get_predicted_shortages('CRITICAL')
print(f"Critical shortages: {len(critical_shortages)}")
for shortage in critical_shortages[:10]:
    print(f"{shortage['yarn_id']}: {shortage['description']}")
    print(f"  Net shortage: {shortage['net_shortage']:.2f} lbs")
    print(f"  Days until shortage: {shortage['days_until_shortage']}")
```

---

## Response Format Examples

### Knit Orders Response
```json
{
  "orders": [
    {
      "serial_number": "KO-2024-001",
      "style": "Style-ABC",
      "customer": "Customer Inc.",
      "completion_percentage": 75.0,
      "days_until_due": 5,
      "balance_lbs": 500.0
    }
  ],
  "total": 42,
  "source": "efab",
  "status": "ok"
}
```

### Yarn Intelligence Response (Normal)
```json
{
  "criticality_analysis": {
    "yarns": [
      {
        "yarn_id": 18865,
        "description": "Cotton Blend 30/1",
        "theoretical_balance": 15353.22,
        "planning_balance": 12853.22,
        "risk_level": "LOW"
      }
    ],
    "summary": {
      "critical_count": 5,
      "high_count": 12,
      "total_yarns": 185
    }
  }
}
```

### Yarn Intelligence Response (Forecast)
```json
{
  "forecast": {
    "predicted_shortages": [
      {
        "yarn_id": 11759,
        "description": "Polyester 150D",
        "net_shortage": -3350.5,
        "days_until_shortage": 0,
        "urgency": "CRITICAL"
      }
    ],
    "total_shortage_count": 15,
    "critical_count": 5
  }
}
```

### Pipeline Summary Response
```json
{
  "status": "success",
  "pipeline": {
    "g00": {
      "stage": "G00 - Raw Greige",
      "total_on_hand": 45000.0
    },
    "g02": {
      "stage": "G02 - Greige Processing",
      "total_on_hand": 28000.0
    }
  },
  "total_inventory_yards": 143500.0
}
```

---

## Troubleshooting

### Server won't start
```bash
# Check session cookie
echo $EFAB_SESSION

# If empty, run login script
python scripts/efab_login.py
```

### Connection refused
```bash
# Verify server is running
curl http://localhost:5006/api/health

# Check port
netstat -an | grep 5006
```

### Empty responses
```bash
# Check eFab connection
curl http://localhost:5006/api/health | jq '.efab_connected'

# Should return: true
```

### Rate limit errors
```python
import time
import requests

def fetch_with_retry(url, max_retries=3):
    """Fetch with automatic retry on rate limit."""
    for attempt in range(max_retries):
        response = requests.get(url)
        if response.status_code == 429:
            time.sleep(60)  # Wait 1 minute
            continue
        return response.json()
    raise Exception("Max retries exceeded")
```

---

## Environment Variables

```bash
# Required
export EFAB_SESSION="your_session_cookie"

# Optional
export EFAB_BASE_URL="https://efab.bkiapps.com"
export ENABLE_RATE_LIMITING="true"
export API_RATE_LIMIT="60 per minute"
export RATE_LIMIT_STORAGE_URI="memory://"
```

---

## Complete Client Example

```python
#!/usr/bin/env python3
"""
Complete eFab API client example.
"""
import requests
from typing import List, Dict, Optional


class EfabClient:
    """Simple eFab API client."""

    def __init__(self, base_url: str = "http://localhost:5006"):
        self.base_url = base_url
        self.session = requests.Session()

    def health(self) -> Dict:
        """Check server health."""
        return self._get("/api/health")

    def knit_orders(self) -> List[Dict]:
        """Get all knit orders."""
        data = self._get("/api/knit-orders")
        return data.get("orders", [])

    def yarn_intelligence(self, forecast: bool = False) -> Dict:
        """Get yarn intelligence data."""
        params = {"forecast": "true" if forecast else "false"}
        return self._get("/api/yarn-intelligence", params=params)

    def pipeline_summary(self) -> Dict:
        """Get inventory pipeline summary."""
        return self._get("/api/inventory/pipeline-summary")

    def critical_yarns(self) -> List[Dict]:
        """Get yarns with CRITICAL risk."""
        data = self.yarn_intelligence()
        yarns = data.get("criticality_analysis", {}).get("yarns", [])
        return [y for y in yarns if y["risk_level"] == "CRITICAL"]

    def overdue_orders(self) -> List[Dict]:
        """Get overdue knit orders."""
        orders = self.knit_orders()
        return [
            o for o in orders
            if o.get("days_until_due") is not None and o["days_until_due"] < 0
        ]

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request."""
        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()


# Usage
if __name__ == "__main__":
    client = EfabClient()

    # Health check
    health = client.health()
    print(f"Server status: {health['status']}")
    print(f"eFab connected: {health['efab_connected']}")

    # Get critical yarns
    critical = client.critical_yarns()
    print(f"\nCritical yarns: {len(critical)}")
    for yarn in critical[:5]:
        print(f"  {yarn['yarn_id']}: {yarn['description']}")
        print(f"    Balance: {yarn['planning_balance']:.2f} lbs")

    # Get overdue orders
    overdue = client.overdue_orders()
    print(f"\nOverdue orders: {len(overdue)}")
    for order in overdue[:5]:
        print(f"  {order['serial_number']}: {order['style']}")
        print(f"    {abs(order['days_until_due'])} days overdue")
        print(f"    {order['completion_percentage']:.1f}% complete")

    # Pipeline summary
    pipeline = client.pipeline_summary()
    print(f"\nTotal inventory: {pipeline['total_inventory_yards']:,.0f} yards")
    for stage_key, stage in pipeline['pipeline'].items():
        print(f"  {stage['stage']}: {stage['total_on_hand']:,.0f} yards")
```

---

## Additional Resources

**Full Documentation:**
- C:\finalee\beverly_knits_erp_v2\.agent-workspace\outputs\analysis\efab_api_integration_guide.md

**Technical Analysis:**
- C:\finalee\beverly_knits_erp_v2\.agent-workspace\outputs\analysis\efab_api_technical_analysis.md

**Source Code:**
- C:\finalee\beverly_knits_erp_v2\src\api\efab_api_server.py

---

**Quick Reference Version:** 1.0
**Last Updated:** 2025-10-19
