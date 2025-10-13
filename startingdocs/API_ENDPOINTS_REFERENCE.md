# Beverly Knits ERP - API Endpoints Reference

**Document Type**: API Reference Guide
**Created**: 2025-09-28
**Version**: V2.0.0
**Purpose**: Comprehensive API endpoint documentation for Beverly Knits textile manufacturing system

## Table of Contents

1. [External System APIs](#external-system-apis)
2. [Internal Microservice APIs](#internal-microservice-apis)
3. [Production & Inventory Endpoints](#production--inventory-endpoints)
4. [Data Mapping & Transformations](#data-mapping--transformations)
5. [Authentication & Security](#authentication--security)

## External System APIs

### eFab ERP Integration

**Base Configuration:**
```yaml
Base URL: https://efab.bkiapps.com
API Prefix: /api
Authentication: Session-based (dancer.session cookie)
Content-Type: application/json
Timeout: 30 seconds
Retry: 3 attempts with exponential backoff
```

**Primary Endpoints:**

| Endpoint | Method | Purpose | Response Format |
|----------|--------|---------|----------------|
| `/api/sales-order/plan/list` | GET | Sales orders with planning data | JSON array of orders |
| `/api/knitorder/list` | GET | Active knit production orders | JSON array of knit orders |
| `/api/yarn/active` | GET | Current yarn inventory levels | JSON with shortage analysis |
| `/api/yarn-po` | GET | Yarn purchase order status | JSON array of POs |
| `/api/styles` | GET | Master fabric styles catalog | JSON array of styles |

### QuadS System Integration

**Base Configuration:**
```yaml
Base URL: https://quads.bkiapps.com
API Prefix: /api
Authentication: Session token
Content-Type: application/json
Timeout: 30 seconds
```

**Style Management Endpoints:**

| Endpoint | Method | Purpose | Response Format |
|----------|--------|---------|----------------|
| `/api/styles/greige/active` | GET | Active greige fabric styles | JSON array of greige styles |
| `/api/styles/finished/active` | GET | Active finished fabric styles | JSON array of finished styles |

## Internal Microservice APIs

### Service Registry

All internal services follow RESTful patterns with standardized responses:

```json
{
  "status": "success|error",
  "data": {},
  "message": "Optional message",
  "timestamp": "2025-09-28T10:00:00Z",
  "request_id": "uuid"
}
```

## Production & Inventory Endpoints

### Inventory Stage Flow

Beverly Knits uses a four-stage inventory management system:

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│   G00   │ --> │   G02   │ --> │   I01   │ --> │   F01   │
│Raw Greige│     │Processed│     │QC/Inspect│     │Finished │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
```

### Stage-Specific Endpoints

#### Greige Inventory (G00 & G02)

| Endpoint | Stage | Description | Key Fields |
|----------|-------|-------------|------------|
| `/api/greige/g00` | G00 | Raw greige off loom | style_number, yards, rolls |
| `/api/greige/g02` | G02 | Dyed/finished greige | style_number, color, yards |

#### Quality & Finished (I01 & F01)

| Endpoint | Stage | Description | Key Fields |
|----------|-------|-------------|------------|
| `/api/finished/i01` | I01 | Awaiting QC inspection | style_number, test_required |
| `/api/finished/f01` | F01 | QC passed, ready to ship | style_number, grade, location |

### Yarn Management Endpoints

| Endpoint | Purpose | Key Metrics |
|----------|---------|-------------|
| `/api/report/yarn_demand` | Total yarn requirements | demand_lbs, lead_time |
| `/api/report/yarn_demand_ko` | Yarn by knit order | order_id, yarn_id, quantity |
| `/api/report/yarn_expected` | Expected deliveries | po_number, eta, quantity |

## Data Mapping & Transformations

### Standard Field Mappings

| Source System | Source Field | API Field | Data Type | Transformation |
|---------------|--------------|-----------|-----------|----------------|
| eFab | `Style #` | `style_number` | VARCHAR(50) | Remove spaces |
| eFab | `cFVersion` + `fBase` | `fabric_version` | VARCHAR(100) | Concatenate |
| QuadS | `Desc#` | `yarn_id` | VARCHAR(50) | Direct map |
| eFab | `Planning Balance` | `planning_balance` | DECIMAL(15,3) | Parse decimal |
| QuadS | `BOM%` | `bom_percent` | DECIMAL(5,2) | Multiply by 100 if <1 |

### Unit Conversions

```javascript
// Textile industry standard conversions
const conversions = {
  // Weight
  lbsToKg: (lbs) => lbs * 0.453592,
  kgToLbs: (kg) => kg * 2.20462,

  // Length
  yardsToMeters: (yards) => yards * 0.9144,
  metersToYards: (meters) => meters * 1.09361,

  // Fabric weight
  gsmToOzYd2: (gsm) => gsm * 0.0295,
  ozYd2ToGsm: (oz) => oz * 33.906
};
```

### Business Calculations

```sql
-- Core inventory calculation
Planning_Balance = On_Hand - Allocated + On_Order

-- Yarn demand for production
Yarn_Required = (Fabric_Yards * BOM_Percent / 100) * Fabric_Weight_Per_Yard

-- Machine utilization
Utilization_Percent = (Running_Hours / Available_Hours) * 100
```

## Authentication & Security

### API Authentication Methods

#### External APIs (eFab/QuadS)
```yaml
Method: Session-based
Flow:
  1. POST to /login with credentials
  2. Receive session cookie
  3. Include cookie in all requests
  4. Refresh before expiry (30 min)
```

#### Internal Microservices
```yaml
Method: JWT Bearer Token
Flow:
  1. Obtain token from Auth Service
  2. Include in Authorization header
  3. Token expires in 1 hour
  4. Refresh using refresh token
```

### Security Headers

All API requests should include:

```http
Authorization: Bearer <jwt-token>
X-Request-ID: <uuid>
X-Correlation-ID: <trace-id>
Content-Type: application/json
Accept: application/json
```

### Rate Limiting

| Service Tier | Requests/Minute | Requests/Hour | Burst Limit |
|--------------|----------------|---------------|-------------|
| Basic | 100 | 1,000 | 10 |
| Premium | 500 | 10,000 | 50 |
| Enterprise | 2,000 | 50,000 | 200 |
| Unlimited | No limit | No limit | 1,000 |

### Error Response Format

```json
{
  "error": {
    "code": "ERR_CODE",
    "message": "Human readable message",
    "details": {
      "field": "specific_field",
      "reason": "validation_failed"
    },
    "timestamp": "2025-09-28T10:00:00Z",
    "request_id": "uuid"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description | Resolution |
|------|-------------|-------------|------------|
| `AUTH_FAILED` | 401 | Authentication failed | Check credentials |
| `FORBIDDEN` | 403 | Insufficient permissions | Verify access rights |
| `NOT_FOUND` | 404 | Resource not found | Check resource ID |
| `RATE_LIMITED` | 429 | Too many requests | Implement backoff |
| `VALIDATION_ERROR` | 400 | Invalid request data | Fix request format |
| `INTERNAL_ERROR` | 500 | Server error | Retry with backoff |

## API Versioning Strategy

All APIs follow semantic versioning:

- **Current Version**: v2
- **URL Pattern**: `/api/v{version}/{resource}`
- **Deprecation Policy**: 6 months notice
- **Backward Compatibility**: Maintained for 1 major version

## WebSocket Events

For real-time updates, connect to WebSocket endpoints:

```javascript
// Connection
ws://gateway:8000/ws

// Subscribe to events
{
  "action": "subscribe",
  "channels": ["inventory", "production", "quality"]
}

// Event format
{
  "event": "inventory.updated",
  "data": {
    "stage": "G00",
    "style_number": "ABC123",
    "quantity_change": -100
  },
  "timestamp": "2025-09-28T10:00:00Z"
}
```

## Health Check Endpoints

All services expose standard health checks:

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `/health` | Basic health | `{"status": "healthy"}` |
| `/ready` | Readiness check | `{"ready": true, "dependencies": {...}}` |
| `/metrics` | Prometheus metrics | Metrics in Prometheus format |

---

*This API reference document serves as the comprehensive guide for all Beverly Knits ERP API integrations. For implementation examples, refer to the specific service documentation.*