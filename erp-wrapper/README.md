`GET`

# eFab ERP Wrapper Service

A FastAPI-based proxy service that handles authentication and session management for the eFab ERP API, now integrated with Beverly Knits ERP v2 for API-only data access.

## Features

- ✅ **Automatic Session Management**: Handles login and cookie refresh automatically
- ✅ **Caching**: 5-minute cache for frequently accessed data
- ✅ **Async API**: Non-blocking FastAPI endpoints
- ✅ **Health Checks**: Built-in health monitoring
- ✅ **Docker Ready**: Containerized for easy deployment
- ✅ **Auto-Retry**: Automatic re-login on session expiry
- ✅ **ERP Integration**: Primary data source for Beverly Knits ERP (replaces file access)
- ✅ **Fallback Support**: Graceful degradation when APIs unavailable

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 2: Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ERP_USERNAME="psytz"
export ERP_PASSWORD="big$cat"

# Run the service
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

### Health & Status

- `GET /` - Service information
- `GET /health` - Health check
- `GET /api/status` - Detailed service status

### Data Endpoints (Beverly Knits ERP Integration)

**Primary wrapper endpoints (used by main ERP):**

- `GET /api/sales-orders` - Fetch sales orders ✅ **ERP Integrated**
- `GET /api/knit-orders` - Fetch knit orders ✅ **ERP Integrated**
- `GET /api/yarn/active` - Active yarn inventory ✅ **ERP Integrated**
- `GET /api/styles` - Fetch styles

**Direct eFab API endpoints:**

- `https://efab.bkiapps.com/api/greige/g02` - Greige stage 2 inventory
- `https://efab.bkiapps.com/api/finished/i01` - QC/Inspection inventory
- `https://efab.bkiapps.com/api/greige/g00` - Greige stage 1 inventory
- `https://efab.bkiapps.com/api/finished/f01` - Finished goods inventory

**Reporting endpoints:**

- `GET /api/report/yarn_demand_ko` - Yarn demand (KO format)
- `GET /api/report/yarn_demand` - Standard yarn demand
- `GET /api/yarn-po` - Yarn purchase orders
- `GET /api/report/yarn_expected` - Expected yarn deliveries

### Management

- `POST /api/sync-all` - Sync all data in background
- `POST /api/cache/clear` - Clear cache

### Query Parameters

All data endpoints support:

- `force_refresh=true` - Bypass cache and fetch fresh data

## Configuration

Create a `.env` file based on `.env.example`:

```env
# eFab ERP Configuration
ERP_BASE_URL=https://efab.bkiapps.com
ERP_USERNAME=psytz
ERP_PASSWORD=big$cat

# Optional: Disable SSL verification for self-signed certs
VERIFY_SSL=False
```

## Integration with Beverly Knits ERP ✅ **COMPLETED**

**As of September 2025**, Beverly Knits ERP v2 now uses this wrapper as the **primary data source**, replacing direct file access.

### Integration Details

**Main ERP** (http://localhost:5006) → **Wrapper** (http://localhost:8000) → **eFab APIs**

```python
# APIDataLoader class in beverly_comprehensive_erp.py
class APIDataLoader:
    def __init__(self, wrapper_url="http://localhost:8000"):
        self.wrapper_url = wrapper_url
        # Setup session with retry strategy
  
    def load_yarn_inventory(self):
        response = self.session.get(f"{self.wrapper_url}/api/yarn/active")
        return pd.DataFrame(response.json()['data'])
      
    def load_knit_orders(self):
        response = self.session.get(f"{self.wrapper_url}/api/knit-orders") 
        return pd.DataFrame(response.json()['data'])
```

### Current Status

- ✅ **Yarn inventory**: API-first with file fallback
- ✅ **Knit orders**: API-first with file fallback
- ✅ **Sales orders**: API-first with file fallback
- ✅ **BOM data**: File-based with API framework ready
- ✅ **246 yarns tracked** via corrected API data
- ✅ **5 critical shortages identified** (64,377 lbs)

## Architecture

```
  Browser/Client
        ↓
  Beverly Knits ERP
        ↓
  ERP Wrapper (This Service)
        ↓
  [Session Management]
        ↓
  eFab ERP API
```

## Session Management

The wrapper handles:

1. Initial login with username/password
2. Cookie storage and reuse
3. Automatic re-login on 401/403 responses
4. Session persistence across restarts

## Caching Strategy

- Default TTL: 5 minutes
- Cache invalidation on force_refresh
- Background sync updates cache
- Separate cache per endpoint/warehouse

## Monitoring

```bash
# Check service health
curl http://localhost:8000/health

# View service status
curl http://localhost:8000/api/status | jq

# Monitor logs
docker-compose logs -f erp-wrapper
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Troubleshooting

### Connection Issues

1. **DNS Resolution**: Ensure the domain resolves

   ```bash
   nslookup efab.bkiapps.com
   ```
2. **SSL Certificate**: If using self-signed certs

   ```env
   VERIFY_SSL=False
   ```
3. **Session Expired**: The wrapper auto-refreshes, but you can force it

   ```bash
   curl -X POST http://localhost:8000/api/cache/clear
   ```

### Performance

- Use caching: Don't set `force_refresh=true` unless needed
- Background sync: Use `/api/sync-all` to pre-populate cache
- Monitor cache hit rates in `/api/status`

## Development

```bash
# Run with auto-reload
uvicorn app.main:app --reload --port 8000 --log-level debug

# Run tests
pytest tests/

# Format code
black app/
```

## Security Notes

- Store credentials in environment variables or secrets manager
- Use HTTPS for production deployment
- Implement rate limiting for production
- Add authentication to wrapper endpoints if exposed publicly

## License

Internal use only - Beverly Knits
