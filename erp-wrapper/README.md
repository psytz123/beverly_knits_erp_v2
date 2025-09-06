# eFab ERP Wrapper Service

A FastAPI-based proxy service that handles authentication and session management for the eFab ERP API.

## Features

- ✅ **Automatic Session Management**: Handles login and cookie refresh automatically
- ✅ **Caching**: 5-minute cache for frequently accessed data
- ✅ **Async API**: Non-blocking FastAPI endpoints
- ✅ **Health Checks**: Built-in health monitoring
- ✅ **Docker Ready**: Containerized for easy deployment
- ✅ **Auto-Retry**: Automatic re-login on session expiry

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

### Data Endpoints

- `GET /api/sales-orders` - Fetch sales orders
- `GET /api/knit-orders` - Fetch knit orders
- `GET /api/inventory/{warehouse}` - Fetch inventory (F01/G00/G02/I01/all)
- `GET /api/styles` - Fetch styles

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
ERP_BASE_URL=https://efab.bklapps.com
ERP_USERNAME=your_username
ERP_PASSWORD=your_password

# Optional: Disable SSL verification for self-signed certs
VERIFY_SSL=False
```

## Integration with Beverly Knits ERP

Update your Beverly Knits ERP to use this wrapper:

```python
# In efab_api_connector.py
class eFabAPIConnector:
    def __init__(self):
        # Use the wrapper instead of direct connection
        self.base_url = "http://localhost:8000"  # Wrapper URL
        self.session = requests.Session()
    
    def get_sales_order_plan_list(self):
        response = self.session.get(f"{self.base_url}/api/sales-orders")
        data = response.json()
        return pd.DataFrame(data['data'])
```

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
   nslookup efab.bklapps.com
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