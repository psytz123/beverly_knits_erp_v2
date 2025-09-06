# eFab API Integration Guide

## Overview

The Beverly Knits ERP system can now integrate directly with the eFab API to fetch real-time data for:
- Sales Orders
- Knit Orders  
- Inventory across warehouses (F01, G00, G02, I01)
- Production Planning

## Setup

### 1. Authentication

The system supports two authentication methods:

#### Method A: Username/Password (Recommended)
```python
from src.data_sync.efab_api_connector import eFabAPIConnector

connector = eFabAPIConnector(
    username="psytz",
    password="Big$cat1"
)
```

#### Method B: Session Cookie
```python
connector = eFabAPIConnector(
    session_cookie="aLfEsRKatML6uTMdgQEvwQchdl6c3LyRbm"
)
```

### 2. Configuration File

Edit `/config/efab_config.json`:
```json
{
    "base_url": "https://efab.bklapps.com",
    "session_cookie": "your_session_cookie_here",
    "sync_interval_minutes": 15,
    "retry_attempts": 3
}
```

## API Endpoints

### Internal ERP Endpoints

Once integrated, the following endpoints are available:

- `GET /api/efab/status` - Check connection status
- `POST /api/efab/sync` - Sync all data from eFab
- `GET /api/efab/sales-orders` - Get sales orders
- `GET /api/efab/knit-orders` - Get knit orders
- `GET /api/efab/inventory/{warehouse}` - Get inventory (F01, G00, G02, I01, or all)
- `GET/POST /api/efab/config` - Manage configuration

### eFab API Endpoints (External)

- `https://efab.bklapps.com/api/sales-order/plan/list` - Sales order planning
- `https://efab.bklapps.com/api/knit-orders` - Knit orders
- `https://efab.bklapps.com/api/inventory/{warehouse}` - Warehouse inventory

## Usage Examples

### Test Connection
```bash
python3 scripts/test_efab_connection.py --test-only
```

### Sync All Data
```bash
python3 scripts/test_efab_connection.py --sync
```

### Via API
```bash
# Check status
curl http://localhost:5006/api/efab/status

# Sync all data
curl -X POST http://localhost:5006/api/efab/sync \
  -H "Content-Type: application/json" \
  -d '{"data_types": ["all"], "force_refresh": true}'

# Get sales orders
curl http://localhost:5006/api/efab/sales-orders

# Get inventory for warehouse F01
curl http://localhost:5006/api/efab/inventory/F01
```

## Data Mapping

The eFab data is automatically mapped to existing ERP structures:

### Sales Orders → eFab_SO_List.csv
- Order details
- Customer information
- Delivery dates
- Quantities

### Knit Orders → eFab_Knit_Orders.csv
- Production orders
- Machine assignments
- Work center mappings
- Production status

### Inventory → eFab_Inventory_{warehouse}.csv
- Stock levels by warehouse
- Material codes
- Available quantities
- Production stages (G00, G02, I01, F01)

## Integration with Existing System

The eFab data integrates seamlessly with:
- **Inventory Intelligence**: Real-time stock levels
- **Production Planning**: Live order data
- **Machine Planning**: Current assignments
- **Yarn Management**: Material availability

## Troubleshooting

### Connection Issues

1. **DNS Resolution Error**
   - Ensure you're on the corporate network/VPN
   - Check firewall settings
   - Verify the domain `efab.bklapps.com` is accessible

2. **Authentication Failed**
   - Check username/password in config
   - Ensure session cookie is valid (expires after inactivity)
   - Try re-authenticating with credentials

3. **No Data Returned**
   - Check API endpoint paths
   - Verify permissions for your account
   - Check network connectivity

### Data Sync Issues

1. **CSV Files Not Created**
   - Check write permissions in data directory
   - Verify path exists: `/data/production/5/ERP Data/`
   - Check disk space

2. **Cached Data**
   - Force refresh: Add `?force_refresh=true` to API calls
   - Clear cache: `rm -rf /tmp/bki_cache/*`

## Security Considerations

- Store credentials securely (use environment variables in production)
- Session cookies expire - implement auto-renewal
- Use HTTPS for all API communications
- Implement rate limiting to avoid overwhelming the API
- Log all API access for audit trails

## Performance

- Data is cached for 15 minutes by default
- Parallel fetching for multiple warehouses
- Retry logic with exponential backoff
- Compression supported for large datasets

## Future Enhancements

- [ ] Webhook support for real-time updates
- [ ] Bi-directional sync (push updates to eFab)
- [ ] Automated scheduling with cron
- [ ] Data validation and reconciliation
- [ ] Historical data archiving