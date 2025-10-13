# eFab API Database Integration

Complete database solution for pulling and storing data from the eFab API.

## Features

- **SQLAlchemy Models** - Full database schema for eFab data
- **API Sync Service** - Pull data from eFab API endpoints
- **Scheduler** - Automated periodic data synchronization
- **RESTful API** - Flask endpoints for database operations
- **Session Management** - Handle eFab authentication cookies

## Quick Start

### 1. Install Dependencies

```bash
pip install sqlalchemy psycopg2-binary schedule
```

### 2. Set Environment Variables

```bash
# Database connection
export DATABASE_URL="postgresql://user:password@localhost/efab_erp"

# eFab session cookie (get from browser)
export EFAB_SESSION="aNEM2YqIXevF7IvZ13r68JFeSlsVo1Lh"
```

### 3. Initialize Database

```bash
# Create tables
python src/database/setup.py --init

# Or reset existing database
python src/database/setup.py --reset
```

### 4. Run Initial Sync

```bash
# Pull data from eFab API
python src/database/setup.py --sync
```

### 5. Start Scheduler

```bash
# Start automated sync (default: every 2 hours)
python src/database/setup.py --schedule

# Or specify interval
python src/database/setup.py --schedule --interval 4
```

## Database Schema

### Core Tables

- **cf_versions** - Core fabric/style versions
- **yarn_requirements** - Yarn requirements per CF version
- **yarn_inventory** - Current yarn stock levels
- **production_orders** - Manufacturing orders
- **machine_assignments** - Machine scheduling
- **knit_orders** - Knitting production orders
- **sales_activity** - Historical sales data
- **api_sync** - Sync history and status

## API Endpoints

### Database Status
```bash
GET /api/db/status
```

### Manual Sync
```bash
POST /api/db/sync/manual
{
  "session_cookie": "your_efab_session_cookie"
}
```

### Scheduler Control
```bash
# Start scheduler
POST /api/db/sync/scheduler/start
{
  "session_cookie": "your_cookie",
  "interval_hours": 2
}

# Stop scheduler
POST /api/db/sync/scheduler/stop

# Get status
GET /api/db/sync/scheduler/status
```

### Data Queries
```bash
# CF Versions
GET /api/db/cf-versions?limit=100&offset=0&style=ABC123

# Production Orders
GET /api/db/production-orders?status=active&machine=161

# Yarn Inventory
GET /api/db/yarn-inventory?yarn_code=Y123&low_stock=true
```

## Integration with Main ERP

Add to your main Flask app:

```python
from database.api import register_database_api
from database.scheduler import init_scheduler

# Register API endpoints
register_database_api(app)

# Start scheduler on app startup
@app.before_first_request
def start_database_sync():
    session_cookie = os.environ.get('EFAB_SESSION')
    if session_cookie:
        init_scheduler(
            get_database_url(),
            session_cookie,
            sync_interval_hours=2,
            auto_start=True
        )
```

## Usage Examples

### Python Code

```python
from database.config import get_session
from database.models import CFVersion, ProductionOrder

# Query CF versions
with get_session() as session:
    versions = session.query(CFVersion).filter(
        CFVersion.style_number.contains('ABC')
    ).all()

    for v in versions:
        print(f"{v.style_number}: {v.description}")

# Get production orders
with get_session() as session:
    orders = session.query(ProductionOrder).filter(
        ProductionOrder.status == 'active'
    ).order_by(ProductionOrder.due_date).all()

    for order in orders:
        print(f"{order.order_number}: Due {order.due_date}")
```

### Manual Sync

```python
from database.efab_api_sync import EFabAPISync

# Initialize sync service
sync = EFabAPISync(database_url, session_cookie)

# Sync CF versions
versions = sync.fetch_cf_versions(base_id=1611, limit=50)
created, updated = sync.sync_cf_versions(versions)
print(f"Synced {created} new, {updated} updated")

# Full sync
results = sync.sync_all()
print(f"Sync complete: {results}")
```

## Session Cookie Management

The eFab session cookie expires after ~24 hours. To get a new cookie:

1. Log into eFab in your browser
2. Open Developer Tools (F12)
3. Go to Application/Storage > Cookies
4. Copy the `dancer.session` cookie value
5. Update environment variable:
   ```bash
   export EFAB_SESSION="new_cookie_value"
   ```

## Troubleshooting

### Database Connection Issues
```bash
# Test connection
python src/database/config.py

# Check PostgreSQL is running
sudo systemctl status postgresql

# Create database if needed
createdb efab_erp
```

### Sync Failures
```bash
# Check status
python src/database/setup.py --status

# View logs
tail -f logs/database_sync.log

# Test API connection
curl -H "Cookie: dancer.session=$EFAB_SESSION" \
     https://efab.bkiapps.com/api/cf_version/from_base/1611
```

### Performance Tuning
```python
# Adjust pool size in config.py
engine = create_engine(
    database_url,
    pool_size=20,  # Increase for more connections
    max_overflow=40,
    pool_pre_ping=True
)
```

## Architecture

```
┌─────────────────┐
│   eFab API      │
└────────┬────────┘
         │ HTTPS
         ▼
┌─────────────────┐
│  Sync Service   │◄───── Scheduler (every 2 hrs)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PostgreSQL    │
│    Database     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Flask API     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ERP Dashboard  │
└─────────────────┘
```

## Security Notes

- Store session cookies securely (use environment variables)
- Rotate cookies regularly (they expire in ~24 hours)
- Use HTTPS for all API communications
- Implement rate limiting for API calls
- Log all sync activities for audit trail

## License

Proprietary - Beverly Knits ERP v2