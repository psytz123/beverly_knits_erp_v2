# Beverly Knits ERP Dashboard - Deployment Guide

## Overview

This guide provides complete instructions for deploying and running the Beverly Knits ERP consolidated dashboard system.

## System Architecture

```
┌─────────────────────────────────────────────────┐
│         Beverly Knits ERP System                │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐         ┌─────────────────┐  │
│  │   Frontend   │         │    Backend      │  │
│  │  Dashboard   │◄────────┤   API Server    │  │
│  │  (Port 8080) │  HTTP   │   (Port 5006)   │  │
│  └──────────────┘         └─────────────────┘  │
│         │                          │            │
│         │                          │            │
│         ▼                          ▼            │
│  consolidated_dashboard    database_api_server │
│  _visual_preserved.html          .py           │
│                                   │             │
│                                   ▼             │
│                            ┌─────────────┐     │
│                            │  Database   │     │
│                            │ PostgreSQL  │     │
│                            └─────────────┘     │
└─────────────────────────────────────────────────┘
```

## Prerequisites

### Required Software

1. **Python 3.8+**
   - Download from: https://www.python.org/downloads/
   - Verify installation: `python --version`

2. **Python Packages** (Already installed in venv)
   - Flask 3.1.1
   - Flask-CORS
   - psycopg2-binary (for PostgreSQL)
   - pandas, numpy

3. **PostgreSQL Database** (Optional)
   - The API server connects to PostgreSQL
   - Configuration in `src/database/database_config.json`

### System Requirements

- **Operating System**: Windows, Linux, or macOS
- **Memory**: 4GB RAM minimum
- **Disk Space**: 500MB free space
- **Network**: Port 5006 and 8080 available

## Quick Start

### Option 1: Windows

1. **Double-click** `start_dashboard.bat`

   The script will:
   - Start the backend API server on port 5006
   - Start the frontend dashboard server on port 8080
   - Open the dashboard in your default browser

2. **Access the dashboard**:
   - URL: http://localhost:8080/consolidated_dashboard_visual_preserved.html

3. **Stop the servers**:
   - Close the command windows
   - Or press Ctrl+C in each window

### Option 2: Linux/Mac

1. **Run the startup script**:
   ```bash
   ./start_dashboard.sh
   ```

2. **Access the dashboard**:
   - URL: http://localhost:8080/consolidated_dashboard_visual_preserved.html

3. **Stop the servers**:
   ```bash
   ./stop_dashboard.sh
   ```

## Manual Deployment

### Step 1: Start Backend API Server

```bash
# Navigate to the project directory
cd C:\finalee\beverly_knits_erp_v2

# Activate virtual environment (if needed)
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Start the API server
python src/api/database_api_server.py
```

The API server will start on **port 5006**.

### Step 2: Start Frontend Dashboard Server

```bash
# In a new terminal, navigate to the web directory
cd C:\finalee\beverly_knits_erp_v2\web

# Start the web server
python server.py 8080
```

The dashboard will be available on **port 8080**.

### Step 3: Access the Dashboard

Open your browser and navigate to:
```
http://localhost:8080/consolidated_dashboard_visual_preserved.html
```

## Dashboard Features

The consolidated dashboard provides:

### 1. **Yarn Intelligence**
   - Real-time yarn inventory tracking
   - Critical yarn alerts
   - Yarn consumption forecasts

### 2. **Knit Orders Management**
   - Production order tracking
   - Machine assignments
   - Order priorities

### 3. **Production Planning**
   - Time-phased production schedules
   - Capacity planning
   - Material requirements planning (MRP)

### 4. **Fabric Forecasting**
   - ML-powered demand forecasts
   - Sales trend analysis
   - Inventory optimization

### 5. **Analytics & Reporting**
   - Real-time KPIs
   - Production metrics
   - Export capabilities (CSV, Excel, JSON)

## Configuration

### API Configuration

The dashboard automatically detects the environment and configures the API base URL:

- **Local Development**: `http://localhost:5006`
- **ngrok/Railway/Render**: Uses current hostname
- **Custom**: Edit `getAPIBaseURI()` in the HTML file (line 3454)

### Backend Configuration

Database connection settings in `src/database/database_config.json`:

```json
{
  "host": "localhost",
  "port": 5432,
  "database": "beverly_erp",
  "user": "postgres",
  "password": "your_password"
}
```

## Troubleshooting

### Issue: Port Already in Use

**Error**: `Address already in use: 5006` or `8080`

**Solution**:
```bash
# Windows - Find and kill process on port
netstat -ano | findstr :5006
taskkill /PID <process_id> /F

# Linux/Mac - Find and kill process on port
lsof -ti:5006 | xargs kill -9
```

### Issue: Cannot Connect to Database

**Error**: `Connection refused` or `Database connection failed`

**Solution**:
1. Check PostgreSQL is running
2. Verify database credentials in `database_config.json`
3. Ensure database exists: `createdb beverly_erp`

### Issue: Dashboard Shows No Data

**Possible Causes**:
1. Backend API not running
2. Database not initialized
3. No data synced from eFab

**Solution**:
1. Check API health: `http://localhost:5006/api/health`
2. Check browser console for errors (F12)
3. Verify API endpoints are responding

### Issue: CORS Errors

**Error**: `Access-Control-Allow-Origin` errors in browser console

**Solution**:
- The `server.py` includes CORS headers
- Ensure you're accessing via `http://localhost:8080`
- Don't use `file://` protocol

## Advanced Deployment

### Using Production WSGI Server

For production, use a proper WSGI server like **Gunicorn**:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5006 src.api.database_api_server:app
```

### Using Reverse Proxy (Nginx)

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Dashboard static files
    location / {
        root /path/to/beverly_knits_erp_v2/web;
        try_files $uri $uri/ =404;
    }

    # API proxy
    location /api/ {
        proxy_pass http://localhost:5006;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5006 8080

CMD ["python", "start_services.py"]
```

Build and run:

```bash
docker build -t beverly-erp .
docker run -p 5006:5006 -p 8080:8080 beverly-erp
```

## Security Considerations

### Production Checklist

- [ ] Change default database passwords
- [ ] Enable HTTPS/TLS
- [ ] Implement authentication
- [ ] Set up firewall rules
- [ ] Enable API rate limiting
- [ ] Configure secure session cookies
- [ ] Regular security updates
- [ ] Implement audit logging

### Environment Variables

Store sensitive configuration in environment variables:

```bash
export DATABASE_URL="postgresql://user:pass@host/db"
export EFAB_SESSION="your_session_cookie"
export SECRET_KEY="your_secret_key"
```

## Monitoring & Maintenance

### Health Checks

```bash
# API health
curl http://localhost:5006/api/health

# Expected response
{"status": "healthy", "database": "connected"}
```

### Logs

Check logs for errors:

```bash
# API server logs
tail -f logs/api_server.log

# Web server logs
tail -f logs/web_server.log
```

### Database Maintenance

```sql
-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Vacuum and analyze
VACUUM ANALYZE;
```

## Support & Documentation

### Additional Documentation

- **API Reference**: `docs/API_REFERENCE.md`
- **Database Setup**: `src/database/README.md`
- **Mapping Guide**: `docs/technical/MAPPING/`
- **Quick Start**: `docs/QUICK_START.md`

### Getting Help

For issues or questions:
1. Check the troubleshooting section above
2. Review error logs
3. Consult the technical documentation
4. Contact system administrator

## Version Information

- **System**: Beverly Knits ERP v2
- **Dashboard**: consolidated_dashboard_visual_preserved.html
- **API Version**: 2025-08-29-fix-double-api-prefix
- **Backend**: Flask 3.1.1
- **Database**: PostgreSQL 12+

## License

Proprietary - Beverly Knits ERP v2

---

**Last Updated**: 2025-10-12
