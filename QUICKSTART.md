# Beverly Knits ERP Dashboard - Quick Start Guide

## 🚀 Launch in 3 Ways

### Option 1: Python Launch Script (Recommended)
```bash
python launch.py
```

### Option 2: Windows Batch Script
```cmd
start_dashboard.bat
```

### Option 3: Linux/Mac Shell Script
```bash
./start_dashboard.sh
```

## ✅ System Status

**Backend API Server**: ✓ Running on http://localhost:5006
- Database: SQLite (automatic fallback from PostgreSQL)
- Health Check: http://localhost:5006/api/health

**Frontend Dashboard**: ✓ Running on http://localhost:8080
- Dashboard URL: http://localhost:8080/consolidated_dashboard_visual_preserved.html

## 📊 Dashboard Features

### 1. Yarn Intelligence
- Real-time inventory tracking
- Critical yarn alerts
- Consumption forecasts

### 2. Knit Orders Management
- Production order tracking
- Machine assignments
- Priority management

### 3. Production Planning
- Time-phased schedules
- Capacity planning
- Material requirements (MRP)

### 4. Fabric Forecasting
- ML-powered demand forecasts
- Sales trend analysis
- Inventory optimization

### 5. Analytics & Reporting
- Real-time KPIs
- Production metrics
- Export capabilities (CSV, Excel, JSON)

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Application
APP_PORT=5006
APP_HOST=0.0.0.0

# Database
DATABASE_TYPE=sqlite
DATABASE_PATH=data/beverly_erp.db

# API Settings
ENABLE_CORS=true
API_RATE_LIMIT=100
```

### Database Configuration (src/database/database_config.json)
```json
{
    "host": "localhost",
    "port": 5432,
    "database": "beverly_knits_erp",
    "user": "erp_user",
    "password": "erp_password"
}
```

## 🛠️ Troubleshooting

### Ports Already in Use
```bash
# Windows - Kill process on port
netstat -ano | findstr :5006
taskkill /PID <process_id> /F

# Linux/Mac
lsof -ti:5006 | xargs kill -9
```

### Cannot Connect to API
1. Check if API server is running: `curl http://localhost:5006/api/health`
2. Check firewall settings
3. Verify port 5006 is not blocked

### Dashboard Shows No Data
1. Verify API health endpoint responds
2. Check browser console (F12) for errors
3. Ensure CORS is enabled

## 📁 Project Structure

```
beverly_knits_erp_v2/
├── launch.py                  # Quick launch script
├── start_dashboard.bat        # Windows launcher
├── start_dashboard.sh         # Linux/Mac launcher
├── web/
│   ├── server.py              # Frontend web server
│   └── consolidated_dashboard_visual_preserved.html
├── src/
│   ├── api/
│   │   ├── lightweight_api_server.py  # Backend API
│   │   └── database_api_server.py     # Full database API
│   └── database/
│       ├── models.py
│       └── database_config.json
└── data/
    └── beverly_erp.db         # SQLite database
```

## 🔐 Security Notes

- Change default database passwords in production
- Enable HTTPS/TLS for production deployments
- Implement authentication for API endpoints
- Set up firewall rules
- Regular security updates

## 📖 Additional Documentation

- **Full Deployment Guide**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **API Reference**: `docs/API_REFERENCE.md`
- **Database Setup**: `src/database/README.md`
- **Technical Mapping**: `docs/technical/MAPPING/`

## 🆘 Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs in `logs/` directory
3. Consult technical documentation
4. Contact system administrator

## 📝 Version Information

- **System**: Beverly Knits ERP v2
- **Dashboard**: consolidated_dashboard_visual_preserved.html (2025-08-29)
- **Backend**: Flask 3.1.1 + SQLite/PostgreSQL
- **Python**: 3.8+

---

**Last Updated**: 2025-10-12
**Status**: ✅ Operational
