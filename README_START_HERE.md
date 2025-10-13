# 🚀 Beverly Knits ERP Dashboard - START HERE

## Quick Start (30 seconds)

### Windows
**Double-click**: `RESTART_SERVERS.bat`

### Linux/Mac
```bash
./start_dashboard.sh
```

Then open: **http://localhost:8080/consolidated_dashboard_visual_preserved.html**

---

## ✅ What's Ready

Your complete ERP dashboard system with:

- ✅ **Real-time eFab data** (via authenticated API proxy)
- ✅ **Dashboard frontend** (HTML + Charts + Analytics)
- ✅ **Turso database** (configured for sync)
- ✅ **Complete documentation**

## 📊 System Components

| Component | Port | Status | Purpose |
|-----------|------|--------|---------|
| **eFab API Server** | 5006 | ✅ Ready | Proxies live data from eFab |
| **Dashboard Server** | 8080 | ✅ Ready | Serves HTML dashboard |
| **Turso Database** | - | ✅ Configured | Cloud data storage |

## 🔐 Credentials

All stored securely in `.env`:

```ini
# eFab Access
EFAB_SESSION="aOxeGQLjJmukatAjrIDT..." (Active)

# Turso Database
TURSO_DATABASE_URL="libsql://efab2..."
TURSO_AUTH_TOKEN="eyJhbGci..."
```

## 📁 Important Files

```
START_CLEAN.bat           ← USE THIS to launch (Windows)
start_dashboard.sh        ← USE THIS to launch (Linux/Mac)

src/api/efab_api_server.py      ← Real eFab data (ACTIVE)
web/server.py                     ← Dashboard web server
web/consolidated_dashboard...    ← Your dashboard HTML

.env                              ← All credentials
FINAL_STATUS.md                   ← Detailed status
DEPLOYMENT_GUIDE.md               ← Full documentation
```

## 🎯 Access Points

After starting servers:

- **Dashboard**: http://localhost:8080/consolidated_dashboard_visual_preserved.html
- **API Health**: http://localhost:5006/api/health
- **API Docs**: See `DEPLOYMENT_GUIDE.md`

## ⚡ Quick Commands

```bash
# Start everything (Windows)
START_CLEAN.bat

# Test API
curl http://localhost:5006/api/health

# Refresh eFab session (if expired)
python scripts/efab_login.py
```

## 🛠️ If Something Goes Wrong

### Dashboard shows mock data?
```bash
# Kill everything and restart
taskkill /F /IM python.exe
START_CLEAN.bat
```

### eFab session expired?
```bash
python scripts/efab_login.py
# Copy new session to .env file
```

### Port already in use?
```bash
# Kill process on port
netstat -ano | findstr :5006
taskkill /F /PID <process_id>
```

## 📖 Documentation

- **Quick Start**: This file
- **Full Guide**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Status**: [FINAL_STATUS.md](FINAL_STATUS.md)
- **API Ref**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

## ✨ What You Get

Your dashboard shows:
- 📊 Real-time production data from eFab
- 🧶 Yarn inventory & criticality
- 🏭 Production orders & scheduling
- 📈 Forecasting & analytics
- 📤 Export capabilities (CSV, Excel, JSON)

## 🎉 Ready to Go!

**Just run**: `START_CLEAN.bat`

Everything is configured and ready to display your real eFab production data!

---

**System**: Beverly Knits ERP v2
**Date**: October 12, 2025
**Status**: ✅ READY FOR USE
