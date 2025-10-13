# Beverly Knits ERP - Quick Start Guide

## Starting the Server

### Option 1: From Windows (Recommended)
**PowerShell:**
```powershell
.\start_server.ps1
```

**Command Prompt:**
```cmd
start_server.bat
```

### Option 2: From WSL/Linux
```bash
./start_server.sh
```

### Option 3: Manual Start in WSL
```bash
export EFAB_SESSION="aMdcwNLa0ov0pcbWcQ_zb5wyPLSkYF_B"
export ENABLE_YARN_SCHEDULER=true
export FILTER_NONPRODUCTION_YARNS=true
python3 src/core/beverly_comprehensive_erp.py
```

## Access Points
- Main Dashboard: http://localhost:5006/consolidated
- API Base: http://localhost:5006/api/

## Troubleshooting Path Issues

If you see: `did not find executable at '/usr/bin\python.exe'`

This is a Windows/WSL path format conflict. Solutions:

1. **Use the provided startup scripts** (start_server.ps1, start_server.bat, start_server.sh)

2. **Ensure you're in the correct environment:**
   - In WSL: Use `python3` (not python.exe)
   - In Windows: Use `python` (not /usr/bin paths)

3. **Check your terminal:**
   - WSL Terminal: Unix paths work (`/mnt/c/...`)
   - PowerShell: Windows paths work (`C:\...`)

## Common Commands

**Check if server is running:**
```bash
curl http://localhost:5006/api/health
```

**Kill existing server:**
```bash
pkill -f "python.*beverly"
```

**Manual Yarn Demand refresh:**
```bash
curl -X POST http://localhost:5006/api/manual-yarn-refresh
```

## Environment Variables
- `EFAB_SESSION`: eFab session cookie (expires ~24 hours)
- `ENABLE_YARN_SCHEDULER`: Enable automatic Yarn Demand downloads
- `FILTER_NONPRODUCTION_YARNS`: Show all yarns with negative balances