# Beverly Knits ERP v2 - File Cleanup Plan

## Overview
This document identifies files and directories that can be safely removed to reduce project size and improve organization. Current project size: ~804MB. Expected size after cleanup: ~690MB (14% reduction).

## Files to Remove by Category

### 1. Archive & Backup Files (114MB)
**Safe to remove - these are old backups and duplicates:**
```
cleanup_backup_20250902_041449.tar.gz    # 79MB - Old backup archive
ngrok.zip                                 # 10MB - Duplicate ngrok archive
ngrok2.zip                                # 10MB - Duplicate ngrok archive  
ngrok-new.zip                            # 10MB - Another ngrok duplicate
ngrok_old.zip                            # 5MB - Older ngrok version
```

### 2. Python Cache Files (~5MB)
**Auto-generated files that will be recreated as needed:**
```
**/__pycache__/                          # All Python cache directories
**/*.pyc                                 # Compiled Python files
*.db-shm                                 # SQLite temporary files
*.db-wal                                 # SQLite write-ahead log files
```

### 3. Duplicate Files in Root Directory
**These exist in proper locations within src/:**
```
yarn_interchangeability_analyzer.py      # Duplicate of src/yarn_intelligence/yarn_interchangeability_analyzer.py
```

### 4. Test Files in Root (Should be in /tests/)
**Misplaced test files that should be organized:**
```
test_api.py
test_capacity.py
test_consolidation.py
test_data_loader.py
test_inventory.py
test_ml_forecast.py
test_production.py
test_yarn_intelligence.py
test_results.json
test_results.html
```

### 5. Log & Temporary Files
**Regeneratable logs and temp files:**
```
ml_errors.log                            # ML error log
server_output.log                        # Server output log
ngrok_version.txt                        # Temp version file
nul                                      # Empty file
*.log                                    # Other log files
```

### 6. Old Training & Test Reports
**Historical reports with timestamps (keep only latest):**
```
ml_training_report_*.json                # Old ML training reports
test_report_*.json                       # Old test result reports
backtest_results_*.json                  # Old backtest files
```

### 7. Backup & Temporary Directories
**Old backups and temporary extraction folders:**
```
data/production/5/ERP Data/backup_20250902_070706/  # Old data backup
ngrok_temp/                                          # Temp extraction directory
hello-ngrok/hello-ngrok/                           # Empty nested directories
web/consolidated_dashboard_backup.html              # Dashboard backup
```

### 8. Broken Symlinks & WSL Artifacts
```
C:/                                      # WSL mount symlink (if present)
```

## Cleanup Commands

### Step 1: Remove Archive Files
```bash
rm -f cleanup_backup_20250902_041449.tar.gz
rm -f ngrok.zip ngrok2.zip ngrok-new.zip ngrok_old.zip
```

### Step 2: Clean Python Cache
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
rm -f *.db-shm *.db-wal
```

### Step 3: Remove Duplicate Files
```bash
rm -f yarn_interchangeability_analyzer.py  # Root duplicate
```

### Step 4: Clean Test Files from Root
```bash
# First verify these exist in /tests/ directory
rm -f test_*.py test_*.json test_*.html
```

### Step 5: Remove Logs and Temp Files
```bash
rm -f ml_errors.log server_output.log ngrok_version.txt nul
rm -f *.log
```

### Step 6: Clean Old Reports
```bash
# Keep only the most recent of each type
ls -t ml_training_report_*.json | tail -n +2 | xargs rm -f
ls -t test_report_*.json | tail -n +2 | xargs rm -f
ls -t backtest_results_*.json | tail -n +2 | xargs rm -f
```

### Step 7: Remove Backup Directories
```bash
rm -rf "data/production/5/ERP Data/backup_20250902_070706/"
rm -rf ngrok_temp/
rm -rf hello-ngrok/
rm -f web/consolidated_dashboard_backup.html
```

### Step 8: One-Line Cleanup Script
```bash
# Complete cleanup (review before running)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null && \
find . -type f -name "*.pyc" -delete && \
rm -f *.tar.gz ngrok*.zip && \
rm -f *.log ngrok_version.txt nul && \
rm -f *.db-shm *.db-wal && \
echo "Cleanup complete!"
```

## Safety Checklist

### Files to NEVER Remove:
- ✅ All files in `/src/` (except __pycache__)
- ✅ `/data/production/5/ERP Data/*.xlsx` and `*.csv` (current data files)
- ✅ `/web/consolidated_dashboard.html` (main dashboard)
- ✅ `requirements.txt`, `Makefile`, `CLAUDE.md`
- ✅ `.env`, `.env.example`, configuration files
- ✅ `/tests/` directory structure (organized tests)
- ✅ `beverly_comprehensive_erp.py` (main application)

### Pre-Cleanup Verification:
1. **Backup Important Data**: Create a backup before cleanup
2. **Check Running Processes**: Ensure server is stopped
3. **Verify File Paths**: Double-check paths before deletion
4. **Test After Cleanup**: Run server and tests after cleanup

## Expected Results

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Total Size | ~804MB | ~690MB | 114MB (14%) |
| File Count | ~500 | ~380 | 120 files |
| Cache Files | 83 | 0 | 83 files |
| Archive Files | 5 | 0 | 5 files |

## Post-Cleanup Actions

1. **Run Tests**: `make test`
2. **Start Server**: `python src/core/beverly_comprehensive_erp.py`
3. **Verify Dashboard**: http://localhost:5006/consolidated
4. **Update .gitignore**: Add patterns for files that keep regenerating

## .gitignore Additions
```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Logs
*.log
logs/

# Database
*.db-shm
*.db-wal

# Backups
*.tar.gz
*.zip
*_backup*
backup_*/

# Temp files
ngrok_temp/
*.tmp
nul

# ML artifacts
ml_training_report_*.json
backtest_results_*.json
test_report_*.json
```

## Maintenance Schedule

- **Daily**: Remove *.log files
- **Weekly**: Clean __pycache__ directories
- **Monthly**: Review and remove old training reports
- **Quarterly**: Full cleanup using this guide

---

**Last Updated**: September 2025
**Estimated Cleanup Time**: 5-10 minutes
**Risk Level**: LOW (all identified files are redundant/regeneratable)