# Quick Start: eFab Inventory Sync

## 🚀 3-Step Setup

### Step 1: Test (Dry Run)
```bash
python scripts/sync_fabric_inventory_from_efab.py --dry-run
```
✅ **Success**: See "DRY RUN" messages, no errors

### Step 2: Sync Real Data
```bash
python scripts/sync_fabric_inventory_from_efab.py
```
✅ **Success**: See "✅ Sync completed successfully!"

### Step 3: Verify Dashboard
```
1. Open: http://localhost:5000/
2. Go to: Production tab
3. Look at: Forecasted Production Orders table
4. Check: "Current Inventory" and "Pipeline (WIP)" show numbers
```

---

## 📋 One-Liner Commands

```bash
# Test connection
python scripts/sync_fabric_inventory_from_efab.py --dry-run

# Sync all stages
python scripts/sync_fabric_inventory_from_efab.py

# Sync one stage
python scripts/sync_fabric_inventory_from_efab.py --stage G00

# Sync from production
python scripts/sync_fabric_inventory_from_efab.py --efab-url http://prod:5006
```

---

## 🔄 What Happens

```
eFab API (G00, G02, I01, F01)
        ↓
  Sync Script
        ↓
Turso: fabric_inventory table
        ↓
Fabric Inquiry API
        ↓
Dashboard Tables
```

---

## ✅ Success Checklist

- [ ] Script runs without errors
- [ ] Database shows records: `SELECT COUNT(*) FROM fabric_inventory;`
- [ ] Dashboard shows real numbers (not "-")
- [ ] Numbers match eFab totals

---

## 🆘 Quick Fixes

| Problem | Solution |
|---------|----------|
| "Cannot connect" | Check eFab is running: `curl localhost:5006/api/greige/g00` |
| "No records" | Verify eFab has data at endpoints |
| "Table not found" | Run: `python scripts/apply_fabric_inventory_schema.py` |
| "Stale data" | Set up hourly cron job |

---

## ⏰ Automate (Optional)

### Linux/Mac Cron (Hourly)
```bash
crontab -e
# Add: 0 * * * * cd /path/to/project && python scripts/sync_fabric_inventory_from_efab.py
```

### Windows Task Scheduler
```
Task → Daily → Repeat every 1 hour
Action → python scripts/sync_fabric_inventory_from_efab.py
```

---

**That's it! You're ready to sync real eFab inventory data to your dashboard.**
