# eFab Fabric Inventory Sync Guide

**Date:** 2025-10-18
**Purpose:** Automatically sync fabric inventory from eFab to dashboard
**Status:** ✅ Ready to Use

---

## Quick Start

### 1. Test Connection (Dry Run)

First, test that the script can connect to eFab without making changes:

```bash
python scripts/sync_fabric_inventory_from_efab.py --dry-run
```

**Expected Output:**
```
╔==========================================================╗
║               eFab Inventory Sync                        ║
╚==========================================================╝

🎯 Target: http://localhost:5006
📊 Stages: G00, G02, I01, F01
🔧 Mode: DRY RUN

============================================================
🔄 Syncing Stage: G00
============================================================
📡 Fetching G00 data from http://localhost:5006/api/greige/g00...
✅ Fetched 45 records from G00

📝 Processing 45 records...
  [DRY RUN] Would upsert: 4025 @ G00 = 500 yards
  [DRY RUN] Would upsert: 2412 @ G00 = 300 yards
  ...
```

### 2. Run Live Sync

Once you've confirmed the dry run works, sync for real:

```bash
python scripts/sync_fabric_inventory_from_efab.py
```

### 3. Verify Data

Check that data was inserted into Turso:

```bash
# Using Turso CLI
turso db shell beverly_erp_db

# Run query
SELECT stage, COUNT(*) as count, SUM(quantity_yards) as total_yards
FROM fabric_inventory
GROUP BY stage;
```

---

## Command Options

### Sync All Stages (Default)

```bash
python scripts/sync_fabric_inventory_from_efab.py
```

Syncs all 4 stages: G00 → G02 → I01 → F01

### Sync One Stage Only

```bash
# Sync only raw greige
python scripts/sync_fabric_inventory_from_efab.py --stage G00

# Sync only finished goods
python scripts/sync_fabric_inventory_from_efab.py --stage F01
```

### Dry Run Mode

```bash
python scripts/sync_fabric_inventory_from_efab.py --dry-run
```

Shows what would be synced without making database changes.

### Custom eFab URL

```bash
python scripts/sync_fabric_inventory_from_efab.py --efab-url http://production-server:5006
```

Sync from a different eFab instance.

---

## What Gets Synced

### Data Mapping

| eFab Field | Database Field | Notes |
|------------|----------------|-------|
| `style_number` | `fabric_id` | Cleaned to 4 digits |
| `yards` | `quantity_yards` | - |
| `lbs` or `pounds` | `quantity_lbs` | - |
| `rolls` | `rolls` | - |
| `location` | `location` | Warehouse location |
| `lot` or `lot_number` | `lot_number` | Batch identifier |
| `grade` | `grade` | Quality grade |
| `color` | `notes` | Stored in notes field |

### Stage Assignments

| eFab Endpoint | Stage | Fabric Type | Description |
|---------------|-------|-------------|-------------|
| `/api/greige/g00` | G00 | greige | Raw greige off loom |
| `/api/greige/g02` | G02 | greige | Dyed/finished greige |
| `/api/finished/i01` | I01 | finished | Awaiting QC |
| `/api/finished/f01` | F01 | finished | QC passed |

---

## Sync Behavior

### Insert vs Update

The script uses **upsert** logic:

- **First time:** Inserts new records
- **Subsequent runs:** Updates existing records based on `(fabric_id, stage)` key

**Example:**

```
Run 1: Fabric 4025 @ G00 = 500 yards  → INSERT
Run 2: Fabric 4025 @ G00 = 450 yards  → UPDATE
```

### Timestamps

- `created_at`: Set on first insert, never changed
- `updated_at`: Updated on every sync

---

## Automation

### Cron Job (Linux/Mac)

Sync every hour:

```bash
# Edit crontab
crontab -e

# Add this line (sync at :00 every hour)
0 * * * * cd /path/to/beverly_knits_erp_v2 && python scripts/sync_fabric_inventory_from_efab.py >> logs/efab_sync.log 2>&1
```

### Task Scheduler (Windows)

1. Open **Task Scheduler**
2. Create Task → **General tab**:
   - Name: "eFab Inventory Sync"
   - Run whether user is logged on or not
3. **Triggers tab** → New:
   - Daily, repeat every 1 hour
4. **Actions tab** → New:
   - Program: `python`
   - Arguments: `scripts/sync_fabric_inventory_from_efab.py`
   - Start in: `C:\finalee\beverly_knits_erp_v2`

### Systemd Service (Linux)

Create `/etc/systemd/system/efab-sync.service`:

```ini
[Unit]
Description=eFab Fabric Inventory Sync
After=network.target

[Service]
Type=oneshot
User=your_user
WorkingDirectory=/path/to/beverly_knits_erp_v2
ExecStart=/usr/bin/python3 scripts/sync_fabric_inventory_from_efab.py
StandardOutput=append:/var/log/efab_sync.log
StandardError=append:/var/log/efab_sync_error.log

[Install]
WantedBy=multi-user.target
```

Create timer `/etc/systemd/system/efab-sync.timer`:

```ini
[Unit]
Description=Run eFab Sync Every Hour

[Timer]
OnBootSec=5min
OnUnitActiveSec=1h

[Install]
WantedBy=timers.target
```

Enable:
```bash
sudo systemctl enable efab-sync.timer
sudo systemctl start efab-sync.timer
```

---

## Monitoring

### Check Sync Logs

If running as cron/scheduler:

```bash
# View last 50 lines
tail -50 logs/efab_sync.log

# Watch live
tail -f logs/efab_sync.log
```

### Database Checks

```sql
-- How many records per stage?
SELECT stage, COUNT(*) FROM fabric_inventory GROUP BY stage;

-- Total inventory yards
SELECT SUM(quantity_yards) FROM fabric_inventory;

-- Last sync time per fabric
SELECT fabric_id, stage, updated_at
FROM fabric_inventory
ORDER BY updated_at DESC
LIMIT 10;

-- Find stale data (not updated in 24 hours)
SELECT fabric_id, stage, updated_at
FROM fabric_inventory
WHERE updated_at < datetime('now', '-1 day');
```

---

## Troubleshooting

### Error: "Cannot connect to eFab"

**Symptom:** Script fails with connection error

**Solutions:**
1. Check eFab is running: `curl http://localhost:5006/api/greige/g00`
2. Verify URL: `--efab-url http://correct-host:port`
3. Check firewall/network access

### Error: "No records fetched"

**Symptom:** `✅ Fetched 0 records from G00`

**Solutions:**
1. Verify eFab endpoint has data
2. Check eFab API response format
3. Enable debug logging (add `print(data)` in `fetch_stage_data()`)

### Error: "Table fabric_inventory not found"

**Symptom:** Database error on insert

**Solution:** Run migration first:
```bash
python scripts/apply_fabric_inventory_schema.py
```

### Warning: "Error transforming record"

**Symptom:** Some records skipped

**Cause:** Record missing `style_number` or invalid format

**Action:** Check eFab data quality, add validation

---

## Performance

### Typical Sync Times

| Records | Time |
|---------|------|
| 50 | ~2 seconds |
| 500 | ~15 seconds |
| 5000 | ~2 minutes |

### Optimization Tips

1. **Sync during off-hours** - Less load on eFab API
2. **Stage-specific sync** - Only sync changed stages
3. **Batch processing** - Already implemented (10 records/batch)

---

## Integration with Dashboard

Once synced, the dashboard automatically uses this data:

### Production → Forecasted Orders Table

```
Current Inventory = I01 + F01
Pipeline (WIP) = G00 + G02
```

Data flows:
1. Sync script populates `fabric_inventory` table
2. Dashboard queries `/api/fabric-inquiry/search`
3. Real data displays in "Current Inventory" and "Pipeline (WIP)" columns

### Verification

After sync, reload dashboard and check:
- ✅ "Current Inventory" shows numbers (not "-")
- ✅ "Pipeline (WIP)" shows numbers (not "-")
- ✅ Values match eFab totals

---

## Example Sync Output

```
╔==========================================================╗
║               eFab Inventory Sync                        ║
╚==========================================================╝

🎯 Target: http://localhost:5006
📊 Stages: G00, G02, I01, F01
🔧 Mode: LIVE

============================================================
🔄 Syncing Stage: G00
============================================================
📡 Fetching G00 data from http://localhost:5006/api/greige/g00...
✅ Fetched 45 records from G00

📝 Processing 45 records...
  ➕ Inserted: 4025 @ G00
  ➕ Inserted: 2412 @ G00
  ➕ Inserted: 5117 @ G00
  Progress: 10/45
  ➕ Inserted: 4585 @ G00
  ...
  Progress: 40/45
✅ Completed G00

============================================================
🔄 Syncing Stage: G02
============================================================
📡 Fetching G02 data from http://localhost:5006/api/greige/g02...
✅ Fetched 32 records from G02

📝 Processing 32 records...
  ➕ Inserted: 4025 @ G02
  ✏️  Updated: 2412 @ G02
  ...
✅ Completed G02

============================================================
🔄 Syncing Stage: I01
============================================================
📡 Fetching I01 data from http://localhost:5006/api/finished/i01...
✅ Fetched 18 records from I01
...

============================================================
🔄 Syncing Stage: F01
============================================================
📡 Fetching F01 data from http://localhost:5006/api/finished/f01...
✅ Fetched 67 records from F01
...

============================================================
📊 SYNC SUMMARY
============================================================
  Total Fetched:     162 records
  Inserted:          148 records
  Updated:            14 records
  Errors:              0 errors
============================================================

✅ Sync completed successfully!
```

---

## Next Steps

1. **Run initial sync:**
   ```bash
   python scripts/sync_fabric_inventory_from_efab.py
   ```

2. **Verify dashboard shows data:**
   - Open Production tab
   - Check Forecasted Orders table
   - Confirm real numbers display

3. **Set up automation:**
   - Choose cron/scheduler/systemd
   - Schedule hourly or daily sync

4. **Monitor:**
   - Check logs regularly
   - Verify data freshness
   - Set up alerts for sync failures

---

**The sync is ready to use! Run the script to populate your dashboard with real eFab inventory data.**
