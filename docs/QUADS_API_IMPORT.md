# QuadS API Import Guide

## Overview

Import fabric specifications directly from QuadS into Turso database using the QuadS REST API.

## Setup

### 1. Configure QuadS Authentication

**Option 1: Automatic Login (Recommended)**

Add your QuadS credentials to `.env` for automatic login:

```bash
QUADS_USERNAME=your_username
QUADS_PASSWORD=your_password
```

The script will automatically login and obtain a session token.

**Option 2: Manual Session Token**

Alternatively, provide a session token directly:

```bash
QUADS_SESSION_TOKEN=your_session_token_here
```

To get a manual token:
1. Log in to QuadS at https://quads.bkiapps.com
2. Open browser DevTools (F12)
3. Go to Application/Storage → Cookies
4. Copy the `session` cookie value

### 2. Verify API Access

Test your connection to the QuadS API:

```bash
python scripts/test_quads_api.py
```

This will:
- Verify your session token works
- Show the structure of data returned from QuadS
- Display sample records

## Import Methods

### Method 1: Import from QuadS API (Recommended)

Import live data from QuadS API:

```bash
python scripts/import_from_quads_api.py
```

**Features:**
- Fetches finished fabric styles
- Fetches greige fabric styles
- Maps QuadS fields to Turso schema
- Batch imports for performance
- Automatic verification

### Method 2: Import from Excel File (Fallback)

If API access is unavailable, import from the Excel export:

```bash
python scripts/import_quads_fabric_specs.py
```

Requires: `startingdocs/QuadS_finishedFabricList_ (6).xlsx`

## QuadS API Endpoints

### Base Configuration

```yaml
Base URL: https://quads.bkiapps.com
API Prefix: /api
Authentication: Session token
Content-Type: application/json
Timeout: 30 seconds
```

### Available Endpoints

| Endpoint | Method | Purpose | Response |
|----------|--------|---------|----------|
| `/api/styles/finished/active` | GET | Active finished fabric styles | JSON array |
| `/api/styles/greige/active` | GET | Active greige fabric styles | JSON array |

## Data Mapping

QuadS fields are mapped to Turso `fabric_specs` table:

| QuadS Field | Turso Column | Type | Notes |
|-------------|--------------|------|-------|
| `id`, `fid`, `F ID` | `style` | TEXT | Style identifier (required) |
| `gsm`, `GSM`, `weight` | `gsm` | INTEGER | Fabric weight |
| `width`, `Overall Width` | `width` | REAL | Fabric width in inches |
| `yds_per_lb`, `Yds/Lbs` | `yds_per_lb` | REAL | Yards per pound (calculated if missing) |
| `type`, `Fabric Type` | `fabric_type` | TEXT | Fabric type/category |
| `name`, `description` | `description` | TEXT | Fabric description |

The script handles multiple possible field name variations to ensure compatibility.

### Automatic Yds/Lbs Calculation

If `yds_per_lb` is not provided in the QuadS data, it will be **automatically calculated** from GSM and width using the standard textile industry formula:

```
yds_per_lb = 16129.032 / (gsm × width_in_inches)
```

**Example:**
- GSM: 263
- Width: 74 inches
- Calculated yds/lb: 16129.032 / (263 × 74) = **0.83 yards/pound**

This ensures all fabric specs have yds/lb values for production planning, even if QuadS doesn't store them directly.

## Verification

After import, verify the data:

```bash
python scripts/verify_quads_import.py
```

Shows:
- Total record count
- Sample fabric specs
- Recent imports

## Turso Schema

The `fabric_specs` table structure:

```sql
CREATE TABLE fabric_specs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    style TEXT NOT NULL UNIQUE,
    yds_per_lb REAL,
    gsm INTEGER,
    width REAL,
    fabric_type TEXT,
    description TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

## Troubleshooting

### Invalid Session Token

**Error:** `HTTP 401 Unauthorized`

**Solution:**
1. Log in to QuadS at https://quads.bkiapps.com
2. Open browser DevTools (F12)
3. Go to Application/Storage → Cookies
4. Copy the `session` cookie value
5. Update `.env` with new token

### No Records Returned

**Error:** `No styles returned from API`

**Possible causes:**
- Invalid session token
- No active styles in QuadS
- API endpoint changed

**Solution:**
1. Run `test_quads_api.py` to see actual response
2. Check QuadS web interface for active styles
3. Verify endpoints with QuadS admin

### Field Mapping Issues

**Error:** `No style ID found in record`

**Solution:**
1. Run `test_quads_api.py` to see actual field names
2. Update `map_quads_style_to_fabric_spec()` in import script
3. Add new field name variations to the mapping logic

### Schema Errors

**Error:** `table fabric_specs has no column named X`

**Solution:**
```bash
python scripts/fix_fabric_specs_schema.py
```

This adds missing columns to the table.

## Automation

### Scheduled Imports

Set up a scheduled task to import fresh data daily:

**Windows (Task Scheduler):**
```bash
cd C:\finalee\beverly_knits_erp_v2
python scripts/import_from_quads_api.py
```

**Linux/Mac (cron):**
```bash
0 2 * * * cd /path/to/beverly_knits_erp_v2 && python scripts/import_from_quads_api.py
```

### Import on Demand

Add to your workflow as needed:

```python
from scripts.import_from_quads_api import import_finished_styles, import_greige_styles

session_token = os.getenv("QUADS_SESSION_TOKEN")

# Import finished styles
count = import_finished_styles(session_token)
print(f"Imported {count} finished styles")

# Import greige styles
count = import_greige_styles(session_token)
print(f"Imported {count} greige styles")
```

## Performance

- **Batch size:** 50 records per batch
- **Timeout:** 30 seconds per request
- **Estimated time:** ~1-2 minutes for 3,500 records

## Security

**Session Token Security:**
- Never commit `.env` file to git
- Rotate tokens regularly
- Use environment variables in production
- Limit token access to necessary users

**Best Practices:**
- Store tokens in secure secrets management
- Use short-lived tokens when possible
- Monitor API access logs
- Implement rate limiting

## Support

For issues with:
- **QuadS API:** Contact QuadS support
- **Import scripts:** Check logs in script output
- **Turso database:** Verify credentials in `.env`

## Changelog

### 2025-10-18
- Initial QuadS API integration
- Support for finished and greige styles
- Automatic field mapping
- Batch import optimization
- Verification utilities
