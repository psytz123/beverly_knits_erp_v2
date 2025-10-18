# QuadS Integration Setup Guide

## Quick Start

### 1. Environment Configuration

Add the following to your `.env` file:

```bash
# QuadS Authentication (Option 1 - Recommended)
QUADS_USERNAME=your_username
QUADS_PASSWORD=your_password

# OR QuadS Authentication (Option 2)
# QUADS_SESSION_TOKEN=your_manual_token

# Turso Database (Required)
TURSO_DATABASE_URL=libsql://your-database.turso.io
TURSO_AUTH_TOKEN=your_turso_token
```

### 2. Test Connection

```bash
python scripts/test_quads_api.py
```

Expected output:
```
Attempting login as your_username...
Login successful! Token: abc123...
QuadS API Connection Test
======================================================================
Testing: /api/styles/finished/active
Status Code: 200
Number of records: 3539
```

### 3. Run Import

```bash
python scripts/import_from_quads_api.py
```

Expected output:
```
QuadS API to Turso Import
======================================================================
Logging in to QuadS as your_username...
Successfully logged in to QuadS
Fetching finished fabric styles from QuadS API...
Fetched 3539 finished styles from QuadS
Calculated yds/lb for 1716: 0.83
Calculated yds/lb for 2469: 0.51
...
Import Complete!
  Finished styles: 3539
  Greige styles: 0
  Total: 3539
```

## Features

### ✅ Automatic Login
- No need to manually extract session tokens
- Script logs in automatically using username/password
- Falls back to manual token if needed

### ✅ Automatic Yds/Lbs Calculation
- Calculates yards per pound from GSM and width
- Uses standard textile industry formula
- Ensures all specs have yds/lb for production planning

**Formula:**
```
yds_per_lb = 16129.032 / (gsm × width_inches)
```

### ✅ Batch Processing
- Imports 50 records per batch
- Handles large datasets efficiently
- Automatic retry on failure

### ✅ Smart Field Mapping
- Handles multiple field name variations
- Adapts to QuadS API changes
- Validates data before import

## API Endpoints

The integration uses these QuadS endpoints:

| Endpoint | Purpose |
|----------|---------|
| `POST /api/auth/login` | Authenticate and get session token |
| `GET /api/styles/finished/active` | Fetch active finished fabric styles |
| `GET /api/styles/greige/active` | Fetch active greige fabric styles |

## Data Flow

```
QuadS API → Python Script → Turso Database
     ↓            ↓              ↓
  Login      Field Map      fabric_specs
  Fetch      Calculate         table
             Validate
```

## Troubleshooting

### Login Failed

**Error:** `Login failed: 401`

**Solutions:**
1. Verify username/password in `.env`
2. Check if QuadS account is active
3. Try manual token as fallback

### No Records Returned

**Error:** `Fetched 0 finished styles from QuadS`

**Solutions:**
1. Verify you have access to styles in QuadS
2. Check QuadS web interface for active styles
3. Verify API endpoint hasn't changed

### Calculation Errors

**Error:** `Cannot calculate yds/lb`

**Cause:** Missing GSM or width data

**Impact:** Record imported without yds/lb value

**Solution:** Add GSM/width in QuadS or manually update Turso

### Import Timeout

**Error:** `Request timeout after 120s`

**Solutions:**
1. Check network connection
2. Reduce batch size in script
3. Run during off-peak hours

## Advanced Usage

### Custom Field Mapping

Edit `map_quads_style_to_fabric_spec()` to add new fields:

```python
def map_quads_style_to_fabric_spec(style_data: Dict) -> Optional[Dict]:
    # Add custom field
    custom_field = style_data.get('CustomField')

    return {
        'style': str(style),
        'yds_per_lb': yds_per_lb_value,
        'gsm': gsm_value,
        'width': width_value,
        'fabric_type': fabric_type,
        'description': description,
        'custom_field': custom_field  # Add to return dict
    }
```

### Scheduled Imports

**Windows Task Scheduler:**
```batch
@echo off
cd C:\finalee\beverly_knits_erp_v2
python scripts\import_from_quads_api.py
```

**Linux/Mac Cron:**
```bash
0 2 * * * cd /path/to/beverly_knits_erp_v2 && python scripts/import_from_quads_api.py
```

### API Rate Limiting

If you hit rate limits, adjust batch size:

```python
# In import_from_quads_api.py
batch_size = 25  # Reduce from 50 to 25
```

## Security Best Practices

### ✅ Do:
- Store credentials in `.env` file
- Add `.env` to `.gitignore`
- Use strong passwords
- Rotate credentials regularly
- Limit API access to necessary users

### ❌ Don't:
- Commit credentials to git
- Share `.env` file
- Use same password across systems
- Store tokens in code
- Log credentials

## Support

**For QuadS API issues:**
- Contact QuadS support team
- Check QuadS API documentation
- Verify account permissions

**For import script issues:**
- Check script logs
- Verify `.env` configuration
- Test with `test_quads_api.py`

**For Turso database issues:**
- Verify credentials
- Check database limits
- Review Turso dashboard
