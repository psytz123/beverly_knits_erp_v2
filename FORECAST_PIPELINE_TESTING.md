# 13-Week Forecast Pipeline - End-to-End Testing Guide

## Overview
This guide provides step-by-step instructions to test the complete forecast pipeline from data import through dashboard visualization.

## Prerequisites

### 1. Environment Setup
```bash
# Verify Python environment
python --version  # Should be 3.9+

# Verify virtual environment is activated
# (venv) should appear in terminal prompt

# Verify required packages
pip list | grep -E "pandas|numpy|httpx|prophet|xgboost|scikit-learn"
```

### 2. Environment Variables
Ensure `.env` file contains:
```env
TURSO_DATABASE_URL=libsql://your-database.turso.io
TURSO_AUTH_TOKEN=your_auth_token_here
EFAB_SESSION=your_efab_session_cookie
```

Test Turso connection:
```bash
curl -X POST https://your-database.turso.io \
  -H "Authorization: Bearer $TURSO_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"q":"SELECT 1"}]}'
```

## Phase 1: Database Setup

### Step 1.1: Create Forecast Tables
```bash
python -c "from src.database.turso_client import get_turso_client; client = get_turso_client(); client.initialize_schema(); print('✓ Schema initialized')"
```

**Expected Output:**
```
✓ Created table: style_mappings
✓ Created table: external_forecasts
✓ Created table: forecast_accuracy
✓ Created table: forecast_blend_weights
✓ Schema initialized
```

**Verify:**
```bash
# Check tables exist
python scripts/test_turso_integration.py
```

### Step 1.2: Import Style Mappings
```bash
python scripts/import_style_mappings_to_turso.py \
  --create-table \
  --file "C:\Users\psytz\Downloads\eFab_Styles_20251018.xlsx"
```

**Expected Output:**
```
✓ Table created successfully
✓ Loaded X rows from Excel
✓ Inserted Y mappings
✓ Total mappings in database: Y
Sample records:
  fStyle: STYLE001 -> gBase: C1B4014
```

**Verify:**
```python
from src.utils.style_mapper import get_style_mapper
mapper = get_style_mapper()
stats = mapper.get_mapping_stats()
print(f"Total fStyles: {stats['total_fstyles']}")
# Should show count > 0
```

### Step 1.3: Import Historical Sales (if available)
```bash
# If you have historical sales CSV
python scripts/import_sales_to_turso.py --file path/to/sales.csv

# Or test with sample data
python -c "from src.forecasting.weekly_forecast_generator import WeeklyForecastGenerator; gen = WeeklyForecastGenerator(); print('✓ Can access historical sales')"
```

### Step 1.4: Import BOM and Fabric Specs
```bash
python scripts/import_bom_and_specs_to_turso.py
```

**Expected Output:**
```
✓ Imported BOM data: X entries
✓ Imported fabric specs: Y entries
```

## Phase 2: Forecast Module Testing

### Step 2.1: Test ML Forecast Generation
```bash
cd C:\finalee\beverly_knits_erp_v2
python src/forecasting/weekly_forecast_generator.py
```

**Expected Output:**
```
=== Test 1: ML Forecast Only (No External Sources) ===
================================================================================
COMPREHENSIVE FORECAST GENERATION
================================================================================
Forecast horizon: Weeks 42 to 54

[1/6] Generating ML forecasts from historical data...
Auto-selected X styles with sufficient history
✓ ML forecasts: X styles

[2/6] Loading external forecasts...
✓ Total external sources: 0

[3/6] Blending forecasts (strategy: weighted_average)...
✓ Blended forecast: X styles

[4/6] Loading actual confirmed orders...
✓ Actual orders: Y styles

[5/6] Comparing forecast vs actual...
✓ High-confidence gaps: Z
✓ Variance alerts: W

[6/6] Creating combined production schedule...
✓ Combined schedule: X styles

================================================================================
FORECAST GENERATION COMPLETE
================================================================================

Results:
  ML forecasts: X styles
  Blended forecasts: X styles
  Actual orders: Y styles
  Combined schedule: X styles
```

**Troubleshooting:**
- If "No styles with history" → Import historical sales data
- If "Turso connection error" → Check .env credentials
- If "Style mapping error" → Run import_style_mappings_to_turso.py

### Step 2.2: Test External Forecast Loader
```bash
python src/forecasting/external_forecast_loader.py
```

**Expected Output:**
```
=== Test 1: Validate External Forecast ===
✓ Validation passed
```

**Test with sample data:**
```python
from src.forecasting.external_forecast_loader import ExternalForecastLoader
import pandas as pd

loader = ExternalForecastLoader()

# Create sample forecast
sample_df = pd.DataFrame([
    {'style': 'STYLE001', 'week_number': 42, 'forecasted_yards': 1000, 'confidence': 0.85},
    {'style': 'STYLE001', 'week_number': 43, 'forecasted_yards': 1050, 'confidence': 0.88}
])

# Upload to Turso
count = loader.upload_to_turso([
    {'style': 'STYLE001', 'week_number': 42, 'forecasted_yards': 1000, 'confidence': 0.85, 'notes': 'Test'},
], 'test_source', 'test_user')

print(f"✓ Uploaded {count} forecasts")
```

### Step 2.3: Test Forecast Blender
```bash
python src/forecasting/forecast_blender.py
```

**Expected Output:**
```
=== Test 1: Basic Weighted Average Blending ===
Blending 2 sources for 1 styles

Sample blended forecast:
  Style: STYLE001
    Week 42: 950.0 yards (confidence: 0.77, dominant: ml_historical)
✓ Blending successful
```

### Step 2.4: Test Forecast-Actual Comparator
```bash
python src/forecasting/forecast_actual_comparator.py
```

**Expected Output:**
```
=== Test 1: Get Actual Orders ===
Loaded X styles with actual orders

Sample: STYLE001
  Week 42: 1000 yards

=== Test 2: Compare Forecast vs Actual ===
Summary:
  total_styles: X
  styles_with_orders: Y
  styles_forecast_only: Z
  high_confidence_gaps: W

=== Test 3: Proactive Production Recommendations ===
Found X recommendations

Top 3 recommendations:
  Priority 1: STYLE001 Week 42
    Forecasted: 1000 yards
    Confidence: 92.0%
    Action: Schedule for production immediately
```

### Step 2.5: Test Accuracy Tracker
```bash
python src/forecasting/forecast_accuracy_tracker.py
```

**Expected Output:**
```
=== Test 1: Track Forecast vs Actual ===
✓ Stored X accuracy records

=== Test 2: Calculate Source Accuracy ===
ML Historical Performance:
  MAPE: 12.50%
  Bias: -2.30%
  Hit Rate: 78.0%
  Sample Size: X

=== Test 3: Weight Recommendations ===
Current weights → Recommended weights:
  ml_historical: 0.40 → 0.42 (+0.02)
  sales_team: 0.35 → 0.37 (+0.02)

Recommended changes:
  - ml_historical weight increased by 5.0% (MAPE: 12.5%, Hit Rate: 78.0%)
```

## Phase 3: API Endpoint Testing

### Step 3.1: Start API Server
```bash
python src/api/efab_api_server.py
```

**Expected Output:**
```
======================================================================
Beverly Knits ERP - eFab API Server
======================================================================
eFab Base URL: https://efab.bkiapps.com
Session: ABC123...
Server: http://0.0.0.0:5006
======================================================================
Press Ctrl+C to stop
======================================================================
```

Keep this terminal open!

### Step 3.2: Test Health Endpoint
Open new terminal:
```bash
curl http://localhost:5006/api/health
```

**Expected:**
```json
{
  "status": "healthy",
  "data_source": "efab_direct",
  "efab_connected": true,
  "timestamp": "2025-10-18T..."
}
```

### Step 3.3: Test Comprehensive Forecast Endpoint
```bash
curl "http://localhost:5006/api/forecast/comprehensive?forecast_weeks=13&blending_strategy=weighted_average"
```

**Expected:**
```json
{
  "status": "success",
  "ml_forecasts": 50,
  "blended_forecasts": 50,
  "actual_orders": 35,
  "proactive_production_count": 12,
  "variance_alerts_count": 5,
  "combined_schedule_count": 50,
  "data": {
    "blended_forecast": {...},
    "combined_schedule": {...},
    "proactive_production": [...],
    "variance_alerts": [...]
  },
  "metadata": {...}
}
```

**Troubleshooting:**
- If 500 error → Check API server logs for Python errors
- If "No forecast data" → Verify historical sales data exists
- If "Style mapping error" → Check style_mappings table

### Step 3.4: Test Upload External Forecast
```bash
curl -X POST http://localhost:5006/api/forecast/upload-external \
  -H "Content-Type: application/json" \
  -d '{
    "source_name": "sales_team",
    "forecasts": [
      {
        "style": "STYLE001",
        "week_number": 42,
        "forecasted_yards": 1200,
        "confidence": 0.90,
        "notes": "Customer pre-order"
      }
    ]
  }'
```

**Expected:**
```json
{
  "status": "success",
  "message": "Successfully uploaded 1 forecasts",
  "source": "sales_team",
  "count": 1
}
```

### Step 3.5: Test Accuracy Report
```bash
curl "http://localhost:5006/api/forecast/accuracy-report?lookback_weeks=13"
```

**Expected:**
```json
{
  "status": "success",
  "report": {
    "period": "Last 13 weeks",
    "sources_analyzed": 4,
    "source_performance": {...},
    "best_performer": {...},
    "worst_performer": {...}
  }
}
```

### Step 3.6: Test Weight Recommendations
```bash
curl "http://localhost:5006/api/forecast/weight-recommendations?lookback_weeks=13"
```

### Step 3.7: Test Proactive Production
```bash
curl "http://localhost:5006/api/forecast/proactive-production?min_confidence=0.85&max_items=20"
```

## Phase 4: Dashboard Integration Testing

### Step 4.1: Start Dashboard Server
New terminal:
```bash
cd C:\finalee\beverly_knits_erp_v2\web
python -m http.server 8000
```

### Step 4.2: Open Dashboard
Browser: http://localhost:8000/consolidated_dashboard.html

### Step 4.3: Verify Forecast Section
**Visual Checks:**
- [ ] 13-Week Forecast section visible below Time-Phased PO
- [ ] 4 summary cards show data (not "0" or "--")
- [ ] Week headers show current week + 13 weeks (e.g., W42, W43... W54)
- [ ] Table has rows with data
- [ ] Color coding visible (green, blue, yellow cells)
- [ ] Icons appear (✅, 🔮, ⚡, ❓)

**Browser Console (F12):**
```
=== Loading 13-Week Comprehensive Forecast ===
Comprehensive forecast response: {...}
✓ 13-week forecast loaded successfully
```

**No Errors Expected!**

### Step 4.4: Test Interactive Features

**Blending Strategy:**
1. Change dropdown: "Weighted Average" → "Highest Confidence"
2. Table should refresh
3. Console: "Loading 13-Week Comprehensive Forecast..."

**Proactive Production Button:**
1. Click "Proactive Production"
2. Modal opens
3. Table shows recommendations with priorities
4. Close modal (X button)

**Variance Alerts Button:**
1. Click "Variance Alerts"
2. Modal opens
3. Table shows forecast vs actual discrepancies
4. Severity colors visible (red/orange/yellow)

**Accuracy Report Button:**
1. Click "Accuracy Report"
2. Modal opens
3. Source performance cards show MAPE
4. Weight recommendations table populated
5. Best/worst performer boxes show data

### Step 4.5: Test Pagination
If >20 styles:
- [ ] Pagination controls appear
- [ ] Click "Next" → table updates
- [ ] Page numbers clickable
- [ ] Click "Previous" → returns

## Phase 5: Data Validation

### Step 5.1: Verify Forecast Accuracy
Compare dashboard data with database:

```python
from src.forecasting.weekly_forecast_generator import WeeklyForecastGenerator

gen = WeeklyForecastGenerator(forecast_weeks=13)
result = gen.generate_comprehensive_forecast()

# Check counts match dashboard
print(f"ML forecasts: {len(result['ml_forecast'])}")
print(f"Combined schedule: {len(result['combined_schedule'])}")
print(f"Proactive production: {len(result['proactive_production'])}")
```

**Dashboard should match these counts!**

### Step 5.2: Verify Color Coding Logic
```python
# Check a specific style-week
combined = result['combined_schedule']
style = list(combined.keys())[0]
week_data = combined[style]

for week, info in list(week_data.items())[:3]:
    print(f"Week {week}:")
    print(f"  Yards: {info['yards']}")
    print(f"  Source: {info['source']}")  # 'actual_order' or 'forecast'
    print(f"  Confidence: {info['confidence']}")

    # Verify color logic
    if info['source'] == 'actual_order':
        assert info['confidence'] == 1.0, "Actual orders should have 100% confidence"
        print(f"  → Should be GREEN ✅")
    elif info['confidence'] >= 0.85:
        print(f"  → Should be BLUE 🔮")
    elif info['confidence'] >= 0.70:
        print(f"  → Should be YELLOW ⚡")
    else:
        print(f"  → Should be GRAY ❓")
```

### Step 5.3: Verify Week Calculations
```python
from datetime import datetime

current_week = datetime.now().isocalendar()[1]
print(f"Current ISO week: {current_week}")

# Dashboard should show:
# W{current_week}, W{current_week+1}, ..., W{current_week+12}

expected_weeks = [(current_week + i) % 53 or 1 for i in range(13)]
print(f"Expected week headers: {expected_weeks}")
```

Check dashboard headers match!

## Phase 6: Performance Testing

### Step 6.1: Measure Forecast Generation Time
```python
import time
from src.forecasting.weekly_forecast_generator import WeeklyForecastGenerator

gen = WeeklyForecastGenerator(forecast_weeks=13)

start = time.time()
result = gen.generate_comprehensive_forecast()
elapsed = time.time() - start

print(f"Forecast generation took: {elapsed:.2f} seconds")
print(f"Styles processed: {len(result['ml_forecast'])}")
print(f"Average time per style: {elapsed/len(result['ml_forecast']):.3f}s")
```

**Benchmarks:**
- < 5 seconds for 50 styles = Excellent
- < 15 seconds for 100 styles = Good
- < 30 seconds for 200 styles = Acceptable
- > 60 seconds = Needs optimization

### Step 6.2: Test API Response Times
```bash
time curl -s "http://localhost:5006/api/forecast/comprehensive" > /dev/null
```

**Target:** < 5 seconds

### Step 6.3: Monitor Database Queries
Check Turso dashboard for query counts and latency.

## Common Issues & Solutions

### Issue 1: "No forecast data available"
**Cause:** No historical sales data
**Solution:**
```bash
python scripts/import_sales_to_turso.py --file your_sales.csv
# OR use eFab integration to pull real sales data
```

### Issue 2: "Style mapping error"
**Cause:** Style not found in mapping table
**Solution:**
```bash
python scripts/import_style_mappings_to_turso.py --create-table --file eFab_Styles.xlsx
```

### Issue 3: Week headers show wrong weeks
**Cause:** System date incorrect or timezone issue
**Solution:**
```python
from datetime import datetime
print(f"System date: {datetime.now()}")
print(f"ISO week: {datetime.now().isocalendar()[1]}")
# Verify this matches expected week
```

### Issue 4: API returns 500 error
**Cause:** Python exception in forecast generation
**Solution:**
- Check API server logs for stack trace
- Test forecast modules individually
- Verify Turso connection

### Issue 5: Colors not showing in dashboard
**Cause:** CSS class conflict or missing TailwindCSS
**Solution:**
- Inspect element in DevTools
- Verify TailwindCSS CDN loaded
- Check for CSS specificity conflicts

## Success Criteria

Pipeline is working correctly if:

✅ All database tables created and populated
✅ ML forecast generation runs without errors
✅ External forecasts can be uploaded
✅ Forecast blending produces combined schedule
✅ Comparator identifies proactive production opportunities
✅ Accuracy tracker calculates metrics
✅ API endpoints return valid JSON
✅ Dashboard displays 13-week table with correct data
✅ Color coding matches confidence levels
✅ Modals open and show data
✅ Week headers are dynamic and correct
✅ Pagination works
✅ Performance acceptable (<30s for 100 styles)

## Next Steps

After successful testing:

1. **Production Deployment:**
   - Set up production Turso database
   - Configure production .env
   - Deploy API server
   - Deploy dashboard

2. **Monitor & Tune:**
   - Track forecast accuracy for 2-3 weeks
   - Review variance alerts
   - Apply weight recommendations
   - Adjust confidence thresholds

3. **User Training:**
   - Show stakeholders the dashboard
   - Explain color coding
   - Demonstrate proactive production workflow
   - Train on uploading external forecasts

4. **Continuous Improvement:**
   - Weekly accuracy reviews
   - Monthly weight adjustments
   - Quarterly model retraining
   - Feedback loops with sales team

## Support

For issues during testing:
- Check logs: Browser console (F12) + API server terminal
- Review error messages carefully
- Test each phase independently
- Verify data exists at each step
- Check Turso database directly if needed

## Testing Checklist

```
[ ] Phase 1: Database Setup
  [ ] Tables created
  [ ] Style mappings imported
  [ ] Historical sales imported
  [ ] BOM/specs imported

[ ] Phase 2: Module Testing
  [ ] ML forecast generator works
  [ ] External loader works
  [ ] Blender works
  [ ] Comparator works
  [ ] Accuracy tracker works

[ ] Phase 3: API Testing
  [ ] Health endpoint responds
  [ ] Comprehensive forecast endpoint works
  [ ] Upload external forecast works
  [ ] Accuracy report works
  [ ] Weight recommendations work
  [ ] Proactive production works

[ ] Phase 4: Dashboard Testing
  [ ] Forecast section visible
  [ ] Data displays correctly
  [ ] Colors work
  [ ] Week headers dynamic
  [ ] Modals functional
  [ ] Blending strategy changes work

[ ] Phase 5: Validation
  [ ] Forecast counts match
  [ ] Color logic correct
  [ ] Week calculations correct

[ ] Phase 6: Performance
  [ ] Generation time acceptable
  [ ] API response time < 5s
  [ ] Database queries efficient

[ ] Final Acceptance
  [ ] End-to-end workflow successful
  [ ] No console errors
  [ ] Data accuracy verified
  [ ] Performance acceptable
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-10-18
**Tested With:** Beverly Knits ERP v2, Python 3.9+, Turso LibSQL
