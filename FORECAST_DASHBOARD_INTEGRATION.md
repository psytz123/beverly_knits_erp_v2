# 13-Week Forecast Dashboard Integration Guide

## Overview
This guide explains how to integrate the new 13-week forecast component into `web/consolidated_dashboard.html`.

## Integration Steps

### 1. Locate the Insertion Point

Find the "Time-Phased Yarn Purchase Orders" section in `consolidated_dashboard.html` (around line 1873).

Look for this closing `</div>` tag after the time-phased section (around line 1960):

```html
                <!-- Pagination for Time-Phased -->
                <div id="timePhasedPagination"></div>

                <!-- Expedite Recommendations -->
                ...
            </div>
            <!-- END Time-Phased Section -->
```

### 2. Insert the Forecast Component

**Option A: Copy-Paste** (Recommended)
1. Open `web/forecast_dashboard_component.html`
2. Copy the ENTIRE contents (excluding the HTML comment at the top)
3. Paste it RIGHT AFTER the closing `</div>` of the Time-Phased section (line ~1960)

**Option B: Include via Script**
Add this script tag before the closing `</body>` tag:

```javascript
// Load forecast component
fetch('forecast_dashboard_component.html')
    .then(response => response.text())
    .then(html => {
        const insertPoint = document.querySelector('#timePhasedPagination').parentElement;
        insertPoint.insertAdjacentHTML('afterend', html);
    });
```

### 3. Verify API Endpoint

Ensure the `fetchAPI` function in consolidated_dashboard.html uses the correct base URL:

```javascript
const API_BASE_URL = 'http://localhost:5006';  // efab_api_server.py port

async function fetchAPI(endpoint) {
    const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;
    const response = await fetch(url);
    return response.json();
}
```

### 4. Test the Integration

1. Start the API server:
   ```bash
   python src/api/efab_api_server.py
   ```

2. Open the dashboard:
   ```
   http://localhost:8000/web/consolidated_dashboard.html
   ```

3. Verify:
   - ✅ 13-Week Forecast section appears
   - ✅ Summary cards show data
   - ✅ Week headers are dynamic (current week + 13)
   - ✅ Color coding works (green/blue/yellow)
   - ✅ Modals open (Proactive Production, Variance Alerts, Accuracy Report)

## Features Included

### Summary Cards
1. **Confirmed Orders** (Green) - Actual orders from eFab
2. **High-Confidence Forecasts** (Blue) - ML + External blended forecasts ≥85%
3. **Proactive Production Items** (Orange) - Gaps requiring action
4. **Forecast Accuracy** (Purple) - Current MAPE metric

### 13-Week Table
- **Dynamic week headers** - Auto-updates based on current ISO week
- **Color-coded cells**:
  - 🟢 Green = Actual confirmed order (100% confidence)
  - 🔵 Blue = High-confidence forecast (≥85%)
  - 🟡 Yellow = Medium-confidence forecast (70-85%)
  - ⚪ Gray = Low-confidence forecast (<70%)
- **Tooltips** show confidence percentages
- **Pagination** for large datasets

### Modals

#### Proactive Production
- Priority-ranked recommendations
- Forecasted demand without orders
- Confidence levels and source agreement
- Risk assessment (very low/low/medium/high)
- Recommended actions

#### Variance Alerts
- Forecast vs actual discrepancies >30%
- Over-forecast vs under-forecast indicators
- Severity levels (high/medium/low)
- Helps identify forecast model issues

#### Accuracy Report
- Performance by source (ML, Sales Team, Customer, Market Intel)
- MAPE, Bias, Hit Rate, RMSE metrics
- Weight adjustment recommendations
- Best/worst performing sources

### Controls
- **Blending Strategy selector**:
  - Weighted Average (default) - Uses configured weights
  - Highest Confidence - Takes most confident source
  - Conservative - Takes minimum forecast (lower inventory risk)
  - Aggressive - Takes maximum forecast (higher service level)

## API Endpoints Used

1. `GET /api/forecast/comprehensive` - Main forecast data
2. `GET /api/forecast/accuracy-report` - Performance metrics
3. `GET /api/forecast/weight-recommendations` - Auto-tuning suggestions

## Customization

### Adjust Confidence Thresholds
In `forecast_dashboard_component.html`, modify:

```javascript
if (confidence >= 0.85) {  // HIGH threshold
    // Blue cell
} else if (confidence >= 0.70) {  // MEDIUM threshold
    // Yellow cell
}
```

### Change Default Blending Strategy
Update the select element:

```html
<select id="blendingStrategy" ...>
    <option value="weighted_average" selected>Weighted Average</option>
    ...
</select>
```

### Modify Pagination
Change `rowsPerPage` variable:

```javascript
const rowsPerPage = 20;  // Adjust rows per page
```

## Troubleshooting

### Forecast data not loading
1. Check browser console for errors
2. Verify API server is running on port 5006
3. Check Turso database has historical sales data
4. Ensure style mappings are imported

### Week numbers incorrect
- Verify ISO week calculation: `new Date().toISOString().split('W')[1]`
- Check system date/timezone settings

### Colors not showing
- Inspect cell classes in browser DevTools
- Verify TailwindCSS is loaded
- Check CSS specificity conflicts

### Modals not opening
- Check for JavaScript errors in console
- Verify modal IDs match function calls
- Ensure `closeModal()` function exists

## Next Steps

After integration:

1. **Import Data**:
   ```bash
   # Import style mappings
   python scripts/import_style_mappings_to_turso.py --create-table --file path/to/eFab_Styles.xlsx

   # Import historical sales
   python scripts/import_sales_to_turso.py

   # Import BOM and fabric specs
   python scripts/import_bom_and_specs_to_turso.py
   ```

2. **Test Forecast Pipeline**:
   ```bash
   # Test forecast generation
   python src/forecasting/weekly_forecast_generator.py

   # Test accuracy tracking
   python src/forecasting/forecast_accuracy_tracker.py

   # Run integration tests
   python scripts/test_turso_integration.py
   ```

3. **Configure Weights**:
   - Monitor accuracy report for 2-3 weeks
   - Apply weight recommendations
   - Track improvement

## Support

For issues or questions:
- Check logs in browser console (F12)
- Review API server logs
- Verify Turso database connection
- Test endpoints with curl/Postman

## Version
Dashboard Component Version: 1.0.0
Last Updated: 2025-10-18
Compatible with: Beverly Knits ERP v2
