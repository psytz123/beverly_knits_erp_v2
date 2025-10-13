# Lessons Learned - Beverly Knits ERP eFab Integration

## Date: 2025-10-13

---

## Key Lessons from eFab API Integration

### 1. **Don't Trust eFab's Pre-calculated Fields**
**Problem**: eFab's aggregated/calculated fields (like `qty_theoretical`, `qty_planning`) are often incorrect or don't match the UI display.

**Solution**:
- Always pull individual component fields directly
- Calculate values ourselves using the component fields
- Formula: `theoretical_balance = reconciled_qty + added + adjustments + consumed`
- Formula: `planning_balance = theoretical_balance + allocated + onorder`

**Example**:
```python
# ❌ Wrong - Don't trust these
theoretical = yarn.get('qty_theoretical')  # This is wrong!
planning = yarn.get('qty_planning')        # This is also wrong!

# ✅ Correct - Calculate from components
theoretical_balance = (
    safe_float(yarn.get('reconciled_qty', 0)) +
    safe_float(yarn.get('added', 0)) +
    safe_float(yarn.get('adjustments', 0)) +
    safe_float(yarn.get('consumed', 0))
)

planning_balance = (
    theoretical_balance +
    safe_float(yarn.get('allocated', 0)) +
    safe_float(yarn.get('onorder', 0))
)
```

### 2. **eFab UI is the Source of Truth**
**Problem**: API documentation or field names may be misleading.

**Solution**:
- Always verify calculated values against the eFab web UI
- Use screenshots from eFab to find the exact values we need to match
- Search for specific values in API responses to identify correct fields

**Process**:
1. Take screenshot of eFab UI showing target value
2. Make API call and log all fields
3. Search response for the value from screenshot
4. Identify which field or calculation produces that value

### 3. **eFab Uses Asynchronous Report Generation**
**Problem**: Not all reports have real-time JSON API endpoints. Some endpoints exist but return no data.

**Discovery**: eFab uses a report queue system that generates Excel files asynchronously.

**Architecture**:
- Reports are queued at `/api/report/report_queue`
- Excel files are stored at `/admin/report_queue/{filename}`
- Filename is nested in `report.notes.filename`, not at top level
- Must download Excel and parse with pandas

**Solution**:
```python
# Step 1: Get report queue
queue_data = fetch_from_efab('api/report/report_queue')

# Step 2: Find target report (note nested filename!)
for report in queue_data:
    if 'yarn_demand' in report.get('report_name', ''):
        filename = report.get('notes', {}).get('filename')  # ⚠️ Nested!
        break

# Step 3: Download Excel
response = requests.get(f"{EFAB_BASE_URL}/admin/report_queue/{filename}")

# Step 4: Parse with pandas
df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
```

### 4. **Excel Files May Have Title Rows**
**Problem**: Pandas reads Excel with "Unnamed" columns when headers aren't in first row.

**Solution**: Implement smart header detection
```python
# Read without assuming header location
df = pd.read_excel(excel_data, engine='openpyxl', header=None)

# Find actual header row (look for keywords)
header_row = None
for idx in range(min(10, len(df))):
    row_values = df.iloc[idx].astype(str).tolist()
    if any('yarn' in str(v).lower() or 'style' in str(v).lower() for v in row_values):
        header_row = idx
        break

# Re-read with correct header
if header_row is not None:
    excel_data.seek(0)
    df = pd.read_excel(excel_data, engine='openpyxl', header=header_row)
```

### 5. **Pandas NaN Values Must Be Converted for JSON**
**Problem**: Pandas NaN values cannot be serialized to JSON, causing `SyntaxError: Unexpected token 'N'` in frontend.

**Root Cause**: NaN is a special float value in Python/pandas, not a valid JSON value. JSON only supports `null` for missing values.

**Solution**: Use `df.replace({np.nan: None})` before converting to dict
```python
import numpy as np

# Replace NaN with None using numpy
df = df.replace({np.nan: None})
data = df.to_dict(orient='records')

# ❌ Wrong - fillna(value=None) doesn't work
df = df.fillna(value=None)  # This fails with "Must specify a fill 'value' or 'method'"
```

**Why This Matters**:
- Empty cells in Excel become NaN in pandas
- NaN in JSON response breaks frontend parsing
- Must convert NaN → None (which becomes null in JSON)

### 6. **Always Use Dynamic Week Calculations for Time-Phased Data**
**Problem**: Hardcoded week numbers (like W36-W43) become outdated as time passes, showing incorrect week references.

**Root Cause**: Time-phased data is inherently time-sensitive. What's "this week" changes every 7 days, so any hardcoded week numbers will be wrong within weeks.

**Solution**: Calculate current ISO week number dynamically and generate all week-based headers and data mappings from it
```javascript
// Calculate ISO week number
function getISOWeek(date) {
    const d = new Date(date);
    d.setHours(0, 0, 0, 0);
    d.setDate(d.getDate() + 4 - (d.getDay() || 7));
    const yearStart = new Date(d.getFullYear(), 0, 1);
    const weekNo = Math.ceil((((d - yearStart) / 86400000) + 1) / 7);
    return weekNo;
}

// Generate headers dynamically
const currentWeek = getISOWeek(new Date());
for (let i = 0; i < 8; i++) {
    const weekNum = currentWeek + i;
    // Create header: W42, W43, W44...
}

// Map data to current week
if (yarnData["Receipts This Week"] !== undefined) {
    yarnData.weekly_receipts[`week_${currentWeek}`] = yarnData["Receipts This Week"];
}
```

**Implementation Points**:
- Calculate `currentWeek` once at start of data loading
- Use `currentWeek` variable throughout for consistency
- Generate table headers dynamically: W42, W43, W44, W45, W46, W47, W48, W49, W50+
- Map "This Week" data to `week_${currentWeek}` not hardcoded `week_36`
- Generate table cells in loop: `for (let i = 0; i < 8; i++) formatWeekCell(\`week_${currentWeek + i}\`, ...)`
- Update shortage detection: check if `first_shortage_week.includes(currentWeek.toString())`

**Why This Matters**:
- Time-phased reports must stay current without code changes
- Users expect "this week" to mean the actual current week
- Hardcoded values create maintenance burden and confusion
- Dynamic calculation ensures data accuracy over time

### 7. **Log Everything During Debugging**
**Why**: eFab API behavior is not always intuitive, and detailed logs help identify issues quickly.

**What to Log**:
- Response status codes
- Response content types (JSON vs HTML)
- First 500 characters of unexpected responses
- All available fields in API responses
- Sample values for key fields
- Calculated vs raw values

**Example**:
```python
if response.status_code == 200:
    try:
        data = response.json()
        logger.info(f"✓ Successfully fetched {endpoint}")
        return data
    except ValueError as json_err:
        logger.error(f"JSON parse error for {endpoint}: {json_err}")
        logger.error(f"Response content type: {response.headers.get('Content-Type')}")
        logger.error(f"Response text (first 500 chars): {response.text[:500]}")
        return None
```

### 8. **Understand Business Logic First**
**Problem**: Technical implementation without business context leads to wrong calculations.

**Solution**:
- Ask about business rules before coding
- Understand what each field represents in manufacturing context
- Verify calculations make business sense
- Example: "Planning balance" includes future orders, "theoretical balance" is current state

---

## Pattern: Working with eFab APIs

### Step-by-Step Process

1. **Identify the data requirement**
   - What value do we need to display?
   - Where is it shown in eFab UI?

2. **Explore eFab API**
   - Check if real-time API endpoint exists
   - If not, check report queue for Excel reports
   - Look at API response structure

3. **Find the correct data source**
   - Try API endpoint first
   - If no data or wrong data, check report queue
   - Verify report exists and is recent

4. **Verify against eFab UI**
   - Take screenshots of target values
   - Log all available fields from API
   - Match API data to UI values

5. **Implement with proper error handling**
   - Handle missing data gracefully
   - Return appropriate status messages
   - Log warnings for debugging

6. **Test thoroughly**
   - Verify calculations match UI
   - Test with multiple data points
   - Check edge cases (zero values, negative values)

---

## Common Pitfalls & Solutions

### ❌ Pitfall: Trusting field names
**Example**: Field named `qty_theoretical` doesn't match "Theoretical Balance" in UI

**Solution**: Always calculate from components and verify against UI

### ❌ Pitfall: Assuming JSON API endpoints
**Example**: `/api/report/yarn_demand_ko` endpoint exists but returns no data

**Solution**: Check report queue for Excel reports

### ❌ Pitfall: Looking for filename at top level
**Example**: `report.get('filename')` returns None

**Solution**: Filename is nested: `report.get('notes', {}).get('filename')`

### ❌ Pitfall: Default pandas Excel reading
**Example**: Getting "Unnamed: 1", "Unnamed: 2" columns

**Solution**: Implement smart header detection to find actual header row

---

## Quick Reference: Field Mapping

### Yarn Inventory Fields (from `/api/yarn/active`)

| eFab Field | Purpose | Notes |
|------------|---------|-------|
| `reconciled_qty` | Physical inventory count | Base starting point |
| `added` | Yarn received since reconciliation | Positive value |
| `consumed` | Yarn used in production | Negative value |
| `adjustments` | Manual inventory adjustments | Can be positive or negative |
| `allocated` | Yarn reserved for future orders | Negative value |
| `onorder` | Yarn ordered but not received | Positive value |
| `qty_theoretical` | ❌ Don't use - incorrect | Calculate yourself |
| `qty_planning` | ❌ Don't use - incorrect | Calculate yourself |

### Calculated Fields (Calculate These Yourself)

```python
theoretical_balance = reconciled_qty + added + adjustments + consumed
planning_balance = theoretical_balance + allocated + onorder
```

---

## File Structure Notes

### Backend API (`src/api/efab_api_server.py`)
- Main proxy server for eFab API
- Port: 5006
- Session-based authentication using `dancer.session` cookie
- CORS enabled for frontend on port 8080

### Frontend (`web/`)
- Static HTML/CSS/JS dashboard
- Served on port 8080
- Makes AJAX calls to backend on port 5006

---

## Testing Checklist

When implementing new eFab API integration:

- [ ] Verify API endpoint exists and returns data
- [ ] Check if data is JSON or requires Excel parsing
- [ ] Log all available fields in response
- [ ] Take screenshot of target value from eFab UI
- [ ] Find matching value in API response
- [ ] Implement calculation using component fields
- [ ] Verify calculation matches eFab UI exactly
- [ ] Test with multiple data points
- [ ] Handle missing/null values gracefully
- [ ] Add comprehensive logging for debugging
- [ ] Test frontend display

---

## Future Considerations

1. **Caching Strategy**: eFab API can be slow; consider caching with appropriate TTL
2. **Error Recovery**: Implement retry logic for transient failures
3. **Data Validation**: Add validation to detect when eFab data structure changes
4. **Performance**: Monitor API response times and optimize as needed
5. **Documentation**: Keep this file updated with new discoveries

---

## Contact & References

- **eFab Base URL**: https://efab.bkiapps.com
- **Backend API**: http://localhost:5006
- **Frontend Dashboard**: http://localhost:8080
- **Report Queue**: https://efab.bkiapps.com/reports/report_queue
