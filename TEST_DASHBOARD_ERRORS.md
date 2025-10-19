# Dashboard Error Handling - Testing Guide

## Quick Test Procedures

### Test 1: Error State (API Server Down)

**Purpose**: Verify error messages display correctly when eFab API is unavailable.

**Steps**:
1. Stop the eFab API server:
   ```bash
   # Find the process running on port 5006
   netstat -ano | findstr :5006

   # Kill the process (replace <PID> with actual process ID)
   taskkill /PID <PID> /F
   ```

2. Open browser and navigate to consolidated dashboard:
   ```
   http://localhost:5000/consolidated_dashboard.html
   ```

3. Click "Load Data" button on Forecasted Production Orders section

**Expected Results**:
- ✅ Red error alert appears in table area
- ✅ Error message: "eFab API unavailable - Cannot connect to server on localhost:5006"
- ✅ Troubleshooting hints are visible
- ✅ Retry button is present
- ✅ Summary cards show "0" (not undefined or NaN)
- ✅ No JavaScript errors in browser console

**Screenshot Location**: Error message should appear where the table normally is

---

### Test 2: Loading State

**Purpose**: Verify loading spinner appears during API calls.

**Steps**:
1. Start the eFab API server:
   ```bash
   cd C:\finalee\beverly_knits_erp_v2
   python efab_consolidated_api.py
   ```

2. Open browser developer tools (F12)
3. Go to Network tab
4. Set throttling to "Slow 3G" (to simulate slow network)
5. Refresh the dashboard
6. Click "Load Data" button

**Expected Results**:
- ✅ Blue loading spinner appears immediately
- ✅ Loading message: "Loading forecasted orders from eFab API..."
- ✅ Spinner animates smoothly (rotates)
- ✅ Table appears after data loads
- ✅ Loading state disappears when data arrives

---

### Test 3: Success State (Normal Operation)

**Purpose**: Verify normal data loading works correctly.

**Steps**:
1. Ensure eFab API server is running on port 5006
2. Open consolidated dashboard
3. Click "Load Data" buttons on both:
   - Forecasted Production Orders
   - Forecasted Fabric Requirements

**Expected Results**:
- ✅ Brief loading spinner appears
- ✅ Tables populate with data
- ✅ No error messages visible
- ✅ Summary cards show correct values (not zeros)
- ✅ All columns have data (no blank cells)
- ✅ No console errors

---

### Test 4: Empty Data State

**Purpose**: Verify handling of valid API response with no data.

**Steps**:
1. Temporarily modify backend to return empty arrays:
   ```python
   # In efab_consolidated_api.py, find get_fabric_forecast()
   # Replace return statement with:
   return {
       "status": "success",
       "forecast_items": [],
       "fabric_forecast": [],
       "summary": {
           "total_yards_forecasted": 0,
           "critical_items": 0,
           ...
       }
   }
   ```

2. Restart eFab API server
3. Refresh dashboard
4. Click "Load Data" button

**Expected Results**:
- ✅ Info message appears (not error)
- ✅ Message: "No fabric forecast data available"
- ✅ Blue/gray styling (not red)
- ✅ Summary cards show "0"
- ✅ No error alerts

**Cleanup**: Revert backend changes after test

---

### Test 5: Retry Functionality

**Purpose**: Verify retry button works correctly.

**Steps**:
1. Stop eFab API server (see Test 1)
2. Load dashboard - should show error
3. Start eFab API server
4. Click "Retry" button on error alert

**Expected Results**:
- ✅ Page reloads
- ✅ Loading spinner appears
- ✅ Data loads successfully
- ✅ Error message disappears

---

### Test 6: Multiple Rapid Clicks

**Purpose**: Verify no race conditions or duplicate errors.

**Steps**:
1. Stop eFab API server
2. Click "Load Data" button 5 times rapidly
3. Wait 5 seconds

**Expected Results**:
- ✅ Only one error message appears
- ✅ No duplicate error alerts
- ✅ No console errors
- ✅ No frozen UI

---

### Test 7: Browser Console Verification

**Purpose**: Ensure proper error logging for debugging.

**Steps**:
1. Open browser console (F12)
2. Stop eFab API server
3. Click "Load Data" button
4. Check console output

**Expected Results**:
- ✅ Error is logged to console with full details
- ✅ Console shows: `Error loading forecasted orders: [error details]`
- ✅ User sees friendly error message (not raw error)
- ✅ Stack trace available in console for debugging

---

## Edge Case Testing

### Edge Case 1: Network Timeout
**Simulate**: Set network throttling to "Offline" in DevTools
**Expected**: "Network error - Please check your connection"

### Edge Case 2: Malformed JSON Response
**Simulate**: Modify backend to return invalid JSON
**Expected**: Generic error message, no page crash

### Edge Case 3: Partial Data Missing
**Simulate**: Remove some fields from API response
**Expected**: Tables show "-" for missing data, no errors

---

## Automated Test Checklist

Run through this checklist for both tables:
- **Forecasted Production Orders** (loadForecastedOrders)
- **Forecasted Fabric Requirements** (loadFabricForecast)

| Test | Forecasted Orders | Fabric Forecast | Notes |
|------|-------------------|-----------------|-------|
| Error state displays | ☐ | ☐ | Red alert with troubleshooting |
| Loading state displays | ☐ | ☐ | Blue spinner with message |
| Success state displays | ☐ | ☐ | Table with data |
| Empty data displays | ☐ | ☐ | Info message (blue) |
| Retry button works | ☐ | ☐ | Reloads page successfully |
| Summary cards reset | ☐ | ☐ | Show "0" on error |
| Console logging works | ☐ | ☐ | Errors logged for debugging |
| No JavaScript errors | ☐ | ☐ | Clean console on success |

---

## Performance Testing

### Test Load Times
1. Open Network tab in DevTools
2. Click "Load Data" button
3. Measure time from click to table display

**Acceptable Thresholds**:
- ✅ Loading state appears: < 100ms
- ✅ API response time: < 2 seconds (local)
- ✅ Table render time: < 500ms
- ✅ Total time to interactive: < 3 seconds

---

## Accessibility Testing

### Keyboard Navigation
1. Tab through the error alert
2. Press Enter on Retry button
3. Verify screen reader announces error

**Expected**:
- ✅ Error alert is focusable
- ✅ Retry button can be activated with keyboard
- ✅ ARIA roles are present

### Screen Reader Testing
1. Enable screen reader (NVDA/JAWS)
2. Trigger error state
3. Listen to announcement

**Expected**:
- ✅ Alert role announces error
- ✅ Error message is read aloud
- ✅ Troubleshooting list is accessible

---

## Regression Testing

### Before/After Comparison

**Before** (old behavior):
```
Error: undefined
[Blank table]
Summary cards: NaN, undefined, 0
```

**After** (new behavior):
```
API Error
eFab API unavailable - Cannot connect to server on localhost:5006

Troubleshooting:
• Check that eFab API server is running on localhost:5006
• Verify network connectivity
• Check browser console for detailed errors

[Retry Button]

Summary cards: 0, 0, 0, 0
```

---

## Bug Reporting Template

If you find issues, report with this format:

```
**Test**: [Test name, e.g., "Error State Test"]
**Expected**: [What should happen]
**Actual**: [What actually happened]
**Steps to Reproduce**:
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Browser**: [Chrome/Firefox/Edge + version]
**Console Errors**: [Paste console output]
**Screenshot**: [Attach if relevant]
```

---

## Success Criteria

All tests must pass for the feature to be considered complete:

- [ ] Error states display user-friendly messages
- [ ] Loading states show animated spinners
- [ ] Success states populate tables correctly
- [ ] Empty data states show info messages
- [ ] Retry buttons reload and recover
- [ ] Summary cards never show NaN/undefined
- [ ] Console errors are logged for debugging
- [ ] No JavaScript errors in production
- [ ] Performance meets thresholds
- [ ] Accessibility standards met (WCAG 2.1 AA)

---

## Contact

**Developer**: Claude (Frontend Developer Agent)
**Date**: 2025-10-19
**Related Files**:
- `web/consolidated_dashboard.html` (modified)
- `dashboard_error_handling_summary.md` (documentation)
- `TEST_DASHBOARD_ERRORS.md` (this file)
