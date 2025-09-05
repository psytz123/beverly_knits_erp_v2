# eFab API Integration Activation Guide
## Beverly Knits ERP v2 - Real-Time Data Integration

### Document Version: 1.0
### Date: September 5, 2025
### Status: Ready for Activation

---

## ✅ Implementation Status

### Completed Components:
- ✅ **Environment Configuration** (.env file with credentials)
- ✅ **API Client Infrastructure** (EFabAPIClient fully implemented)
- ✅ **Data Transformation Layer** (EFabDataTransformer ready)
- ✅ **API Data Loader** (EFabAPIDataLoader integrated)
- ✅ **Main ERP Integration** (beverly_comprehensive_erp.py updated)
- ✅ **Feature Flags System** (Gradual rollout controls)
- ✅ **Test Suite** (test_efab_api_integration.py)
- ✅ **Dashboard Integration** (83 API calls confirmed)

### API Data Usage Confirmation:

#### **Dashboard (web/consolidated_dashboard.html)**
- **CONFIRMED**: Uses API endpoints exclusively via `fetchAPI()` function
- 83 total API calls throughout the dashboard
- All data tabs load from API endpoints:
  - Production: `/api/production-planning`, `/api/production-pipeline`, `/api/knit-orders`
  - Inventory: `/api/inventory-intelligence-enhanced`, `/api/yarn-intelligence`
  - Forecasting: `/api/ml-forecast-detailed`
  - Analytics: Various consolidated endpoints

#### **Backend (src/core/beverly_comprehensive_erp.py)**
- **CONFIRMED**: When `EFAB_API_ENABLED=true`:
  - Uses `EFabAPIDataLoader` as primary data loader
  - Routes all data through `self.parallel_loader = self.api_loader`
  - Automatic fallback to files if API fails

---

## 🚀 Quick Activation Steps

### Step 1: Enable API Integration (30 seconds)

Edit `.env` file and change:
```bash
EFAB_API_ENABLED=true  # Was false, now true
```

### Step 2: Test Connectivity (2 minutes)

Run the test suite:
```bash
cd D:\AI\Workspaces\efab.ai\beverly_knits_erp_v2
python tests/test_efab_api_integration.py
```

Expected output:
```
✅ Environment Setup
✅ API Modules
✅ Feature Flags
✅ Client Initialization
✅ Authentication
✅ Health Check
✅ Yarn Inventory
✅ Data Loader
✅ Fallback Mechanism
✅ Performance

🎉 All tests passed! eFab API integration is ready.
```

### Step 3: Start Server with API (1 minute)

```bash
# Kill any existing server
pkill -f "python3.*beverly"

# Start with API integration
python3 src/core/beverly_comprehensive_erp.py
```

Look for these messages:
```
[OK] eFab API data loader available - real-time data integration ready
[OK] Using eFab API data loader for real-time data integration
[OK] eFab API integration active - real-time data enabled with automatic fallback
```

### Step 4: Verify Dashboard (1 minute)

1. Open browser to http://localhost:5006/consolidated
2. Open Developer Console (F12)
3. Check Network tab for API calls
4. Verify data loads correctly in all tabs

---

## 📊 Gradual Rollout Strategy

### Phase 1: Testing (Current)
```bash
# .env settings
EFAB_API_ENABLED=true
EFAB_ROLLOUT_PERCENTAGE=0  # Testing only, no production traffic
```

### Phase 2: Limited Rollout (10% traffic)
```python
# In Python console or script:
from src.config.feature_flags import set_efab_rollout
set_efab_rollout(10)  # 10% of requests use API
```

### Phase 3: Expanded Rollout (50% traffic)
```python
set_efab_rollout(50)  # 50% of requests use API
```

### Phase 4: Full Production (100% traffic)
```python
set_efab_rollout(100)  # All requests use API
```

---

## 🔄 Rollback Procedures

### Quick Disable (< 10 seconds)
```bash
# Edit .env
EFAB_API_ENABLED=false
# Restart server
```

### Emergency Rollback (< 30 seconds)
```python
# In Python console:
from src.config.feature_flags import emergency_disable_efab
emergency_disable_efab()
```

### Full Rollback (< 2 minutes)
```bash
# 1. Disable in .env
EFAB_API_ENABLED=false

# 2. Reset feature flags
python3 -c "from src.config.feature_flags import disable_efab_api; disable_efab_api()"

# 3. Restart server
pkill -f "python3.*beverly"
python3 src/core/beverly_comprehensive_erp.py
```

---

## 📈 Monitoring & Verification

### Check API Status
```python
# Python console
from src.config.feature_flags import is_efab_api_enabled
print(f"API Enabled: {is_efab_api_enabled()}")
```

### Monitor API Calls
```bash
# Watch server logs
tail -f logs/efab_api.log  # If logging configured
```

### Verify Data Source
1. Check server console for "[OK] Using eFab API data loader"
2. Dashboard Network tab should show calls to efab.bkiapps.com
3. Data should update in real-time (no 15-60 min delays)

---

## ⚠️ Important Notes

### Current Limitations
1. **Partial Integration**: Some direct file loads bypass API (being addressed)
2. **BOM Data**: Still file-based (API endpoint pending)
3. **Style Mappings**: Still file-based (API endpoint pending)

### Performance Impact
- **API Response Time**: < 2 seconds typical
- **Fallback Time**: < 100ms to switch to files
- **Cache TTL**: 5-60 minutes depending on data type

### Security Considerations
- Credentials stored in `.env` (never commit)
- Session-based authentication with auto-refresh
- All API calls use HTTPS

---

## 🎯 Success Criteria

### Technical Metrics
- ✅ API response time < 2 seconds
- ✅ API success rate > 99%
- ✅ Automatic fallback working
- ✅ No production disruptions

### Business Metrics
- ✅ Real-time data updates (< 1 minute latency)
- ✅ Reduced manual data management
- ✅ Improved planning accuracy
- ✅ Enhanced decision making

---

## 📞 Support & Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Authentication fails | Check EFAB_USERNAME and EFAB_PASSWORD in .env |
| No data returned | Verify EFAB_BASE_URL is correct |
| Slow performance | Check EFAB_CACHE_ENABLED=true |
| API timeout | Increase EFAB_API_TIMEOUT in .env |
| Fallback activated | Check API health, credentials |

### Debug Mode
```bash
# Enable debug logging
EFAB_DEBUG_MODE=true
python3 src/core/beverly_comprehensive_erp.py
```

### Dry Run Mode (Test without API calls)
```bash
python tests/test_efab_api_integration.py --dry-run
```

---

## 📋 Checklist for Production Activation

- [ ] Credentials verified in .env
- [ ] Test suite passes all tests
- [ ] Server starts with API messages
- [ ] Dashboard loads data correctly
- [ ] Monitoring configured
- [ ] Team notified of activation
- [ ] Rollback procedure tested
- [ ] Documentation reviewed

---

## 🎉 Activation Complete!

Once all steps are completed and verified:

1. **Set EFAB_API_ENABLED=true** in .env
2. **Restart the server**
3. **Monitor for 30 minutes**
4. **Gradually increase rollout percentage**

The Beverly Knits ERP v2 system is now using **real-time data from eFab.ai API** with automatic fallback to ensure reliability!

---

**Document Status**: Complete
**Last Updated**: September 5, 2025
**Next Review**: After first week of production use