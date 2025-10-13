# Architecture Decision Record: Production Relevance Filtering for Time-Phased Reports

## Date
2025-09-15

## Status
Implemented

## Context
The time-phased yarn reporting system was showing ALL yarns with PO receipts scheduled, regardless of whether they would be used in production. This created noise in reports and made it difficult to identify actual issues that could impact production.

## Problem
- Reports included yarns that wouldn't be used during the reporting period
- Dashboard showed 20+ yarns when only 4-5 were actually critical
- Users had to manually filter through irrelevant data
- Early issue detection was hampered by noise

## Decision
Implement production relevance filtering in the `/api/time-phased-yarn-po` endpoint to only show yarns that meet specific criteria.

## Implementation

### Filtering Criteria
A yarn is included in the time-phased report if it meets ANY of these conditions:
1. **Active Production Demand**: Yarn is required for styles with active knit orders
2. **Critical Shortage**: Has shortage within next 4 weeks (weeks 36-39)
3. **Negative Balance**: Current balance < 0 (already impacting production)
4. **Negative Planning Balance**: Planning balance < 0 (will impact production)

### Technical Approach
```python
# 1. Load active knit orders
active_orders = knit_orders[(balance > 0) | (in_production == True)]

# 2. Map to required yarns via BOM
active_styles = active_orders['Style'].unique()
required_yarns = BOM[BOM['Style#'].isin(active_styles)]['YarnID']

# 3. Apply filter based on criteria
if yarn_id in required_yarns or has_critical_shortage or negative_balance:
    include_yarn()
```

### Feature Flag
Environment variable `FILTER_NONPRODUCTION_YARNS` controls behavior:
- `true` (default): Apply production relevance filtering
- `false`: Show all yarns (legacy behavior)

## Results
- **Before**: 20+ yarns shown regardless of production relevance
- **After**: 4 yarns shown (all with critical issues)
- **Reduction**: 80% noise reduction in reports
- **Impact**: Faster issue identification, cleaner dashboard

## Trade-offs

### Pros
- Focused reporting on production-critical items
- Earlier issue detection for relevant yarns
- Cleaner dashboard interface
- Better prioritization of actions

### Cons
- May miss long-term planning issues (>4 weeks out)
- Requires BOM data to be accurate for style-to-yarn mapping
- Active production styles must be properly maintained

## Rollback Plan
To disable filtering and return to legacy behavior:
```bash
export FILTER_NONPRODUCTION_YARNS=false
```
Then restart the server.

## Alternatives Considered
1. **Client-side filtering**: Rejected - would still load all data
2. **Separate filtered endpoint**: Rejected - adds complexity
3. **Configuration file**: Rejected - environment variable simpler

## Code Changes
- Modified: `src/core/beverly_comprehensive_erp.py` lines 12860-12962
- Added: Production relevance filtering logic (~50 LOC)
- Added: Feature flag check for compatibility

## Validation
Testing confirmed:
- With filtering: 4 yarns shown (all critical)
- Without filtering: 20+ yarns shown
- All critical yarns still appear in filtered view
- No performance impact

## Follow Operating Charter
- **Less is More**: Minimal code change (~50 LOC)
- **Document Everything**: This ADR captures decision
- **Check Before Create**: Reused existing BOM/KO data
- **Phase Gate Reviews**: Tested with/without filtering
- **Plan Before Act**: Planned approach before implementation