# Time-Phased PO Delivery Integration Plan

## Executive Summary

This document outlines the plan to integrate time-phased Purchase Order (PO) delivery schedules into the Beverly Knits ERP system, addressing a critical gap where the system currently shows only total "On Order" quantities without delivery timing information.

## Current State Analysis

### Manual/CSV Process (Working)
- **Data Sources**: 
  - `Yarn_Demand_By_Style_KO.csv` - Production demand by style
  - `Expected_Yarn_Report.xlsx` - PO delivery schedules with weekly buckets
  - `Yarn_Demand.csv` - Consolidated report
- **Coverage**: 184 yarns with 9-week time horizon
- **Strengths**: Weekly visibility, accurate shortage timing prediction
- **Weaknesses**: Manual process, limited yarn coverage, point-in-time snapshot

### ERP System (Gap)
- **Current Capability**: 
  - Shows total "On Order" as single value
  - Formula: `Planning Balance = Theoretical + On Order + Allocated`
  - Real-time updates with 1-2 second cache
- **Missing**: Time-phased receipt visibility
- **Impact**: Cannot predict WHEN shortages occur, only IF they exist

### Business Impact Example

**Yarn 18884 Case Study:**
- Total On Order: 36,161 lbs
- Total Demand: 30,860 lbs
- **Without Timing**: Appears sufficient (36,161 > 30,860)
- **With Timing Reality**:
  - Past Due Available: 20,161 lbs
  - Week 43 Delivery: 4,000 lbs
  - Week 44 Delivery: 4,000 lbs
  - Later: 8,000 lbs
  - **Result**: SHORTAGE in Weeks 37-42 despite having POs

## Integration Architecture

### Data Flow
```
Expected_Yarn_Report.xlsx (PO Deliveries)
           +
eFab_Knit_Orders.csv (Production Demand)
           +
BOM_updated.csv (Yarn Requirements)
           ↓
    [Time-Phased Planning Engine]
           ↓
    Weekly Planning Balance
           ↓
    Shortage Timeline Prediction
```

## Implementation Plan

### Phase 1: Data Integration Layer (Week 1)

#### 1.1 PO Delivery Loader Module
**New File**: `src/data_loaders/po_delivery_loader.py`

```python
class PODeliveryLoader:
    """
    Loads and processes PO delivery schedules with time buckets
    """
    def __init__(self):
        self.delivery_columns = [
            'Unscheduled or Past Due',
            'This Week',
            '9/12/2025', '9/19/2025', '9/26/2025',
            '10/3/2025', '10/10/2025', '10/17/2025',
            '10/24/2025', '10/31/2025',
            'Later'
        ]
    
    def load_po_deliveries(self, file_path):
        """Load Expected_Yarn_Report.xlsx with delivery timing"""
        pass
    
    def map_to_weekly_buckets(self, po_data):
        """Convert date columns to week numbers"""
        pass
    
    def aggregate_by_yarn(self, po_data):
        """Group PO deliveries by yarn with weekly totals"""
        pass
```

#### 1.2 Data Model Extension
**Modified File**: `src/core/beverly_comprehensive_erp.py`

Add to yarn data structure:
```python
yarn_data = {
    'yarn_id': 18884,
    'theoretical_balance': 2506.18,
    'on_order': 36161.30,  # Keep for backward compatibility
    'on_order_weekly': {    # NEW: Time-phased receipts
        'past_due': 20161.30,
        'week_36': 0,
        'week_37': 0,
        'week_38': 0,
        'week_39': 0,
        'week_40': 0,
        'week_41': 0,
        'week_42': 0,
        'week_43': 4000,
        'week_44': 4000,
        'later': 8000
    },
    'allocated': -30859.80,
    'planning_balance': 7807.68
}
```

### Phase 2: Time-Phased Calculation Engine (Week 1-2)

#### 2.1 Weekly Planning Calculator
**New File**: `src/production/time_phased_planning.py`

```python
class TimePhasedPlanning:
    """
    Calculates weekly planning balance and shortage timeline
    """
    
    def calculate_weekly_balance(self, yarn_id, start_week=36, horizon=9):
        """
        Calculate rolling balance for each week
        Balance[Week N] = Balance[Week N-1] + Receipts[N] - Demand[N]
        """
        pass
    
    def identify_shortage_periods(self, weekly_balances):
        """
        Find weeks where balance < 0
        Return: [(week_num, shortage_amount, recovery_week)]
        """
        pass
    
    def calculate_expedite_requirements(self, shortage_timeline):
        """
        Determine which POs need expediting to prevent shortages
        """
        pass
```

#### 2.2 Integration with Existing Systems
**Modified**: `src/core/beverly_comprehensive_erp.py`

```python
def calculate_yarn_shortages_enhanced(self):
    """Enhanced shortage detection with timing"""
    shortages = []
    
    for yarn in self.yarn_data:
        # Existing calculation
        current_shortage = yarn['planning_balance'] < 0
        
        # NEW: Time-phased analysis
        weekly_balance = self.calculate_weekly_balance(yarn)
        shortage_weeks = [w for w, bal in weekly_balance.items() if bal < 0]
        
        shortage_data = {
            'yarn_id': yarn['yarn_id'],
            'current_shortage': current_shortage,
            'shortage_timeline': shortage_weeks,  # NEW
            'first_shortage_week': min(shortage_weeks) if shortage_weeks else None,
            'recovery_week': self.find_recovery_week(weekly_balance),
            'expedite_needed': len(shortage_weeks) > 0
        }
        shortages.append(shortage_data)
    
    return shortages
```

### Phase 3: API Enhancement (Week 2)

#### 3.1 New Endpoints
**New File**: `src/api/blueprints/time_phased_bp.py`

```python
@bp.route('/api/yarn-shortage-timeline')
def yarn_shortage_timeline():
    """
    Returns weekly shortage progression for all yarns
    Response format:
    {
        "yarns": {
            "18884": {
                "weekly_balance": {
                    "week_36": -6349.71,
                    "week_37": -14943.50,
                    "week_38": -20355.43,
                    "week_43": 4000,  # Recovery
                },
                "shortage_weeks": [36, 37, 38, 39, 40, 41, 42],
                "recovery_week": 43
            }
        }
    }
    """
    pass

@bp.route('/api/po-delivery-schedule')
def po_delivery_schedule():
    """
    Returns PO receipt timing by yarn
    """
    pass

@bp.route('/api/time-phased-planning')
def time_phased_planning():
    """
    Complete weekly planning view with demand and receipts
    """
    pass
```

#### 3.2 Enhanced Existing Endpoints
**Modified**: `src/api/blueprints/yarn_bp.py`

```python
@bp.route('/api/yarn-intelligence')
def yarn_intelligence_enhanced():
    include_timing = request.args.get('include_timing', 'false') == 'true'
    
    response = existing_yarn_intelligence()
    
    if include_timing:
        # Add time-phased data
        for yarn in response['yarn_analysis']:
            yarn['next_receipt_week'] = get_next_receipt_week(yarn['yarn_id'])
            yarn['weeks_until_receipt'] = calculate_weeks_until_receipt(yarn['yarn_id'])
            yarn['shortage_timeline'] = get_shortage_timeline(yarn['yarn_id'])
    
    return response
```

### Phase 4: ML Forecast Integration (Week 2-3)

#### 4.1 Forecast-Driven Planning
**Modified**: `src/forecasting/enhanced_forecasting_engine.py`

```python
def generate_forecasted_demand_weekly(self, yarn_id, horizon_weeks=13):
    """
    Extend demand forecast beyond confirmed orders
    Weeks 1-9: Actual orders
    Weeks 10-13: ML forecast
    """
    actual_demand = self.get_actual_weekly_demand(yarn_id, weeks=9)
    forecasted_demand = self.ml_forecast_weekly(yarn_id, weeks=4)
    
    return {
        'confirmed': actual_demand,
        'forecasted': forecasted_demand,
        'confidence': self.calculate_forecast_confidence()
    }
```

#### 4.2 Predictive PO Generation
**New**: `src/production/po_recommendation_engine.py`

```python
class PORecommendationEngine:
    """
    Generates optimal PO timing recommendations
    """
    
    def calculate_reorder_point(self, yarn_id):
        """
        ROP = (Lead Time Demand) + Safety Stock
        """
        pass
    
    def recommend_po_timing(self, yarn_id):
        """
        Based on:
        - Current shortage timeline
        - Lead times
        - Forecasted demand
        - Safety stock requirements
        """
        pass
```

### Phase 5: Dashboard Integration (Week 3)

#### 5.1 Time-Phased View Tab
**Modified**: `web/consolidated_dashboard.html`

```javascript
// New tab for time-phased planning
function renderTimePhasedView() {
    const weekColumns = ['Current', 'W37', 'W38', 'W39', 'W40', 'W41', 'W42', 'W43', 'W44', 'Later'];
    
    // Create weekly grid
    data.forEach(yarn => {
        const row = createYarnRow(yarn);
        
        weekColumns.forEach(week => {
            const balance = yarn.weekly_balance[week];
            const cell = createBalanceCell(balance);
            
            // Color coding
            if (balance < 0) {
                cell.classList.add('shortage');  // Red
            } else if (balance < yarn.safety_stock) {
                cell.classList.add('low');       // Yellow
            } else {
                cell.classList.add('ok');        // Green
            }
            
            row.appendChild(cell);
        });
    });
}
```

#### 5.2 Enhanced Yarn Intelligence Display
Add columns:
- "Weeks Until Shortage"
- "Next Receipt Date"
- "Coverage Weeks"
- "Expedite Required"

### Phase 6: Testing & Validation (Week 3-4)

#### 6.1 Unit Tests
**New File**: `tests/test_time_phased_planning.py`

```python
def test_weekly_balance_calculation():
    """Verify weekly balance matches manual Excel"""
    pass

def test_shortage_timeline_detection():
    """Confirm shortage weeks identified correctly"""
    pass

def test_po_delivery_aggregation():
    """Validate PO receipts sum to correct totals"""
    pass
```

#### 6.2 Integration Tests
```python
def test_manual_excel_comparison():
    """
    Compare all 184 yarns with manual calculation
    Tolerance: < 0.01 lbs difference
    """
    pass

def test_api_performance():
    """
    Ensure < 2 second response with time-phased data
    """
    pass
```

## Success Metrics

### Quantitative Metrics
- ✅ **Accuracy**: 100% match with manual Excel calculations
- ✅ **Performance**: <2 second API response time
- ✅ **Coverage**: All 184 priority yarns, expandable to 1,199
- ✅ **Horizon**: 9-week confirmed + 4-week forecast (13 weeks total)
- ✅ **Reduction**: 30% fewer false shortage alerts

### Qualitative Metrics
- ✅ **Visibility**: Clear view of WHEN shortages occur
- ✅ **Actionability**: Specific POs to expedite identified
- ✅ **Proactivity**: 9-week advance warning of shortages
- ✅ **Integration**: Seamless with existing ERP workflows

## Risk Mitigation

### Technical Risks
1. **Performance Impact**
   - Mitigation: Implement aggressive caching for weekly calculations
   - Cache TTL: 5 minutes for time-phased data

2. **Data Quality**
   - Mitigation: Validate PO dates, handle missing/invalid entries
   - Fallback: Use "Later" bucket for unparseable dates

3. **Backward Compatibility**
   - Mitigation: Keep existing "On Order" field
   - Add new fields without breaking existing APIs

### Business Risks
1. **User Adoption**
   - Mitigation: Parallel run with manual process for 2 weeks
   - Training: Create user guide with examples

2. **Data Synchronization**
   - Mitigation: Daily validation against source systems
   - Alert: Email if discrepancies > 1%

## Implementation Timeline

### Week 1: Foundation
- ✓ Create PO delivery loader
- ✓ Extend data model
- ✓ Basic weekly calculation engine

### Week 2: Integration
- ✓ API endpoints
- ✓ ML forecast integration
- ✓ Enhanced shortage detection

### Week 3: User Interface
- ✓ Dashboard updates
- ✓ Testing suite
- ✓ Documentation

### Week 4: Deployment
- ✓ UAT with business users
- ✓ Performance optimization
- ✓ Production deployment

## Conclusion

This integration will transform the ERP system from showing static "On Order" totals to providing dynamic, time-phased material planning. By incorporating PO delivery schedules, the system will match and exceed the capabilities of the current manual Excel process while maintaining real-time data accuracy and automated calculations.

The result: **Proactive shortage prevention** instead of reactive shortage discovery.