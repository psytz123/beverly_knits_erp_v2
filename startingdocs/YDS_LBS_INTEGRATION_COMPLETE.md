# Yds/Lbs Integration Complete

## Summary
Successfully integrated updated Yds/Lbs conversion rates from QuadS_finishedFabricList_ (2) (1).csv into the Production dashboard.

## Data Source
Using updated conversion rates from:
- **Primary**: `/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5/QuadS_finishedFabricList_ (2) (1).csv`
- **Backup**: `/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/BKI_comp/Docs/DATA/QuadS_finishedFabricList_ (2) (1).xlsx`

## Updated Yds/Lbs Values

### Sample Conversion Rates (Updated):
| Style# | Previous | **Updated** | GSM | Width |
|--------|----------|-------------|-----|-------|
| 6191-BK | 0.63 | **1.58** yds/lb | 187 | 66" |
| 80393C-DS | 0.91 | **1.10** yds/lb | 250 | 70" |
| 72762-GS | 0.73 | **1.36** yds/lb | 245 | 63" |
| 72762-DS | 0.73 | **1.36** yds/lb | 245 | 63" |
| 71320-BK | 0.65 | **1.55** yds/lb | 185 | 69" |

### Statistics:
- **Min**: 0.14 yds/lb
- **Max**: 3.67 yds/lb  
- **Mean**: 1.31 yds/lb
- **Median**: 1.29 yds/lb

## Dashboard Changes

### 1. Added Yds/Lbs Column
The forecast netting table now displays:
- **Style#**: Fabric style code
- **Customer**: Customer name
- **Fabric Type**: GSM and width specifications
- **→ Yds/Lbs**: Conversion rate (highlighted in purple)
- **Order Qty**: Shows both yards AND pounds
- **Delivery**: Delivery date
- **Forecast**: Forecasted quantity
- **On Order**: Quantity on order
- **Net Position**: Supply vs demand
- **Coverage**: Days of coverage
- **Status**: ON TRACK, TIGHT, CRITICAL

### 2. Dual Unit Display
Order quantities now show both units:
```
10,106 yds
6,395 lbs
```

### 3. Live Data Examples

| Style | Customer | Yds/Lbs | Order (yds/lbs) | Status |
|-------|----------|---------|-----------------|--------|
| 6191-BK | Zonkd Limitada | 1.58 | 10,106 / 6,395 | ON TRACK |
| 72762-GS | Reliefmart | 1.36 | 11,535 / 8,481 | ON TRACK |
| 71320-BK | Behrens | 1.55 | 2,924 / 1,887 | TIGHT |

## Calculation Formula

For each fabric order:
```
Pounds = Yards ÷ Yds/Lbs

Example for 6191-BK:
10,106 yards ÷ 1.58 yds/lb = 6,395 lbs
```

## How to View

1. Open **http://localhost:5005/**
2. Click **"Production"** tab
3. Look for the **purple Yds/Lbs column**
4. See dual units (yards/pounds) in Order Qty column

## Technical Implementation

### JavaScript Updates:
```javascript
// Each fabric item now includes:
{
  "style": "6191-BK",
  "yds_lbs": 1.58,        // Conversion rate
  "order_qty": 10106,     // Yards
  "order_lbs": 6395,      // Calculated pounds
  ...
}
```

### Display Format:
- Yds/Lbs shown in purple for visibility
- Order quantity shows yards with pounds below in gray
- All calculations use updated conversion rates

## Benefits

1. **Accurate Conversions**: Using real Yds/Lbs from QuadS data
2. **Dual Unit Visibility**: See both yards and pounds
3. **Better Planning**: Accurate weight calculations for shipping
4. **Inventory Management**: Proper conversion for yarn requirements

## Files Updated

1. **consolidated_dashboard.html** - Added Yds/Lbs display
2. **update_dashboard_with_yds_lbs.py** - Script to generate live data
3. **Live data embedded** with 10 fabric styles and conversion rates

---

**Implementation Date**: 2025-08-16  
**Data Source**: QuadS_finishedFabricList_ (2) (1).csv  
**Total Fabrics**: 3,481 with Yds/Lbs values  
**Dashboard URL**: http://localhost:5005/