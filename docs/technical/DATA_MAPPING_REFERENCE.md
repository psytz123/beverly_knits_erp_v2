# Beverly Knits ERP - Data Mapping & Flow Reference

**Last Updated**: September 6, 2025  
**System Version**: Beverly Knits ERP v2

## Overview

This comprehensive reference documents all data flows, field mappings, and column standardizations in the Beverly Knits ERP system. The system integrates multiple data sources to manage textile manufacturing operations from sales to production to inventory management.

---

## Primary Data Directory Structure

**Current Path**: `/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/`

**Latest Data Subset**: `8-28-2025/` (most current data files)

### Core Data Files
```
ERP Data/
├── 8-28-2025/                              # Latest production data
│   ├── eFab_Knit_Orders.csv               # Production orders (194 total: 154 assigned, 40 unassigned)
│   ├── eFab_SO_List.csv                   # Sales orders
│   ├── yarn_inventory.csv                 # Current yarn inventory (1,199 items)
│   ├── eFab_Inventory_F01.csv             # Finished goods inventory
│   ├── eFab_Inventory_G00.csv             # Greige goods inventory  
│   ├── eFab_Inventory_G02.csv             # Secondary greige inventory
│   ├── eFab_Inventory_I01.csv             # Inspection queue
│   ├── Yarn_Demand.csv                    # Yarn demand forecast
│   ├── Yarn_Demand_By_Style.csv           # Style-based yarn demand
│   └── Expected_Yarn_Report.csv           # Expected yarn deliveries
├── BOM_updated.csv                         # Bill of Materials (28,653 entries)
├── QuadS_finishedFabricList.csv           # Fabric specifications & work center mappings
├── Machine Report fin1.csv                # Machine to Work Center mappings
├── Sales Activity Report.csv              # Historical sales data (10,338+ records)
├── Yarn_ID_Master.csv                     # Master yarn database
├── Supplier_ID.csv                        # Supplier master data
└── styles_master.csv                      # Master style reference
```

---

## Column Standardization & Mappings

### Yarn Identifier Standardization

**Standard Column**: `Desc#` (Universal yarn identifier)

| Data Source | Original Column | Maps To | Notes |
|------------|----------------|---------|-------|
| **BOM_updated** | `Desc#` | `Desc#` | ✅ Already standard |
| **Yarn_ID_Master** | `Desc#` | `Desc#` | ✅ Already standard |
| **yarn_inventory** | `Desc#` | `Desc#` | ✅ Already standard |
| **Expected_Yarn_Report** | `Desc` | `Desc#` | Maps Desc → Desc# |
| **Yarn_Demand** | `Yarn` | `Desc#` | Maps Yarn → Desc# |
| **Yarn_Demand_By_Style** | `Yarn` | `Desc#` | Maps Yarn → Desc# |

#### Common Yarn Column Variations Handled
- `Yarn_ID`, `YarnID`, `Yarn ID` → `Desc#`
- `Description`, `desc#`, `desc_num` → `Desc#`
- `Yarn` (in demand reports) → `Desc#`
- `Desc` (in expected reports) → `Desc#`

### Style/Fabric Identifier Standardization

**Primary Columns**:
- `Style#` - Standard production style identifier
- `fStyle#` - Fabric style identifier (specific contexts)

| Data Source | Style Column | Maps To | Context |
|------------|--------------|---------|---------|
| **eFab_Knit_Orders** | `Style #` | `Style#` | Production orders |
| **eFab_SO_List** | `cFVersion` + `fBase` | Combined | Sales orders |
| **eFab_Inventory_F01** | `Style #` | `Style#` | Finished goods |
| **eFab_Inventory_G00** | `Style #` | `Style#` | Greige stage 1 |
| **eFab_Inventory_G02** | `fStyle` | `fStyle#` | Greige stage 2 |
| **eFab_Inventory_I01** | `Style #` | `Style#` | QC inspection |
| **QuadS_finishedFabric** | `Style#` | `Style#` | Fabric specs |
| **BOM_updated** | `Style#` | `Style#` | Bill of materials |
| **Sales Activity Report** | `Style` | `cFVersion` | ⚠️ Special mapping |

#### Style Column Variations Handled
- `Style #`, `Style Number` → `Style#`
- `fStyle`, `Fabric Style` → `fStyle#`
- Remove "Style " prefix, standardize spacing

### Critical Column Names (Planning Balance)

**⚠️ Important**: The planning balance column appears with variations:
- **Correct**: `Planning Balance` (with space) - Used in actual data
- **System handles**: Both `Planning Balance` and `Planning_Balance`
- **Formula**: `Planning Balance = Theoretical Balance + Allocated + On Order`

---

## Data Flow Mappings

### Production Flow Stages
```
Sales → SO List → Production Orders → Machine Assignment → Inventory Tracking

G00 (Greige) → G02 (Greige Stage 2) → I01 (QC Inspection) → F01 (Finished Goods)
```

### Work Center & Machine Mapping

**Work Center Pattern**: `x.xx.xx.X`
- **First digit**: Knit construction
- **Second pair**: Machine diameter  
- **Third pair**: Needle cut
- **Letter**: Type (F/M/C/V etc.)
- **Example**: `9.38.20.F` = construction 9, diameter 38, needle 20, type F

**Machine Mapping Process**:
1. `QuadS_finishedFabricList.csv` - Style to Work Center mapping (columns C=style, D=work_center)
2. `Machine Report fin1.csv` - Work Center to Machine mapping (WC=pattern, MACH=machine IDs)
3. **91 Work Centers** with **285 total machines**

### Yarn Flow Tracking
```
BOM Requirements → Yarn Inventory → Demand Forecasting → Shortage Analysis

BOM (Desc#) → Inventory (Desc#) → Demand Reports (Yarn → Desc#)
```

---

## System Integration Points

### Critical Relationships

1. **Style Production Planning**
   ```
   Sales Activity (Style) → QuadS Mapping (Style# → Work Center) → Machine Assignment
   ```

2. **Yarn Requirement Calculation**
   ```
   Production Orders (Style#) → BOM Lookup (Style# + Desc#) → Inventory Check (Desc#)
   ```

3. **Inventory Shortage Detection**
   ```
   Planning Balance < 0 OR Theoretical Balance < 0 = Yarn Shortage
   ```

### Data Consistency Rules

**Implemented in**: `src/data_consistency/consistency_manager.py`

1. **Column Alias System**: Handles all column name variations automatically
2. **Unified Shortage Calculation**: Consistent logic using both Planning and Theoretical Balance
3. **Standardized Risk Levels**:
   - **CRITICAL**: -1000+ lbs shortage
   - **HIGH**: -500+ lbs shortage  
   - **MEDIUM**: -100+ lbs shortage
   - **LOW**: <0 lbs shortage

---

## Processing Pipeline

### Data Loading Sequence
1. **Path Resolution**: Dynamic path finder locates data files in multiple locations
2. **Column Mapping**: Apply standardized column mappings based on file type
3. **Data Validation**: Validate data types and business rules
4. **Consistency Check**: Ensure cross-file data consistency
5. **Cache Storage**: Store processed data with TTL for performance

### File Identification Patterns
- **Style/Fabric Files**: Contains `eFab_`, `QuadS_`, `BOM`, `Sales Activity`
- **Yarn Files**: Contains `yarn_inventory`, `Yarn_ID`, `Yarn_Demand`, `Expected_Yarn`
- **Machine Files**: Contains `Machine Report`, `QuadS`

---

## API Data Access

### Critical Endpoints
- `/api/inventory-intelligence-enhanced` - Uses standardized Planning Balance calculation
- `/api/yarn-intelligence` - Applies consistent shortage detection logic
- `/api/production-planning` - Uses standardized style and work center mappings
- `/api/machine-assignment-suggestions` - Uses QuadS style-to-work-center mappings

### Parameter Support
```
GET /api/inventory-intelligence-enhanced?view=summary&analysis=shortage&realtime=true
GET /api/yarn-intelligence?analysis=shortage&forecast=true
GET /api/production-planning?view=orders&forecast=true
```

---

## Troubleshooting Data Issues

### Common Column Issues
1. **Column not found**: Check both `Column Name` and `Column_Name` variations
2. **Planning Balance errors**: System handles both space and underscore versions
3. **Style mapping failures**: Verify both `Style#` and `fStyle#` patterns
4. **Yarn ID mismatches**: All yarn references should use `Desc#`

### Validation Commands
```bash
# Check data consistency
curl http://localhost:5006/api/data-consistency-check

# Reload data with fresh mappings
curl http://localhost:5006/api/reload-data

# Check API health
curl http://localhost:5006/api/comprehensive-kpis
```

### Emergency Fixes Available
**Script**: `scripts/day0_emergency_fixes.py`
- **Dynamic Path Resolution**: Finds data files automatically
- **Column Alias System**: Maps all column variations
- **Data Validation**: Comprehensive checks with fallbacks

---

## Data Quality Benefits

1. **Consistency**: Same identifiers across entire pipeline
2. **Traceability**: Track items from sales through production to inventory
3. **Integration**: Seamless joins between all data sources  
4. **Automation**: No manual column renaming required
5. **Validation**: Early detection of data quality issues
6. **Performance**: Optimized with caching and parallel loading

---

## Current System Statistics

- **Yarn Items Tracked**: 1,199
- **BOM Entries**: 28,653 (style-to-yarn mappings)
- **Production Orders**: 194 (154 assigned to machines, 40 pending)
- **Work Centers**: 91 with 285 total machines
- **Total Production Workload**: 557,671 lbs
- **Sales Records**: 10,338+ for forecasting

---

**Note**: This mapping system ensures all data files use consistent column names regardless of source variations, enabling reliable data integration across the entire ERP system.